import json
import logging
import os
import secrets
from uuid import uuid4

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import text
from pywebpush import WebPushException, webpush

logger = logging.getLogger("medimind.push")


class PushSubscription(BaseModel):
    endpoint: str = Field(min_length=1, max_length=2048)
    expirationTime: int | None = None
    keys: dict[str, str]


def create_push_router(engine, get_current_user):
    router = APIRouter(prefix="/api/v1/push", tags=["Desktop push notifications"])
    with engine.begin() as connection:
        connection.execute(text("""
            CREATE TABLE IF NOT EXISTS reminder_push_subscriptions (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                user_id UUID NOT NULL,
                endpoint TEXT NOT NULL UNIQUE,
                subscription JSONB NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
        """))
        connection.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_push_subscriptions_user
            ON reminder_push_subscriptions (user_id)
        """))

    @router.post("/subscribe", status_code=204)
    def subscribe(payload: PushSubscription, current_user: dict = Depends(get_current_user)):
        with engine.begin() as connection:
            connection.execute(text("""
                INSERT INTO reminder_push_subscriptions (user_id, endpoint, subscription)
                VALUES (CAST(:user_id AS UUID), :endpoint, CAST(:subscription AS JSONB))
                ON CONFLICT (endpoint) DO UPDATE
                SET user_id = EXCLUDED.user_id, subscription = EXCLUDED.subscription
            """), {
                "user_id": current_user["id"],
                "endpoint": payload.endpoint,
                "subscription": json.dumps(payload.model_dump(by_alias=True)),
            })
        return None

    @router.delete("/subscribe", status_code=204)
    def unsubscribe(payload: PushSubscription, current_user: dict = Depends(get_current_user)):
        with engine.begin() as connection:
            connection.execute(text("""
                DELETE FROM reminder_push_subscriptions
                WHERE user_id = CAST(:user_id AS UUID) AND endpoint = :endpoint
            """), {"user_id": current_user["id"], "endpoint": payload.endpoint})
        return None

    return router


def create_push_sender_router(engine):
    router = APIRouter(prefix="/api/v1/internal", tags=["Internal jobs"])

    @router.post("/send-reminder-push")
    def send_reminder_push(authorization: str | None = Header(default=None)):
        secret = os.getenv("REMINDER_CRON_SECRET", "")
        supplied = (authorization or "").removeprefix("Bearer ").strip()
        if not secret or not secrets.compare_digest(supplied, secret):
            raise HTTPException(status_code=401, detail="Unauthorized.")
        public_key = os.getenv("VAPID_PUBLIC_KEY", "")
        private_key = os.getenv("VAPID_PRIVATE_KEY", "")
        subject = os.getenv("VAPID_SUBJECT", "")
        if not public_key or not private_key or not subject:
            raise HTTPException(status_code=503, detail="Web push is not configured.")

        with engine.connect() as connection:
            due = connection.execute(text("""
                SELECT r.id AS reminder_id, r.medication_name, r.dosage,
                       r.instructions, r.user_id, subs.subscriptions,
                       occurrence.scheduled_for
                FROM medication_reminders r
                CROSS JOIN LATERAL (
                    SELECT jsonb_agg(jsonb_build_object(
                        'endpoint', s.endpoint,
                        'subscription', s.subscription
                    )) AS subscriptions
                    FROM reminder_push_subscriptions s
                    WHERE s.user_id = r.user_id
                ) subs
                CROSS JOIN LATERAL (
                    SELECT ((day_value::date + r.reminder_time) AT TIME ZONE r.timezone) AS scheduled_for
                    FROM generate_series(
                        GREATEST(r.start_date, (NOW() AT TIME ZONE r.timezone)::date - 30)::timestamp,
                        LEAST(r.end_date, (NOW() AT TIME ZONE r.timezone)::date)::timestamp,
                        INTERVAL '1 day'
                    ) AS day_value
                    WHERE ((day_value::date + r.reminder_time) AT TIME ZONE r.timezone) <= NOW()
                    ORDER BY scheduled_for DESC
                    LIMIT 1
                ) occurrence
                WHERE r.enabled = TRUE
                  AND occurrence.scheduled_for > NOW() - INTERVAL '31 days'
                  AND subs.subscriptions IS NOT NULL
                  AND NOT EXISTS (
                    SELECT 1 FROM reminder_deliveries d
                    WHERE d.reminder_id = r.id
                      AND d.scheduled_for = occurrence.scheduled_for
                      AND d.delivered_at IS NOT NULL
                  )
                ORDER BY occurrence.scheduled_for
                LIMIT 500
            """)).mappings().all()

        sent = 0
        skipped = 0
        for item in due:
            token = str(uuid4())
            with engine.begin() as connection:
                claim = connection.execute(text("""
                    INSERT INTO reminder_deliveries (reminder_id, scheduled_for, claim_token, lease_expires_at)
                    VALUES (:reminder_id, :scheduled_for, :token, NOW() + INTERVAL '3 minutes')
                    ON CONFLICT (reminder_id, scheduled_for) DO UPDATE
                    SET claim_token = EXCLUDED.claim_token,
                        lease_expires_at = EXCLUDED.lease_expires_at
                    WHERE reminder_deliveries.delivered_at IS NULL
                      AND (reminder_deliveries.lease_expires_at IS NULL
                           OR reminder_deliveries.lease_expires_at < NOW())
                    RETURNING claim_token
                """), {
                    "reminder_id": item["reminder_id"],
                    "scheduled_for": item["scheduled_for"],
                    "token": token,
                }).first()
            if not claim:
                skipped += 1
                continue

            body = {
                "title": f"Medication reminder: {item['medication_name']}",
                "body": " ".join(part for part in [
                    item["dosage"] or "",
                    item["instructions"] or "",
                ] if part) or "It is time to take your scheduled medication.",
                "url": "/",
            }
            successful_delivery = False
            for subscription_record in item["subscriptions"]:
                try:
                    body["tag"] = f"medimind-{item['reminder_id']}-{item['scheduled_for'].isoformat()}"
                    webpush(
                        subscription_info=subscription_record["subscription"],
                        data=json.dumps(body),
                        vapid_private_key=private_key,
                        vapid_claims={"sub": subject},
                        ttl=86400,
                    )
                    successful_delivery = True
                except WebPushException as error:
                    status = getattr(getattr(error, "response", None), "status_code", None)
                    if status in (404, 410):
                        with engine.begin() as connection:
                            connection.execute(text("""
                                DELETE FROM reminder_push_subscriptions WHERE endpoint = :endpoint
                            """), {"endpoint": subscription_record["endpoint"]})
                    logger.warning("Push delivery failed with status %s", status)
            with engine.begin() as connection:
                if successful_delivery:
                    connection.execute(text("""
                        UPDATE reminder_deliveries
                        SET delivered_at = NOW(), lease_expires_at = NULL
                        WHERE reminder_id = :reminder_id
                          AND scheduled_for = :scheduled_for
                          AND claim_token = :token
                    """), {
                        "reminder_id": item["reminder_id"],
                        "scheduled_for": item["scheduled_for"],
                        "token": token,
                    })
                    sent += 1
                else:
                    connection.execute(text("""
                        UPDATE reminder_deliveries
                        SET lease_expires_at = NOW()
                        WHERE reminder_id = :reminder_id
                          AND scheduled_for = :scheduled_for
                          AND claim_token = :token
                    """), {
                        "reminder_id": item["reminder_id"],
                        "scheduled_for": item["scheduled_for"],
                        "token": token,
                    })
        return {"sent": sent, "skipped": skipped}

    return router

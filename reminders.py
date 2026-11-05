from datetime import date, datetime, time
from typing import Optional
from uuid import UUID, uuid4
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import text


class ReminderCreate(BaseModel):
    medication_name: str = Field(min_length=1, max_length=255)
    dosage: str = Field(default="", max_length=255)
    instructions: str = Field(default="", max_length=4000)
    reminder_time: time
    start_date: date
    end_date: date
    timezone: str = Field(default="Asia/Kolkata", min_length=1, max_length=100)
    prescription_id: Optional[int] = None

    @field_validator("medication_name")
    @classmethod
    def clean_medication_name(cls, value):
        value = value.strip()
        if not value:
            raise ValueError("Medication name cannot be blank.")
        return value

    @field_validator("timezone")
    @classmethod
    def validate_timezone(cls, value):
        try:
            ZoneInfo(value)
        except (ZoneInfoNotFoundError, ValueError):
            raise ValueError("Use a valid IANA timezone, such as Asia/Kolkata.")
        return value

    @field_validator("end_date")
    @classmethod
    def validate_date_range(cls, value, info):
        start = info.data.get("start_date")
        if start and value < start:
            raise ValueError("End date cannot be before start date.")
        return value


class ReminderUpdate(BaseModel):
    enabled: bool


class ReminderEdit(BaseModel):
    medication_name: str = Field(min_length=1, max_length=255)
    dosage: str = Field(default="", max_length=255)
    instructions: str = Field(default="", max_length=4000)
    reminder_time: time
    start_date: date
    end_date: date
    timezone: str = Field(default="Asia/Kolkata", min_length=1, max_length=100)
    enabled: bool = True

    @field_validator("medication_name")
    @classmethod
    def clean_medication_name(cls, value):
        value = value.strip()
        if not value:
            raise ValueError("Medication name cannot be blank.")
        return value

    @field_validator("timezone")
    @classmethod
    def validate_timezone(cls, value):
        try:
            ZoneInfo(value)
        except (ZoneInfoNotFoundError, ValueError):
            raise ValueError("Use a valid IANA timezone, such as Asia/Kolkata.")
        return value

    @field_validator("end_date")
    @classmethod
    def validate_date_range(cls, value, info):
        start = info.data.get("start_date")
        if start and value < start:
            raise ValueError("End date cannot be before start date.")
        return value


class ReminderClaim(BaseModel):
    scheduled_for: datetime

    @field_validator("scheduled_for")
    @classmethod
    def require_timezone(cls, value):
        if value.tzinfo is None:
            raise ValueError("scheduled_for must include a timezone.")
        return value


def create_reminder_router(engine, get_current_user):
    router = APIRouter(
        prefix="/api/v1/reminders",
        tags=["Medication reminders"],
    )

    @router.post("", status_code=201)
    def create_reminder(
        payload: ReminderCreate,
        current_user: dict = Depends(get_current_user),
    ):
        if payload.prescription_id is not None:
            with engine.connect() as connection:
                owned = connection.execute(
                    text("""
                        SELECT id
                        FROM prescriptions
                        WHERE id = :prescription_id
                          AND user_id = CAST(:user_id AS UUID)
                    """),
                    {
                        "prescription_id": payload.prescription_id,
                        "user_id": current_user["id"],
                    },
                ).first()

            if not owned:
                raise HTTPException(
                    status_code=404,
                    detail="Prescription not found.",
                )

        with engine.begin() as connection:
            row = connection.execute(
                text("""
                    INSERT INTO medication_reminders (
                        user_id,
                        prescription_id,
                        medication_name,
                        dosage,
                        instructions,
                        reminder_time,
                        start_date,
                        end_date,
                        timezone
                    )
                    VALUES (
                        CAST(:user_id AS UUID),
                        :prescription_id,
                        :medication_name,
                        :dosage,
                        :instructions,
                        :reminder_time,
                        :start_date,
                        :end_date,
                        :timezone
                    )
                    RETURNING
                        id,
                        medication_name,
                        dosage,
                        instructions,
                        reminder_time,
                        start_date,
                        end_date,
                        timezone,
                        enabled,
                        prescription_id,
                        created_at
                """),
                {
                    "user_id": current_user["id"],
                    "prescription_id": payload.prescription_id,
                    "medication_name": payload.medication_name,
                    "dosage": payload.dosage,
                    "instructions": payload.instructions,
                    "reminder_time": payload.reminder_time,
                    "start_date": payload.start_date,
                    "end_date": payload.end_date,
                    "timezone": payload.timezone,
                },
            ).mappings().one()

        return dict(row)

    @router.get("")
    def list_reminders(
        current_user: dict = Depends(get_current_user),
    ):
        with engine.connect() as connection:
            rows = connection.execute(
                text("""
                    SELECT
                        id,
                        medication_name,
                        dosage,
                        instructions,
                        reminder_time,
                        start_date,
                        end_date,
                        timezone,
                        enabled,
                        prescription_id,
                        created_at
                    FROM medication_reminders
                    WHERE user_id = CAST(:user_id AS UUID)
                    ORDER BY
                        enabled DESC,
                        reminder_time,
                        medication_name
                """),
                {"user_id": current_user["id"]},
            ).mappings().all()

        return [dict(row) for row in rows]

    @router.patch("/{reminder_id}")
    def update_reminder(
        reminder_id: UUID,
        payload: ReminderUpdate,
        current_user: dict = Depends(get_current_user),
    ):
        with engine.begin() as connection:
            row = connection.execute(
                text("""
                    UPDATE medication_reminders
                    SET
                        enabled = :enabled,
                        updated_at = NOW()
                    WHERE id = :reminder_id
                      AND user_id = CAST(:user_id AS UUID)
                    RETURNING id, medication_name, enabled
                """),
                {
                    "enabled": payload.enabled,
                    "reminder_id": str(reminder_id),
                    "user_id": current_user["id"],
                },
            ).mappings().first()

        if not row:
            raise HTTPException(
                status_code=404,
                detail="Reminder not found.",
            )

        return dict(row)

    @router.put("/{reminder_id}")
    def edit_reminder(
        reminder_id: UUID,
        payload: ReminderEdit,
        current_user: dict = Depends(get_current_user),
    ):
        with engine.begin() as connection:
            row = connection.execute(
                text("""
                    UPDATE medication_reminders
                    SET
                        medication_name = :medication_name,
                        dosage = :dosage,
                        instructions = :instructions,
                        reminder_time = :reminder_time,
                        start_date = :start_date,
                        end_date = :end_date,
                        timezone = :timezone,
                        enabled = :enabled,
                        updated_at = NOW()
                    WHERE id = :reminder_id
                      AND user_id = CAST(:user_id AS UUID)
                    RETURNING
                        id,
                        medication_name,
                        dosage,
                        instructions,
                        reminder_time,
                        start_date,
                        end_date,
                        timezone,
                        enabled,
                        prescription_id,
                        created_at
                """),
                {
                    **payload.model_dump(),
                    "reminder_id": str(reminder_id),
                    "user_id": current_user["id"],
                },
            ).mappings().first()

        if not row:
            raise HTTPException(
                status_code=404,
                detail="Reminder not found.",
            )

        return dict(row)

    @router.delete("/{reminder_id}", status_code=204)
    def delete_reminder(
        reminder_id: UUID,
        current_user: dict = Depends(get_current_user),
    ):
        with engine.begin() as connection:
            result = connection.execute(
                text("""
                    DELETE FROM medication_reminders
                    WHERE id = :reminder_id
                      AND user_id = CAST(:user_id AS UUID)
                    RETURNING id
                """),
                {
                    "reminder_id": str(reminder_id),
                    "user_id": current_user["id"],
                },
            ).first()

        if not result:
            raise HTTPException(
                status_code=404,
                detail="Reminder not found.",
            )

        return None

    @router.get("/due")
    def get_due_reminders(
        current_user: dict = Depends(get_current_user),
    ):
        # Recover unhandled occurrences from the previous 30 local calendar
        # days, respecting each reminder's actual start and end dates.
        # This also allows recovery after the reminder's end date has passed.

        with engine.connect() as connection:
            rows = connection.execute(
                text("""
                    SELECT
                        r.id,
                        r.medication_name,
                        r.dosage,
                        r.instructions,
                        r.timezone,
                        occurrences.scheduled_for,
                        CASE
                            WHEN occurrences.scheduled_for >
                                 NOW() - INTERVAL '5 minutes'
                            THEN 'due'
                            ELSE 'missed'
                        END AS occurrence_status

                    FROM medication_reminders r

                    CROSS JOIN LATERAL (
                        SELECT
                            (
                                (day_value::date + r.reminder_time)
                                AT TIME ZONE r.timezone
                            ) AS scheduled_for

                        FROM generate_series(
                            GREATEST(
                                r.start_date,
                                (
                                    NOW() AT TIME ZONE r.timezone
                                )::date - 29
                            )::timestamp,

                            LEAST(
                                r.end_date,
                                (
                                    NOW() AT TIME ZONE r.timezone
                                )::date
                            )::timestamp,

                            INTERVAL '1 day'
                        ) AS day_value

                        WHERE (
                            (day_value::date + r.reminder_time)
                            AT TIME ZONE r.timezone
                        ) <= NOW()
                    ) AS occurrences

                    LEFT JOIN reminder_deliveries d
                      ON d.reminder_id = r.id
                     AND d.scheduled_for = occurrences.scheduled_for

                    WHERE r.user_id = CAST(:user_id AS UUID)
                      AND r.enabled = TRUE

                      AND (
                          d.reminder_id IS NULL

                          OR (
                              d.delivered_at IS NULL

                              AND (
                                  d.lease_expires_at IS NULL
                                  OR d.lease_expires_at < NOW()
                              )
                          )
                      )

                    ORDER BY occurrences.scheduled_for DESC

                    LIMIT 100
                """),
                {"user_id": current_user["id"]},
            ).mappings().all()

        return [dict(row) for row in rows]

    @router.post("/{reminder_id}/claim")
    def claim_delivery(
        reminder_id: UUID,
        payload: ReminderClaim,
        current_user: dict = Depends(get_current_user),
    ):
        token = uuid4()

        with engine.begin() as connection:
            owned = connection.execute(
                text("""
                    SELECT id
                    FROM medication_reminders
                    WHERE id = :reminder_id
                      AND user_id = CAST(:user_id AS UUID)
                      AND enabled = TRUE

                      AND :scheduled_for = (
                          (
                              (
                                  :scheduled_for AT TIME ZONE timezone
                              )::date + reminder_time
                          ) AT TIME ZONE timezone
                      )

                      AND (
                          :scheduled_for AT TIME ZONE timezone
                      )::date BETWEEN start_date AND end_date

                      AND :scheduled_for <= NOW()
                """),
                {
                    "reminder_id": str(reminder_id),
                    "user_id": current_user["id"],
                    "scheduled_for": payload.scheduled_for,
                },
            ).first()

            if not owned:
                raise HTTPException(
                    status_code=404,
                    detail=(
                        "Reminder occurrence not found, not due, "
                        "disabled, or not owned by this account."
                    ),
                )

            row = connection.execute(
                text("""
                    INSERT INTO reminder_deliveries (
                        reminder_id,
                        scheduled_for,
                        claim_token,
                        lease_expires_at
                    )
                    VALUES (
                        :reminder_id,
                        :scheduled_for,
                        :claim_token,
                        NOW() + INTERVAL '90 seconds'
                    )

                    ON CONFLICT (reminder_id, scheduled_for)

                    DO UPDATE SET
                        claim_token = EXCLUDED.claim_token,
                        lease_expires_at = EXCLUDED.lease_expires_at

                    WHERE reminder_deliveries.delivered_at IS NULL

                      AND (
                          reminder_deliveries.lease_expires_at IS NULL
                          OR reminder_deliveries.lease_expires_at < NOW()
                      )

                    RETURNING claim_token, lease_expires_at
                """),
                {
                    "reminder_id": str(reminder_id),
                    "scheduled_for": payload.scheduled_for,
                    "claim_token": str(token),
                },
            ).mappings().first()

        if not row:
            return {
                "claimed": False,
                "reason": "already_claimed_or_delivered",
            }

        return {
            "claimed": True,
            "claim_token": str(row["claim_token"]),
            "lease_expires_at": row["lease_expires_at"],
        }

    @router.post("/{reminder_id}/acknowledge")
    def acknowledge_delivery(
        reminder_id: UUID,
        claim_token: UUID,
        current_user: dict = Depends(get_current_user),
    ):
        with engine.begin() as connection:
            row = connection.execute(
                text("""
                    UPDATE reminder_deliveries d

                    SET
                        delivered_at = NOW(),
                        lease_expires_at = NULL

                    FROM medication_reminders r

                    WHERE d.reminder_id = r.id
                      AND r.id = :reminder_id
                      AND r.user_id = CAST(:user_id AS UUID)
                      AND d.claim_token = :claim_token
                      AND d.delivered_at IS NULL

                    RETURNING d.id
                """),
                {
                    "reminder_id": str(reminder_id),
                    "user_id": current_user["id"],
                    "claim_token": str(claim_token),
                },
            ).first()

        if not row:
            raise HTTPException(
                status_code=409,
                detail=(
                    "Claim expired, already acknowledged, "
                    "or reminder not owned by this account."
                ),
            )

        return {"acknowledged": True}

    return router
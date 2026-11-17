import json
import logging
from uuid import uuid4

import fitz
from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile
from sqlalchemy import text
from starlette.concurrency import run_in_threadpool

from parser import parse_prescription_image

MAX_PDF_BYTES = 50 * 1024 * 1024
MAX_PDF_PAGES = 100
CREATOR_EMAIL = "murarichaparla10@gmail.com"
logger = logging.getLogger("medimind.creator_evaluation")


def create_creator_evaluation_router(engine, get_current_user):
    router = APIRouter(
        prefix="/api/v1/creator/evaluations",
        tags=["Creator evaluation"],
    )

    with engine.begin() as connection:
        connection.execute(text("""
            CREATE TABLE IF NOT EXISTS creator_evaluation_jobs (
                id UUID PRIMARY KEY,
                user_id UUID NOT NULL,
                status VARCHAR(20) NOT NULL DEFAULT 'queued',
                total_pages INTEGER NOT NULL,
                completed_pages INTEGER NOT NULL DEFAULT 0,
                results JSONB NOT NULL DEFAULT '[]'::jsonb,
                error TEXT,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
        """))
        connection.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_creator_evaluation_user_created
            ON creator_evaluation_jobs (user_id, created_at DESC)
        """))

    def require_creator(user: dict) -> None:
        email = (user.get("email") or "").strip().casefold()
        if email != CREATOR_EMAIL:
            raise HTTPException(status_code=404, detail="Not found.")

    def update_job(job_id, **values):
        assignments = ", ".join(f"{column} = :{column}" for column in values)
        values["job_id"] = str(job_id)
        with engine.begin() as connection:
            connection.execute(
                text(f"""
                    UPDATE creator_evaluation_jobs
                    SET {assignments}, updated_at = NOW()
                    WHERE id = :job_id
                """),
                values,
            )

    def run_evaluation(job_id: str, pdf_bytes: bytes):
        update_job(job_id, status="running")
        try:
            with fitz.open(stream=pdf_bytes, filetype="pdf") as document:
                for page_number, page in enumerate(document, start=1):
                    try:
                        pixmap = page.get_pixmap(
                            matrix=fitz.Matrix(2, 2),
                            alpha=False,
                        )
                        result = parse_prescription_image(
                            pixmap.tobytes("png"),
                            "image/png",
                        ).model_dump()
                        row = {
                            "page": page_number,
                            "status": "processed",
                            "extraction": result,
                        }
                    except Exception as error:
                        logger.exception(
                            "Creator evaluation %s failed on page %s",
                            job_id,
                            page_number,
                        )
                        row = {
                            "page": page_number,
                            "status": "error",
                            "error": str(error)[:1000],
                        }

                    with engine.begin() as connection:
                        connection.execute(
                            text("""
                                UPDATE creator_evaluation_jobs
                                SET completed_pages = :page,
                                    results = results || CAST(:result AS JSONB),
                                    updated_at = NOW()
                                WHERE id = :job_id
                            """),
                            {
                                "page": page_number,
                                "result": json.dumps([row]),
                                "job_id": job_id,
                            },
                        )

            update_job(job_id, status="completed")
        except Exception as error:
            logger.exception("Creator evaluation %s failed", job_id)
            update_job(job_id, status="failed", error=str(error)[:2000])

    @router.post("", status_code=202)
    async def start_evaluation(
        background_tasks: BackgroundTasks,
        file: UploadFile = File(...),
        current_user: dict = Depends(get_current_user),
    ):
        require_creator(current_user)
        if file.content_type != "application/pdf":
            raise HTTPException(status_code=400, detail="Upload a PDF file.")
        pdf_bytes = await file.read(MAX_PDF_BYTES + 1)
        await file.close()
        if len(pdf_bytes) > MAX_PDF_BYTES:
            raise HTTPException(status_code=413, detail="PDF must be 50 MB or smaller.")
        if not pdf_bytes.startswith(b"%PDF-"):
            raise HTTPException(status_code=400, detail="The uploaded file is not a valid PDF.")
        try:
            with fitz.open(stream=pdf_bytes, filetype="pdf") as document:
                page_count = len(document)
        except Exception as error:
            raise HTTPException(status_code=400, detail="Could not read this PDF.") from error
        if page_count < 1 or page_count > MAX_PDF_PAGES:
            raise HTTPException(
                status_code=400,
                detail=f"PDF must contain between 1 and {MAX_PDF_PAGES} pages.",
            )

        job_id = str(uuid4())
        with engine.begin() as connection:
            connection.execute(
                text("""
                    INSERT INTO creator_evaluation_jobs (id, user_id, total_pages)
                    VALUES (:id, CAST(:user_id AS UUID), :total_pages)
                """),
                {
                    "id": job_id,
                    "user_id": current_user["id"],
                    "total_pages": page_count,
                },
            )
        background_tasks.add_task(run_evaluation, job_id, pdf_bytes)
        return {"id": job_id, "status": "queued", "total_pages": page_count}

    @router.get("/{job_id}")
    def get_evaluation(
        job_id: str,
        current_user: dict = Depends(get_current_user),
    ):
        require_creator(current_user)
        with engine.connect() as connection:
            row = connection.execute(
                text("""
                    SELECT id, status, total_pages, completed_pages,
                           results, error, created_at, updated_at
                    FROM creator_evaluation_jobs
                    WHERE id = :job_id AND user_id = CAST(:user_id AS UUID)
                """),
                {"job_id": job_id, "user_id": current_user["id"]},
            ).mappings().first()
        if not row:
            raise HTTPException(status_code=404, detail="Evaluation not found.")
        return dict(row)

    return router


import os
import base64
import logging
from io import BytesIO

from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image, ImageEnhance, ImageOps
from pydantic import BaseModel, Field


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

MODEL = os.getenv("AIROUTER_MODEL", "openai/gpt-6-luna")
AIROUTER_BASE_URL = "https://api.airouter.in/v1"

logger = logging.getLogger(__name__)


class MedicationExtraction(BaseModel):
    name: str = "Unknown"
    dosage: str = ""
    frequency_per_day: int = 0
    times_of_day: list[str] = Field(default_factory=list)
    duration_days: int = 0
    instructions: str = ""
    verification_note: str = ""

    raw_transcription: str = ""
    name_confidence: str = "low"
    dosage_confidence: str = "low"
    frequency_confidence: str = "low"


class PrescriptionExtraction(BaseModel):
    doctor_notes: str = ""
    medications: list[MedicationExtraction] = Field(
        default_factory=list
    )


def encode_image(image_bytes: bytes, mime_type: str) -> str:
    encoded = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def preprocess_image(image_bytes: bytes) -> bytes:
    """
    Create an enhanced image while preserving the original.

    Operations:
    - Correct EXIF orientation.
    - Convert to RGB.
    - Improve local readability using autocontrast.
    - Improve contrast and sharpness.
    - Resize very large images.
    """

    with Image.open(BytesIO(image_bytes)) as source:
        image = ImageOps.exif_transpose(source)
        image = image.convert("RGB")

        image = ImageOps.autocontrast(
            image,
            cutoff=1
        )

        image = ImageEnhance.Contrast(image).enhance(1.25)
        image = ImageEnhance.Sharpness(image).enhance(1.35)

        max_dimension = 2200

        if max(image.size) > max_dimension:
            image.thumbnail(
                (max_dimension, max_dimension),
                Image.Resampling.LANCZOS
            )

        output = BytesIO()

        image.save(
            output,
            format="JPEG",
            quality=92,
            optimize=True
        )

        return output.getvalue()


def normalize_confidence(value: str) -> str:
    value = str(value).strip().lower()

    if value not in ("high", "medium", "low"):
        return "low"

    return value


def parse_prescription_image(
    image_bytes: bytes,
    mime_type: str
) -> PrescriptionExtraction:

    api_key = os.getenv("AIROUTER_API_KEY")

    if not api_key:
        raise RuntimeError("AIROUTER_API_KEY is missing.")

    if mime_type not in (
        "image/jpeg",
        "image/png",
        "image/webp"
    ):
        raise ValueError(
            "Only JPEG, PNG and WEBP images are supported."
        )

    if not image_bytes:
        raise ValueError("Uploaded image is empty.")

    try:
        enhanced_bytes = preprocess_image(image_bytes)
    except Exception as error:
        logger.exception("Prescription image preprocessing failed.")
        raise ValueError(
            "The uploaded image could not be processed."
        ) from error

    original_image_url = encode_image(
        image_bytes,
        mime_type
    )

    enhanced_image_url = encode_image(
        enhanced_bytes,
        "image/jpeg"
    )

    instructions = """
You are an advanced medical prescription handwriting
transcription assistant.

Your task is to accurately transcribe the information
visible in the supplied prescription images.

You receive two views of the SAME prescription:
1. The original photograph.
2. An enhanced version of that photograph.

Use both views together. The original is the primary
source of truth. The enhanced version is supplementary.

CRITICAL TRANSCRIPTION PRINCIPLES:

1. Read the handwriting carefully, character by character.
2. Use visual context, handwriting patterns and nearby
   prescription information to interpret difficult text.
3. Do not require every character to be perfectly clear
   before attempting a transcription.
4. Preserve partially readable text instead of replacing
   the entire entry with "Unknown".
5. Never invent a medication or silently substitute
   a different medication.
6. Do not turn a possible interpretation into a confirmed
   medication name.
7. Extract every medication entry separately.
8. Do not omit entries because handwriting is difficult.

PARTIAL TRANSCRIPTION:

If only part of a medication name is readable:
- Put the readable transcription in name.
- Put the visible handwriting in raw_transcription.
- Mark name_confidence as "low" or "medium".
- Explain the uncertain characters in verification_note.
- Do not invent the missing characters.

If no meaningful part of the name can be read:
- Use "Unknown" for name.
- Preserve other readable fields.
- Explain why identification is uncertain.

FIELD-BY-FIELD EXTRACTION:

Medication name:
- Transcribe the visible name as accurately as possible.
- Do not replace a partially readable name with Unknown.

Dosage:
- Extract strength, units and formulation independently.
- Examples: 20 mg, 7.5 mg, 5 ml.
- Do not confuse medicine strength with administered amount.
- A readable dosage must be preserved even if the
  medication name is uncertain.

Frequency:
- Extract frequency only when supported by the handwriting.
- Use 0 when frequency cannot be determined.

Administration times:
- Use Morning, Afternoon or Night only when supported.
- Use HH:MM for explicitly written clock times.
- For clear 1-0-1 notation, use frequency 2 and
  ["Morning", "Night"].
- Do not infer schedules from ambiguous marks.

Duration:
- Extract the number of days only when visible.
- Use 0 when unknown.

Instructions:
- Preserve the doctor's written instructions.
- Do not invent missing instructions.

CONFIDENCE SCORES:

Assign a separate confidence score to each field:
- "high": clearly readable.
- "medium": mostly readable but some ambiguity exists.
- "low": substantially uncertain or partially unreadable.

Confidence must describe visual transcription certainty,
not whether a medication is medically appropriate.

VERIFICATION:

- Flag uncertain names, strengths, schedules and durations.
- Include possible readings only as explicitly unconfirmed
  possibilities in verification_note.
- Do not silently select between multiple possible readings.
- Use an empty verification_note only when the extracted
  information is sufficiently clear.
- Do not provide medical advice.

OTHER RULES:

- Extract doctor notes when visible.
- If the image is not a prescription, return no medications.
- Do not infer strength from a brand name.
- Do not calculate or modify a doctor's prescription.
- Never manufacture missing information.

OUTPUT:

Return valid JSON with exactly this structure:

{
  "doctor_notes": "",
  "medications": [
    {
      "name": "",
      "dosage": "",
      "frequency_per_day": 0,
      "times_of_day": [],
      "duration_days": 0,
      "instructions": "",
      "verification_note": "",
      "raw_transcription": "",
      "name_confidence": "low",
      "dosage_confidence": "low",
      "frequency_confidence": "low"
    }
  ]
}

All fields are required for every medication.
Use lowercase confidence values: high, medium or low.
Use empty strings for unavailable text.
Use 0 for unknown numeric values.
Return JSON only.
"""

    client = OpenAI(
        api_key=api_key,
        base_url=AIROUTER_BASE_URL,
        timeout=75.0,
        max_retries=1
    )

    logger.info(
        "Starting prescription extraction using model %s",
        MODEL
    )

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": instructions
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Inspect both views of this prescription. "
                            "Transcribe every medication separately. "
                            "Preserve partial readings, extract each "
                            "field independently, and flag uncertainty."
                        )
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": original_image_url
                        }
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": enhanced_image_url
                        }
                    }
                ]
            }
        ],
        response_format={"type": "json_object"},
        temperature=0.1,
        max_tokens=5000
    )

    result = response.choices[0].message.content

    if not result:
        raise ValueError(
            "GPT-6 Luna returned an empty response."
        )

    extraction = PrescriptionExtraction.model_validate_json(
        result
    )

    for medication in extraction.medications:
        medication.name_confidence = normalize_confidence(
            medication.name_confidence
        )
        medication.dosage_confidence = normalize_confidence(
            medication.dosage_confidence
        )
        medication.frequency_confidence = normalize_confidence(
            medication.frequency_confidence
        )

        if medication.frequency_per_day < 0:
            medication.frequency_per_day = 0

        if medication.duration_days < 0:
            medication.duration_days = 0

        if not medication.name.strip():
            medication.name = "Unknown"

    logger.info(
        "Prescription extraction completed: %s medications",
        len(extraction.medications)
    )

    return extraction

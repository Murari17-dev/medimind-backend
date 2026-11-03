
import os
import base64

import cv2
import numpy as np

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel
from rapidocr import RapidOCR

from ocr_test import group_medication_lines
from medication_extractor import select_medication_lines


load_dotenv()

MODEL = os.getenv("AIROUTER_MODEL", "openai/gpt-6-luna")
AIROUTER_BASE_URL = "https://api.airouter.in/v1"

ocr_engine = RapidOCR()


# --------------------------------------------------
# EXTRACTION MODELS
# --------------------------------------------------

class MedicationExtraction(BaseModel):
    name: str
    dosage: str
    frequency_per_day: int
    times_of_day: list[str]
    duration_days: int
    instructions: str
    verification_note: str


class PrescriptionExtraction(BaseModel):
    doctor_notes: str
    medications: list[MedicationExtraction]


# --------------------------------------------------
# OCR
# --------------------------------------------------

def extract_ocr_lines(image_bytes: bytes) -> list[dict]:

    image_array = np.frombuffer(image_bytes, dtype=np.uint8)

    image = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR
    )

    if image is None:
        raise ValueError(
            "Could not decode uploaded prescription image."
        )

    result = ocr_engine(image)

    if not result.txts:
        return []

    grouped_lines = group_medication_lines(
        result.boxes,
        result.txts,
        result.scores
    )

    return select_medication_lines(grouped_lines)


# --------------------------------------------------
# AI PRESCRIPTION EXTRACTION
# --------------------------------------------------

def parse_prescription_image(
    image_bytes: bytes,
    mime_type: str
) -> PrescriptionExtraction:

    api_key = os.getenv("AIROUTER_API_KEY")

    if not api_key:
        raise RuntimeError(
            "AIROUTER_API_KEY is missing from .env"
        )

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

    # Step 1: Extract OCR candidates.

    ocr_lines = extract_ocr_lines(image_bytes)

    if ocr_lines:
        ocr_text = "\n".join(
            (
                f"Line {index}: {line['text']}\n"
                f"OCR confidence: {line['confidence']:.3f}\n"
                f"Medication indicators detected: "
                f"{line['contains_medication_marker']}"
            )
            for index, line in enumerate(ocr_lines, start=1)
        )
    else:
        ocr_text = (
            "No medication-related OCR lines were detected. "
            "Inspect the original image independently."
        )

    # Step 2: Encode the original image.

    encoded_image = base64.b64encode(
        image_bytes
    ).decode("utf-8")

    image_url = f"data:{mime_type};base64,{encoded_image}"

    # Step 3: Initialize AI Router client.

    client = OpenAI(
        api_key=api_key,
        base_url=AIROUTER_BASE_URL,
        timeout=90.0
    )

    # Step 4: Prescription extraction instructions.

    instructions = """
You are an expert prescription transcription assistant.

Your task is to accurately read handwritten and printed
medical prescriptions using both the original image and OCR text.

EVIDENCE PRIORITY:

1. The original prescription image is the primary source.
2. OCR text is supporting evidence, not ground truth.
3. Compare OCR candidates with the actual handwriting.
4. Never blindly copy OCR errors.
5. Inspect the entire prescription, including all medication lines.

MEDICATION NAME RULES:

1. Transcribe the medication name as it appears to be written.
2. Do not silently replace the written name with a different
   medication name.
3. Use surrounding context only as supporting evidence.
4. Do not identify a medicine solely because it makes
   clinical sense.
5. If the name is unclear, preserve the best visual transcription
   and explain the uncertainty.
6. If the name cannot reasonably be read, use "Unknown".
7. Do not invent missing medication entries.

VERIFICATION RULES:

- Clearly readable medication names do not need a warning.
- Uncertain medication names must have a verification_note
  beginning with "Needs verification:".
- Explain why the name is uncertain.
- If a likely corrected medication name can be suggested,
  clearly identify it as a possibility, not a confirmed identity.
- Never invent confidence percentages.
- Never silently convert an uncertain guess into a confirmed result.
- Flag unclear strengths, formulations and instructions too.

DOSAGE AND SCHEDULE RULES:

- Extract the strength and units exactly when visible.
- Never guess dosage, frequency or treatment duration.
- Use 0 for unknown frequency or duration.
- Use an empty list if administration times cannot be determined.
- Preserve the doctor's written instructions.
- Use HH:MM for explicitly written clock times.
- Use Morning, Afternoon or Night when explicitly stated.
- For an unambiguous schedule such as 1-0-1, frequency_per_day
  may be 2 and times_of_day may be ["Morning", "Night"].
- Do not invent a schedule when the notation is unclear.
- Distinguish a medicine's strength from the amount to administer.

OTHER RULES:

- Extract doctor notes when visible.
- If the image is not a prescription, return no medications.
- Extract only information supported by the image.
- Do not provide new medical advice or change the doctor's orders.
- Do not infer a medicine's strength from its brand name.

OUTPUT REQUIREMENTS:

Return a valid JSON object with exactly this structure:

{
  "doctor_notes": "string",
  "medications": [
    {
      "name": "string",
      "dosage": "string",
      "frequency_per_day": 0,
      "times_of_day": [],
      "duration_days": 0,
      "instructions": "string",
      "verification_note": "string"
    }
  ]
}

Every medication must contain all seven fields.

Use an empty string for an unavailable dosage or instruction.
Use an empty string for verification_note when no warning is needed.

Return JSON only. Do not include Markdown or explanations.
"""

    user_message = f"""
OCR-extracted medication candidates:

{ocr_text}

Compare every candidate against the original prescription image.

Preserve unclear entries and explicitly flag uncertainty.
Do not omit medication lines simply because OCR is uncertain.
"""

    # Step 5: Call GPT-6 Luna.

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
                        "text": user_message
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_url
                        }
                    }
                ]
            }
        ],
        response_format={"type": "json_object"},
        temperature=0.1,
        max_tokens=4000
    )

    result = response.choices[0].message.content

    if not result:
        raise ValueError(
            "GPT-6 Luna returned an empty response."
        )

    # Step 6: Validate the AI response.

    return PrescriptionExtraction.model_validate_json(result)

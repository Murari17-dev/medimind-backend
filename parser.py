
import os
import base64

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

MODEL = os.getenv("AIROUTER_MODEL", "openai/gpt-6-luna")
AIROUTER_BASE_URL = "https://api.airouter.in/v1"


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

    encoded_image = base64.b64encode(
        image_bytes
    ).decode("utf-8")

    image_url = f"data:{mime_type};base64,{encoded_image}"

    instructions = """
You are an expert prescription transcription assistant.

Read the original prescription image carefully and extract
all visible medication information.

ACCURACY RULES:

1. The original image is the only source of truth.
2. Never invent medication names or missing information.
3. Never silently replace an unclear medicine with another.
4. Preserve the best visual transcription when uncertain.
5. If a medication name cannot reasonably be read, use "Unknown".
6. Do not omit medication entries merely because handwriting
   is difficult to read.
7. Flag uncertainty in verification_note.

VERIFICATION RULES:

- Uncertain medication names must have a verification_note
  beginning with "Needs verification:".
- Explain what is unclear.
- Clearly distinguish possible corrections from confirmed names.
- Flag unclear strengths, formulations and instructions.
- Use an empty verification_note when no warning is needed.

DOSAGE AND SCHEDULE RULES:

- Extract strength and units exactly when visible.
- Never guess dosage, frequency or duration.
- Use 0 for unknown frequency or duration.
- Use an empty list when administration times are unknown.
- Preserve the doctor's written instructions.
- Use HH:MM for explicitly written clock times.
- Use Morning, Afternoon or Night when explicitly stated.
- For an unambiguous schedule such as 1-0-1,
  frequency_per_day may be 2 and times_of_day may be
  ["Morning", "Night"].
- Do not invent a schedule when notation is unclear.
- Distinguish medicine strength from amount administered.

OTHER RULES:

- Extract doctor notes when visible.
- If the image is not a prescription, return no medications.
- Do not provide new medical advice or modify doctor's orders.
- Do not infer medicine strength from a brand name.

OUTPUT REQUIREMENTS:

Return valid JSON with exactly this structure:

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

Use an empty string for unavailable dosage or instruction.
Use an empty string for verification_note when no warning
is needed.

Return JSON only. No Markdown or explanations.
"""

    client = OpenAI(
        api_key=api_key,
        base_url=AIROUTER_BASE_URL,
        timeout=90.0
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
                            "Carefully inspect the entire original "
                            "prescription image. Extract every "
                            "medication and flag uncertainty."
                        )
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

    return PrescriptionExtraction.model_validate_json(result)

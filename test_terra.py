
import os
import base64
import mimetypes
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

API_KEY = os.getenv("AIROUTER_API_KEY")

if not API_KEY:
    raise ValueError("AIROUTER_API_KEY not found in .env")

client = OpenAI(
    api_key=API_KEY,
    base_url="https://api.airouter.in/v1"
)

# Enter the path of the ONE image you want to analyse.
IMAGE_PATH = input("Enter prescription image path: ").strip().strip('"')

image_path = Path(IMAGE_PATH)

if not image_path.is_file():
    raise FileNotFoundError(f"Image not found: {image_path}")

mime_type, _ = mimetypes.guess_type(str(image_path))

if not mime_type or not mime_type.startswith("image/"):
    raise ValueError("Unsupported image format.")

with open(image_path, "rb") as image_file:
    encoded_image = base64.b64encode(
        image_file.read()
    ).decode("utf-8")

image_data = f"data:{mime_type};base64,{encoded_image}"

prompt = """
You are an expert prescription-reading assistant.

Carefully inspect the entire prescription image.

Identify every medication visible.

For each medication, provide:
1. Exact name as visually written.
2. Likely corrected medication name, if reasonably identifiable.
3. Strength and units.
4. Dosage form.
5. Frequency.
6. Duration.
7. Instructions.
8. Confidence: High, Medium, or Low.

IMPORTANT:
- Inspect the original image, not just individual words.
- Use medical knowledge to assess whether a candidate name
  is a real medication.
- Consider surrounding prescription context.
- Never invent a medication to fill in an unclear entry.
- Do not silently correct uncertain spellings.
- If a name cannot be identified, mark it Unreadable.
- Distinguish visual observations from medical inferences.
- Flag contradictions between medicine name, strength,
  and dosage form.
- Return all identifiable medications.
- Return valid JSON only.
"""

response = client.chat.completions.create(
    model="openai/gpt-6.1-terra",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": image_data}
                }
            ]
        }
    ],
    max_tokens=4000
)

print("\n========== GPT-6.1 TERRA RESULT ==========\n")
print(response.choices[0].message.content)
print("\n==========================================")

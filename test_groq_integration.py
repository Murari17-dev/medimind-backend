from pathlib import Path
from parser import parse_prescription_image


image_path = Path("sample_prescription.jpg")

image_bytes = image_path.read_bytes()

print("Sending prescription through integrated pipeline...")

result = parse_prescription_image(
    image_bytes,
    "image/jpeg"
)

print("\n========== GROQ EXTRACTION RESULTS ==========\n")

print("Doctor notes:", result.doctor_notes)

for index, medication in enumerate(result.medications, start=1):
    print(f"\nMedication {index}")
    print("Name:", medication.name)
    print("Dosage:", medication.dosage)
    print("Frequency:", medication.frequency_per_day)
    print("Times:", medication.times_of_day)
    print("Duration:", medication.duration_days)
    print("Instructions:", medication.instructions)
    print("Verification:", medication.verification_note)

print("\nTotal medications:", len(result.medications))
from pathlib import Path

from parser import extract_ocr_lines


image_path = Path("sample_prescription.jpg")

image_bytes = image_path.read_bytes()

lines = extract_ocr_lines(image_bytes)

print("\n========== INTEGRATED OCR RESULTS ==========\n")

for index, line in enumerate(lines, start=1):
    print("Medication line:", index)
    print("Text:", line["text"])
    print("Confidence:", round(line["confidence"], 3))
    print(
        "Medication indicators:",
        line["contains_medication_marker"]
    )
    print("-" * 40)

print("\nTotal selected lines:", len(lines))
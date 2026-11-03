from pathlib import Path

from rapidocr import RapidOCR

from ocr_test import group_medication_lines
from medication_extractor import select_medication_lines


def main():
    image_path = Path("sample_prescription.jpg")

    if not image_path.exists():
        raise FileNotFoundError("Sample prescription not found.")

    print("Initializing RapidOCR...")
    engine = RapidOCR()

    print("Reading prescription...")
    result = engine(str(image_path))

    if not result.txts:
        print("No text detected.")
        return

    grouped_lines = group_medication_lines(
        result.boxes,
        result.txts,
        result.scores
    )

    medication_lines = select_medication_lines(grouped_lines)

    print("\n========== SELECTED MEDICATION LINES ==========\n")

    for index, line in enumerate(medication_lines, start=1):
        print(f"Medication line {index}: {line['text']}")
        print(f"OCR confidence: {line['confidence']:.3f}")
        print(
            "Contains medication indicators:",
            line["contains_medication_marker"]
        )
        print("-" * 45)

    print("\nTotal selected lines:", len(medication_lines))


if __name__ == "__main__":
    main()
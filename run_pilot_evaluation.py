
import csv
import json
import mimetypes
from pathlib import Path

from parser import parse_prescription_image


CSV_PATH = Path(
    r"C:\Users\SDB LAPTOPS\Downloads\Prescription Dataset.csv"
)

IMAGE_FOLDER = Path(
    r"C:\Users\SDB LAPTOPS\Documents\Prescription images.jpg\Prescription images.jpg"
)

OUTPUT_PATH = Path("pilot_evaluation.csv")
PILOT_SIZE = 10


def main():
    with CSV_PATH.open(
        "r", encoding="utf-8-sig", newline=""
    ) as file:
        records = list(csv.DictReader(file))

    image_index = {
        image.stem.lower(): image
        for image in IMAGE_FOLDER.iterdir()
        if image.suffix.lower() in [".jpg", ".jpeg"]
    }

    # Keep only the first record for each unique image.
    unique_records = []
    seen = set()

    for record in records:
        filename = record["image_filename"].strip().replace(",", ".")
        image_stem = Path(filename).stem.lower()

        if image_stem in seen:
            continue

        seen.add(image_stem)

        if image_stem in image_index:
            unique_records.append((record, image_index[image_stem]))

        if len(unique_records) == PILOT_SIZE:
            break

    if not unique_records:
        print("No matching images found.")
        return

    fieldnames = [
        "image_filename",
        "ground_truth_english",
        "predicted_medications",
        "doctor_notes",
        "error"
    ]

    with OUTPUT_PATH.open(
        "w", encoding="utf-8-sig", newline=""
    ) as output_file:

        writer = csv.DictWriter(
            output_file,
            fieldnames=fieldnames
        )
        writer.writeheader()
        output_file.flush()

        for index, (record, image_path) in enumerate(
            unique_records, start=1
        ):
            print(f"\nProcessing {index}/{len(unique_records)}:")
            print("Image:", image_path.name)

            row = {
                "image_filename": image_path.name,
                "ground_truth_english": record[
                    "description_english"
                ],
                "predicted_medications": "",
                "doctor_notes": "",
                "error": ""
            }

            try:
                image_bytes = image_path.read_bytes()
                mime_type = (
                    mimetypes.guess_type(image_path.name)[0]
                    or "image/jpeg"
                )

                result = parse_prescription_image(
                    image_bytes,
                    mime_type
                )

                medications = [
                    medication.model_dump()
                    for medication in result.medications
                ]

                row["predicted_medications"] = json.dumps(
                    medications,
                    ensure_ascii=False
                )

                row["doctor_notes"] = result.doctor_notes

                print("Extracted medications:", len(medications))

            except Exception as error:
                row["error"] = str(error)
                print("ERROR:", error)

            writer.writerow(row)
            output_file.flush()

    print("\n========== PILOT COMPLETE ==========")
    print("Processed:", len(unique_records))
    print("Results saved to:", OUTPUT_PATH.resolve())


if __name__ == "__main__":
    main()

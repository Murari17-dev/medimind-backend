import csv
from pathlib import Path
from collections import Counter

csv_path = Path(
    r"C:\Users\SDB LAPTOPS\Downloads\Prescription Dataset.csv"
)

image_folder = Path(
    r"C:\Users\SDB LAPTOPS\Documents\Prescription images.jpg\Prescription images.jpg"
)

with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
    records = list(csv.DictReader(file))

image_index = {
    image.stem.lower(): image
    for image in image_folder.iterdir()
    if image.suffix.lower() in [".jpg", ".jpeg"]
}

filenames = [
    record["image_filename"].strip().lower().replace(",", ".")
    for record in records
]

counts = Counter(filenames)

duplicates = {
    name: count
    for name, count in counts.items()
    if count > 1
}

matched = [
    name for name in counts
    if Path(name).stem in image_index
]

missing = [
    name for name in counts
    if Path(name).stem not in image_index
]

print("\n========== DATASET AUDIT ==========")
print("CSV records:", len(records))
print("Unique CSV filenames:", len(counts))
print("Image files:", len(image_index))
print("Unique matched filenames:", len(matched))
print("Missing filenames:", missing)
print("Duplicate filenames:", duplicates)
print("==================================")
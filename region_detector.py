
import cv2
import numpy as np
from pathlib import Path


def detect_prescription_regions(image_bytes: bytes):
    if not image_bytes:
        raise ValueError("Image is empty.")

    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("Could not decode image.")

    # Upscale the image
    enlarged = cv2.resize(
        image,
        None,
        fx=2,
        fy=2,
        interpolation=cv2.INTER_CUBIC
    )

    gray = cv2.cvtColor(enlarged, cv2.COLOR_BGR2GRAY)

    # Improve contrast
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )
    enhanced = clahe.apply(gray)

    # Detect handwriting and printed text
    binary = cv2.adaptiveThreshold(
        enhanced,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        10
    )

    # Remove tiny isolated noise
    num_labels, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            binary,
            connectivity=8
        )
    )

    cleaned = np.zeros_like(binary)

    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]

        if area >= 3 and h >= 2:
            cleaned[labels == i] = 255

    # Group nearby characters into text lines
    line_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (25, 5)
    )

    grouped = cv2.dilate(
        cleaned,
        line_kernel,
        iterations=1
    )

    contours, _ = cv2.findContours(
        grouped,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    height, width = grouped.shape

    candidates = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        if w < 20 or h < 8:
            continue

        if w * h < 100:
            continue

        # Extract the original handwriting pixels
        padding = 12

        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(width, x + w + padding)
        y2 = min(height, y + h + padding)

        region_width = x2 - x1
        region_height = y2 - y1

        # Calculate text density
        region = cleaned[y1:y2, x1:x2]

        density = cv2.countNonZero(region) / (
            region_width * region_height
        )

        candidates.append({
            "box": (x1, y1, x2, y2),
            "score": (
                (region_width / width) * 2
                + min(region_width / max(region_height, 1), 10) * 0.1
                + density
            )
        })

    # Merge vertically adjacent detections
    candidates.sort(key=lambda item: item["box"][1])

    merged = []

    for item in candidates:
        x1, y1, x2, y2 = item["box"]

        matched = False

        for existing in merged:
            ex1, ey1, ex2, ey2 = existing["box"]

            vertical_gap = y1 - ey2
            horizontal_overlap = (
                min(x2, ex2) - max(x1, ex1)
            )

            if vertical_gap <= 35 and horizontal_overlap > -30:
                existing["box"] = (
                    min(x1, ex1),
                    min(y1, ey1),
                    max(x2, ex2),
                    max(y2, ey2)
                )
                matched = True
                break

        if not matched:
            merged.append(item.copy())

    merged.sort(
        key=lambda item: item["score"],
        reverse=True
    )

    regions = [
        item["box"] for item in merged
    ]

    crops = []

    annotated = enlarged.copy()

    for index, (x1, y1, x2, y2) in enumerate(regions):

        crops.append(
            enlarged[y1:y2, x1:x2]
        )

        cv2.rectangle(
            annotated,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            annotated,
            f"Region {index + 1}",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1
        )

    return annotated, regions, crops


if __name__ == "__main__":

    input_path = Path("sample_prescription.jpg")

    if not input_path.exists():
        raise FileNotFoundError(
            "sample_prescription.jpg was not found."
        )

    annotated, regions, crops = detect_prescription_regions(
        input_path.read_bytes()
    )

    cv2.imwrite("detected_regions.png", annotated)

    # Remove stale crop files from earlier runs
    for old_file in Path(".").glob("detected_region_*.png"):
        old_file.unlink()

    for index, crop in enumerate(crops):
        cv2.imwrite(
            f"detected_region_{index + 1}.png",
            crop
        )

    print("Detection completed.")
    print("Regions detected:", len(regions))
    print("Annotated preview: detected_regions.png")

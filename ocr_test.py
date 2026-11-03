
import cv2
import numpy as np
from pathlib import Path
from rapidocr import RapidOCR


def group_medication_lines(boxes, texts, scores):
    """
    Groups OCR detections into nearby horizontal text lines.
    Uses relative distances rather than fixed image coordinates.
    """

    items = []

    for box, text, score in zip(boxes, texts, scores):
        points = np.array(box)

        x1 = float(points[:, 0].min())
        y1 = float(points[:, 1].min())
        x2 = float(points[:, 0].max())
        y2 = float(points[:, 1].max())

        items.append({
            "text": text,
            "score": float(score),
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "height": y2 - y1,
            "center_y": (y1 + y2) / 2
        })

    # Sort detections from top to bottom.
    items.sort(key=lambda item: item["center_y"])

    if not items:
        return []

    # Estimate typical text height.
    typical_height = np.median([
        item["height"] for item in items
        if item["height"] > 0
    ])

    # Group detections that occupy approximately the same line.
    lines = []

    for item in items:
        matched_line = None

        for line in lines:
            line_center = np.mean([
                entry["center_y"] for entry in line
            ])

            if abs(item["center_y"] - line_center) <= (
                typical_height * 0.8
            ):
                matched_line = line
                break

        if matched_line is None:
            lines.append([item])
        else:
            matched_line.append(item)

    # Combine text detections within each line.
    grouped = []

    for line in lines:
        line.sort(key=lambda item: item["x1"])

        grouped.append({
            "text": " ".join(item["text"] for item in line),
            "confidence": float(np.mean([
                item["score"] for item in line
            ])),
            "box": (
                min(item["x1"] for item in line),
                min(item["y1"] for item in line),
                max(item["x2"] for item in line),
                max(item["y2"] for item in line)
            )
        })

    return grouped


def main():

    image_path = Path("sample_prescription.jpg")

    if not image_path.exists():
        raise FileNotFoundError(
            "sample_prescription.jpg was not found."
        )

    image = cv2.imread(str(image_path))

    print("Initializing RapidOCR...")
    engine = RapidOCR()

    print("Processing prescription...")
    result = engine(str(image_path))

    if not result.txts:
        print("No text detected.")
        return

    grouped_lines = group_medication_lines(
        result.boxes,
        result.txts,
        result.scores
    )

    print("\n========== GROUPED OCR LINES ==========\n")

    for index, line in enumerate(grouped_lines, start=1):
        print(f"Line {index}: {line['text']}")
        print(f"Confidence: {line['confidence']:.3f}")
        print(f"Coordinates: {line['box']}")
        print("-" * 40)

    # Visualize grouped OCR lines.
    annotated = image.copy()

    for index, line in enumerate(grouped_lines, start=1):

        x1, y1, x2, y2 = line["box"]

        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)

        cv2.rectangle(
            annotated,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            1
        )

        cv2.putText(
            annotated,
            str(index),
            (x1, max(12, y1 - 3)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (0, 0, 255),
            1
        )

    cv2.imwrite("grouped_ocr_result.png", annotated)

    print("\nGrouped lines:", len(grouped_lines))
    print("Preview saved: grouped_ocr_result.png")


if __name__ == "__main__":
    main()

import pytest

from medication_extractor import select_medication_lines


def test_grouped_rapidocr_detections_can_be_selected():
    # RapidOCR is optional: the production parser does not depend on it.
    pytest.importorskip("rapidocr")
    from ocr_test import group_medication_lines

    boxes = [
        [[0, 0], [30, 0], [30, 10], [0, 10]],
        [[35, 0], [60, 0], [60, 10], [35, 10]],
        [[0, 20], [70, 20], [70, 30], [0, 30]],
    ]
    texts = ["Aspirin", "75 mg", "Take daily"]
    scores = [0.9, 0.8, 0.7]

    grouped = group_medication_lines(boxes, texts, scores)
    selected = select_medication_lines(grouped)

    assert [line["text"] for line in grouped] == [
        "Aspirin 75 mg",
        "Take daily",
    ]
    assert [line["text"] for line in selected] == [
        "Aspirin 75 mg",
        "Take daily",
    ]
    assert all(
        line["contains_medication_marker"] for line in selected
    )
    assert grouped[0]["confidence"] == pytest.approx(0.85)

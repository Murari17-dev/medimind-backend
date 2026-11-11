import pytest

from medication_extractor import (
    is_medication_candidate,
    is_section_end,
    select_medication_lines,
)


def make_line(text, top, bottom, confidence=0.9):
    return {
        "text": text,
        "confidence": confidence,
        "box": (0, top, 100, bottom),
    }


@pytest.mark.parametrize(
    "text",
    [
        "Aspirin 75 mg",
        "TAB example",
        "Take twice daily",
        "Take 2 times a day",
    ],
)
def test_recognises_existing_medication_markers(text):
    assert is_medication_candidate(text)


def test_text_without_marker_is_not_a_candidate():
    assert not is_medication_candidate("Patient name")


@pytest.mark.parametrize("text", ["Signature", "Refill", "Dr. Smith"])
def test_recognises_existing_section_end_markers(text):
    assert is_section_end(text)


def test_empty_and_unmarked_input_select_nothing():
    assert select_medication_lines([]) == []
    assert select_medication_lines([make_line("Patient name", 0, 10)]) == []


def test_selects_lines_in_order_between_medication_candidates():
    lines = [
        make_line("Drug B 10 mg", 40, 50),
        make_line("Patient name", 0, 10),
        make_line("Take after food", 30, 40),
        make_line("Drug A 5 mg", 20, 30),
    ]

    selected = select_medication_lines(lines)

    assert [line["text"] for line in selected] == [
        "Drug A 5 mg",
        "Take after food",
        "Drug B 10 mg",
    ]
    assert [
        line["contains_medication_marker"] for line in selected
    ] == [True, False, True]
    assert selected[0]["confidence"] == 0.9
    assert selected[0]["box"] == (0, 20, 100, 30)


def test_stops_before_footer_even_if_later_line_has_marker():
    lines = [
        make_line("Drug A 5 mg", 0, 10),
        make_line("Signature", 12, 22),
        make_line("Drug B 10 mg", 24, 34),
    ]

    selected = select_medication_lines(lines)

    assert [line["text"] for line in selected] == ["Drug A 5 mg"]


def test_existing_large_gap_rule_skips_distant_line():
    lines = [
        make_line("Drug A 5 mg", 0, 10),
        make_line("Drug B 10 mg", 100, 110),
    ]

    selected = select_medication_lines(lines)

    assert [line["text"] for line in selected] == ["Drug A 5 mg"]

import re
from statistics import median


MEDICATION_PATTERNS = [
    r"\b\d+(?:\.\d+)?\s*(?:mg|mcg|g|ml|iu|units?)\b",
    r"\b(?:tab|tabs|tablet|tablets|cap|caps|capsule|capsules)\b",
    r"\b(?:bid|tid|qid|od|qd|bd|prn|once|twice|daily)\b",
    r"\b\d+\s*(?:x|times?)\s*(?:daily|a day)\b",
]

MEDICATION_REGEX = re.compile(
    "|".join(MEDICATION_PATTERNS),
    re.IGNORECASE
)

SECTION_END_MARKERS = [
    "signature",
    "refill",
    "label",
    "doctor",
    "dr.",
]


def is_medication_candidate(text: str) -> bool:
    return bool(MEDICATION_REGEX.search(text))


def is_section_end(text: str) -> bool:
    normalized = text.strip().lower()

    return any(
        marker in normalized
        for marker in SECTION_END_MARKERS
    )


def select_medication_lines(grouped_lines: list[dict]) -> list[dict]:

    if not grouped_lines:
        return []

    ordered_lines = sorted(
        grouped_lines,
        key=lambda line: line["box"][1]
    )

    candidate_indices = [
        index
        for index, line in enumerate(ordered_lines)
        if is_medication_candidate(line["text"])
    ]

    if not candidate_indices:
        return []

    first_index = candidate_indices[0]

    # Stop at the first detected signature or footer marker.
    last_index = len(ordered_lines) - 1

    for index in range(first_index, len(ordered_lines)):
        if is_section_end(ordered_lines[index]["text"]):
            last_index = index - 1
            break

    # Find the last medication candidate before the footer.
    valid_candidates = [
        index
        for index in candidate_indices
        if first_index <= index <= last_index
    ]

    if not valid_candidates:
        return []

    last_index = valid_candidates[-1]

    heights = [
        line["box"][3] - line["box"][1]
        for line in ordered_lines
        if line["box"][3] > line["box"][1]
    ]

    typical_height = median(heights) if heights else 1

    selected = []
    previous_bottom = None

    for index in range(first_index, last_index + 1):

        line = ordered_lines[index]

        top = line["box"][1]
        bottom = line["box"][3]

        if previous_bottom is not None:
            gap = top - previous_bottom

            if gap > typical_height * 3:
                continue

        selected.append({
            "text": line["text"],
            "confidence": line["confidence"],
            "box": line["box"],
            "contains_medication_marker": (
                is_medication_candidate(line["text"])
            )
        })

        previous_bottom = bottom

    return selected
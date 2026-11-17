import base64
import json
from io import BytesIO
from types import SimpleNamespace

import pytest
from PIL import Image
from pydantic import ValidationError

import parser


def make_png():
    output = BytesIO()
    Image.new("RGB", (8, 8), "white").save(output, format="PNG")
    return output.getvalue()


def complete_medication(**overrides):
    medication = {
        "name": "Example",
        "dosage": "5 mg",
        "frequency_per_day": 2,
        "times_of_day": ["Morning", "Night"],
        "duration_days": 5,
        "instructions": "After food",
        "raw_transcription": "Example 5 mg",
        "name_confidence": "high",
        "dosage_confidence": "high",
        "frequency_confidence": "medium",
    }
    medication.update(overrides)
    return medication


def fake_client(responses, calls):
    """Return a client whose completions yield each response in turn."""
    queue = list(responses)

    def create(**kwargs):
        calls.append(kwargs)
        content = queue.pop(0)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )

    return SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )


def test_prescription_model_supplies_defaults():
    extraction = parser.PrescriptionExtraction.model_validate({})

    assert extraction.doctor_notes == ""
    assert extraction.medications == []


def test_medication_model_requires_every_field_so_the_model_must_retry():
    with pytest.raises(ValidationError, match="missing required medication fields"):
        parser.MedicationExtraction.model_validate({})


def test_models_reject_extra_fields_and_wrong_types():
    with pytest.raises(ValidationError):
        parser.PrescriptionExtraction.model_validate({"extra": "nope"})

    with pytest.raises(ValidationError):
        parser.MedicationExtraction.model_validate(
            complete_medication(frequency_per_day="2")
        )


@pytest.mark.parametrize(
    "payload",
    [
        "not json",
        '{"medications": "not a list"}',
        '{"medications": [{"frequency_per_day": "not an integer"}]}',
    ],
)
def test_models_reject_invalid_json_or_field_types(payload):
    with pytest.raises(ValidationError):
        parser.PrescriptionExtraction.model_validate_json(payload)


def test_times_of_day_are_normalised_and_validated():
    medication = parser.MedicationExtraction.model_validate(
        complete_medication(times_of_day=["morning", "20:30"])
    )

    assert medication.times_of_day == ["Morning", "20:30"]

    with pytest.raises(ValidationError):
        parser.MedicationExtraction.model_validate(
            complete_medication(times_of_day=["teatime"])
        )


def test_missing_information_is_recorded_not_invented():
    medication = parser.MedicationExtraction.model_validate(
        complete_medication(
            name=" ",
            dosage="",
            frequency_per_day=0,
            duration_days=0,
        )
    )

    assert medication.name == "Unknown"
    assert "Medication name could not be read" in medication.verification_note
    assert "Dosage was not visible" in medication.verification_note
    assert "Frequency could not be determined" in medication.verification_note
    assert "Duration was not visible" in medication.verification_note


def test_encode_image_preserves_bytes_and_mime_type():
    result = parser.encode_image(b"sample", "image/png")

    assert result == (
        "data:image/png;base64,"
        + base64.b64encode(b"sample").decode("utf-8")
    )


def test_parse_sends_original_and_enhanced_views(monkeypatch):
    calls = []
    response_json = json.dumps({
        "doctor_notes": "Review",
        "medications": [complete_medication()],
    })
    monkeypatch.setattr(parser, "client", fake_client([response_json], calls))

    extraction = parser.parse_prescription_image(make_png(), "image/png")

    medication = extraction.medications[0]
    assert extraction.doctor_notes == "Review"
    assert medication.name == "Example"
    assert medication.frequency_per_day == 2
    assert medication.frequency_confidence == "medium"

    assert len(calls) == 1
    content = calls[0]["messages"][1]["content"]
    images = [item for item in content if item["type"] == "image_url"]
    assert len(images) == 2
    assert images[0]["image_url"]["url"].startswith("data:image/png;base64,")
    assert images[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_parse_repairs_invalid_output_with_a_follow_up_request(monkeypatch):
    calls = []
    valid = json.dumps({"doctor_notes": "", "medications": [complete_medication()]})
    monkeypatch.setattr(
        parser, "client", fake_client(["not json", valid], calls)
    )

    extraction = parser.parse_prescription_image(make_png(), "image/png")

    assert len(extraction.medications) == 1
    assert len(calls) == 2
    repair_request = calls[1]["messages"][-1]["content"]
    assert "failed validation" in repair_request


def test_parse_gives_up_after_the_repair_budget(monkeypatch):
    calls = []
    attempts = parser.MAX_REPAIR_ATTEMPTS + 1
    monkeypatch.setattr(
        parser, "client", fake_client(["not json"] * attempts, calls)
    )

    with pytest.raises(ValueError, match="failed validation after"):
        parser.parse_prescription_image(make_png(), "image/png")

    assert len(calls) == attempts


def test_parse_rejects_unsupported_mime_before_calling_provider(monkeypatch):
    def unexpected_call(**kwargs):
        raise AssertionError("Provider must not be called")

    monkeypatch.setattr(
        parser,
        "client",
        SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=unexpected_call)
            )
        ),
    )

    with pytest.raises(ValueError, match="Only JPEG"):
        parser.parse_prescription_image(make_png(), "text/plain")


def test_parse_rejects_undecodable_image_before_calling_provider(monkeypatch):
    def unexpected_call(**kwargs):
        raise AssertionError("Provider must not be called")

    monkeypatch.setattr(
        parser,
        "client",
        SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=unexpected_call)
            )
        ),
    )

    with pytest.raises(ValueError, match="could not be processed"):
        parser.parse_prescription_image(b"not an image", "image/png")

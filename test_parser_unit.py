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


def test_prescription_model_supplies_defaults_but_medications_must_be_complete():
    extraction = parser.PrescriptionExtraction.model_validate({})

    assert extraction.doctor_notes == ""
    assert extraction.medications == []

    with pytest.raises(ValidationError, match="missing required medication fields"):
        parser.MedicationExtraction.model_validate({})


def test_medication_models_reject_extra_fields_and_wrong_types():
    medication = {
        "name": "Aspirin",
        "dosage": "75 mg",
        "frequency_per_day": 2,
        "times_of_day": ["Morning", "Night"],
        "duration_days": 5,
        "instructions": "After food",
        "verification_note": "",
        "raw_transcription": "Aspirin 75 mg 1-0-1 x 5 days",
        "name_confidence": "high",
        "dosage_confidence": "high",
        "frequency_confidence": "medium",
    }
    parsed = parser.MedicationExtraction.model_validate(medication)
    assert parsed.frequency_per_day == 2

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        parser.MedicationExtraction.model_validate({**medication, "extra": "rejected"})

    with pytest.raises(ValidationError):
        parser.MedicationExtraction.model_validate({**medication, "frequency_per_day": "2"})


@pytest.mark.parametrize(
    "payload",
    [
        "not json",
        '{"medications": "not a list"}',
        '{"medications": [{"frequency_per_day": "not an integer"}]}',
    ],
)
def test_current_models_reject_invalid_json_or_field_types(payload):
    with pytest.raises(ValidationError):
        parser.PrescriptionExtraction.model_validate_json(payload)


def test_encode_image_preserves_bytes_and_mime_type():
    result = parser.encode_image(b"sample", "image/png")

    assert result == (
        "data:image/png;base64,"
        + base64.b64encode(b"sample").decode("utf-8")
    )


def test_parse_preserves_valid_fields_and_sends_two_views(monkeypatch):
    calls = []
    response_json = json.dumps({
        "doctor_notes": "Review",
        "medications": [{
            "name": "Aspirin",
            "dosage": "75 mg",
            "frequency_per_day": 2,
            "times_of_day": ["Morning", "Night"],
            "duration_days": 5,
            "instructions": "After food",
            "verification_note": "",
            "raw_transcription": "Aspirin 75 mg 1-0-1 x 5 days",
            "name_confidence": "high",
            "dosage_confidence": "high",
            "frequency_confidence": "medium",
        }],
    })

    def fake_create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=response_json)
                )
            ]
        )

    monkeypatch.setattr(parser.client.chat.completions, "create", fake_create)

    extraction = parser.parse_prescription_image(make_png(), "image/png")

    medication = extraction.medications[0]
    assert extraction.doctor_notes == "Review"
    assert medication.name == "Aspirin"
    assert medication.frequency_per_day == 2
    assert medication.duration_days == 5
    assert medication.name_confidence == "high"
    assert medication.dosage_confidence == "high"
    assert medication.frequency_confidence == "medium"

    assert len(calls) == 1
    content = calls[0]["messages"][1]["content"]
    images = [item for item in content if item["type"] == "image_url"]
    assert len(images) == 2
    assert images[0]["image_url"]["url"].startswith("data:image/png;base64,")
    assert images[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_parse_repairs_invalid_model_output(monkeypatch):
    calls = []
    valid_response = json.dumps({
        "doctor_notes": "",
        "medications": [{
            "name": "Aspirin",
            "dosage": "75 mg",
            "frequency_per_day": 1,
            "times_of_day": ["Morning"],
            "duration_days": 5,
            "instructions": "",
            "verification_note": "",
            "raw_transcription": "Aspirin 75 mg once daily",
            "name_confidence": "high",
            "dosage_confidence": "high",
            "frequency_confidence": "high",
        }],
    })
    responses = iter(["not json", valid_response])

    def fake_create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=next(responses))
                )
            ]
        )

    monkeypatch.setattr(parser.client.chat.completions, "create", fake_create)

    extraction = parser.parse_prescription_image(make_png(), "image/png")
    assert extraction.medications[0].name == "Aspirin"
    assert len(calls) == 2


def test_parse_rejects_unsupported_mime_before_calling_provider(monkeypatch):
    monkeypatch.setenv("AIROUTER_API_KEY", "test-key")

    def unexpected_request(**kwargs):
        raise AssertionError("Provider must not be called")

    monkeypatch.setattr(parser.client.chat.completions, "create", unexpected_request)

    with pytest.raises(ValueError, match="Only JPEG"):
        parser.parse_prescription_image(make_png(), "text/plain")


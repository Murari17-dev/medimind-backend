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


def test_current_models_supply_defaults():
    extraction = parser.PrescriptionExtraction.model_validate({})

    assert extraction.doctor_notes == ""
    assert extraction.medications == []

    medication = parser.MedicationExtraction.model_validate({})
    assert medication.name == "Unknown"
    assert medication.frequency_per_day == 0
    assert medication.times_of_day == []
    assert medication.name_confidence == "low"


def test_current_models_coerce_integer_strings_and_ignore_extra_fields():
    extraction = parser.PrescriptionExtraction.model_validate_json(
        json.dumps({
            "extra": "ignored",
            "medications": [{
                "frequency_per_day": "2",
                "extra": "ignored",
            }],
        })
    )

    assert extraction.medications[0].frequency_per_day == 2
    assert "extra" not in extraction.model_dump()
    assert "extra" not in extraction.medications[0].model_dump()


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


def test_parse_normalises_existing_fields_and_sends_two_views(monkeypatch):
    monkeypatch.setenv("AIROUTER_API_KEY", "test-key")
    calls = []
    response_json = json.dumps({
        "doctor_notes": "Review",
        "medications": [{
            "name": " ",
            "frequency_per_day": -2,
            "duration_days": -3,
            "name_confidence": " HIGH ",
            "dosage_confidence": "unexpected",
            "frequency_confidence": "Medium",
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

    def fake_openai(**kwargs):
        assert kwargs["api_key"] == "test-key"
        return SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=fake_create)
            )
        )

    monkeypatch.setattr(parser, "OpenAI", fake_openai)

    extraction = parser.parse_prescription_image(make_png(), "image/png")

    medication = extraction.medications[0]
    assert extraction.doctor_notes == "Review"
    assert medication.name == "Unknown"
    assert medication.frequency_per_day == 0
    assert medication.duration_days == 0
    assert medication.name_confidence == "high"
    assert medication.dosage_confidence == "low"
    assert medication.frequency_confidence == "medium"

    assert len(calls) == 1
    content = calls[0]["messages"][1]["content"]
    images = [item for item in content if item["type"] == "image_url"]
    assert len(images) == 2
    assert images[0]["image_url"]["url"].startswith("data:image/png;base64,")
    assert images[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_parse_rejects_invalid_model_output_without_repair(monkeypatch):
    monkeypatch.setenv("AIROUTER_API_KEY", "test-key")
    calls = []

    def fake_create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="not json")
                )
            ]
        )

    monkeypatch.setattr(
        parser,
        "OpenAI",
        lambda **kwargs: SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=fake_create)
            )
        ),
    )

    with pytest.raises(ValidationError):
        parser.parse_prescription_image(make_png(), "image/png")

    assert len(calls) == 1


def test_parse_rejects_unsupported_mime_before_calling_provider(monkeypatch):
    monkeypatch.setenv("AIROUTER_API_KEY", "test-key")

    def unexpected_client(**kwargs):
        raise AssertionError("Provider must not be called")

    monkeypatch.setattr(parser, "OpenAI", unexpected_client)

    with pytest.raises(ValueError, match="Only JPEG"):
        parser.parse_prescription_image(make_png(), "text/plain")

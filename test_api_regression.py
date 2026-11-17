import importlib.util
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import sqlalchemy
from fastapi.testclient import TestClient


USER_A = "11111111-1111-1111-1111-111111111111"
USER_B = "22222222-2222-2222-2222-222222222222"


class InitialisationGuard:
    """Allow mocked startup DDL, but reject unexpected database access."""

    def __init__(self):
        self.initialising = True

    @contextmanager
    def begin(self):
        yield self

    @contextmanager
    def connect(self):
        yield self

    def execute(self, *args, **kwargs):
        if not self.initialising:
            raise AssertionError("Unexpected database access")
        return None


class FakeResult:
    def __init__(self, values):
        self.values = values

    def fetchall(self):
        return [SimpleNamespace(id=value) for value in self.values]

    def scalar_one_or_none(self):
        if not self.values:
            return None
        assert len(self.values) == 1
        return self.values[0]

    def scalars(self):
        return self

    def all(self):
        return list(self.values)


class FakePrescriptionDatabase:
    """In-memory substitute for the prescription queries used in these tests."""

    def __init__(self, records):
        self.records = dict(records)
        self.calls = []

    @contextmanager
    def begin(self):
        yield self

    @contextmanager
    def connect(self):
        yield self

    def execute(self, statement, parameters):
        sql = str(statement)
        self.calls.append((sql, dict(parameters)))
        user_id = parameters["user_id"]

        if "SELECT id" in sql and "FROM prescriptions" in sql:
            ids = [
                prescription_id
                for prescription_id, owner in self.records.items()
                if owner == user_id
            ]
            return FakeResult(sorted(ids, reverse=True))

        if "DELETE FROM prescriptions" in sql:
            if "prescription_id" in parameters:
                prescription_id = parameters["prescription_id"]
                ids = (
                    [prescription_id]
                    if self.records.get(prescription_id) == user_id
                    else []
                )
            else:
                ids = [
                    prescription_id
                    for prescription_id, owner in self.records.items()
                    if owner == user_id
                ]

            for prescription_id in ids:
                del self.records[prescription_id]

            return FakeResult(ids)

        raise AssertionError(f"Unexpected SQL: {sql}")


@pytest.fixture
def api(monkeypatch):
    # Import main only after replacing create_engine, so startup migrations
    # cannot connect to any PostgreSQL database, regardless of local .env.
    guard = InitialisationGuard()
    monkeypatch.setenv("DATABASE_URL", "postgresql://test:test@invalid/test")
    monkeypatch.setenv("SUPABASE_URL", "https://supabase.invalid")
    monkeypatch.setenv("SUPABASE_PUBLISHABLE_KEY", "test-publishable-key")
    monkeypatch.setattr(sqlalchemy, "create_engine", lambda *args, **kwargs: guard)

    module_path = Path(__file__).resolve().parent / "main.py"
    spec = importlib.util.spec_from_file_location(
        "medimind_test_main",
        module_path,
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    guard.initialising = False

    module.app.dependency_overrides[module.get_current_user] = (
        lambda: {"id": USER_A}
    )

    with TestClient(module.app) as client:
        yield module, client

    module.app.dependency_overrides.clear()


def prescription_response(module, prescription_id):
    return module.PrescriptionResponse(
        id=prescription_id,
        doctor_notes="Review",
        medications=[
            module.MedicationResponse(
                name="Example",
                dosage="5 mg",
                instructions="",
                times_of_day=["Morning"],
                frequency_per_day=1,
                duration_days=5,
                verification_note="",
                raw_transcription="Example 5 mg",
                name_confidence="high",
                dosage_confidence="high",
                frequency_confidence="medium",
            )
        ],
    )


def test_home_response(api):
    _, client = api

    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {
        "message": "MediMind API is running",
        "version": "4.0.0",
        "database": "Supabase PostgreSQL",
    }


def test_prescriptions_require_authentication(api):
    module, client = api
    module.app.dependency_overrides.clear()

    response = client.get("/api/v1/prescriptions")

    assert response.status_code == 401
    assert response.json() == {"detail": "Sign in to continue."}


def test_list_prescriptions_uses_authenticated_user(api, monkeypatch):
    module, client = api
    calls = []

    def fake_fetch(user_id, prescription_id=None):
        calls.append((user_id, prescription_id))
        return [prescription_response(module, 3), prescription_response(module, 1)]

    monkeypatch.setattr(module, "fetch_prescriptions", fake_fetch)

    response = client.get("/api/v1/prescriptions")

    assert response.status_code == 200
    assert [item["id"] for item in response.json()] == [3, 1]
    assert calls == [(USER_A, None)]


class FakeMappingResult:
    def __init__(self, rows):
        self.rows = rows

    def mappings(self):
        return self

    def all(self):
        return self.rows


class FakeBatchDatabase:
    """Records every query so the test can prove the list is not N+1."""

    def __init__(self, prescriptions, medications):
        self.prescriptions = prescriptions
        self.medications = medications
        self.calls = []

    @contextmanager
    def connect(self):
        yield self

    def execute(self, statement, parameters):
        sql = str(statement)
        self.calls.append((sql, dict(parameters)))

        if "FROM prescriptions" in sql:
            return FakeMappingResult(self.prescriptions)

        if "FROM medications" in sql:
            return FakeMappingResult(self.medications)

        raise AssertionError(f"Unexpected SQL: {sql}")


def medication_row(prescription_id, name):
    return {
        "prescription_id": prescription_id,
        "name": name,
        "dosage": "5 mg",
        "instructions": None,
        "times_of_day": '["Morning"]',
        "frequency_per_day": 1,
        "duration_days": 5,
        "verification_note": None,
        "raw_transcription": None,
        "name_confidence": None,
        "dosage_confidence": "high",
        "frequency_confidence": "medium",
    }


def test_fetch_prescriptions_loads_everything_in_two_queries(api, monkeypatch):
    module, _ = api
    database = FakeBatchDatabase(
        prescriptions=[
            {"id": 3, "doctor_notes": "Second", "created_at": None},
            {"id": 1, "doctor_notes": None, "created_at": None},
        ],
        medications=[
            medication_row(3, "Alpha"),
            medication_row(3, "Beta"),
            medication_row(1, "Gamma"),
        ],
    )
    monkeypatch.setattr(module, "engine", database)

    results = module.fetch_prescriptions(USER_A)

    assert len(database.calls) == 2
    assert database.calls[0][1]["user_id"] == USER_A
    assert database.calls[1][1] == {"ids": [3, 1]}
    assert [item.id for item in results] == [3, 1]
    assert [m.name for m in results[0].medications] == ["Alpha", "Beta"]
    assert [m.name for m in results[1].medications] == ["Gamma"]
    assert results[0].medications[0].times_of_day == ["Morning"]
    assert results[0].medications[0].name_confidence == "low"


def test_fetch_prescriptions_skips_medication_query_when_empty(api, monkeypatch):
    module, _ = api
    database = FakeBatchDatabase(prescriptions=[], medications=[])
    monkeypatch.setattr(module, "engine", database)

    assert module.fetch_prescriptions(USER_A) == []
    assert len(database.calls) == 1


def test_health_endpoint(api):
    _, client = api

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_delete_prescription_cannot_delete_another_users_record(api, monkeypatch):
    module, client = api
    database = FakePrescriptionDatabase({1: USER_A, 2: USER_B})
    monkeypatch.setattr(module, "engine", database)

    response = client.delete("/api/v1/prescriptions/2")

    assert response.status_code == 404
    assert response.json() == {"detail": "Prescription not found."}
    assert database.records == {1: USER_A, 2: USER_B}
    assert database.calls[0][1] == {
        "prescription_id": 2,
        "user_id": USER_A,
    }


def test_delete_owned_prescription_preserves_response(api, monkeypatch):
    module, client = api
    database = FakePrescriptionDatabase({1: USER_A, 2: USER_B})
    monkeypatch.setattr(module, "engine", database)

    response = client.delete("/api/v1/prescriptions/1")

    assert response.status_code == 200
    assert response.json() == {"deleted": True, "prescription_id": 1}
    assert database.records == {2: USER_B}


def test_delete_all_only_deletes_authenticated_users_records(api, monkeypatch):
    module, client = api
    database = FakePrescriptionDatabase({1: USER_A, 2: USER_B, 3: USER_A})
    monkeypatch.setattr(module, "engine", database)

    response = client.delete("/api/v1/prescriptions")

    assert response.status_code == 200
    assert response.json() == {"deleted": True, "deleted_count": 2}
    assert database.records == {2: USER_B}
    assert database.calls[0][1] == {"user_id": USER_A}


def test_parse_prescription_preserves_response_format(api, monkeypatch):
    module, client = api
    extraction = SimpleNamespace(medications=[object()])
    calls = []

    def fake_parse(image_bytes, mime_type):
        calls.append(("parse", image_bytes, mime_type))
        return extraction

    def fake_save(value, user_id):
        assert value is extraction
        calls.append(("save", user_id))
        return 42

    def fake_build(prescription_id, user_id):
        calls.append(("build", prescription_id, user_id))
        return prescription_response(module, prescription_id)

    monkeypatch.setattr(module, "parse_prescription_image", fake_parse)
    monkeypatch.setattr(module, "save_prescription", fake_save)
    monkeypatch.setattr(module, "build_prescription_response", fake_build)

    response = client.post(
        "/api/v1/parse-prescription",
        files={"file": ("sample.png", b"image bytes", "image/png")},
    )

    assert response.status_code == 200
    assert response.json() == {
        "id": 42,
        "doctor_notes": "Review",
        "created_at": None,
        "medications": [{
            "name": "Example",
            "dosage": "5 mg",
            "instructions": "",
            "times_of_day": ["Morning"],
            "frequency_per_day": 1,
            "duration_days": 5,
            "verification_note": "",
            "raw_transcription": "Example 5 mg",
            "name_confidence": "high",
            "dosage_confidence": "high",
            "frequency_confidence": "medium",
        }],
    }
    assert calls == [
        ("parse", b"image bytes", "image/png"),
        ("save", USER_A),
        ("build", 42, USER_A),
    ]


@pytest.mark.parametrize(
    ("content", "mime_type", "expected_status"),
    [
        (b"image", "text/plain", 400),
        (b"", "image/png", 400),
        (b"x" * (10 * 1024 * 1024 + 1), "image/png", 413),
    ],
)
def test_parse_rejects_invalid_uploads_without_processing(
    api, monkeypatch, content, mime_type, expected_status
):
    module, client = api

    def unexpected_parse(*args, **kwargs):
        raise AssertionError("Parser must not be called")

    monkeypatch.setattr(module, "parse_prescription_image", unexpected_parse)

    response = client.post(
        "/api/v1/parse-prescription",
        files={"file": ("sample", content, mime_type)},
    )

    assert response.status_code == expected_status


def test_parse_validation_error_does_not_save(api, monkeypatch):
    module, client = api
    saved = []

    def invalid_extraction(*args):
        raise ValueError("Invalid image")

    monkeypatch.setattr(module, "parse_prescription_image", invalid_extraction)
    monkeypatch.setattr(
        module,
        "save_prescription",
        lambda *args: saved.append(args),
    )

    response = client.post(
        "/api/v1/parse-prescription",
        files={"file": ("sample.png", b"image", "image/png")},
    )

    assert response.status_code == 422
    assert response.json() == {"detail": "Invalid image"}
    assert saved == []


def test_token_verification_is_cached_per_token(api, monkeypatch):
    import asyncio

    module, _ = api
    module._auth_cache.clear()
    lookups = []

    class FakeResponse:
        status_code = 200

        def json(self):
            return {"id": USER_A}

    class FakeAsyncClient:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def get(self, url, headers):
            lookups.append(headers["Authorization"])
            return FakeResponse()

    monkeypatch.setattr(module.httpx, "AsyncClient", FakeAsyncClient)

    def credentials(token):
        return SimpleNamespace(scheme="Bearer", credentials=token)

    first = asyncio.run(module.get_current_user(credentials("token-1")))
    second = asyncio.run(module.get_current_user(credentials("token-1")))
    asyncio.run(module.get_current_user(credentials("token-2")))

    assert first == second == {"id": USER_A}
    assert lookups == ["Bearer token-1", "Bearer token-2"]

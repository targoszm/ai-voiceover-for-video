from io import BytesIO
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setattr("app.main.UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr("app.main.GENERATED_DIR", tmp_path / "generated")
    (tmp_path / "uploads").mkdir()
    (tmp_path / "generated").mkdir()
    return TestClient(app)


def test_home_page(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "Generate AI Voiceover" in response.text


def test_upload_rejects_non_mp4(client: TestClient) -> None:
    response = client.post(
        "/generate",
        files={"video": ("sample.txt", BytesIO(b"x"), "text/plain")},
    )
    assert response.status_code == 400


def test_lead_capture(client: TestClient) -> None:
    response = client.post(
        "/lead-capture",
        json={"email": "a@b.com", "name": "Jane", "company": "Acme", "company_size": "11-50"},
    )
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

import io
import os
import zipfile

import pytest
from fastapi.testclient import TestClient

from conftest import FakeChatClient, fake_mock_llm, make_pdf

import app as app_module
from src.core import llm_mockgen, vision


@pytest.fixture
def client():
    app_module.RATE_BUCKETS.clear()
    return TestClient(app_module.app)


@pytest.fixture
def fake_llms(monkeypatch):
    seen = {"deepseek_keys": []}
    deepseek = FakeChatClient(fake_mock_llm)

    def fake_openai(**kwargs):
        seen["deepseek_keys"].append(kwargs.get("api_key"))
        return deepseek

    seen["vision"] = FakeChatClient(lambda kw: "1. Solve $x^2 - 5x + 6 = 0$. [3 marks]")
    monkeypatch.setattr(llm_mockgen, "OpenAI", fake_openai)
    monkeypatch.setattr("openai.OpenAI", fake_openai)
    monkeypatch.setattr(vision, "make_vision_client", lambda provider, timeout: seen["vision"])
    return seen


# ---------- filename sanitising ----------
@pytest.mark.parametrize("raw, expected", [
    ("paper.pdf", "paper.pdf"),
    ("../../etc/passwd", "passwd"),
    ("..\\..\\x\\exam.docx", "exam.docx"),
    ("/tmp/Final Exam 2024!.PDF", "Final_Exam_2024.pdf"),
    (".env", "env"),
    ("", "upload"),
    (None, "upload"),
])
def test_safe_upload_name(raw, expected):
    assert app_module.safe_upload_name(raw) == expected


def test_unique_upload_name():
    name = app_module.unique_upload_name("../../a b.pdf")
    assert name.endswith("_a_b.pdf") and "/" not in name and len(name.split("_", 1)[0]) == 12


# ---------- endpoints ----------
def test_healthz(client):
    r = client.get("/healthz")
    assert r.status_code == 200 and r.json() == {"ok": True}


def test_generate_happy_path_with_mocked_llm_and_vision(client, fake_llms, tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "server-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "vision-key")
    pdf = make_pdf(tmp_path / "in.pdf", ["1. Solve x^2 - 5x + 6 = 0. [3 marks]"])

    with pdf.open("rb") as fh:
        r = client.post(
            "/generate",
            files={"file": ("../../../etc/Exam 1.pdf", fh, "application/pdf")},
            data={"num_mocks": "1", "difficulty": "same", "language": "en", "ocr_engine": "easyocr"},
        )

    assert r.status_code == 200, r.text
    assert r.headers["content-type"] == "application/zip"
    names = zipfile.ZipFile(io.BytesIO(r.content)).namelist()
    assert sorted(names) == ["mock_1.pdf", "mock_1_answers.pdf"]
    assert len(fake_llms["vision"].calls) == 1
    assert set(fake_llms["deepseek_keys"]) == {"server-key"}


def test_generate_passes_vision_text_to_generator(client, fake_llms, tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "server-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "vision-key")
    captured = {}
    real = llm_mockgen.generate_mock_specs

    def spy(paper_text, *args, **kwargs):
        captured["text"] = paper_text
        return real(paper_text, *args, **kwargs)

    monkeypatch.setattr(llm_mockgen, "generate_mock_specs", spy)
    pdf = make_pdf(tmp_path / "in.pdf", ["native"])
    with pdf.open("rb") as fh:
        r = client.post("/generate", files={"file": ("in.pdf", fh, "application/pdf")})
    assert r.status_code == 200, r.text
    assert "$x^2 - 5x + 6 = 0$" in captured["text"]


def test_form_key_is_used_for_request_only(client, fake_llms, tmp_path):
    pdf = make_pdf(tmp_path / "in.pdf", ["1. Define a prime number. [2 marks]"])
    with pdf.open("rb") as fh:
        r = client.post(
            "/generate",
            files={"file": ("in.pdf", fh, "application/pdf")},
            data={"openai_api_key": "request-key"},
        )
    assert r.status_code == 200, r.text
    assert set(fake_llms["deepseek_keys"]) == {"request-key"}
    assert "DEEPSEEK_API_KEY" not in os.environ
    assert not fake_llms["vision"].calls  # no vision key -> text layer only


def test_generate_rejects_wrong_type(client):
    r = client.post("/generate", files={"file": ("notes.txt", b"hi", "text/plain")})
    assert r.status_code == 400


def test_generate_missing_key(client, tmp_path):
    pdf = make_pdf(tmp_path / "in.pdf", ["text"])
    with pdf.open("rb") as fh:
        r = client.post("/generate", files={"file": ("in.pdf", fh, "application/pdf")})
    assert r.status_code == 400 and "DEEPSEEK_API_KEY" in r.json()["message"]


def test_rate_limit_still_enforced(client, monkeypatch):
    monkeypatch.setitem(app_module.RATE_LIMITS, ("POST", "/generate"), (2, 60))
    codes = [client.post("/generate", files={"file": ("x.txt", b"x", "text/plain")}).status_code for _ in range(3)]
    assert codes == [400, 400, 429]


def test_cors_only_allows_portfolio_origin(client):
    ok = client.options("/generate", headers={"Origin": "https://ngyuhang.com", "Access-Control-Request-Method": "POST"})
    bad = client.options("/generate", headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "POST"})
    assert ok.headers.get("access-control-allow-origin") == "https://ngyuhang.com"
    assert "access-control-allow-origin" not in bad.headers

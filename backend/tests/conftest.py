import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SERVICE_ROOT = Path(__file__).resolve().parents[1]
if str(SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVICE_ROOT))

PROVIDER_ENV = (
    "VISION_API_KEY", "VISION_BASE_URL", "VISION_MODEL",
    "OPENROUTER_API_KEY", "OPENAI_API_KEY", "DEEPSEEK_API_KEY",
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """No test may see real keys or make a live call."""
    for name in PROVIDER_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("VISION_RETRY_BACKOFF_SECONDS", "0")
    yield


def make_pdf(path: Path, pages):
    import fitz

    doc = fitz.open()
    for text in pages:
        page = doc.new_page()
        if text:
            page.insert_text((72, 72), text)
    doc.save(str(path))
    doc.close()
    return path


def _response(content: str):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


class FakeChatClient:
    """Mimics openai.OpenAI().chat.completions.create; records calls, never touches the network."""

    def __init__(self, handler):
        self.calls = []
        self._handler = handler
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return _response(self._handler(kwargs))


def fake_mock_llm(kwargs):
    """Answers the structured mock-paper prompt with a valid one-question spec."""
    return json.dumps({"mocks": [{
        "title": "Mock Exam Paper",
        "instructions": "Answer all questions.",
        "sections": [{"title": "Algebra", "questions": [
            {"id": "1", "type": "free", "marks": 3, "text": "Solve x^2 - 5x + 6 = 0."},
        ]}],
        "answer_key": [{"id": "1", "answer": "x = 2 or x = 3", "workings": "Factor: (x-2)(x-3) = 0."}],
    }]})

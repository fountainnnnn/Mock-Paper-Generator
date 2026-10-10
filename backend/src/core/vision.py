# -*- coding: utf-8 -*-
"""
Vision-LLM page reader for exam papers (replaces EasyOCR).

- Renders PDF pages (PyMuPDF) and DOCX documents (docx2pdf/LibreOffice when
  available, otherwise the document's embedded pictures) to PNG.
- Asks an OpenAI-compatible vision model to transcribe each page exactly:
  question numbering, sub-parts, options and marks kept, math as LaTeX ($...$).
- Falls back to the document's embedded text layer (PyMuPDF get_text /
  python-docx) when no vision provider is configured, for pages past the page
  cap, and for any page whose vision call fails.

Provider resolution (first match wins):
  1) VISION_API_KEY  (+ optional VISION_BASE_URL, VISION_MODEL)
  2) OPENROUTER_API_KEY -> https://openrouter.ai/api/v1, qwen/qwen2.5-vl-72b-instruct
  3) OPENAI_API_KEY     -> OpenAI, gpt-4o-mini
  4) none               -> text layer only (degraded, logged as a warning)
"""

from __future__ import annotations

import base64
import io
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Mapping, Optional

log = logging.getLogger(__name__)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_DEFAULT_MODEL = "qwen/qwen2.5-vl-72b-instruct"
OPENAI_DEFAULT_MODEL = "gpt-4o-mini"

VISION_SYSTEM_PROMPT = (
    "You transcribe one page of an exam paper so a generator can write a new mock paper from it.\n"
    "- Transcribe every question EXACTLY as written, in reading order. Do not paraphrase, solve, "
    "answer, or add commentary.\n"
    "- Keep question numbering and sub-part labels exactly (e.g. 1, 2(a), (ii), Q3b) and keep every "
    "marks allocation exactly as printed (e.g. [3 marks], (2), [Total: 10]).\n"
    "- Keep MCQ options with their labels (A, B, C, D) on separate lines.\n"
    "- Write ALL mathematics as LaTeX: inline as $...$, display equations as $$...$$ "
    "(fractions, powers, roots, integrals, matrices, units with symbols).\n"
    "- Reproduce tables as Markdown tables.\n"
    "- For a diagram, graph or figure a question depends on, add one line '[Figure: ...]' with its "
    "labelled values, axes and key features.\n"
    "- Skip running headers/footers, page numbers, barcodes, 'Turn over', 'Do not write in this margin'.\n"
    "- The embedded text layer is given as a hint and may be incomplete, garbled or out of order; "
    "trust the image.\n"
    "- Return only the transcription. If the page has no exam content, return an empty string."
)


# =========================
# Settings
# =========================
def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def vision_settings() -> dict:
    """Read tunables at call time so tests and deployments can override via env."""
    return {
        "concurrency": max(1, _env_int("VISION_CONCURRENCY", 4)),
        "timeout": max(5.0, _env_float("VISION_PAGE_TIMEOUT_SECONDS", 60.0)),
        "retries": max(0, _env_int("VISION_MAX_RETRIES", 2)),
        "backoff": max(0.0, _env_float("VISION_RETRY_BACKOFF_SECONDS", 1.5)),
        "max_pages": max(1, _env_int("VISION_MAX_PAGES", 40)),
        "max_images_per_page": max(1, _env_int("VISION_MAX_IMAGES_PER_PAGE", 4)),
    }


# =========================
# Provider resolution
# =========================
@dataclass(frozen=True)
class VisionProvider:
    name: str
    api_key: str
    base_url: Optional[str]
    model: str


def resolve_vision_provider(env: Optional[Mapping[str, str]] = None) -> Optional[VisionProvider]:
    env = os.environ if env is None else env
    model_override = (env.get("VISION_MODEL") or "").strip()

    key = (env.get("VISION_API_KEY") or "").strip()
    if key:
        base_url = (env.get("VISION_BASE_URL") or "").strip() or None
        return VisionProvider("custom", key, base_url, model_override or OPENAI_DEFAULT_MODEL)

    key = (env.get("OPENROUTER_API_KEY") or "").strip()
    if key:
        return VisionProvider("openrouter", key, OPENROUTER_BASE_URL, model_override or OPENROUTER_DEFAULT_MODEL)

    key = (env.get("OPENAI_API_KEY") or "").strip()
    if key:
        return VisionProvider("openai", key, None, model_override or OPENAI_DEFAULT_MODEL)

    return None


def make_vision_client(provider: VisionProvider, timeout: float):
    """OpenAI-compatible client. Retries are handled by read_pages, not the SDK."""
    from openai import OpenAI

    kwargs: dict = {"api_key": provider.api_key, "timeout": timeout, "max_retries": 0}
    if provider.base_url:
        kwargs["base_url"] = provider.base_url
    return OpenAI(**kwargs)


# =========================
# Page model
# =========================
@dataclass
class Page:
    index: int                                          # 1-based
    text: str = ""                                      # embedded text layer
    images: List[bytes] = field(default_factory=list)   # PNG bytes sent to the vision model


def _clean(page: Page, body: str) -> str:
    return (body or "").strip()


def _to_png(blob: bytes, max_side: int = 2000, min_side: int = 64) -> Optional[bytes]:
    """Normalise an embedded picture to PNG; skip icons/bullets that are too small to matter."""
    try:
        from PIL import Image

        img = Image.open(io.BytesIO(blob))
        if min(img.size) < min_side:
            return None
        if img.mode not in ("RGB", "RGBA", "L"):
            img = img.convert("RGBA")
        img.thumbnail((max_side, max_side))
        out = io.BytesIO()
        img.save(out, format="PNG")
        return out.getvalue()
    except Exception:
        return None


# =========================
# Document loaders
# =========================
def load_pdf_pages(path: str | Path, work_dir: str | Path, dpi: int, render: bool) -> List[Page]:
    import fitz  # PyMuPDF
    from .render import pdf_to_png

    with fitz.open(str(path)) as doc:
        texts = [page.get_text("text").strip() for page in doc]

    pngs: List[Path] = []
    if render and texts:
        max_pages = vision_settings()["max_pages"]
        pngs = pdf_to_png(path, Path(work_dir) / "pages", dpi=dpi, max_pages=max_pages)

    pages: List[Page] = []
    for i, text in enumerate(texts, 1):
        images = [pngs[i - 1].read_bytes()] if i <= len(pngs) else []
        pages.append(Page(index=i, text=text, images=images))
    return pages


def load_docx_pages(path: str | Path, work_dir: str | Path, dpi: int, render: bool) -> List[Page]:
    from docx import Document

    settings = vision_settings()
    doc = Document(str(path))
    text = "\n".join(p.text.strip() for p in doc.paragraphs if p.text.strip())

    if render:
        # Page render needs docx2pdf (Windows/macOS + Word) or LibreOffice; the Docker
        # image has neither, so this usually falls through to embedded pictures.
        try:
            from .render import render_paper_to_images

            pngs = render_paper_to_images(path, str(Path(work_dir) / "pages"), dpi=dpi)[: settings["max_pages"]]
            if pngs:
                return [Page(index=i, text=text if i == 1 else "", images=[p.read_bytes()])
                        for i, p in enumerate(pngs, 1)]
        except Exception as e:
            log.info("DOCX page rendering unavailable; sending embedded pictures instead (%s)", str(e).splitlines()[0])

    images: List[bytes] = []
    if render:
        for rel in doc.part.rels.values():
            if "image" not in rel.reltype:
                continue
            try:
                png = _to_png(rel.target_part.blob)
            except Exception:
                png = None
            if png:
                images.append(png)
            if len(images) >= settings["max_images_per_page"]:
                break
    return [Page(index=1, text=text, images=images)]


# =========================
# Vision calls
# =========================
def _data_url(png: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


def _transcribe_page(client: Any, provider: VisionProvider, page: Page, language: str, timeout: float) -> str:
    hint = (page.text or "").strip()[:4000] or "(empty)"
    content: List[dict] = [{
        "type": "text",
        "text": (
            f"Exam page {page.index}. Document language: {language or 'en'}.\n"
            f"Embedded text layer (hint):\n{hint}"
        ),
    }]
    content.extend({"type": "image_url", "image_url": {"url": _data_url(png)}} for png in page.images)
    resp = client.chat.completions.create(
        model=provider.model,
        messages=[
            {"role": "system", "content": VISION_SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
        temperature=0,
        timeout=timeout,
    )
    return (resp.choices[0].message.content or "").strip() if resp.choices else ""


def _read_one(client: Any, provider: VisionProvider, page: Page, language: str, settings: dict) -> str:
    if not page.images:
        return _clean(page, page.text)

    last_error: Optional[Exception] = None
    for attempt in range(settings["retries"] + 1):
        try:
            text = _transcribe_page(client, provider, page, language, settings["timeout"])
            if text:
                return _clean(page, text)
            last_error = RuntimeError("empty response")
        except Exception as e:  # network, timeout, rate limit, bad response
            last_error = e
        if attempt < settings["retries"]:
            time.sleep(settings["backoff"] * (2 ** attempt))

    log.warning(
        "Vision read failed for page %d after %d attempt(s) (%s); using the text layer for this page.",
        page.index, settings["retries"] + 1, type(last_error).__name__ if last_error else "unknown",
    )
    return _clean(page, page.text)


def read_pages(
    pages: List[Page],
    provider: Optional[VisionProvider],
    language: str = "en",
    client: Any = None,
) -> List[str]:
    """Return one transcription per page, in page order."""
    if provider is None:
        return [_clean(p, p.text) for p in pages]

    settings = vision_settings()
    if client is None:
        client = make_vision_client(provider, settings["timeout"])
    with ThreadPoolExecutor(max_workers=settings["concurrency"]) as pool:
        return list(pool.map(lambda p: _read_one(client, provider, p, language, settings), pages))


def read_document(
    path: str | Path,
    work_dir: str | Path,
    dpi: int = 220,
    language: str = "en",
    provider: Optional[VisionProvider] = None,
    client: Any = None,
) -> List[str]:
    """
    Read a .pdf or .docx into per-page Markdown/LaTeX.
    `provider` defaults to resolve_vision_provider(); pass a client to inject a fake in tests.
    """
    if provider is None:
        provider = resolve_vision_provider()
    if provider is None:
        log.warning(
            "No vision provider configured (set VISION_API_KEY, OPENROUTER_API_KEY or OPENAI_API_KEY). "
            "Falling back to the document's embedded text layer; scanned pages and diagrams will be missed."
        )

    ext = Path(path).suffix.lower()
    dpi = max(72, min(int(dpi), 300))
    render = provider is not None
    if ext == ".pdf":
        pages = load_pdf_pages(path, work_dir, dpi, render)
    elif ext in {".docx", ".doc"}:
        pages = load_docx_pages(path, work_dir, dpi, render)
    else:
        raise ValueError(f"Unsupported file type: {ext or 'unknown'}. Only PDF and DOCX are supported.")

    return read_pages(pages, provider, language=language, client=client)

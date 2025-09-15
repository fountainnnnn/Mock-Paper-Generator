# backend/src/core/mock_upload.py
# -*- coding: utf-8 -*-
"""
Robust text extraction utilities for uploaded mock exam papers (PDF/DOCX).

- PDF: prefer native text layer, fallback to OCR (EasyOCR) for scanned/math regions.
- DOCX: use python-docx for text extraction.
- OCR preprocessing: grayscale, binarize, denoise.
- Outputs both plain text and simple HTML (with math spans preserved).
"""

from pathlib import Path
from typing import List, Dict

import fitz  # PyMuPDF
from docx import Document
import numpy as np
import cv2

from .ocr import get_ocr_engine, ocr_image_easy


# =========================
# Preprocessing helper
# =========================
def _preprocess_for_ocr(img: np.ndarray) -> np.ndarray:
    """Convert to grayscale + binarize + denoise for sharper OCR."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img
    th = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 35, 11
    )
    th = cv2.fastNlMeansDenoising(th, h=15)
    return th


# =========================
# Internal helpers
# =========================
def _wrap_html_paragraphs(text: str) -> str:
    """
    Wrap plain text into HTML paragraphs, preserving math spans.
    Math expressions like \( ... \) or \[ ... \] are wrapped in <span>/<div>.
    """
    html_lines: List[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        # Replace LaTeX-style math delimiters with spans
        line = line.replace("\\(", '<span class="math">').replace("\\)", "</span>")
        line = line.replace("\\[", '<div class="math">').replace("\\]", "</div>")

        html_lines.append(f"<p>{line}</p>")

    return "\n".join(html_lines)


def _extract_text_from_pdf(path: str, lang: str = "en", dpi: int = 400) -> str:
    """
    Extract text from a PDF file.
    - Try native PDF text layer first.
    - Fallback to OCR (EasyOCR) for scanned pages.
    - Returns plain text string.
    """
    text_blocks: List[str] = []
    doc = fitz.open(path)

    # OCR engine (lazy init)
    reader = None

    for i, page in enumerate(doc):
        native_text = page.get_text("text").strip()
        if native_text:
            print(f"[DEBUG] Page {i}: native text ({len(native_text)} chars)")
            text_blocks.append(native_text)
            continue

        # Init OCR engine only if needed
        if reader is None:
            reader = get_ocr_engine(lang)

        # Render page to image
        pix = page.get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72))
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        if pix.n == 4:  # RGBA → RGB
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

        img = _preprocess_for_ocr(img)

        # OCR fallback
        results = ocr_image_easy(reader, img)
        page_texts = [r["text"] for r in results if r.get("text")]

        if page_texts:
            joined = " ".join(page_texts)
            print(f"[DEBUG] Page {i}: OCR extracted {len(joined)} chars")
            text_blocks.append(joined)
        else:
            print(f"[WARNING] Page {i}: OCR returned nothing")

    return "\n".join(text_blocks)


def _extract_text_from_docx(path: str) -> str:
    """Extract plain text from a DOCX file."""
    doc = Document(path)
    paras = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    return "\n".join(paras)


# =========================
# Public API
# =========================
def papers_to_clean_text(
    files: List[str],
    out_dir: str,
    lang: str = "en",
    dpi: int = 400,
) -> Dict[str, str]:
    """
    Extract text from uploaded PDF/DOCX mock papers,
    concatenate into plain text + HTML files, and return their paths.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_plain: List[str] = []
    all_html: List[str] = []

    for f in files:
        p = Path(f)
        if p.suffix.lower() == ".pdf":
            plain = _extract_text_from_pdf(str(p), lang=lang, dpi=dpi)
        elif p.suffix.lower() in {".docx", ".doc"}:
            plain = _extract_text_from_docx(str(p))
        else:
            raise ValueError(f"Unsupported file type: {p.suffix}")

        if plain.strip():
            all_plain.append(plain)
            all_html.append(_wrap_html_paragraphs(plain))
        else:
            print(f"[WARNING] No text extracted from {p.name}")

    # Save plain text
    concat_plain = "\n\n".join(all_plain).strip()
    concat_txt_path = out_dir / "reference_concat.txt"
    concat_txt_path.write_text(concat_plain or "[EMPTY DOCUMENT: No text extracted]", encoding="utf-8")

    # Save HTML
    concat_html = "\n<hr/>\n".join(all_html)
    html_doc = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<title>Extracted Mock Paper</title>
<style>
  body {{ font-family: 'Georgia', serif; line-height: 1.5; }}
  .math {{ font-family: 'Cambria Math', 'STIX', serif; }}
</style>
</head>
<body>
{concat_html if concat_html.strip() else "<p>[EMPTY DOCUMENT: No text extracted]</p>"}
</body>
</html>"""
    concat_html_path = out_dir / "reference_concat.html"
    concat_html_path.write_text(html_doc, encoding="utf-8")

    return {
        "concat_txt": str(concat_txt_path),
        "concat_html": str(concat_html_path),
    }

# backend/src/core/mock_upload.py
# -*- coding: utf-8 -*-
"""
Text extraction for uploaded mock exam papers (PDF/DOCX).

- Each page is read by the vision page reader (vision.py): exact question text,
  numbering and marks, math as LaTeX.
- Without a vision provider, or for pages whose vision call fails, the native
  text layer is used (PyMuPDF for PDF, python-docx for DOCX).
- Outputs both plain text and simple HTML (with math spans preserved).
"""

from pathlib import Path
from typing import List, Dict
import logging

from .vision import read_document

log = logging.getLogger(__name__)


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


def _extract_text(path: Path, out_dir: Path, lang: str = "en", dpi: int = 220) -> str:
    """Per-page transcriptions from the vision reader (text-layer fallback), in page order."""
    work_dir = out_dir / f"pages_{path.stem}"
    pages = read_document(path, work_dir, dpi=dpi, language=lang)
    return "\n\n".join(t for t in pages if t.strip())


import re

_HARD_WRAP = re.compile(r"(?<=\S)-\s*\n(?=\S)")  # e.g., "com-\npute" -> "compute"
_LINE_SP   = re.compile(r"[ \t]+\n")             # trailing spaces before newline
_MULTI_NL  = re.compile(r"\n{3,}")               # collapse >2 blank lines

def _dehyphenate(s: str) -> str:
    # join hyphenated line breaks *only* when there's no space around the break
    return _HARD_WRAP.sub("", s)

def _normalize_ws(s: str) -> str:
    # trim trailing spaces at line ends, and keep paragraph breaks tidy
    s = _LINE_SP.sub("\n", s)
    s = _MULTI_NL.sub("\n\n", s)
    return s.strip()


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
        if p.suffix.lower() not in {".pdf", ".docx", ".doc"}:
            raise ValueError(f"Unsupported file type: {p.suffix}")
        plain = _extract_text(p, out_dir, lang=lang, dpi=dpi)

        if plain.strip():
            all_plain.append(plain)
            all_html.append(_wrap_html_paragraphs(plain))
        else:
            log.warning("No text extracted from %s", p.name)

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

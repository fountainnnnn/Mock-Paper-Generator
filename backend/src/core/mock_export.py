# backend/src/core/mock_export.py
# -*- coding: utf-8 -*-
"""
PDF export for generated mock exam papers.

Primary path:
- HTML + KaTeX + Playwright  -> crisp math, clean layout, lists/tables
Fallback:
- ReportLab text-only PDF     -> no browser needed, legacy-compatible

Public API:
- build_mockpaper_pdf(text, out_path, title="...", source_name=None, instructions=None)
- build_mockpaper_pdf_from_spec(spec: dict, out_path: str)
"""

from __future__ import annotations

from typing import Optional, List, Tuple, Dict, Any
from pathlib import Path
import re
import html

# -------- Optional imports --------
try:
    from playwright.sync_api import sync_playwright  # type: ignore
    _HAS_PLAYWRIGHT = True
except Exception:
    _HAS_PLAYWRIGHT = False

try:
    from markdown_it import MarkdownIt  # type: ignore
    from mdit_py_plugins.footnote import footnote_plugin  # type: ignore
    from mdit_py_plugins.anchors import anchors_plugin  # type: ignore
    _HAS_MD = True
except Exception:
    _HAS_MD = False

try:
    from jinja2 import Environment, BaseLoader, select_autoescape  # type: ignore
    _HAS_JINJA = True
except Exception:
    _HAS_JINJA = False


# ============================================================
# HTML / KaTeX rendering (preferred)
# ============================================================
_CSS = """
@page { size: A4; margin: 20mm 18mm 20mm 18mm; }
body {
  font-family: 'Segoe UI', Helvetica, Arial, sans-serif;
  color: #222;
  font-size: 12pt;
  line-height: 1.6;
}

h1 {
  font-size: 24pt;
  margin: 0 0 10mm 0;
  color: #1a3d7c;
  border-bottom: 3px solid #1a3d7c;
  padding-bottom: 4mm;
}
h2 {
  font-size: 16pt;
  margin: 8mm 0 4mm 0;
  color: #4d2c91;
  border-left: 5px solid #4d2c91;
  padding-left: 6px;
}

.meta {
  color: #555;
  background: #f3f3f3;
  border-left: 4px solid #999;
  padding: 4px 8px;
  margin: 0 0 8mm 0;
  font-style: italic;
}

.section { page-break-inside: avoid; }

.item {
  margin: 6mm 0 6mm 0;
  padding: 4px 0;
}

.points {
  color: #444;
  font-style: italic;
  font-size: 10pt;
  background: #eef;
  padding: 2px 6px;
  border-radius: 4px;
  display: inline-block;
  margin-top: 3mm;
}

.mcq {
  background: #fafafa;
  border: 1px solid #ddd;
  border-radius: 6px;
  padding: 4mm;
  margin-top: 2mm;
}
.mcq ol {
  list-style-type: upper-alpha;
  padding-left: 7mm;
  margin: 0;
}
.mcq li { margin: 1mm 0; }

.figure { text-align:center; margin: 5mm 0; }

.katex { font-size: 1.06em; }
.katex-display { margin: 5mm 0 !important; }

pre, code {
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
  background: #f7f7f7;
  padding: 2px 4px;
  border-radius: 3px;
}

.hr {
  border: 0;
  height: 1px;
  background: #ccc;
  margin: 10mm 0;
}
"""

_HTML_TMPL = """<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <style>""" + _CSS + """</style>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css"/>
  <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>
  <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js"></script>
  <script>
    document.addEventListener("DOMContentLoaded", function() {
      renderMathInElement(document.body, {
        delimiters: [
          {left: "$$", right: "$$", display: true},
          {left: "\\[", right: "\\]", display: true},
          {left: "$", right: "$", display: false},
          {left: "\\(", right: "\\)", display: false}
        ],
        throwOnError: false,
        strict: false
      });
    });
  </script>
</head>
<body>

<h1>{{ title }}</h1>
{% if source_name %}<div class="meta">Generated from {{ source_name }}</div>{% endif %}
{% if instructions %}<div class="meta"><em>{{ instructions|safe }}</em></div>{% endif %}

{{ body|safe }}

</body>
</html>
"""

def _mk_md():
    if not _HAS_MD:
        class _Dummy:
            def render(self, s: str) -> str: return html.escape(s).replace("\n", "<br/>")
        return _Dummy()
    return (MarkdownIt("commonmark", {"html": False, "linkify": True})
            .use(footnote_plugin)
            .use(anchors_plugin, permalink=False))

def _mk_env():
    if not _HAS_JINJA:
        raise RuntimeError("Jinja2 not available; cannot render HTML template.")
    return Environment(loader=BaseLoader(), autoescape=select_autoescape())


# ---------------- Plain-text → HTML (heuristic) ----------------
_OPT_RE    = re.compile(r"^\s*([A-Da-d])[\.\)]\s+(.+)")
_Q_RE      = re.compile(r"^\s*((?:Q\s*)?\d+[\.\)]\s+.+)", re.I)
_SEC_RE    = re.compile(r"^\s*section\b", re.I)
_MARKS_RE  = re.compile(r"\bmark\b", re.I)

def _group_lines_into_html(paper_text: str) -> str:
    lines = paper_text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    md = _mk_md()

    html_parts: List[str] = []
    buffer_question: List[str] = []
    buffer_opts: List[Tuple[str, str]] = []
    in_opts = False

    def flush_question():
        nonlocal buffer_question, buffer_opts, in_opts
        if not buffer_question and not buffer_opts:
            return
        q_html = md.render("\n".join(buffer_question).strip())
        if buffer_opts:
            html_parts.append("<div class='item'>")
            html_parts.append(f"<div class='stem'>{q_html}</div>")
            html_parts.append("<div class='mcq'><ol>")
            for key, txt in buffer_opts:
                html_parts.append(f"<li>{md.render(txt)}</li>")
            html_parts.append("</ol></div>")
            html_parts.append("</div>")
        else:
            html_parts.append(f"<div class='item'>{q_html}</div>")
        buffer_question = []
        buffer_opts = []
        in_opts = False

    for raw in lines:
        ln = raw.strip()
        if not ln:
            flush_question()
            html_parts.append("<div style='height:5mm'></div>")
            continue

        if _SEC_RE.match(ln):
            flush_question()
            html_parts.append(f"<h2>{html.escape(ln)}</h2>")
            continue

        mopt = _OPT_RE.match(ln)
        if mopt:
            in_opts = True
            key = mopt.group(1).upper()
            txt = mopt.group(2)
            buffer_opts.append((key, txt))
            continue

        if _Q_RE.match(ln):
            flush_question()
            buffer_question.append(ln)
            continue

        if _MARKS_RE.search(ln):
            flush_question()
            html_parts.append(f"<div class='points'>{html.escape(ln)}</div>")
            continue

        if in_opts:
            k, prev = buffer_opts[-1]
            buffer_opts[-1] = (k, prev + " " + ln)
        else:
            if buffer_question:
                buffer_question.append(ln)
            else:
                html_parts.append(f"<div class='item'>{md.render(ln)}</div>")

    flush_question()
    return "\n".join(html_parts)


def _html_wrapper(body_html: str, title: str, source_name: Optional[str], instructions: Optional[str] = None) -> str:
    env = _mk_env()
    tpl = env.from_string(_HTML_TMPL)
    return tpl.render(title=title, source_name=source_name, instructions=instructions, body=body_html)


def _html_to_pdf(html_str: str, out_pdf: str, title: str = "Mock Paper") -> bool:
    if not _HAS_PLAYWRIGHT:
        return False
    out = Path(out_pdf)
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page()
            page.set_content(html_str, wait_until="load")
            page.pdf(
                path=str(out),
                format="A4",
                display_header_footer=True,
                header_template=(
                    "<div style='font-size:10px; width:100%; text-align:left; "
                    "padding-left:14mm; color:#4d3d84;'>"
                    + html.escape(title) +
                    "</div>"
                ),
                footer_template=(
                    "<div style='font-size:10px; width:100%; text-align:center;'>"
                    "Page <span class='pageNumber'></span></div>"
                ),
                margin={"top": "20mm", "right": "18mm", "bottom": "20mm", "left": "18mm"},
                print_background=True,
            )
            browser.close()
        return True
    except Exception:
        return False


# ============================================================
# Fallback: ReportLab text-only
# ============================================================
from reportlab.lib.pagesizes import A4 as _A4
from reportlab.platypus import SimpleDocTemplate as _SimpleDocTemplate, Paragraph as _Paragraph, Spacer as _Spacer
from reportlab.lib.styles import getSampleStyleSheet as _getSampleStyleSheet, ParagraphStyle as _ParagraphStyle
from reportlab.lib.units import mm as _mm

def _reportlab_text_pdf(text: str, out_pdf: str, title: str, source_name: Optional[str], instructions: Optional[str]) -> None:
    out = Path(out_pdf)
    out.parent.mkdir(parents=True, exist_ok=True)

    styles = _getSampleStyleSheet()
    base = styles["BodyText"]
    base.fontName = "Times-Roman"
    base.fontSize = 11
    base.leading = 14

    title_style = _ParagraphStyle(
        "Title", parent=base, fontSize=16, leading=20, spaceAfter=6*_mm, textColor="#233d7b"
    )
    meta_style = _ParagraphStyle(
        "Meta", parent=base, fontSize=10, leading=12, textColor="#666666", spaceAfter=4*_mm, italic=True
    )

    story: List[Any] = []
    story.append(_Paragraph(html.escape(title), title_style))
    if source_name:
        story.append(_Paragraph(f"Generated from {html.escape(source_name)}", meta_style))
    if instructions:
        story.append(_Paragraph(html.escape(instructions), meta_style))
    story.append(_Spacer(1, 6*_mm))

    paras = re.split(r"\n\s*\n", text.strip())
    for para in paras:
        para_html = "<br/>".join(html.escape(line) for line in para.splitlines())
        story.append(_Paragraph(para_html, base))
        story.append(_Spacer(1, 3*_mm))

    doc = _SimpleDocTemplate(str(out), pagesize=_A4,
                             leftMargin=18*_mm, rightMargin=18*_mm,
                             topMargin=20*_mm, bottomMargin=20*_mm)
    doc.build(story)


# ============================================================
# Public API
# ============================================================

def build_mockpaper_pdf(
    text: str,
    out_path: str,
    title: str = "Mock Exam Paper",
    source_name: Optional[str] = None,
    is_answer_key: bool = False,
    instructions: Optional[str] = None,
) -> str:
    eff_title = f"{title}{' — Answers' if is_answer_key else ''}"

    body_html = _group_lines_into_html(text)
    html_doc = _html_wrapper(body_html, title=eff_title, source_name=source_name, instructions=instructions)

    ok = _html_to_pdf(html_doc, out_path, title=eff_title)
    if not ok:
        _reportlab_text_pdf(text, out_path, eff_title, source_name, instructions)

    return str(out_path)


def build_mockpaper_pdf_from_spec(spec: Dict[str, Any], out_path: str) -> str:
    title = spec.get("title") or "Mock Exam Paper"
    source_name = spec.get("source_name")
    instructions = spec.get("instructions")
    is_answer_key = bool(spec.get("is_answer_key", False))

    parts: List[str] = []
    for sec in spec.get("sections", []):
        heading = sec.get("heading")
        if heading:
            parts.append(heading)
        for item in sec.get("items", []):
            parts.append(item)
        parts.append("")

    plain = "\n".join(parts).strip() or "(empty)"
    return build_mockpaper_pdf(
        text=plain,
        out_path=out_path,
        title=title,
        source_name=source_name,
        is_answer_key=is_answer_key,
        instructions=instructions,
    )

# backend/src/core/pdf_builder.py
# -*- coding: utf-8 -*-
"""
ReportLab-only PDF builder for mock exam papers with stitched LaTeX math.

- Coalesces multi-line math blocks (\[...\], \(...\), $$...$$).
- Renders LaTeX math as baseline-aligned images (matplotlib).
- OCR normalization (• · × − → LaTeX-safe).
- Styles for sections, questions, answers, marks.
- MCQ options a./b./c./d. are printed on one line.
"""

from typing import Optional, List, Union
from pathlib import Path
import re, io

from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak,
    Flowable, Image
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.lib import colors

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------- Layout constants ----------------
LEFT_MARGIN  = 56
RIGHT_MARGIN = 56
TOP_MARGIN   = 64
BOTTOM_MARGIN= 64
BASE_FONTSIZE = 12
BASE_LEADING  = 16

# ---------------- Palette ----------------
ACCENT         = colors.HexColor("#1a3d7c")
SECTION        = colors.HexColor("#4d2c91")
OK_GREEN       = colors.HexColor("#1e9e62")
SOFT_GREY      = colors.HexColor("#555555")
HAIRLINE       = colors.HexColor("#DDDDDD")
LIGHT_GREEN_BG = colors.HexColor("#e6f9f0")
LIGHT_BLUE_BG  = colors.HexColor("#eef3ff")
LIGHT_BOX_BG   = colors.HexColor("#fafafa")

# ---------------- Styles ----------------
styles = getSampleStyleSheet()

style_cover_title = ParagraphStyle(
    "CoverTitle",
    parent=styles["Title"],
    fontName="Helvetica-Bold",
    fontSize=26,
    leading=32,
    alignment=TA_LEFT,
    textColor=ACCENT,
    spaceAfter=12,
)

style_cover_sub = ParagraphStyle(
    "CoverSub",
    parent=styles["Normal"],
    fontName="Helvetica-Oblique",
    fontSize=12,
    leading=16,
    alignment=TA_LEFT,
    textColor=SOFT_GREY,
    spaceAfter=10,
)

style_instr_head = ParagraphStyle(
    "InstrHead",
    parent=styles["Normal"],
    fontName="Helvetica-Bold",
    fontSize=12,
    leading=16,
    textColor=SECTION,
    spaceAfter=4,
)

style_instr_body = ParagraphStyle(
    "InstrBody",
    parent=styles["Normal"],
    fontName="Helvetica",
    fontSize=11.5,
    leading=16,
    textColor=colors.black,
    backColor=LIGHT_BOX_BG,
    spaceBefore=2,
    spaceAfter=10,
    leftIndent=6,
    rightIndent=6,
)

style_body = ParagraphStyle(
    "Body",
    parent=styles["Normal"],
    fontName="Helvetica",
    fontSize=BASE_FONTSIZE,
    leading=BASE_LEADING,
    spaceBefore=0,
    spaceAfter=6,
)

style_question = ParagraphStyle(
    "Question",
    parent=style_body,
    fontName="Helvetica-Bold",
    spaceBefore=8,
    spaceAfter=4,
)

style_option = ParagraphStyle(
    "Option",
    parent=style_body,
    leftIndent=18,
    spaceBefore=0,
    spaceAfter=2,
)

style_answer = ParagraphStyle(
    "Answer",
    parent=style_body,
    leftIndent=10,
    backColor=LIGHT_GREEN_BG,
    textColor=OK_GREEN,
    spaceBefore=4,
    spaceAfter=8,
)

style_marks = ParagraphStyle(
    "Marks",
    parent=style_body,
    alignment=TA_LEFT,
    textColor=ACCENT,
    backColor=LIGHT_BLUE_BG,
    spaceBefore=4,
    spaceAfter=8,
    leftIndent=6,
    rightIndent=6,
)

style_section = ParagraphStyle(
    "SectionHeader",
    parent=style_body,
    fontName="Helvetica-Bold",
    fontSize=15,
    leading=20,
    textColor=SECTION,
    spaceBefore=12,
    spaceAfter=8,
)

style_blockmath = ParagraphStyle(
    "BlockMath",
    parent=style_body,
    alignment=TA_CENTER,
    spaceBefore=6,
    spaceAfter=6,
)

# ---------------- Footer & Header ----------------
def _footer(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(HAIRLINE)
    canvas.setLineWidth(0.5)
    canvas.line(LEFT_MARGIN, 52, A4[0]-RIGHT_MARGIN, 52)
    canvas.setFont("Helvetica", 9)
    canvas.setFillColor(colors.black)
    canvas.drawString(LEFT_MARGIN, 40, "Mock Paper Generator")
    canvas.drawRightString(A4[0]-RIGHT_MARGIN, 40, f"Page {doc.page}")
    canvas.restoreState()

def _header(canvas, doc, title: str):
    canvas.saveState()
    canvas.setFillColor(ACCENT)
    canvas.setFont("Helvetica-Bold", 10.5)
    canvas.drawString(LEFT_MARGIN, A4[1]-42, title)
    canvas.setStrokeColor(ACCENT)
    canvas.setLineWidth(2)
    canvas.line(LEFT_MARGIN, A4[1]-46, A4[0]-RIGHT_MARGIN, A4[1]-46)
    canvas.restoreState()


# ---------------- Helpers ----------------
def _ocr_normalize(s: str) -> str:
    return (s.replace("•", "\\cdot").replace("·", "\\cdot").replace("×", "\\cdot")
              .replace("−", "-").replace("–", "-").replace("—", "-")
              .replace("⁄", "/").replace("°", "^{\\circ}"))

# Inline math: \( ... \) or $...$
INLINE_RE = re.compile(r"(?:\\\((.*?)\\\)|\$(.+?)\$)")
# Block math: \[ ... \] or $$ ... $$  — must fill whole line after trimming
BLOCK_LINE_RE = re.compile(r"^\s*(?:\\\[(.*?)\\\]|\$\$(.+?)\$\$)\s*$")

def _sanitize_math(expr: str) -> str:
    expr = expr.strip()
    # be forgiving: users sometimes include delimiters inside captures
    if expr.startswith("\\(") and expr.endswith("\\)"):
        expr = expr[2:-2].strip()
    if expr.startswith("\\[") and expr.endswith("\\]"):
        expr = expr[2:-2].strip()
    if expr.startswith("$") and expr.endswith("$"):
        expr = expr[1:-1].strip()
    # % must be escaped for mathtext
    expr = expr.replace("%", r"\%")
    # \text{...} is not fully supported by mathtext; map to \mathrm{...}
    expr = re.sub(r"\\text\{([^}]*)\}", r"\\mathrm{\1}", expr)
    return _ocr_normalize(expr)

def _render_math(expr: str, fontsize: int = 12, height: int = 14) -> Image:
    fig = plt.figure(figsize=(0.01, 0.01))
    fig.text(0, 0, f"${expr}$", fontsize=fontsize)
    buf = io.BytesIO()
    plt.axis("off")
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.05,
                dpi=300, transparent=True)
    plt.close(fig)
    buf.seek(0)
    return Image(buf, height=height, kind="proportional")


# --------- Coalesce multi-line math blocks ----------
def _stitch_broken_math_lines(lines: List[str]) -> List[str]:
    """
    Joins lines when a LaTeX span starts on one line and ends on a later line.
    Handles: \( ... \), \[ ... \], $$ ... $$
    """
    out: List[str] = []
    buf: List[str] = []
    mode = None  # "inline_paren" | "block_bracket" | "block_dollar"

    def opens(s: str) -> Optional[str]:
        if "\\(" in s and "\\)" not in s:
            return "inline_paren"
        if "\\[" in s and "\\]" not in s:
            return "block_bracket"
        # $$ span start (odd number of $$ across the line = open)
        if s.count("$$") == 1:
            return "block_dollar"
        return None

    def closes(s: str, m: str) -> bool:
        if m == "inline_paren":
            return "\\)" in s
        if m == "block_bracket":
            return "\\]" in s
        if m == "block_dollar":
            return s.count("$$") == 1
        return False

    for raw in lines:
        s = raw.rstrip()
        if mode is None:
            m = opens(s)
            if m:
                mode = m
                buf = [s]
            else:
                out.append(s)
        else:
            buf.append(s)
            if closes(s, mode):
                out.append(" ".join(buf))  # join with a space to avoid word-glue
                buf = []
                mode = None
    # flush if unterminated (best effort)
    if buf:
        out.append(" ".join(buf))
    return out


# ---------------- Build ----------------
def build_mockpaper_pdf(
    text: str,
    out_path: str,
    title: str = "Generated Mock Exam Paper",
    source_name: Optional[str] = None,
    is_answer_key: bool = False,
):
    raw_lines = text.replace("\r\n", "\n").replace("\r", "\n").splitlines()
    raw_lines = [_ocr_normalize(s) for s in raw_lines]
    # stitch broken \( ... \), \[ ... \], $$ ... $$ across line breaks
    lines = _stitch_broken_math_lines(raw_lines)

    story: List[Union[Flowable, Paragraph]] = []

    # --- Cover ---
    story.append(Spacer(1, 22))
    story.append(Paragraph(title, style_cover_title))
    if source_name:
        story.append(Paragraph(source_name, style_cover_sub))
    story.append(Paragraph("Instructions", style_instr_head))
    story.append(Paragraph(
        "Answer all questions. Show full working. Round off appropriately.",
        style_instr_body
    ))
    story.append(PageBreak())

    # --- Body ---
    i = 0
    q_counter = 0
    while i < len(lines):
        line = lines[i].strip()

        if not line:
            story.append(Spacer(1, 8))
            i += 1
            continue

        # Section headers
        if re.match(r"^\s*\d+\.\s*[A-Za-z]", line) and not line.lower().startswith(("q", "q1", "q2")):
            # e.g. "1. Differentiation and Tangents"
            story.append(Spacer(1, 6))
            story.append(Paragraph(line, style_section))
            i += 1
            continue

        # Numbered questions: "q1.", "1.", "(1)", "Q1"
        if re.match(r"^\s*(?:q\s*\d+|\(?\d+\)?[.)])", line, flags=re.I):
            q_counter += 1
            story.append(Paragraph(line, style_question))
            i += 1
            continue

        # MCQ options a./b./c./d. on one line
        if re.match(r"^[a-d]\.", line, flags=re.I):
            options = [line.strip()]
            j = i + 1
            while j < len(lines) and re.match(r"^[a-d]\.", lines[j].strip(), flags=re.I):
                options.append(lines[j].strip())
                j += 1
            story.append(Paragraph("    ".join(options), style_option))
            i = j
            continue

        # Marks
        if "mark" in line.lower():
            story.append(Paragraph(line, style_marks))
            i += 1
            continue

        # Answers page
        if is_answer_key and line.lower().startswith(("answer", "ans:", "solution")):
            story.append(Paragraph(line, style_answer))
            i += 1
            continue

        # Full-line block math
        m_block = BLOCK_LINE_RE.match(line)
        if m_block:
            expr = next(g for g in m_block.groups() if g)
            try:
                story.append(_render_math(_sanitize_math(expr), fontsize=14, height=22))
            except Exception:
                story.append(Paragraph(line, style_blockmath))
            i += 1
            continue

        # Inline math within text
        parts: List[Flowable] = []
        pos = 0
        for m in INLINE_RE.finditer(line):
            if m.start() > pos:
                parts.append(Paragraph(line[pos:m.start()], style_body))
            expr = next(g for g in m.groups() if g)
            try:
                parts.append(_render_math(_sanitize_math(expr)))
            except Exception:
                parts.append(Paragraph(expr, style_body))
            pos = m.end()
        if pos < len(line):
            parts.append(Paragraph(line[pos:], style_body))
        if parts:
            story.extend(parts)
        else:
            story.append(Paragraph(line, style_body))
        i += 1

    # --- Build ---
    out = Path(out_path); out.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(out),
        pagesize=A4,
        topMargin=TOP_MARGIN, bottomMargin=BOTTOM_MARGIN,
        leftMargin=LEFT_MARGIN, rightMargin=RIGHT_MARGIN
    )

    doc.build(
        story,
        onFirstPage=lambda c, d: (_header(c, d, title), _footer(c, d)),
        onLaterPages=lambda c, d: (_header(c, d, title), _footer(c, d)),
    )

    return str(out)
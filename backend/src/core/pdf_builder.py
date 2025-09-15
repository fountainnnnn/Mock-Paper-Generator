# backend/src/core/pdf_builder.py
# -*- coding: utf-8 -*-
"""
ReportLab-only PDF builder for mock exam papers with a cleaner, colorful design.

- Coalesces multi-line math blocks (\[...\], \(...\), $$...$$).
- Renders math as baseline-aligned images (matplotlib), auto-fits width.
- Conservative math-line stitcher prevents vertical stacks WITHOUT touching normal sentences.
- OCR normalization (• · × − → LaTeX-safe); optional square→π disabled by default.
- Only “unsquashes” when a line is pure letters with ZERO spaces (very safe).
- Inline/block math handled: $...$, \(...\), \[...\], $$...$$
- Gentle punctuation spacing fix for plain text lines.

Design upgrades:
- Left-aligned content (no “right-handed” feel).
- Section headers with color accent.
- Shaded answer boxes and marks badges.
- More breathing room (consistent vertical rhythm).
- Updated header/footer styling.
"""

from typing import Optional, List, Union
from pathlib import Path
import os, re, tempfile

from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak,
    Flowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.lib import colors
from reportlab.lib.utils import ImageReader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Optional LaTeX sniff
try:
    from sympy.parsing.latex import parse_latex
    HAS_SYMPY = True
except Exception:
    HAS_SYMPY = False


# ---------------- TUNABLES ----------------
JOIN_MATH_RUNS_MODE = "strict"     # "off" | "strict" | "loose"
REQUIRE_OPERATOR_FOR_BLOCK = True
NORMALIZE_SQUARE_TO_PI = False
ENABLE_UNSQUASH_SAFE = True

# Visual (math)
MATH_H_SHRINK = 0.90

# ---------------- Layout constants ----------------
LEFT_MARGIN  = 56
RIGHT_MARGIN = 56
TOP_MARGIN   = 64
BOTTOM_MARGIN= 64
USABLE_WIDTH = A4[0] - LEFT_MARGIN - RIGHT_MARGIN

# Typography
BASE_FONTSIZE = 12
BASE_LEADING  = 16
MATH_IMG_H    = 12
MATH_IMG_H_FRAC = 15

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
    spaceAfter=6,   # extra breathing room
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

style_smallmeta = ParagraphStyle(
    "SmallMeta",
    parent=style_body,
    fontName="Helvetica-Oblique",
    fontSize=9.5,
    leading=12,
    textColor=SOFT_GREY,
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


# ---------------- Math handling ----------------
INLINE_RE       = re.compile(r"(?:\\\((.*?)\\\)|\$(.+?)\$)")
BLOCK_LINE_RE   = re.compile(r"^(?:\\\[(.*?)\\\]|\$\$(.+?)\$\$|\$(.+)\$)$")
INLINE_BLOCK_RE = re.compile(r"(\\\[(.+?)\\\]|\$\$(.+?)\$\$)")
_FRACISH_RE     = re.compile(r"(\\(?:d|t)?frac\b|\\over|\d+\s*/\s*[\da-zA-Z(])")

def _ocr_normalize(s: str) -> str:
    s = (s.replace("•", "\\cdot").replace("·", "\\cdot").replace("×", "\\cdot")
           .replace("−", "-").replace("–", "-").replace("—", "-")
           .replace("⁄", "/").replace("°", "^{\\circ}"))
    if NORMALIZE_SQUARE_TO_PI:
        s = s.replace("■", "\\pi").replace("□", "\\pi")
    return s

def _sanitize_math(expr: str) -> str:
    expr = expr.strip()
    if expr.startswith("$") and expr.endswith("$"):
        expr = expr[1:-1].strip()
    if expr.startswith("\\(") and expr.endswith("\\)"):
        expr = expr[2:-2].strip()

    # --- Fix: escape percent signs ---
    expr = expr.replace("%", r"\%")
    # --- Fix: wrap long words in \text{} ---
    expr = re.sub(r"([A-Za-z]{3,})", r"\\text{\1}", expr)

    return _ocr_normalize(expr)

def _looks_like_real_math(expr: str) -> bool:
    s = _ocr_normalize(expr.strip())
    if not s:
        return False
    # too many words → treat as text
    if len(re.findall(r"[A-Za-z]{3,}", s)) > 6:
        return False
    if any(tok in s for tok in ("\\frac","\\sqrt","\\pi","\\cdot","\\sum","\\int","\\lim",
                                "\\sin","\\cos","\\tan","\\ln","\\log")): return True
    if re.search(r"[_^]", s): return True
    if re.search(r"[=+\-*/><]", s) and re.search(r"\d", s): return True
    if HAS_SYMPY:
        try: parse_latex(s); return True
        except Exception: pass
    return False


# (… keep all other helper functions as in your last version …)


# ---------------- Build ----------------
def build_mockpaper_pdf(
    text: str,
    out_path: str,
    title: str = "Generated Mock Exam Paper",
    source_name: Optional[str] = None,
    is_answer_key: bool = False
):
    raw_lines = text.replace("\r\n", "\n").replace("\r", "\n").splitlines()
    raw_lines = [_ocr_normalize(s) for s in raw_lines]
    lines     = [_ocr_normalize(s) for s in raw_lines]

    story: List[Union[Flowable, Paragraph]] = []

    # --- Cover ---
    story.append(Spacer(1, 22))
    story.append(Paragraph(title, style_cover_title))
    if source_name:
        story.append(Paragraph(source_name, style_cover_sub))
    story.append(Paragraph("Instructions", style_instr_head))
    story.append(Paragraph("Answer all questions. Show full working. Round off appropriately.", style_instr_body))
    story.append(PageBreak())

    # --- Body ---
    for raw in lines:
        line = raw.strip()
        if not line:
            story.append(Spacer(1, 8))
            continue

        # Section headers
        if re.match(r"^\s*section\b", line, re.I):
            story.append(Spacer(1, 6))
            story.append(Paragraph(line, style_section))
            continue

        # (… body parsing same as before …)

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

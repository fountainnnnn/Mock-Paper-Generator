# backend/src/core/pdf_builder.py
# -*- coding: utf-8 -*-
"""
ReportLab-only PDF builder for mock exam papers (Unicode math mode).

- ASCII-safe math (x^2, H2O, pi, theta) is converted into Unicode characters (², ₁, θ, π).
- Preserves valid Unicode math (π, θ, √, ∑, ∫, ∞, etc.) if present.
- OCR normalization for basic symbols and junk stripping (■, ▮, █ → *).
- Styles for sections, questions, answers, marks.
- MCQ options a./b./c./d. are printed line by line.
- Markdown-like tables |a|b| → rendered as ReportLab tables (works in both question and answer key).
"""

from typing import Optional, List, Union
from pathlib import Path
import re

from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak,
    Flowable, Table, TableStyle
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# ---------------- Register Unicode font ----------------
FONT_PATH = Path(__file__).resolve().parent.parent / "assets" / "fonts" / "STIXTwoMath-Regular.ttf"
if FONT_PATH.exists():
    pdfmetrics.registerFont(TTFont("STIXTwoMath", str(FONT_PATH)))
    DEFAULT_FONT = "STIXTwoMath"
else:
    DEFAULT_FONT = "Helvetica"  # fallback

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
    fontName=DEFAULT_FONT,
    fontSize=26,
    leading=32,
    alignment=TA_LEFT,
    textColor=ACCENT,
    spaceAfter=12,
)

style_cover_sub = ParagraphStyle(
    "CoverSub",
    parent=styles["Normal"],
    fontName=DEFAULT_FONT,
    fontSize=12,
    leading=16,
    alignment=TA_LEFT,
    textColor=SOFT_GREY,
    spaceAfter=10,
)

style_instr_head = ParagraphStyle(
    "InstrHead",
    parent=styles["Normal"],
    fontName=DEFAULT_FONT,
    fontSize=12,
    leading=16,
    textColor=SECTION,
    spaceAfter=4,
)

style_instr_body = ParagraphStyle(
    "InstrBody",
    parent=styles["Normal"],
    fontName=DEFAULT_FONT,
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
    fontName=DEFAULT_FONT,
    fontSize=BASE_FONTSIZE,
    leading=BASE_LEADING,
    spaceBefore=0,
    spaceAfter=6,
)

style_question = ParagraphStyle(
    "Question",
    parent=style_body,
    fontName=DEFAULT_FONT,
    spaceBefore=8,
    spaceAfter=4,
)

style_option = ParagraphStyle(
    "Option",
    parent=style_body,
    fontName=DEFAULT_FONT,
    leftIndent=18,
    spaceBefore=0,
    spaceAfter=2,
)

style_answer = ParagraphStyle(
    "Answer",
    parent=style_body,
    fontName=DEFAULT_FONT,
    leftIndent=14,
    backColor=LIGHT_GREEN_BG,
    textColor=OK_GREEN,
    spaceBefore=6,
    spaceAfter=10,
    borderWidth=0.5,
    borderColor=OK_GREEN,
    borderPadding=4,
    leading=BASE_LEADING,
)

style_marks = ParagraphStyle(
    "Marks",
    parent=style_body,
    fontName=DEFAULT_FONT,
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
    fontName=DEFAULT_FONT,
    fontSize=15,
    leading=20,
    textColor=SECTION,
    spaceBefore=12,
    spaceAfter=8,
)

style_blockmath = ParagraphStyle(
    "BlockMath",
    parent=style_body,
    fontName=DEFAULT_FONT,
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
    canvas.setFont(DEFAULT_FONT, 9)
    canvas.setFillColor(colors.black)
    canvas.drawString(LEFT_MARGIN, 40, "Mock Paper Generator")
    canvas.drawRightString(A4[0]-RIGHT_MARGIN, 40, f"Page {doc.page}")
    canvas.restoreState()

def _header(canvas, doc, title: str):
    canvas.saveState()
    canvas.setFillColor(ACCENT)
    canvas.setFont(DEFAULT_FONT, 10.5)
    canvas.drawString(LEFT_MARGIN, A4[1]-42, title)
    canvas.setStrokeColor(ACCENT)
    canvas.setLineWidth(2)
    canvas.line(LEFT_MARGIN, A4[1]-46, A4[0]-RIGHT_MARGIN, A4[1]-46)
    canvas.restoreState()

# ---------------- Unicode escape decoder ----------------
def decode_unicode_escapes(text: str) -> str:
    """Decode sequences like 'U+03C0' into real Unicode characters."""
    if not text:
        return text
    def repl(m):
        try:
            return chr(int(m.group(1), 16))
        except Exception:
            return m.group(0)
    return re.sub(r"U\+([0-9A-Fa-f]{4,6})", repl, text)

# ---------------- Math prettifier ----------------
_SUPERSCRIPT_MAP = str.maketrans("0123456789+-=", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼")
_SUBSCRIPT_MAP   = str.maketrans("0123456789+-=", "₀₁₂₃₄₅₆₇₈₉₊₋₌")

def prettify_ascii_math(expr: str) -> str:
    """Convert ASCII-safe math (x^2, H2O, pi, theta) to Unicode pretty math."""
    expr = decode_unicode_escapes(expr)
    expr = expr.replace("\\cdot", "·").replace("\\times", "·")
    expr = expr.replace("*", "·")
    expr = expr.replace("sqrt", "√")
    expr = expr.replace("exp", "e")  

    expr = re.sub(r"\^([0-9+\-=]+)", lambda m: m.group(1).translate(_SUPERSCRIPT_MAP), expr)
    expr = re.sub(r"_([0-9+\-=]+)", lambda m: m.group(1).translate(_SUBSCRIPT_MAP), expr)

    expr = expr.replace("pi", "π").replace("theta", "θ")
    expr = expr.replace("{", "").replace("}", "")
    expr = expr.replace("[", "").replace("]", "")

    return expr

def _ocr_normalize(s: str) -> str:
    """Normalize OCR/LLM quirks into safe text/math symbols."""
    s = decode_unicode_escapes(s)
    s = (s.replace("•", ".")
           .replace("·", ".")
           .replace("×", "*")
           .replace("−", "-").replace("–", "-").replace("—", "-")
           .replace("⁄", "/").replace("°", " deg"))
    keep = {"π", "θ", "√", "∑", "∫", "∞"}
    s2 = []
    for ch in s:
        if ch in keep:
            s2.append(ch)
        elif ch in {"■", "▮", "█", "▪", "▫", "◼", "◾", "◽"}:
            s2.append("*")
        else:
            s2.append(ch)
    return "".join(s2)

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
    lines = raw_lines

    story: List[Union[Flowable, Paragraph]] = []

    story.append(Spacer(1, 22))
    # Differentiate cover page for QP vs Answer key
    if is_answer_key:
        story.append(Paragraph(f"{title} — Answer Key", style_cover_title))
    else:
        story.append(Paragraph(f"{title} — Question Paper", style_cover_title))
    if source_name:
        story.append(Paragraph(source_name, style_cover_sub))
    story.append(Paragraph("Instructions", style_instr_head))
    story.append(Paragraph(
        "Answer all questions. Show full working. Round off appropriately.",
        style_instr_body
    ))
    story.append(PageBreak())

    i = 0
    q_counter = 0
    while i < len(lines):
        line = lines[i].strip()

        if not line:
            story.append(Spacer(1, 8))
            i += 1
            continue

        if re.match(r"^\s*\d+\.\s*[A-Za-z]", line) and not line.lower().startswith(("q", "q1", "q2")):
            story.append(Spacer(1, 6))
            story.append(Paragraph(prettify_ascii_math(line), style_section))
            i += 1
            continue

        if re.match(r"^\s*(?:q\s*\d+|\(?\d+\)?[.)])", line, flags=re.I):
            q_counter += 1
            story.append(Paragraph(prettify_ascii_math(line), style_question))
            i += 1
            continue

        if re.match(r"^[a-d]\.", line, flags=re.I):
            j = i
            while j < len(lines) and re.match(r"^[a-d]\.", lines[j].strip(), flags=re.I):
                story.append(Paragraph(prettify_ascii_math(lines[j].strip()), style_option))
                j += 1
            i = j
            continue

        if "mark" in line.lower():
            story.append(Paragraph(prettify_ascii_math(line), style_marks))
            i += 1
            continue

        if is_answer_key:
            if re.match(r"^\|.+\|$", line):
                table_lines = []
                while i < len(lines) and re.match(r"^\|.+\|$", lines[i].strip()):
                    row = [prettify_ascii_math(c.strip()) for c in lines[i].strip().strip("|").split("|")]
                    table_lines.append(row)
                    i += 1
                tbl = Table(table_lines, style=TableStyle([
                    ("GRID", (0,0), (-1,-1), 0.5, colors.black),
                    ("FONTNAME", (0,0), (-1,-1), DEFAULT_FONT),
                    ("FONTSIZE", (0,0), (-1,-1), BASE_FONTSIZE),
                    ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
                    ("ALIGN", (0,0), (-1,-1), "CENTER"),
                ]))
                story.append(tbl)
                continue

            # For answer keys: keep question number only on the first line of each answer
            if re.match(r"^\s*(?:\d+\s*[.)])", line):
                # First line of answer → keep number
                story.append(Paragraph(prettify_ascii_math(line), style_answer))
            else:
                # Subsequent lines → strip any leftover numbering
                clean_line = re.sub(r"^\s*(?:[a-d][.)])\s*", "", line, flags=re.I)
                story.append(Paragraph(prettify_ascii_math(clean_line), style_answer))
            i += 1
            continue

        if re.match(r"^\|.+\|$", line):
            table_lines = []
            while i < len(lines) and re.match(r"^\|.+\|$", lines[i].strip()):
                row = [prettify_ascii_math(c.strip()) for c in lines[i].strip().strip("|").split("|")]
                table_lines.append(row)
                i += 1
            tbl = Table(table_lines, style=TableStyle([
                ("GRID", (0,0), (-1,-1), 0.5, colors.black),
                ("FONTNAME", (0,0), (-1,-1), DEFAULT_FONT),
                ("FONTSIZE", (0,0), (-1,-1), BASE_FONTSIZE),
                ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
                ("ALIGN", (0,0), (-1,-1), "CENTER"),
            ]))
            story.append(tbl)
            continue

        story.append(Paragraph(prettify_ascii_math(line), style_body))
        i += 1

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

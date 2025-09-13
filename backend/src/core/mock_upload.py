# backend/src/core/pdf_builder.py
# -*- coding: utf-8 -*-
"""
Styled PDF builder for generated mock exam papers and answer keys.
"""

from typing import Optional
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch


# =========================
# Styles
# =========================
styles = getSampleStyleSheet()
style_title = styles["Title"]
style_subtitle = styles["Italic"]

style_section = ParagraphStyle(
    "SectionHeader",
    parent=styles["Heading2"],
    fontSize=14,
    leading=18,
    spaceBefore=18,
    spaceAfter=12,
    bold=True,
)

style_question = ParagraphStyle(
    "Question",
    parent=styles["Normal"],
    fontSize=12,
    leading=16,
    spaceBefore=10,
    bold=True,
)

style_option = ParagraphStyle(
    "Option",
    parent=styles["Normal"],
    fontSize=12,
    leading=16,
    leftIndent=20,
)

style_answer = ParagraphStyle(
    "Answer",
    parent=styles["Normal"],
    fontSize=12,
    leading=16,
    spaceBefore=6,
    leftIndent=20,
    textColor="green",
)

style_marks = ParagraphStyle(
    "Marks",
    parent=styles["Italic"],
    fontSize=11,
    leading=14,
    textColor="grey",
)

style_normal = styles["Normal"]


# =========================
# Footer
# =========================
def _footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 9)
    page_num = f"Page {doc.page}"
    canvas.drawCentredString(A4[0] / 2.0, 0.5 * inch, page_num)
    canvas.restoreState()


# =========================
# Core builder
# =========================
def build_mockpaper_pdf(
    text: str,
    out_path: str,
    title: str = "Generated Mock Exam Paper",
    source_name: Optional[str] = None,
    is_answer_key: bool = False,
):
    """
    Export the generated exam text or answer key into a styled PDF.

    Args:
        text: plain text of exam or answers
        out_path: where to save the PDF
        title: document title
        source_name: optional "generated from ..." note
        is_answer_key: if True, render with "Answer Key" in the header
    """
    story = []

    # Title
    doc_title = f"{title} {'— Answer Key' if is_answer_key else ''}"
    story.append(Paragraph(doc_title, style_title))
    if source_name:
        story.append(Spacer(1, 6))
        story.append(Paragraph(f"Generated from {source_name}", style_subtitle))
    story.append(Spacer(1, 24))

    # Body
    for line in text.splitlines():
        clean = line.strip()
        if not clean:
            story.append(Spacer(1, 12))
            continue

        # Section headers (force new page for new section, except first)
        if clean.lower().startswith("section"):
            story.append(PageBreak())
            story.append(Paragraph(clean, style_section))
            continue

        # Questions (Q1, 1., etc.)
        if clean.lower().startswith("q") or clean.split(" ")[0].rstrip(".").isdigit():
            story.append(Paragraph(clean, style_question))
            continue

        # MCQ options (A., B., etc.)
        if len(clean) > 2 and clean[1] == "." and clean[0].isalpha():
            story.append(Paragraph(clean, style_option))
            continue

        # Marks
        if "mark" in clean.lower():
            story.append(Paragraph(clean, style_marks))
            continue

        # Answers (for answer key docs)
        if is_answer_key and (clean.lower().startswith("answer") or clean.lower().startswith("ans:")):
            story.append(Paragraph(clean, style_answer))
            continue

        # Default
        story.append(Paragraph(clean, style_normal))

    # Export PDF
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(str(out_file), pagesize=A4)
    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)

    return str(out_file)

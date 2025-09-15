# core/__init__.py

# ---- Rendering (PDF/DOCX → PNG for OCR) ----
from .render import (
    pdf_to_png,
    docx_to_pdf,
    render_paper_to_images,
)

# ---- Upload / OCR text extraction ----
from .mock_upload import (
    papers_to_clean_text,
)

# ---- LLM mock paper generation ----
from .llm_mockgen import (
    configure_openai,
    generate_mock_papers,
)

# ---- PDF export (exam paper + answers) ----
from .mock_export import build_mockpaper_pdf

# ---- Orchestration pipeline ----
from .pipeline import run_pipeline_end_to_end


__all__ = [
    # render
    "pdf_to_png", "docx_to_pdf", "render_paper_to_images",
    # upload / ocr
    "papers_to_clean_text",
    # llm mockgen
    "configure_openai", "generate_mock_papers",
    # pdf export
    "build_mockpaper_pdf",
    # pipeline
    "run_pipeline_end_to_end",
]

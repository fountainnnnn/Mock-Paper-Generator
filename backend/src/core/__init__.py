# core/__init__.py

# ---- Rendering (PDF/DOCX → PNG for the vision page reader) ----
from .render import (
    pdf_to_png,
    docx_to_pdf,
    render_paper_to_images,
)

# ---- Vision page reader (replaces EasyOCR) ----
from .vision import (
    VisionProvider,
    resolve_vision_provider,
    read_document,
    read_pages,
)

# ---- Upload text extraction ----
from .mock_upload import (
    papers_to_clean_text,
)

# ---- LLM mock paper generation ----
from .llm_mockgen import (
    configure_deepseek,
    generate_mock_papers,
)

# ---- PDF export (exam paper + answers) ----
from .mock_export import build_mockpaper_pdf

# ---- Orchestration pipeline ----
from .pipeline import run_pipeline_end_to_end


__all__ = [
    # render
    "pdf_to_png", "docx_to_pdf", "render_paper_to_images",
    # vision page reader
    "VisionProvider", "resolve_vision_provider", "read_document", "read_pages",
    # upload text extraction
    "papers_to_clean_text",
    # llm mockgen
    "configure_deepseek", "generate_mock_papers",
    # pdf export
    "build_mockpaper_pdf",
    # pipeline
    "run_pipeline_end_to_end",
]

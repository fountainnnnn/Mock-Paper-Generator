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

# ---- OCR engines ----
from .ocr import (
    init_easyocr_reader,
    init_paddleocr,
    get_ocr_engine,
    ocr_image_easy,
    ocr_image_paddle,
)

# ---- LLM mock paper generation ----
from .llm_mockgen import (
    configure_openai,
    build_mockpaper_prompt,
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
    # ocr
    "init_easyocr_reader", "init_paddleocr", "get_ocr_engine",
    "ocr_image_easy", "ocr_image_paddle",
    # llm mockgen
    "configure_openai", "build_mockpaper_prompt", "generate_mock_papers",
    # pdf export
    "build_mockpaper_pdf",
    # pipeline
    "run_pipeline_end_to_end",
]

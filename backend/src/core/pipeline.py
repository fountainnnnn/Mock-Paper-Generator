# backend/src/core/pipeline.py
# -*- coding: utf-8 -*-
"""
End-to-end pipeline for mock exam paper generation.
- Extract text from uploaded DOCX/PDFs
- Generate N new mock exam papers + answer keys with OpenAI
- Export each as paired styled PDFs
"""

from typing import Optional, Tuple, List
from pathlib import Path
import tempfile

from .llm_mockgen import generate_mock_papers
from .mock_upload import papers_to_clean_text
from .pdf_builder import build_mockpaper_pdf


def run_pipeline_end_to_end(
    files: List,                         # list of file-like objects or paths
    language: str = "en",
    dpi: int = 220,
    openai_api_key: Optional[str] = None,
    model_name: str = "gpt-4o-mini",
    difficulty: str = "same",
    num_mocks: int = 1,
    out_dir: Optional[str] = None,       # now optional
) -> Tuple[List[str], str, str]:
    """
    End-to-end pipeline:
      1. Extract text from uploaded DOCX/PDF mock papers
      2. Generate 1–3 new mock exam paper(s) + answer key(s) with OpenAI
      3. Export each as paired PDFs

    Args:
        files: list of uploaded file-like objects or file paths
        language: OCR language
        dpi: resolution for OCR rasterization
        openai_api_key: OpenAI API key (user-supplied > backend .env)
        model_name: OpenAI model (default gpt-4o-mini)
        difficulty: "easy" | "same" | "harder"
        num_mocks: number of mock paper versions (1–3)
        out_dir: output directory for PDFs and text. If None, use a temp dir.

    Returns:
        (list of generated PDF paths, concat_txt_path, out_dir)
    """
    # --- Use temp directory if not provided
    if out_dir is None:
        out = Path(tempfile.mkdtemp(prefix="mockpaper_"))
    else:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)

    # --- Save uploads to disk
    saved_paths: List[str] = []
    for f in files:
        if hasattr(f, "read"):  # file-like (UploadFile, BytesIO, etc.)
            name_hint = getattr(f, "filename", getattr(f, "name", "upload.pdf"))
            ext = Path(name_hint).suffix.lower()
            tmp_path = out / f"upload_{len(saved_paths)}{ext}"
            tmp_path.write_bytes(f.read())
            saved_paths.append(str(tmp_path))
        else:  # already a path
            saved_paths.append(str(f))

    if not saved_paths:
        raise ValueError("No input files provided for processing.")
    
    # ensure the easyocr is already downloaded
    from .bootstrap import ensure_easyocr_weights
    ensure_easyocr_weights()

    # --- Extract reference text
    extract_result = papers_to_clean_text(
        saved_paths,
        out_dir=str(out),
        lang=language,
        dpi=dpi,
    )

    # Make sure concat_txt exists
    concat_txt_path = (
        extract_result.get("concat_txt")
        or extract_result.get("text_path")
        or extract_result.get("txt_file")
    )
    if not concat_txt_path:
        raise ValueError("papers_to_clean_text did not return a 'concat_txt' key or equivalent.")

    reference_text = Path(concat_txt_path).read_text(encoding="utf-8")
    if not reference_text.strip():
        raise ValueError("No text extracted from uploaded documents.")

    # --- Generate new mock papers + answers
    mock_pairs = generate_mock_papers(
        paper_text=reference_text,
        difficulty=difficulty,
        num_mocks=num_mocks,
        model_name=model_name,
        api_key=openai_api_key,
    )
    if not mock_pairs:
        raise ValueError("Mock paper generation returned no results.")

    # --- Export to PDFs
    generated_paths: List[str] = []
    for idx, (paper_text, answer_text) in enumerate(mock_pairs, start=1):
        paper_pdf = out / f"mock_{idx}.pdf"
        answers_pdf = out / f"mock_{idx}_answers.pdf"

        build_mockpaper_pdf(
            text=paper_text,
            out_path=str(paper_pdf),
            title=f"Mock Exam Paper {idx}",
            source_name="Reference Upload",
            is_answer_key=False,
        )
        build_mockpaper_pdf(
            text=answer_text,
            out_path=str(answers_pdf),
            title=f"Mock Exam Paper {idx}",
            source_name="Reference Upload",
            is_answer_key=True,
        )

        generated_paths.extend([str(paper_pdf), str(answers_pdf)])

    return generated_paths, concat_txt_path, str(out)

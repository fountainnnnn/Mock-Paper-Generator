# backend/src/core/pipeline.py

from typing import Optional, Tuple, List
from pathlib import Path
import os

from .openai_qg import generate_mock_papers
from .paper_extractor import papers_to_clean_text
from .pdf_builder import build_mockpaper_pdf


def run_pipeline_end_to_end(
    files: List,                         # list of file-like objects or paths
    language: str = "en",
    dpi: int = 220,
    openai_api_key: Optional[str] = None,
    model_name: str = "gpt-4o-mini",
    difficulty: str = "same",
    num_mocks: int = 1,
    out_dir: str = "mockpaper_outputs",
) -> Tuple[List[str], str, str]:
    """
    End-to-end pipeline:
    - Extract text from uploaded DOCX/PDF mock papers
    - Generate 1–3 new mock exam paper(s) + answer key(s) with OpenAI
    - Export each as paired PDFs

    Returns:
        (list of generated pdf paths, concat_txt_path, out_dir)
    """
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

    # --- Extract reference text
    extract_result = papers_to_clean_text(saved_paths, out_dir=str(out), lang=language, dpi=dpi)
    concat_txt_path = extract_result["concat_txt"]

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
            paper_text,
            out_path=str(paper_pdf),
            title=f"Mock Exam Paper {idx}",
            source_name="Reference Upload",
            is_answer_key=False,
        )
        build_mockpaper_pdf(
            answer_text,
            out_path=str(answers_pdf),
            title=f"Mock Exam Paper {idx}",
            source_name="Reference Upload",
            is_answer_key=True,
        )

        generated_paths.extend([str(paper_pdf), str(answers_pdf)])

    return generated_paths, concat_txt_path, str(out)

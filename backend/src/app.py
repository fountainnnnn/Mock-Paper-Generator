# backend/src/app.py
# -*- coding: utf-8 -*-
"""
Mock Paper Generator API
- Accepts PDF/DOCX uploads
- Extracts + cleans text (EasyOCR fallback for scanned PDFs)
- Generates mock exam papers using OpenAI models
- Returns generated papers as a zip download
"""

import os
import io
import tempfile
import zipfile
import traceback
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from dotenv import load_dotenv

# --------------------------------------------------------
# Load .env from either backend/.env or repo root
# --------------------------------------------------------
candidates = [
    Path(__file__).resolve().parent.parent / ".env",         # backend/.env
    Path(__file__).resolve().parent.parent.parent / ".env",  # repo root .env
]
for env_path in candidates:
    if env_path.exists():
        load_dotenv(dotenv_path=env_path, override=True)
        print(f"[DEBUG] Loaded .env from {env_path}")
        break
else:
    load_dotenv(override=True)  # fallback: current working dir
print("[DEBUG] OPENAI_API_KEY after load:", os.getenv("OPENAI_API_KEY"))

try:
    from src.core.pipeline import run_pipeline_end_to_end
except Exception as e:
    raise RuntimeError(f"Failed to import pipeline from src.core: {e}")

# =========================
# App setup
# =========================
app = FastAPI(title="Mock Paper Generator API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def index():
    """Root endpoint with API info."""
    return {
        "message": "Mock Paper Generator API is running",
        "endpoints": ["/generate", "/healthz"],
    }


@app.post("/generate")
async def generate(
    file: UploadFile = File(...),
    language: str = Form("en"),
    dpi: int = Form(220),
    openai_api_key: Optional[str] = Form(None),
    model_name: str = Form("gpt-4o-mini"),
    num_mocks: int = Form(1),
    difficulty: str = Form("same"),
):
    """
    Upload a source exam paper (PDF/DOCX).
    Generates cleaned text and new mock exam papers.
    Returns a zip file containing generated PDFs.
    """
    # --- create temp dir for this request
    tmpdir = Path(tempfile.mkdtemp(prefix="mockpaper_"))

    # Save uploaded file inside temp dir
    content = await file.read()
    tmp_in = tmpdir / f"upload_{file.filename}"
    tmp_in.write_bytes(content)

    # API key resolution (ignore Swagger’s "string")
    if openai_api_key and openai_api_key.strip().lower() != "string":
        key = openai_api_key.strip()
    else:
        key = (os.getenv("OPENAI_API_KEY") or "").strip()

    if not key:
        return JSONResponse(
            {"status": "error", "message": "OPENAI_API_KEY missing. Provide via .env or form."},
            status_code=400,
        )
    os.environ["OPENAI_API_KEY"] = key

    # Run pipeline
    try:
        pdf_paths, concat_txt_path, out_dir = run_pipeline_end_to_end(
            files=[tmp_in],
            language=language,
            dpi=dpi,
            openai_api_key=key,
            model_name=model_name,
            num_mocks=num_mocks,
            difficulty=difficulty,
            out_dir=str(tmpdir),
        )
    except Exception as e:
        print("----- PIPELINE ERROR -----")
        print(traceback.format_exc())
        print("--------------------------")
        raise HTTPException(status_code=500, detail=f"Pipeline error: {e}")

    if not pdf_paths:
        raise HTTPException(status_code=500, detail="Generation failed: no PDFs produced")

    # --- Bundle all generated PDFs into a zip
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zipf:
        for p in pdf_paths:
            zipf.write(p, arcname=Path(p).name)
    zip_buffer.seek(0)

    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=mockpapers.zip"},
    )


@app.get("/healthz")
def healthz():
    """Health check endpoint."""
    return {"ok": True}

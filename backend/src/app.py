import os
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from dotenv import load_dotenv

load_dotenv()

try:
    from src.core.pipeline import run_pipeline_end_to_end
except Exception as e:
    raise RuntimeError(f"Failed to import pipeline from src.core: {e}")

app = FastAPI(title="Mock Paper Generator API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@app.get("/")
def index():
    return {
        "message": "Mock Paper Generator API is running",
        "endpoints": ["/generate", "/files/{filename}", "/healthz"],
    }


@app.post("/generate")
async def generate(
    file: UploadFile = File(...),
    language: str = Form("en"),
    dpi: int = Form(220),
    openai_api_key: str = Form(""),
    model_name: str = Form("gpt-4o-mini"),
    num_mocks: int = Form(1),   # how many mock papers (1–3)
    difficulty: str = Form("same"),
):
    # Save uploaded file
    content = await file.read()
    tmp_in = OUTPUT_DIR / f"upload_{file.filename}"
    tmp_in.write_bytes(content)

    # API key check
    key = openai_api_key or os.getenv("OPENAI_API_KEY", "")
    if not key:
        return JSONResponse(
            {"status": "error", "message": "OPENAI_API_KEY missing. Provide via form or .env."},
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
            out_dir=str(OUTPUT_DIR),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {e}")

    if not pdf_paths:
        raise HTTPException(status_code=500, detail="Generation failed, no PDFs produced")

    # Build public URLs
    space_url = os.getenv("SPACE_URL", "https://crystallizedcrust-mockpaper-generator.hf.space")
    urls = [f"{space_url}/files/{Path(p).name}" for p in pdf_paths]

    return {
        "status": "ok",
        "generated_files": [Path(p).name for p in pdf_paths],
        "urls": urls,
        "text_extracted": concat_txt_path,
        "out_dir": out_dir,
    }


@app.get("/files/{filename}")
def get_file(filename: str):
    path = OUTPUT_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(
        path,
        media_type="application/pdf",
        filename=filename,
    )


@app.get("/healthz")
def healthz():
    return {"ok": True}

# Mock Paper Generator

Turns one past exam paper into fresh mock papers for extra practice. Upload a `.pdf` or `.docx` paper and get back a ZIP of 1–3 new papers in the same style, each with its own answer key.

**Live demo:** https://ngyuhang.com/mock-paper-generator/

## How it works

- **Transcribes the paper with a vision model:** exact question text, numbering and sub-parts, marks, tables, and math as LaTeX. Without a vision key it falls back to the paper's embedded text.
- **Generates new papers with an LLM** (DeepSeek by default) as structured specs, at the same, easier or harder difficulty.
- **Builds the PDFs** with ReportLab and returns them zipped with answer keys.

## Keys

`DEEPSEEK_API_KEY` for generation, plus one of `VISION_API_KEY`, `OPENROUTER_API_KEY` or `OPENAI_API_KEY` for page reading (needed for scanned papers).

## Run locally

```bash
# API
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export DEEPSEEK_API_KEY=...   # plus the optional keys listed above
CORS_ORIGINS=http://localhost:5173 uvicorn app:app --port 7860

# Frontend (second terminal)
cd frontend
python -m http.server 5173
```

Open http://localhost:5173. The page calls the API at `http://localhost:7860` by default; add `?api=<url>` to point it elsewhere.

## Tests

```bash
cd backend
pip install -r requirements-dev.txt 2>/dev/null || pip install pytest httpx
pytest
```

All model calls are mocked in the tests, so they run offline and cost nothing.

See [`backend/README.md`](backend/README.md) for every endpoint and environment variable.

## Known limitations

- Scanned papers need a vision key; text-only mode can't read them.
- DOCX equations stored natively by Word aren't seen unless LibreOffice is available to render the pages.

## Project layout

- `backend/`: FastAPI service (the same code that runs the live demo)
- `frontend/`: static page (HTML, CSS, JavaScript)

## License

MIT

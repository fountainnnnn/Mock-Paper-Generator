# Mock Paper Generator API

FastAPI service behind the portfolio's `/mock-paper-generator/` page. Upload an
existing exam paper (`.pdf` or `.docx`) and it returns a ZIP of 1-3 new mock
papers with matching answer keys (`mock_N.pdf`, `mock_N_answers.pdf`).

## How it works

1. **Page reading** (`src/core/vision.py`). Each page is rendered to PNG and a
   vision LLM transcribes it exactly: question text, numbering and sub-parts
   (`1`, `2(a)`, `(ii)`), MCQ options, marks allocations, math as LaTeX
   (`$...$`, `$$...$$`), tables as Markdown, and `[Figure: ...]` notes for
   diagrams a question depends on.
   - PDF pages are rendered with PyMuPDF.
   - DOCX is rendered whole only when docx2pdf (Word) or LibreOffice is
     available. The Docker image has neither, so in production the DOCX text
     layer plus its embedded pictures (equation images, diagrams) go to the
     vision model.
   - Up to `VISION_CONCURRENCY` pages are read at once. Each call has a timeout
     and is retried with exponential backoff. Any page that still fails falls back
     to its embedded text layer, and so do pages past `VISION_MAX_PAGES`.
   - With **no vision key** the service still works in degraded mode: it uses
     the embedded text layer only (PyMuPDF `get_text` / python-docx) and logs a
     warning. Scanned papers come out empty in this mode.
2. **Generation** (`src/core/llm_mockgen.py`): DeepSeek through the OpenAI SDK,
   structured JSON specs with a legacy text fallback.
3. **Export** (`src/core/pdf_builder.py`): ReportLab PDFs, zipped.

EasyOCR, OpenCV, torch and the EasyOCR warm-up/bootstrap were removed. An
`ocr_engine` form field is accepted and ignored.

## Endpoints

| Method | Path | Notes |
| --- | --- | --- |
| `POST` | `/generate` | multipart: `file` (pdf/docx, max 20 MB), `num_mocks` (1-3), `difficulty` (`easy`/`same`/`harder`), `language`, `dpi` (120-300, default 220), `model_name`, optional `openai_api_key` (DeepSeek key for this request only), `ocr_engine` (ignored). Returns `application/zip`. |
| `GET` | `/healthz` | health check used by Railway |

CORS allows only `https://ngyuhang.com`. `POST /generate` is rate limited per client IP.

Uploads are saved under a per-request temp directory with a sanitised,
uuid-prefixed name and the directory is deleted after the response is built.
A key sent in the form is passed to the clients for that request and never
written to the process environment.

## Environment variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `DEEPSEEK_API_KEY` | (required) | text generation |
| `DEEPSEEK_BASE_URL` | `https://api.deepseek.com` | |
| `DEEPSEEK_MODEL` | `deepseek-v4-flash` | |
| `VISION_API_KEY` / `VISION_BASE_URL` / `VISION_MODEL` | | any OpenAI-compatible vision endpoint (checked first; model defaults to `gpt-4o-mini`) |
| `OPENROUTER_API_KEY` | | used if no `VISION_API_KEY`; base `https://openrouter.ai/api/v1`, model `qwen/qwen2.5-vl-72b-instruct` |
| `OPENAI_API_KEY` | | used if neither of the above; model `gpt-4o-mini` |
| `VISION_MODEL` | | overrides the vision model for any provider |
| `VISION_CONCURRENCY` | `4` | pages read in parallel |
| `VISION_PAGE_TIMEOUT_SECONDS` | `60` | per vision call |
| `VISION_MAX_RETRIES` | `2` | retries per page (backoff `VISION_RETRY_BACKOFF_SECONDS`, default `1.5`, doubled each time) |
| `VISION_MAX_PAGES` | `40` | pages sent to the vision model; later pages use the text layer |
| `VISION_MAX_IMAGES_PER_PAGE` | `4` | embedded pictures sent for a DOCX |
| `MOCK_GENERATOR_MAX_UPLOAD_BYTES` | `20971520` | upload cap |
| `MOCK_GENERATOR_RATE_MAX` / `_RATE_WINDOW_SECONDS` | `3` / `60` | `/generate` rate limit |
| `MOCK_GENERATOR_MAX_PARALLEL_CALLS` | `3` | parallel mock generations |
| `PORT` | `7860` | |

## Run locally

```bash
python3.12 -m venv ../../.venv-mock && source ../../.venv-mock/bin/activate
pip install -r requirements.txt
export DEEPSEEK_API_KEY=...            # plus a vision key, e.g. OPENROUTER_API_KEY=...
uvicorn app:app --port 7860
curl localhost:7860/healthz
```

Or with Docker: `docker build -t mock-generator . && docker run -p 7860:7860 -e DEEPSEEK_API_KEY=... mock-generator`.

## Tests

```bash
pip install -r requirements.txt -r requirements-dev.txt
python -m pytest -q
```

The tests mock every LLM client (DeepSeek and vision) and clear provider keys,
so they make no network calls. They cover filename sanitising, provider
resolution order, PDF/DOCX text-layer fallback, the vision path (PDF pages and
DOCX pictures), retries, page cap, `/healthz`, `/generate`, rate limiting and CORS.

`CORS_ORIGINS` (comma-separated, default `https://ngyuhang.com`) sets which browser origins may call the API. For local development with the static frontend, use `CORS_ORIGINS=http://localhost:5173`.

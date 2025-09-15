
# Backend — Slides → Quiz Deck API

Windows-friendly build. Absolute download URLs returned from `/generate`.

## Setup

```bat
cd backend
python -m venv .venv
.\.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
uvicorn src.app:app --reload --host 0.0.0.0 --port 8000

```

Alternatively, using docker:<br>
To build image: <br>
docker build -t mock-paper-backend .

docker run -it --rm -p 8000:8000 mock-paper-backend

# FROM python:3.11-slim

# WORKDIR /app

# # Add swig + pkg-config so PyMuPDF can compile if no wheels are found
# RUN apt-get update && apt-get install -y \
#     build-essential \
#     swig \
#     pkg-config \
#     libglib2.0-0 \
#     libsm6 \
#     libxrender1 \
#     libxext6 \
#     libgl1 \
#     poppler-utils \
#     && rm -rf /var/lib/apt/lists/*

# COPY requirements.txt .

# # Install deps normally so fastapi/starlette/etc. are resolved,
# # then override PyMuPDF to a version compatible with Python 3.12
# RUN pip install --upgrade pip setuptools wheel \
#     && pip install --no-cache-dir -r requirements.txt \
#     && pip install --no-cache-dir "pymupdf==1.24.10"

# COPY . .

# EXPOSE 8000

# CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]

FROM python:3.11-slim

WORKDIR /app

# System libs for PyMuPDF, OpenCV, etc.
RUN apt-get update && apt-get install -y \
    build-essential \
    swig \
    pkg-config \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgl1 \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]

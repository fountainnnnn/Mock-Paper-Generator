FROM python:3.11-slim

WORKDIR /app

# --------------------------------------------------------
# Install system libraries + fonts (Chromium runtime libs for Playwright).
# No OCR/OpenCV stack: pages are read by a vision LLM (src/core/vision.py).
# --------------------------------------------------------
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libxrender1 \
    libxext6 \
    wget \
    curl \
    unzip \
    fonts-liberation \
    fonts-dejavu \
    fonts-freefont-ttf \
    fonts-unifont \
    libasound2 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libatspi2.0-0 \
    libcups2 \
    libdbus-1-3 \
    libdrm2 \
    libgbm1 \
    libnspr4 \
    libnss3 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxkbcommon0 \
    libpango-1.0-0 \
    libxrandr2 \
    && rm -rf /var/lib/apt/lists/*

# --------------------------------------------------------
# Copy requirements and install Python deps
# --------------------------------------------------------
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# --------------------------------------------------------
# Install Chromium for Playwright (headless only)
# --------------------------------------------------------
RUN playwright install chromium

# --------------------------------------------------------
# Copy source code
# --------------------------------------------------------
COPY . .

# Hugging Face requires port 7860
EXPOSE 7860

# Run FastAPI from root-level app.py
CMD ["sh", "-c", "exec uvicorn app:app --host 0.0.0.0 --port ${PORT:-7860}"]

# If you instead have a shim app.py at project root, swap the above with:
# CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]

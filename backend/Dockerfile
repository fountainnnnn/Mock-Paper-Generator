FROM python:3.11-slim

WORKDIR /app

# --------------------------------------------------------
# Install system libraries
# --------------------------------------------------------
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
    wget \
    curl \
    unzip \
    fonts-liberation \
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
# Force CPU-only PyTorch first, then everything else
# --------------------------------------------------------
COPY requirements.txt .

RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir \
      torch==2.4.1+cpu torchvision==0.19.1+cpu torchaudio==2.4.1+cpu \
      --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# --------------------------------------------------------
# Install Chromium for Playwright
# --------------------------------------------------------
RUN playwright install --with-deps chromium

# --------------------------------------------------------
# Copy source code
# --------------------------------------------------------
COPY . .

EXPOSE 8000

CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]

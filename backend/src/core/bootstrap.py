# backend/src/core/bootstrap.py
import os
from pathlib import Path
import easyocr

# Paths
EASYOCR_MODELS = Path(__file__).resolve().parent.parent / "models" / "easyocr"
EASYOCR_CACHE = Path("/tmp/.easyocr_cache")

def ensure_easyocr_weights(lang: str = "en"):
    """
    Initialize EasyOCR, forcing cache to /tmp to avoid permission errors.
    """
    # Ensure directories exist
    EASYOCR_MODELS.mkdir(parents=True, exist_ok=True)
    EASYOCR_CACHE.mkdir(parents=True, exist_ok=True)

    print("[BOOTSTRAP] ensure_easyocr_weights() CALLED")
    print(f"[BOOTSTRAP] model dir: {EASYOCR_MODELS}")
    print(f"[BOOTSTRAP] cache dir: {EASYOCR_CACHE}")

    # Force EasyOCR to use our dirs
    reader = easyocr.Reader(
        [lang],
        model_storage_directory=str(EASYOCR_MODELS),
        user_network_directory=str(EASYOCR_CACHE),
        download_enabled=False,
        gpu=False,
    )

    print("[BOOTSTRAP] EasyOCR reader initialized with /tmp cache")
    return reader

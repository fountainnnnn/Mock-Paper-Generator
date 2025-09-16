# backend/src/core/bootstrap.py
import os
from pathlib import Path
import easyocr

# Define directories
EASYOCR_MODELS = Path(__file__).resolve().parent.parent / "models" / "easyocr"
EASYOCR_CACHE = Path("/tmp/.easyocr_cache")  # Hugging Face writable dir

def ensure_easyocr_weights(lang: str = "en"):
    """
    Ensure EasyOCR model weights are available and usable.
    - Weights: kept in src/models/easyocr (bundled with repo)
    - Cache:   written to /tmp/.easyocr_cache (writable on Hugging Face)
    """
    # Make sure directories exist
    EASYOCR_MODELS.mkdir(parents=True, exist_ok=True)
    EASYOCR_CACHE.mkdir(parents=True, exist_ok=True)

    print(f"[DEBUG] EasyOCR model dir: {EASYOCR_MODELS}")
    print(f"[DEBUG] EasyOCR cache dir: {EASYOCR_CACHE}")

    # Initialize EasyOCR reader with explicit dirs
    reader = easyocr.Reader(
        [lang],
        model_storage_directory=str(EASYOCR_MODELS),
        user_network_directory=str(EASYOCR_CACHE),  # force /tmp
        download_enabled=True,
        gpu=False
    )

    print("[BOOTSTRAP] EasyOCR reader initialized successfully")
    return reader

if __name__ == "__main__":
    ensure_easyocr_weights()

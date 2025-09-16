# backend/src/core/bootstrap.py
import os
from pathlib import Path
import easyocr

# Directories
EASYOCR_MODELS = Path(__file__).resolve().parent.parent / "models" / "easyocr"
EASYOCR_CACHE = Path("/tmp/.easyocr_cache")  # Hugging Face writable dir

def ensure_easyocr_weights(lang: str = "en"):
    """
    Ensure EasyOCR model weights are available and usable.
    - Weights: kept in src/models/easyocr (read-only bundled)
    - Cache:   written to /tmp/.easyocr_cache (writable)
    """
    EASYOCR_MODELS.mkdir(parents=True, exist_ok=True)
    EASYOCR_CACHE.mkdir(parents=True, exist_ok=True)

    reader = easyocr.Reader(
        [lang],
        model_storage_directory=str(EASYOCR_MODELS),
        user_network_directory=str(EASYOCR_CACHE),  # <-- force writable cache
        download_enabled=True,
        gpu=False
    )

    print(f"[BOOTSTRAP] EasyOCR weights loaded from: {EASYOCR_MODELS}")
    print(f"[BOOTSTRAP] EasyOCR cache dir: {EASYOCR_CACHE}")
    return reader

if __name__ == "__main__":
    ensure_easyocr_weights()

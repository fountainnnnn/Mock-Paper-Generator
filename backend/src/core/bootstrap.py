# backend/src/core/bootstrap.py
import os
from pathlib import Path
import easyocr

# Directories
EASYOCR_MODELS = Path(__file__).resolve().parent.parent / "models" / "easyocr"
EASYOCR_CACHE = Path("/tmp/.easyocr_cache")  # Hugging Face writable dir

def ensure_easyocr_weights(lang: str = "en"):
    """
    Initialize EasyOCR with bundled weights and forced /tmp cache.
    This version replaces all old ones so pipeline always uses it.
    """
    # Ensure dirs exist
    EASYOCR_MODELS.mkdir(parents=True, exist_ok=True)
    EASYOCR_CACHE.mkdir(parents=True, exist_ok=True)

    print(f"[DEBUG] EasyOCR model dir (read-only): {EASYOCR_MODELS}")
    print(f"[DEBUG] EasyOCR cache dir (writable temp): {EASYOCR_CACHE}")

    # Reader uses /tmp cache, no downloads, no GPU
    reader = easyocr.Reader(
        [lang],
        model_storage_directory=str(EASYOCR_MODELS),
        user_network_directory=str(EASYOCR_CACHE),
        download_enabled=False,
        gpu=False
    )

    print("[BOOTSTRAP] EasyOCR reader initialized (no downloads, /tmp cache)")
    return reader

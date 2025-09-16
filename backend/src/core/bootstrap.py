# backend/src/core/bootstrap.py
import os
from pathlib import Path
import easyocr

# Directories
EASYOCR_MODELS = Path(__file__).resolve().parent.parent / "models" / "easyocr"
EASYOCR_CACHE = Path("/tmp/.easyocr_cache")

def ensure_easyocr_weights_v2(lang: str = "en"):
    """
    Initialize EasyOCR with bundled weights and /tmp cache.
    - Reads weights from src/models/easyocr
    - Writes only ephemeral files into /tmp/.easyocr_cache
    - No auto-downloads
    """
    EASYOCR_MODELS.mkdir(parents=True, exist_ok=True)
    EASYOCR_CACHE.mkdir(parents=True, exist_ok=True)

    print(f"[DEBUG] EasyOCR model dir (read-only): {EASYOCR_MODELS}")
    print(f"[DEBUG] EasyOCR cache dir (writable temp): {EASYOCR_CACHE}")

    reader = easyocr.Reader(
        [lang],
        model_storage_directory=str(EASYOCR_MODELS),
        user_network_directory=str(EASYOCR_CACHE),
        download_enabled=False,
        gpu=False
    )

    print("[BOOTSTRAP] EasyOCR reader initialized (forced /tmp cache, no downloads)")
    return reader

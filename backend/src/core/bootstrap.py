# backend/src/core/bootstrap.py
import os
from pathlib import Path
import easyocr

# Directories
EASYOCR_MODELS = Path(__file__).resolve().parent.parent / "models" / "easyocr"
EASYOCR_CACHE = Path("/tmp/.easyocr_cache")  # Hugging Face writable dir

def ensure_easyocr_weights_v2(lang: str = "en"):
    """
    Initialize EasyOCR with bundled weights and /tmp cache.
    - Reads weights from src/models/easyocr
    - Writes only ephemeral files into /tmp/.easyocr_cache
    - No downloads allowed
    """
    # Make sure directories exist
    EASYOCR_MODELS.mkdir(parents=True, exist_ok=True)
    EASYOCR_CACHE.mkdir(parents=True, exist_ok=True)

    print(f"[DEBUG] EasyOCR model dir (read-only): {EASYOCR_MODELS}")
    print(f"[DEBUG] EasyOCR cache dir (writable temp): {EASYOCR_CACHE}")

    # Initialize EasyOCR reader
    reader = easyocr.Reader(
        [lang],
        model_storage_directory=str(EASYOCR_MODELS),
        user_network_directory=str(EASYOCR_CACHE),
        download_enabled=False,   # 🚫 don’t fetch anything new
        gpu=False
    )

    print("[BOOTSTRAP] EasyOCR reader initialized (forced /tmp cache, no downloads)")
    return reader

if __name__ == "__main__":
    ensure_easyocr_weights_v2()

import os
from pathlib import Path
import easyocr

# Define model + cache dirs
EASYOCR_MODELS = Path(__file__).resolve().parent.parent / "models" / "easyocr"
EASYOCR_CACHE = Path("/tmp/.easyocr_cache")  # writable on Hugging Face

def ensure_easyocr_weights(lang: str = "en"):
    """
    Ensure EasyOCR model weights are available and usable.
    - Reads weights from src/models/easyocr (bundled with repo)
    - Writes runtime cache/user_network to /tmp/.easyocr_cache
    """
    EASYOCR_MODELS.mkdir(parents=True, exist_ok=True)
    EASYOCR_CACHE.mkdir(parents=True, exist_ok=True)

    reader = easyocr.Reader(
        [lang],
        model_storage_directory=str(EASYOCR_MODELS),
        user_network_directory=str(EASYOCR_CACHE),
        download_enabled=True,
        gpu=False  # disable GPU (HF Spaces usually CPU only)
    )

    print(f"[BOOTSTRAP] EasyOCR weights loaded from: {EASYOCR_MODELS}")
    print(f"[BOOTSTRAP] EasyOCR cache dir: {EASYOCR_CACHE}")
    return reader

if __name__ == "__main__":
    ensure_easyocr_weights()

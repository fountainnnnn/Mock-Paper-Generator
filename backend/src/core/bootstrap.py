# backend/src/core/bootstrap.py
import os
from pathlib import Path
import easyocr

def ensure_easyocr_weights(lang: str = "en"):
    """
    Ensure EasyOCR model weights are available in the project-local directory.
    Downloads them once and keeps them inside backend/models/easyocr/.
    """
    # Local model storage inside repo
    model_dir = Path(__file__).resolve().parent.parent / "models" / "easyocr"
    model_dir.mkdir(parents=True, exist_ok=True)

    # Explicitly tell EasyOCR where to keep models
    reader = easyocr.Reader(
        [lang],
        model_storage_directory=str(model_dir),
        download_enabled=True
    )

    print(f"[BOOTSTRAP] EasyOCR weights are ready in: {model_dir}")
    return reader

if __name__ == "__main__":
    ensure_easyocr_weights()

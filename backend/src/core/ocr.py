# backend/src/core/ocr.py
# -*- coding: utf-8 -*-
"""
OCR utilities for mock exam paper extraction (math/science compatible).
- EasyOCR (fast, general text)
- Preprocessing for sharper OCR
- Persistent cache for weights
- Confidence filtering + math symbol normalization
"""

from __future__ import annotations
from typing import Any, Dict, List, Tuple
from pathlib import Path
import os, tempfile, re

# --- Third-party ---
try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    np = None  # type: ignore

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore

try:
    from PIL import Image  # type: ignore
except Exception:  # pragma: no cover
    Image = None  # type: ignore

try:
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None  # type: ignore


# =========================
# Cache + GPU helpers
# =========================
def _mk_temp_dir(prefix: str = "easyocr_") -> Path:
    """Create a temporary directory for EasyOCR model storage."""
    return Path(tempfile.mkdtemp(prefix=prefix))

def _choose_storage_dir() -> Path:
    """Choose where to store EasyOCR models (persistent or ephemeral)."""
    if os.getenv("PAPERS_OCR_EPHEMERAL") == "1":
        return _mk_temp_dir("easyocr_")
    base = Path.home() / ".cache" / "easyocr_models"
    base.mkdir(parents=True, exist_ok=True)
    return base

def _gpu_allowed(force_cpu: bool) -> bool:
    """Check if GPU can be used for EasyOCR."""
    if force_cpu:
        return False
    if torch is None:
        return False
    try:
        return torch.cuda.is_available()
    except Exception:
        return False


# =========================
# EasyOCR init
# =========================
_EASYOCR_CACHE: Dict[Tuple[Tuple[str, ...], bool, str], Any] = {}

def init_easyocr_reader(lang_list: List[str] = ["en"], force_cpu: bool = True):
    """Initialize and cache an EasyOCR Reader instance."""
    import easyocr
    storage_dir = _choose_storage_dir()
    use_gpu = _gpu_allowed(force_cpu=force_cpu)

    key = (tuple(lang_list), use_gpu, str(storage_dir))
    if key in _EASYOCR_CACHE:
        return _EASYOCR_CACHE[key]

    reader = easyocr.Reader(
        lang_list,
        gpu=use_gpu,
        model_storage_directory=str(storage_dir),
        user_network_directory=str(storage_dir),
        download_enabled=True,
        verbose=False,
    )
    _EASYOCR_CACHE[key] = reader
    return reader


# =========================
# Engine selector (EasyOCR only)
# =========================
def get_ocr_engine(lang: str = "en"):
    """
    Return an EasyOCR reader for the given language.
    Simplified after removing PaddleOCR.
    """
    return init_easyocr_reader([lang])


# =========================
# Preprocessing
# =========================
def _preprocess_for_ocr(img: Any) -> Any:
    """Convert to grayscale, binarize, and denoise for sharper OCR."""
    if cv2 is None or np is None:
        return img

    if isinstance(img, Image.Image):
        img = np.array(img.convert("RGB"))

    if img.ndim == 3 and img.shape[2] == 4:  # RGBA → RGB
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img
    th = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 35, 11
    )
    return cv2.fastNlMeansDenoising(th, h=15)


# =========================
# Math normalization
# =========================
def _normalize_math_text(text: str) -> str:
    """Normalize common OCR misreads and math symbols."""
    replacements = {
        "O": "0",
        "l": "1",
        "×": "x",
        "−": "-",
        "--": "–",
        "<=": "≤",
        ">=": "≥",
        "√ ": "√",
        "∑ ": "∑",
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return re.sub(r"\s+", " ", text).strip()


def _sort_by_coordinates(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Sort OCR results top-to-bottom, left-to-right."""
    return sorted(items, key=lambda x: (x["bbox"][0][1], x["bbox"][0][0]))


# =========================
# OCR runner (EasyOCR only)
# =========================
def ocr_image_easy(reader, image, conf_threshold: float = 0.3):
    """
    Run OCR on an image using EasyOCR.
    Returns list of dicts with bbox, text, and conf.
    """
    if reader is None:
        raise RuntimeError("EasyOCR reader is None.")
    if Image is None or np is None:
        raise RuntimeError("Pillow and numpy are required.")

    try:
        img = _preprocess_for_ocr(image)
        res = reader.readtext(
            img,
            detail=1,
            paragraph=True,
            contrast_ths=0.05,
            adjust_contrast=0.7,
            text_threshold=0.6,
            low_text=0.3,
            width_ths=0.7,
            slope_ths=0.2,
            ycenter_ths=0.5,
            height_ths=0.7,
            mag_ratio=1.5,
        )
    except Exception as e:
        raise RuntimeError(f"EasyOCR failed: {e}")

    out: List[Dict[str, Any]] = []
    for item in res:
        try:
            bbox, text, conf = item
            text = _normalize_math_text(text or "")
            if text and conf >= conf_threshold:
                out.append({"bbox": bbox, "text": text, "conf": float(conf)})
        except Exception:
            continue
    return _sort_by_coordinates(out)

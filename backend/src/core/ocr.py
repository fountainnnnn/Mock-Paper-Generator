# -*- coding: utf-8 -*-
"""
OCR utilities for mock exam paper extraction (math/science compatible).
- Persistent EasyOCR cache
- PaddleOCR optional fallback
- Tuned thresholds for printed/scanned exam PDFs
- Confidence filtering, coordinate sorting, math symbol normalization
"""

from __future__ import annotations
from typing import Any, Dict, List, Tuple
from pathlib import Path
import os, tempfile, re

# --- Third-party (guarded) ---
try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    np = None  # type: ignore

try:
    from PIL import Image  # type: ignore
except Exception:  # pragma: no cover
    Image = None  # type: ignore

# torch only for CUDA check (optional)
try:
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None  # type: ignore


# =========================
# Internal helpers / cache
# =========================
def _mk_temp_dir(prefix: str = "easyocr_") -> Path:
    p = Path(tempfile.mkdtemp(prefix=prefix))
    return p

_EASYOCR_CACHE: Dict[Tuple[Tuple[str, ...], bool, str], Any] = {}


def _choose_storage_dir() -> Path:
    if os.getenv("PAPERS_OCR_EPHEMERAL") == "1":
        return _mk_temp_dir("easyocr_")
    base = Path.home() / ".cache" / "easyocr_models"
    base.mkdir(parents=True, exist_ok=True)
    return base


def _gpu_allowed(force_cpu: bool) -> bool:
    if force_cpu:
        return False
    if torch is None:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


# =========================
# Public API
# =========================
def init_easyocr_reader(lang_list: List[str] = ["en"], force_cpu: bool = True):
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


def init_paddleocr(lang: str = "en"):
    try:
        from paddleocr import PaddleOCR
    except Exception as e:
        raise RuntimeError(f"PaddleOCR not available: {e}")
    return PaddleOCR(use_angle_cls=True, lang=lang)


def get_ocr_engine(engine_name: str, lang: str):
    eng = (engine_name or "").lower()
    if eng == "paddle":
        try:
            ocr = init_paddleocr(lang)
            return ocr, True
        except Exception as e:
            print("[WARN] PaddleOCR unavailable; falling back to EasyOCR:", e)
    return init_easyocr_reader([lang], force_cpu=True), False


# =========================
# OCR helpers
# =========================
def _normalize_math_text(text: str) -> str:
    """
    Normalize OCR quirks in math/science text.
    """
    replacements = {
        "O": "0",  # common misread
        "l": "1",  # lowercase L → one
        "×": "x",
        "−": "-",  # minus variants
        "--": "–",
        "<=": "≤",
        ">=": "≥",
        "√ ": "√",
        "∑ ": "∑",
    }
    for k, v in replacements.items():
        text = text.replace(k, v)

    # collapse multiple spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def _sort_by_coordinates(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Sort OCR results top-to-bottom, then left-to-right.
    """
    return sorted(items, key=lambda x: (x["bbox"][0][1], x["bbox"][0][0]))


# =========================
# OCR runners
# =========================
def ocr_image_easy(reader, image, conf_threshold: float = 0.3):
    if reader is None:
        raise RuntimeError("EasyOCR reader is None.")
    if Image is None or np is None:
        raise RuntimeError("Pillow and numpy are required.")

    try:
        if isinstance(image, (str, bytes, Path)):
            res = reader.readtext(str(image), detail=1, paragraph=True)
        else:
            if isinstance(image, Image.Image):
                image = np.array(image.convert("RGB"))
            res = reader.readtext(
                image,
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
            if not text or conf < conf_threshold:
                continue
            out.append({"bbox": bbox, "text": text, "conf": float(conf)})
        except Exception:
            continue

    return _sort_by_coordinates(out)


def ocr_image_paddle(ocr, image, conf_threshold: float = 0.3):
    if ocr is None:
        raise RuntimeError("PaddleOCR engine is None.")

    try:
        res = ocr.ocr(image, cls=True)
    except Exception as e:
        raise RuntimeError(f"PaddleOCR failed: {e}")

    out: List[Dict[str, Any]] = []
    try:
        for page in res:
            for det in page:
                bbox, meta = det
                text, conf = meta
                text = _normalize_math_text(text or "")
                if not text or conf < conf_threshold:
                    continue
                out.append({"bbox": bbox, "text": str(text), "conf": float(conf)})
    except Exception:
        for bbox, (text, conf) in res:
            text = _normalize_math_text(text or "")
            if not text or conf < conf_threshold:
                continue
            out.append({"bbox": bbox, "text": str(text), "conf": float(conf)})

    return _sort_by_coordinates(out)

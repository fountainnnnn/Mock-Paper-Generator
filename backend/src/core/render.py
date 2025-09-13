# backend/src/core/rasterizer.py
# -*- coding: utf-8 -*-
"""
Rasterization utilities for exam papers.
- PDF → PNG per page (for OCR on scanned math/printed papers)
- DOCX → PDF → PNG (optional, requires docx2pdf on Windows/macOS)
"""

from __future__ import annotations
from pathlib import Path
from typing import List
import sys
import subprocess


# ---------------------------
# PDF → PNG (PyMuPDF)
# ---------------------------
def pdf_to_png(pdf_path: str | Path, out_dir: str | Path, dpi: int = 220) -> List[Path]:
    """
    Rasterize a PDF to PNG pages using PyMuPDF at a target DPI.
    """
    from fitz import open as fitz_open, Matrix

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    doc = fitz_open(str(pdf_path))
    imgs: List[Path] = []

    zoom = dpi / 72.0
    mat = Matrix(zoom, zoom)

    for i, page in enumerate(doc, 1):
        pix = page.get_pixmap(matrix=mat)
        p = out / f"page_{i:03d}.png"
        pix.save(str(p))
        imgs.append(p)

    if not imgs:
        raise RuntimeError("PDF rasterization produced no images.")
    return imgs


# ---------------------------
# DOCX → PDF → PNG
# ---------------------------
def docx_to_pdf(docx_path: str | Path, out_pdf: str | Path) -> Path:
    """
    Convert DOCX → PDF using docx2pdf (Windows/macOS only).
    On Linux, requires LibreOffice installed (fallback).
    """
    docx_path, out_pdf = Path(docx_path), Path(out_pdf)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)

    # Prefer docx2pdf if available
    try:
        from docx2pdf import convert
        convert(str(docx_path), str(out_pdf))
        return out_pdf
    except Exception:
        pass

    # Fallback to LibreOffice
    soffice = shutil.which("soffice") or shutil.which("libreoffice")
    if not soffice:
        raise RuntimeError("No docx2pdf or LibreOffice found for DOCX→PDF conversion.")

    cmd = [
        soffice, "--headless", "--convert-to", "pdf",
        "--outdir", str(out_pdf.parent),
        str(docx_path.resolve())
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"LibreOffice DOCX→PDF failed: {proc.stderr or proc.stdout}")

    produced = out_pdf.parent / (docx_path.stem + ".pdf")
    if not produced.exists():
        raise RuntimeError("DOCX→PDF did not produce expected output.")
    produced.rename(out_pdf)
    return out_pdf


# ---------------------------
# Unified entry point
# ---------------------------
def render_paper_to_images(path: str | Path, out_dir: str, dpi: int = 220) -> List[Path]:
    """
    Render a DOCX or PDF exam paper into per-page PNG images.
    - If input is PDF → direct rasterization.
    - If input is DOCX → convert to PDF, then rasterize.
    """
    path = Path(path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    ext = path.suffix.lower()
    if ext == ".pdf":
        return pdf_to_png(path, out, dpi=dpi)
    elif ext == ".docx":
        tmp_pdf = out / (path.stem + "_tmp.pdf")
        docx_to_pdf(path, tmp_pdf)
        return pdf_to_png(tmp_pdf, out, dpi=dpi)
    else:
        raise ValueError(f"Unsupported file type: {ext}. Only PDF and DOCX are supported.")

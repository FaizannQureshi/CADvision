"""
PDF Processing Utilities
"""
from __future__ import annotations

from pdf2image import convert_from_path
from pathlib import Path
import os
from typing import Optional


def _default_pdf_dpi() -> int:
    try:
        d = int(os.getenv("CADVISION_PDF_DPI", "150"))
        return max(72, min(d, 300))
    except ValueError:
        return 150


def pdf_to_image(pdf_path: str, output_dir: str, dpi: Optional[int] = None) -> str:
    """
    Convert first page of PDF to PNG image
    Returns: path to converted image
    """
    if dpi is None:
        dpi = _default_pdf_dpi()

    print(f"Converting PDF to image: {pdf_path} (dpi={dpi})")

    images = convert_from_path(pdf_path, dpi=dpi)
    
    if not images:
        raise RuntimeError(f"No pages found in PDF: {pdf_path}")
    
    # Save first page
    output_path = os.path.join(output_dir, f"{Path(pdf_path).stem}_page1.png")
    images[0].save(output_path, "PNG")
    
    print(f"✓ PDF converted to: {output_path}")
    return output_path
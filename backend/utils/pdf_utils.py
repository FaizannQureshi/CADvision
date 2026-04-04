"""
PDF Processing Utilities
"""
from pdf2image import convert_from_path
from pathlib import Path
import os


def pdf_to_image(pdf_path: str, output_dir: str, dpi: int = 200) -> str:
    """
    Convert first page of PDF to PNG image
    Returns: path to converted image
    """
    print(f"Converting PDF to image: {pdf_path}")
    
    images = convert_from_path(pdf_path, dpi=dpi)
    
    if not images:
        raise RuntimeError(f"No pages found in PDF: {pdf_path}")
    
    # Save first page
    output_path = os.path.join(output_dir, f"{Path(pdf_path).stem}_page1.png")
    images[0].save(output_path, "PNG")
    
    print(f"✓ PDF converted to: {output_path}")
    return output_path
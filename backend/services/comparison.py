"""
Main Comparison Service - Orchestrates the entire comparison pipeline
"""
import gc
import cv2
import numpy as np
import os
import base64
from pathlib import Path
from services.detector import detect_objects
from services.clip_matcher import match_and_highlight
from services.summarizer import generate_ai_summary
from utils.pdf_utils import pdf_to_image
from utils.image_utils import create_combined_output
from utils.image_ops import downscale_if_needed

async def compare_cad_files(file1_path: str, file2_path: str, temp_dir: str) -> dict:
    """
    Main comparison pipeline
    Returns: dict with base64 images and AI summary
    """
    print(f"\n{'='*60}")
    print("CAD COMPARISON PIPELINE")
    print(f"{'='*60}")
    
    # Step 1: Convert PDFs to images if needed
    print("\n[1/4] Processing input files...")
    img1_path = _prepare_input(file1_path, temp_dir)
    img2_path = _prepare_input(file2_path, temp_dir)
    img1_path = downscale_if_needed(img1_path, temp_dir)
    img2_path = downscale_if_needed(img2_path, temp_dir)

    # Step 2: Detect objects
    print("\n[2/4] Detecting components...")
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    
    if img1 is None or img2 is None:
        raise RuntimeError("Could not read input images")
    
    _, objects1, _ = detect_objects(
        img1,
        bin_threshold=200,
        edge_threshold=50,
        min_area=10000,
        max_area_ratio=0.75,
        merge_gap=0,
        draw_debug=False,
    )
    
    _, objects2, _ = detect_objects(
        img2,
        bin_threshold=200,
        edge_threshold=50,
        min_area=10000,
        max_area_ratio=0.75,
        merge_gap=0,
        draw_debug=False,
    )

    del img1
    del img2
    gc.collect()
    
    print(f"Found {len(objects1)} objects in image 1")
    print(f"Found {len(objects2)} objects in image 2")
    
    # Step 3: Match and highlight
    print("\n[3/4] Matching components and highlighting differences...")
    highlighted_img = match_and_highlight(
        img1_path, img2_path,
        objects1, objects2,
        temp_dir
    )
    
    # Step 4: Create combined output
    print("\n[4/4] Creating combined output...")
    combined_path = os.path.join(temp_dir, "combined.png")
    create_combined_output(
        img1_path, img2_path, highlighted_img,
        combined_path
    )
    
    # Convert to base64
    with open(combined_path, "rb") as f:
        combined_base64 = base64.b64encode(f.read()).decode('utf-8')
    
    with open(img1_path, "rb") as f:
        img1_base64 = base64.b64encode(f.read()).decode('utf-8')
    
    with open(img2_path, "rb") as f:
        img2_base64 = base64.b64encode(f.read()).decode('utf-8')
    
    # Generate AI summary
    print("\n[5/5] Generating AI summary...")
    ai_summary = await generate_ai_summary(img1_base64, img2_base64, combined_base64)
    
    print(f"\n{'='*60}")
    print("COMPARISON COMPLETE")
    print(f"{'='*60}\n")

    gc.collect()

    return {
        "images": {
            "highlighted_1": combined_base64,
            "input_1": img1_base64,
            "input_2": img2_base64
        },
        "ai_summary": ai_summary
    }


def _prepare_input(path: str, temp_dir: str) -> str:
    """Convert PDF to image if needed, otherwise return original path"""
    p = Path(path)
    if p.suffix.lower() == ".pdf":
        return pdf_to_image(str(p), temp_dir)
    return str(p)
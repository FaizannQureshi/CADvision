"""
Resize / normalize images for lower memory use on constrained hosts (e.g. Render).
"""
from __future__ import annotations

import os
import cv2
from pathlib import Path
from typing import Optional


def _max_side_from_env() -> int:
    raw = os.getenv("CADVISION_MAX_IMAGE_SIDE", "1600")
    try:
        v = int(raw)
        return max(512, min(v, 8192))
    except ValueError:
        return 1600


def downscale_if_needed(input_path: str, temp_dir: str, max_side: Optional[int] = None) -> str:
    """
    If the longest edge exceeds max_side, write a resized PNG under temp_dir and return its path.
    Otherwise return input_path unchanged.
    """
    if max_side is None:
        max_side = _max_side_from_env()

    img = cv2.imread(input_path)
    if img is None:
        return input_path

    h, w = img.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        del img
        return input_path

    scale = max_side / float(longest)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    del img

    out = str(Path(temp_dir) / f"downscaled_{Path(input_path).stem}.png")
    cv2.imwrite(out, resized)
    del resized
    print(f"  Downscaled for processing: {w}x{h} -> {new_w}x{new_h} (max_side={max_side})")
    return out

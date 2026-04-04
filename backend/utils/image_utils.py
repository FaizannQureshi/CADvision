"""
Image Processing Utilities
"""
import cv2
import numpy as np
from pathlib import Path


def create_combined_output(img1_path: str, img2_path: str, highlighted_img: np.ndarray,
                          output_path: str):
    """
    Create a single combined image with 3 panels:
    - Input Image 1
    - Input Image 2  
    - Highlighted Differences
    """
    # Read input images
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    
    if img1 is None or img2 is None:
        raise RuntimeError("Could not read input images")
    
    # Get dimensions
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    h3, w3 = highlighted_img.shape[:2]
    
    # Resize all to same height
    max_height = max(h1, h2, h3)
    
    def resize_to_height(img, target_height):
        h, w = img.shape[:2]
        if h == target_height:
            return img
        aspect = w / h
        new_width = int(target_height * aspect)
        return cv2.resize(img, (new_width, target_height), interpolation=cv2.INTER_AREA)
    
    img1_resized = resize_to_height(img1, max_height)
    img2_resized = resize_to_height(img2, max_height)
    highlighted_resized = resize_to_height(highlighted_img, max_height)
    
    # Add labels
    label_height = 60
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.2
    font_thickness = 2
    text_color = (255, 255, 255)
    bg_color = (50, 50, 50)
    
    def add_label(img, label_text):
        h, w = img.shape[:2]
        labeled = np.ones((h + label_height, w, 3), dtype=np.uint8) * np.array(bg_color, dtype=np.uint8)
        labeled[label_height:, :] = img
        
        text_size = cv2.getTextSize(label_text, font, font_scale, font_thickness)[0]
        text_x = (w - text_size[0]) // 2
        text_y = (label_height + text_size[1]) // 2
        cv2.putText(labeled, label_text, (text_x, text_y), font, font_scale,
                   text_color, font_thickness, cv2.LINE_AA)
        return labeled
    
    # Get filenames for labels
    label1 = f"Input: {Path(img1_path).stem}"
    label2 = f"Input: {Path(img2_path).stem}"
    
    img1_labeled = add_label(img1_resized, label1)
    img2_labeled = add_label(img2_resized, label2)
    highlighted_labeled = add_label(highlighted_resized, "Highlighted Differences")
    
    # Add spacing
    spacing = 20
    spacer = np.ones((max_height + label_height, spacing, 3), dtype=np.uint8) * 255
    
    # Combine horizontally
    combined = np.hstack([img1_labeled, spacer, img2_labeled, spacer, highlighted_labeled])
    
    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, combined)
    print(f"✓ Combined output saved: {output_path}")
    
    return combined
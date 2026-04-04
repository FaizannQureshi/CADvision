"""
Image Alignment and Difference Highlighting
"""
import cv2
import numpy as np


def align_and_highlight_region(img1, img2, result_img, bx1, bx2, 
                               img1_h, img1_w, img2_h, img2_w):
    """
    Align and highlight differences for a matched region pair
    """
    x1, y1, w1, h1 = [int(v) for v in bx1]
    x2, y2, w2, h2 = [int(v) for v in bx2]
    
    max_w = max(w1, w2)
    max_h = max(h1, h2)
    
    # Determine corner matching
    def corners(b):
        x, y, w, h = b
        return {
            'tl': np.array([x, y]),
            'tr': np.array([x + w, y]),
            'bl': np.array([x, y + h]),
            'br': np.array([x + w, y + h]),
        }
    
    area1, area2 = w1 * h1, w2 * h2
    larger_bb = bx1 if area1 >= area2 else bx2
    smaller_bb = bx2 if area1 >= area2 else bx1
    
    large_c, small_c = corners(larger_bb), corners(smaller_bb)
    dists = {k: np.linalg.norm(large_c[k] - small_c[k]) for k in ['tl', 'tr', 'bl', 'br']}
    matched_corner = min(dists, key=dists.get)
    
    # Compute new coordinates
    def compute_new_xy(x, y, w, h, anchor):
        if anchor == 'tl':
            return int(x), int(y)
        elif anchor == 'tr':
            return int(max(0, (x + w) - max_w)), int(y)
        elif anchor == 'bl':
            return int(x), int(max(0, (y + h) - max_h))
        else:  # br
            return int(max(0, (x + w) - max_w)), int(max(0, (y + h) - max_h))
    
    new_x1, new_y1 = compute_new_xy(x1, y1, w1, h1, matched_corner)
    new_x2, new_y2 = compute_new_xy(x2, y2, w2, h2, matched_corner)
    
    # Clamp to bounds
    final_w = int(min(min(max_w, img1_w - new_x1), min(max_w, img2_w - new_x2)))
    final_h = int(min(min(max_h, img1_h - new_y1), min(max_h, img2_h - new_y2)))
    
    if final_w <= 0 or final_h <= 0:
        return result_img
    
    # Crop regions
    crop1 = img1[new_y1:new_y1+final_h, new_x1:new_x1+final_w].copy()
    crop2 = img2[new_y2:new_y2+final_h, new_x2:new_x2+final_w].copy()
    
    if crop1.shape[:2] != (final_h, final_w):
        crop1 = cv2.resize(crop1, (final_w, final_h), interpolation=cv2.INTER_CUBIC)
    if crop2.shape[:2] != (final_h, final_w):
        crop2 = cv2.resize(crop2, (final_w, final_h), interpolation=cv2.INTER_CUBIC)
    
    if crop1.size == 0 or crop2.size == 0:
        return result_img
    
    try:
        # Align crop2 to crop1
        aligned_crop2 = align_images(crop1, crop2)
        
        # Highlight differences
        highlighted_crop = highlight_differences(crop1, aligned_crop2)
        
        # Place back into result
        result_img[new_y1:new_y1+final_h, new_x1:new_x1+final_w] = highlighted_crop
    except Exception as e:
        print(f"Warning: region processing failed: {e}")
    
    return result_img


def align_images(img1, img2, max_features=5000, good_match_percent=0.15):
    """
    Align img2 to img1 using ORB feature matching
    """
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    gray1 = cv2.GaussianBlur(gray1, (3, 3), 0)
    gray2 = cv2.GaussianBlur(gray2, (3, 3), 0)
    
    # ORB detector
    detector = cv2.ORB_create(max_features)
    kp1, desc1 = detector.detectAndCompute(gray1, None)
    kp2, desc2 = detector.detectAndCompute(gray2, None)
    
    if desc1 is None or desc2 is None:
        raise ValueError("No descriptors found")
    
    # Match features
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(desc1, desc2)
    
    if len(matches) == 0:
        raise ValueError("No matches found")
    
    matches = sorted(matches, key=lambda x: x.distance)
    num_good = max(10, int(len(matches) * good_match_percent))
    matches = matches[:num_good]
    
    # Extract point correspondences
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)
    
    for i, match in enumerate(matches):
        points1[i, :] = kp1[match.queryIdx].pt
        points2[i, :] = kp2[match.trainIdx].pt
    
    # Estimate affine transform
    h, mask = cv2.estimateAffinePartial2D(points2, points1, method=cv2.RANSAC)
    
    if h is None:
        raise ValueError("Transform estimation failed")
    
    # Warp img2
    height, width = img1.shape[:2]
    aligned = cv2.warpAffine(img2, h, (width, height),
                            borderMode=cv2.BORDER_CONSTANT, borderValue=(255,255,255))
    
    return aligned


def highlight_differences(img1, img2_aligned, diff_threshold=25, min_area=15, white_threshold=240):
    """
    Highlight color differences between two aligned images
    Shows additions (green), deletions (red), overlaps (yellow)
    """
    # Convert to LAB color space for perceptual difference
    lab1 = cv2.cvtColor(img1, cv2.COLOR_BGR2LAB)
    lab2 = cv2.cvtColor(img2_aligned, cv2.COLOR_BGR2LAB)
    
    # Compute differences
    diff_added = cv2.subtract(lab1, lab2)
    diff_removed = cv2.subtract(lab2, lab1)
    
    diff_added_intensity = np.linalg.norm(diff_added.astype(np.float32), axis=2)
    diff_removed_intensity = np.linalg.norm(diff_removed.astype(np.float32), axis=2)
    
    diff_added_intensity = cv2.normalize(diff_added_intensity, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    diff_removed_intensity = cv2.normalize(diff_removed_intensity, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Threshold
    _, mask_added = cv2.threshold(diff_added_intensity, diff_threshold, 255, cv2.THRESH_BINARY)
    _, mask_removed = cv2.threshold(diff_removed_intensity, diff_threshold, 255, cv2.THRESH_BINARY)
    
    # Filter small regions
    mask_added = _filter_small_regions(mask_added, min_area)
    mask_removed = _filter_small_regions(mask_removed, min_area)
    
    # Clean up masks
    mask_added = cv2.medianBlur(mask_added, 3)
    mask_removed = cv2.medianBlur(mask_removed, 3)
    mask_added = cv2.erode(mask_added, np.ones((2, 2), np.uint8), iterations=1)
    mask_removed = cv2.erode(mask_removed, np.ones((2, 2), np.uint8), iterations=1)
    
    # Find overlaps
    overlap = cv2.bitwise_and(mask_added, mask_removed)
    
    # Detect white replacements
    gray2 = cv2.cvtColor(img2_aligned, cv2.COLOR_BGR2GRAY)
    white_mask = (gray2 > white_threshold).astype(np.uint8) * 255
    b, g, r = cv2.split(img2_aligned)
    true_white = ((b > white_threshold) & (g > white_threshold) & (r > white_threshold)).astype(np.uint8) * 255
    white_areas = cv2.bitwise_or(white_mask, true_white)
    
    # Adjust masks for white replacements
    white_replacement = cv2.bitwise_and(overlap, white_areas)
    overlap = cv2.bitwise_and(overlap, cv2.bitwise_not(white_replacement))
    mask_removed = cv2.bitwise_or(mask_removed, white_replacement)
    
    # Create colored overlay
    overlay = img2_aligned.copy()
    overlay[mask_added > 0] = [0, 255, 0]      # Green: additions
    overlay[mask_removed > 0] = [0, 0, 255]    # Red: deletions
    overlay[overlap > 0] = [0, 255, 255]       # Yellow: overlaps
    
    # Blend with original
    highlighted = cv2.addWeighted(img2_aligned, 0.6, overlay, 0.4, 0)
    
    return highlighted


def _filter_small_regions(mask, min_area):
    """Remove small connected components from binary mask"""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    filtered = np.zeros_like(mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            filtered[labels == i] = 255
    return filtered
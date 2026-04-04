"""
CLIP-based Component Matching and Highlighting
"""
import cv2
import numpy as np
import torch
from PIL import Image
from services.alignment import align_and_highlight_region

# Try open_clip then fallback to clip
try:
    import open_clip
    _CLIP_BACKEND = "open_clip"
except Exception:
    try:
        import clip
        _CLIP_BACKEND = "clip"
    except Exception:
        raise ImportError("No CLIP backend available. Install 'open_clip_torch' or 'clip'")


def match_and_highlight(img1_path: str, img2_path: str, 
                       objects1: list, objects2: list,
                       temp_dir: str,
                       clip_threshold: float = 0.20,
                       min_area: int = 10000) -> np.ndarray:
    """
    Match components using CLIP and highlight differences
    Returns: highlighted image (numpy array)
    """
    # Load CLIP model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, preprocess = _load_clip_model(device)
    
    # Load images
    pil1 = Image.open(img1_path).convert("RGB")
    pil2 = Image.open(img2_path).convert("RGB")
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    
    # Filter by area
    objects1 = [o for o in objects1 if o['bbox'][2] * o['bbox'][3] >= min_area]
    objects2 = [o for o in objects2 if o['bbox'][2] * o['bbox'][3] >= min_area]
    
    print(f"After filtering: {len(objects1)} and {len(objects2)} objects")
    
    # Get embeddings
    bboxes1 = [tuple(o['bbox']) for o in objects1]
    bboxes2 = [tuple(o['bbox']) for o in objects2]
    
    emb1 = _get_embeddings(model, preprocess, pil1, bboxes1, device)
    emb2 = _get_embeddings(model, preprocess, pil2, bboxes2, device)
    
    # Auto-select matching direction
    if len(objects1) <= len(objects2):
        direction = "1_to_many"
        print(f"Using 1→many matching (img1:{len(objects1)} ≤ img2:{len(objects2)})")
    else:
        direction = "many_to_1"
        print(f"Using many→1 matching (img1:{len(objects1)} > img2:{len(objects2)})")
    
    # Match components
    matches = _match_bidirectional(emb1, emb2, clip_threshold, direction)
    print(f"Found {len(matches)} matched groups")
    
    # Create highlighted image
    result = img1.copy()
    img1_h, img1_w = img1.shape[:2]
    img2_h, img2_w = img2.shape[:2]
    
    if direction == "1_to_many":
        for i, match_list in matches.items():
            bx1 = bboxes1[i]
            img2_boxes = [bboxes2[j] for j, sim in match_list]
            combined_bx2 = _compute_combined_bbox(img2_boxes)
            
            # Align and highlight this region
            result = align_and_highlight_region(
                img1, img2, result, bx1, combined_bx2,
                img1_h, img1_w, img2_h, img2_w
            )
    else:  # many_to_1
        for j, match_list in matches.items():
            bx2 = bboxes2[j]
            img1_boxes = [bboxes1[i] for i, sim in match_list]
            combined_bx1 = _compute_combined_bbox(img1_boxes)
            
            # Align and highlight this region
            result = align_and_highlight_region(
                img1, img2, result, combined_bx1, bx2,
                img1_h, img1_w, img2_h, img2_w
            )
    
    return result


def _load_clip_model(device):
    """Load CLIP model"""
    if _CLIP_BACKEND == "open_clip":
        model, _, preprocess = open_clip.create_model_and_transforms(
            'ViT-B-32', pretrained='openai'
        )
        model.to(device).eval()
        return model, preprocess
    else:
        import clip as _clip
        model, preprocess = _clip.load("ViT-B/32", device=device)
        model.to(device).eval()
        return model, preprocess


def _get_embeddings(model, preprocess, pil_img, bboxes, device, batch_size=32):
    """Extract CLIP embeddings for bounding boxes"""
    crops = [_crop_bbox(pil_img, bb) for bb in bboxes]
    if not crops:
        return np.zeros((0, 512), dtype=np.float32)
    
    tensors = torch.cat([preprocess(c).unsqueeze(0) for c in crops], dim=0).to(device)
    
    with torch.no_grad():
        embeds = []
        for i in range(0, len(tensors), batch_size):
            batch = tensors[i:i+batch_size]
            if _CLIP_BACKEND == "open_clip":
                emb = model.encode_image(batch)
            else:
                emb = model.encode_image(batch)
            emb = emb.float()
            emb = emb / emb.norm(dim=-1, keepdim=True)
            embeds.append(emb.cpu().numpy())
    
    return np.vstack(embeds)


def _crop_bbox(pil_img, bbox):
    """Crop image region for bbox"""
    x, y, w, h = bbox
    img_w, img_h = pil_img.size
    x1, y1 = max(0, int(x)), max(0, int(y))
    x2, y2 = min(img_w, int(x + w)), min(img_h, int(y + h))
    if x2 <= x1 or y2 <= y1:
        return pil_img.crop((0, 0, 1, 1))
    return pil_img.crop((x1, y1, x2, y2))


def _match_bidirectional(emb1, emb2, threshold, direction):
    """Match embeddings bidirectionally"""
    if emb1.shape[0] == 0 or emb2.shape[0] == 0:
        return {}
    
    sims = np.dot(emb1, emb2.T)
    
    if direction == "1_to_many":
        matches = {}
        matched_img2 = set()
        
        for j in range(sims.shape[1]):
            best_i = int(np.argmax(sims[:, j]))
            best_sim = float(sims[best_i, j])
            
            if best_sim >= threshold:
                if best_i not in matches:
                    matches[best_i] = []
                matches[best_i].append((j, best_sim))
                matched_img2.add(j)
        
        # Last resort matching
        unmatched1 = [i for i in range(emb1.shape[0]) if i not in matches]
        unmatched2 = [j for j in range(emb2.shape[0]) if j not in matched_img2]
        
        if len(unmatched1) == 1 and len(unmatched2) == 1:
            i, j = unmatched1[0], unmatched2[0]
            matches[i] = [(j, float(sims[i, j]))]
        
        return matches
    
    else:  # many_to_1
        matches = {}
        matched_img1 = set()
        
        for i in range(sims.shape[0]):
            best_j = int(np.argmax(sims[i, :]))
            best_sim = float(sims[i, best_j])
            
            if best_sim >= threshold:
                if best_j not in matches:
                    matches[best_j] = []
                matches[best_j].append((i, best_sim))
                matched_img1.add(i)
        
        # Last resort matching
        unmatched1 = [i for i in range(emb1.shape[0]) if i not in matched_img1]
        unmatched2 = [j for j in range(emb2.shape[0]) if j not in matches]
        
        if len(unmatched1) == 1 and len(unmatched2) == 1:
            i, j = unmatched1[0], unmatched2[0]
            matches[j] = [(i, float(sims[i, j]))]
        
        return matches


def _compute_combined_bbox(bboxes):
    """Compute union bbox of multiple boxes"""
    if not bboxes:
        return (0, 0, 0, 0)
    
    min_x = min(bb[0] for bb in bboxes)
    min_y = min(bb[1] for bb in bboxes)
    max_x = max(bb[0] + bb[2] for bb in bboxes)
    max_y = max(bb[1] + bb[3] for bb in bboxes)
    
    return (min_x, min_y, max_x - min_x, max_y - min_y)
"""
Object Detection Service
"""
import cv2
import numpy as np

def detect_objects(image_bgr,
                   bin_threshold=200,
                   edge_threshold=50,
                   min_area=1000,
                   max_area_ratio=0.8,
                   merge_gap=0,
                   draw_debug=True):
    """
    Detect objects in CAD drawing using edge detection and connected components
    
    Returns: (output_image, objects_list, metadata)
    objects_list: list of dicts with 'bbox': (x,y,w,h) and 'points_count': n
    """
    h, w = image_bgr.shape[:2]
    out_img = image_bgr.copy() if draw_debug else image_bgr
    img_area = h * w

    # Convert to grayscale
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # Binary threshold: Separates drawing lines from background
    # Pixels > bin_threshold become white (255), others become black (0)
    # Higher values = only very bright lines, Lower values = include more gray pixels
    _, binary = cv2.threshold(gray, bin_threshold, 255, cv2.THRESH_BINARY)

    # Edge detection using Sobel
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)

    bin_f = binary.astype(np.float32)
    gx = cv2.filter2D(bin_f, -1, kx)
    gy = cv2.filter2D(bin_f, -1, ky)
    edges = np.sqrt(gx * gx + gy * gy)
    edges = np.clip((edges / edges.max()) * 255, 0, 255).astype(np.uint8)

    # Flood fill to find connected components
    visited = np.zeros_like(edges, dtype=np.uint8)
    objects = []

    def flood_fill(start_y, start_x):
        """Iterative flood fill returning bbox and pixel count"""
        stack = [(start_y, start_x)]
        min_x, max_x = w, 0
        min_y, max_y = h, 0
        count = 0

        while stack:
            y, x = stack.pop()
            if x < 0 or x >= w or y < 0 or y >= h:
                continue
            # Skip if already visited or edge strength is too weak
            if visited[y, x] or edges[y, x] <= edge_threshold:
                continue

            visited[y, x] = 1
            count += 1
            min_x, max_x = min(min_x, x), max(max_x, x)
            min_y, max_y = min(min_y, y), max(max_y, y)

            # Check 8 neighbors
            for ny in (y-1, y, y+1):
                for nx in (x-1, x, x+1):
                    if ny == y and nx == x:
                        continue
                    if 0 <= nx < w and 0 <= ny < h and not visited[ny, nx]:
                        stack.append((ny, nx))

        if count == 0:
            return None

        width = max_x - min_x + 1
        height = max_y - min_y + 1

        return {
            'bbox': (min_x, min_y, width, height),
            'points_count': count,
        }

    # Find all edge pixels and flood fill
    # edge_threshold: Filters edge pixels by strength
    # Only pixels with edge intensity > edge_threshold are considered for object detection
    # Higher values = only strong edges (less noise), Lower values = include weak edges (more noise)
    ys, xs = np.where(edges > edge_threshold)
    for y, x in zip(ys, xs):
        if visited[y, x]:
            continue
        obj = flood_fill(y, x)
        if obj is None:
            continue
        x0, y0, ww, hh = obj['bbox']
        area = ww * hh
        if area > min_area and area < (img_area * max_area_ratio):
            objects.append(obj)

    # Merge overlapping/nearby boxes if merge_gap > 0
    if merge_gap > 0 and objects:
        objects = _merge_boxes(objects, merge_gap, min_area, img_area, max_area_ratio)

    # Draw bounding boxes on output image (skip when only bbox list is needed — saves a full copy)
    if draw_debug:
        for idx, obj in enumerate(objects):
            x0, y0, ww, hh = obj['bbox']
            cv2.rectangle(out_img, (x0, y0), (x0 + ww - 1, y0 + hh - 1), (0, 255, 0), 2)
            label = str(idx + 1)
            font_scale = max(0.5, min(2.0, w / 800.0))
            cv2.putText(out_img, label, (x0 + 5, y0 + int(20 * font_scale)),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), 2, cv2.LINE_AA)

    meta = {
        'original_shape': (h, w),
        'binary_threshold': bin_threshold,
        'edge_threshold': edge_threshold,
        'min_area': min_area,
        'max_area_ratio': max_area_ratio,
        'merge_gap': merge_gap
    }

    return out_img, objects, meta


def _merge_boxes(boxes, merge_gap, min_area, img_area, max_area_ratio):
    """Merge overlapping or nearby boxes using Union-Find"""
    n = len(boxes)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    def rect_gap(a, b):
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        dx = max(0, max(bx - (ax + aw), ax - (bx + bw)))
        dy = max(0, max(by - (ay + ah), ay - (by + bh)))
        return max(dx, dy)

    def rect_intersect(a, b):
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        ix1, iy1 = max(ax, bx), max(ay, by)
        ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
        return ix2 > ix1 and iy2 > iy1

    # Build edges
    for i in range(n):
        for j in range(i + 1, n):
            a, b = boxes[i]['bbox'], boxes[j]['bbox']
            if rect_intersect(a, b) or rect_gap(a, b) <= merge_gap:
                union(i, j)

    # Collect clusters
    clusters = {}
    for i in range(n):
        r = find(i)
        clusters.setdefault(r, []).append(i)

    # Merge each cluster
    merged = []
    for indices in clusters.values():
        bx = boxes[indices[0]]['bbox']
        pts = 0
        ux1, uy1 = bx[0], bx[1]
        ux2, uy2 = bx[0] + bx[2], bx[1] + bx[3]
        for idx in indices:
            b = boxes[idx]['bbox']
            pts += boxes[idx]['points_count']
            ux1 = min(ux1, b[0])
            uy1 = min(uy1, b[1])
            ux2 = max(ux2, b[0] + b[2])
            uy2 = max(uy2, b[1] + b[3])
        merged_bbox = (ux1, uy1, ux2 - ux1, uy2 - uy1)
        area = (ux2 - ux1) * (uy2 - uy1)
        if area > min_area and area < (img_area * max_area_ratio):
            merged.append({'bbox': merged_bbox, 'points_count': pts})

    return merged
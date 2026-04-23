"""Layer 2 — Intersection Patch Extraction.

Input:  rectified board image + list of N×N intersection pixel coordinates
Output: (N*N, patch_size, patch_size, 3) uint8 array of normalised patches
"""

import cv2
import numpy as np


def extract_patches(
    img: np.ndarray,
    intersections: list[tuple[float, float]],
    board_size: int,
    patch_size: int = 32,
) -> np.ndarray:
    """
    Crop one square patch per intersection, normalise lighting, resize to
    *patch_size × patch_size*.

    Returns shape (N*N, patch_size, patch_size, 3) uint8.
    """
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    h, w = img.shape[:2]

    # Derive radius from actual inter-line spacing.
    if board_size > 1 and len(intersections) >= 2:
        x0, y0 = intersections[0]
        x1, y1 = intersections[1]
        spacing = abs(x1 - x0)
    else:
        spacing = w / max(board_size, 2)
    radius = max(int(spacing * 0.45), 4)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    patches = []

    for x, y in intersections:
        ix, iy = int(round(x)), int(round(y))
        x0c = max(0, ix - radius)
        y0c = max(0, iy - radius)
        x1c = min(w, ix + radius + 1)
        y1c = min(h, iy + radius + 1)

        patch = img[y0c:y1c, x0c:x1c]
        if patch.size == 0:
            patches.append(np.zeros((patch_size, patch_size, 3), dtype=np.uint8))
            continue

        patch = cv2.resize(patch, (patch_size, patch_size))

        # CLAHE on L channel to handle shadows / board-colour variation.
        lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        lab = cv2.merge([clahe.apply(l), a, b])
        patches.append(cv2.cvtColor(lab, cv2.COLOR_LAB2BGR))

    return np.array(patches, dtype=np.uint8)

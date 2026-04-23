"""Layer 1 — Board Localisation.

Input:  raw BGR image (np.ndarray)
Output: LocalizeResult containing a rectified square crop, N×N intersection
        pixel coordinates, and the detected board size (9 / 13 / 19).
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LocalizeResult:
    rectified: np.ndarray
    intersections: list[tuple[float, float]]  # (x, y) pixel coords, row-major
    board_size: int
    h_lines: list[float] = field(default_factory=list)
    v_lines: list[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _order_points(pts: np.ndarray) -> np.ndarray:
    """Order 4 points: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # TL: smallest x+y
    rect[2] = pts[np.argmax(s)]   # BR: largest x+y
    diff = np.diff(pts, axis=1).ravel()  # y − x
    rect[1] = pts[np.argmin(diff)]  # TR: smallest y−x (large x, small y)
    rect[3] = pts[np.argmax(diff)]  # BL: largest  y−x (small x, large y)
    return rect


def _find_board_quad(gray: np.ndarray) -> Optional[np.ndarray]:
    """Return the 4-corner quad of the board, or None if not found."""
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    min_area = gray.shape[0] * gray.shape[1] * 0.05

    for contour in contours[:10]:
        if cv2.contourArea(contour) < min_area:
            break
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        if len(approx) == 4:
            return _order_points(approx.reshape(4, 2).astype(np.float32))
    return None


def _rectify(img: np.ndarray, quad: np.ndarray, out_size: int = 800) -> np.ndarray:
    dst = np.array(
        [[0, 0], [out_size - 1, 0], [out_size - 1, out_size - 1], [0, out_size - 1]],
        dtype=np.float32,
    )
    M = cv2.getPerspectiveTransform(quad, dst)
    return cv2.warpPerspective(img, M, (out_size, out_size))


def _cluster_lines(positions: list[float], min_gap: float) -> list[float]:
    if not positions:
        return []
    clusters: list[list[float]] = [[sorted(positions)[0]]]
    for pos in sorted(positions)[1:]:
        if pos - clusters[-1][-1] < min_gap:
            clusters[-1].append(pos)
        else:
            clusters.append([pos])
    return [float(np.mean(c)) for c in clusters]


def _detect_grid_lines(gray: np.ndarray) -> tuple[list[float], list[float]]:
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blurred, 30, 100)
    min_len = int(gray.shape[0] * 0.3)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=50, minLineLength=min_len, maxLineGap=20
    )
    h_pos: list[float] = []
    v_pos: list[float] = []
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0]:
            angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
            if angle < 10 or angle > 170:
                h_pos.append((y1 + y2) / 2.0)
            elif 80 < angle < 100:
                v_pos.append((x1 + x2) / 2.0)
    return h_pos, v_pos


def _snap_board_size(n: int) -> int:
    for size in (9, 13, 19):
        if abs(n - size) <= 2:
            return size
    return 19


def _even_grid(img_size: int, n: int, margin_frac: float = 0.05) -> list[float]:
    margin = img_size * margin_frac
    step = (img_size - 2 * margin) / (n - 1)
    return [margin + i * step for i in range(n)]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def localize(img: np.ndarray, hint_size: Optional[int] = None) -> LocalizeResult:
    """
    Detect and rectify a Go board in *img*.

    Parameters
    ----------
    img:       BGR (or grayscale) image array.
    hint_size: Known board size (9/13/19).  Auto-detected when None.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()

    quad = _find_board_quad(gray)
    if quad is not None:
        rectified = _rectify(img, quad)
    else:
        side = min(gray.shape[:2])
        rectified = cv2.resize(img, (side, side))

    rect_gray = (
        cv2.cvtColor(rectified, cv2.COLOR_BGR2GRAY) if rectified.ndim == 3 else rectified
    )

    h_raw, v_raw = _detect_grid_lines(rect_gray)
    spacing = rect_gray.shape[0] / 22.0
    h_lines = _cluster_lines(h_raw, spacing)
    v_lines = _cluster_lines(v_raw, spacing)

    detected_n = max(len(h_lines), len(v_lines), 1)
    board_size = hint_size if hint_size else _snap_board_size(detected_n)

    if len(h_lines) != board_size:
        h_lines = _even_grid(rect_gray.shape[0], board_size)
    if len(v_lines) != board_size:
        v_lines = _even_grid(rect_gray.shape[1], board_size)

    intersections: list[tuple[float, float]] = [
        (x, y) for y in h_lines for x in v_lines
    ]

    return LocalizeResult(
        rectified=rectified,
        intersections=intersections,
        board_size=board_size,
        h_lines=h_lines,
        v_lines=v_lines,
    )

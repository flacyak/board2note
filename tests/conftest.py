"""Shared fixtures and synthetic board helpers."""

import cv2
import numpy as np
import pytest


def make_board_image(
    board_size: int = 9,
    img_size: int = 400,
    black_stones: list[tuple[int, int]] | None = None,
    white_stones: list[tuple[int, int]] | None = None,
) -> np.ndarray:
    """
    Render a synthetic Go board with optional stones.

    Stones are specified as (row, col) in 0-indexed top-to-bottom / left-to-right.
    Returns a BGR uint8 image of shape (img_size, img_size, 3).
    """
    # Wood-coloured background (BGR)
    img = np.full((img_size, img_size, 3), (80, 160, 200), dtype=np.uint8)

    margin = int(img_size * 0.08)
    step = (img_size - 2 * margin) / (board_size - 1)

    # Grid lines
    for i in range(board_size):
        pos = int(margin + i * step)
        cv2.line(img, (margin, pos), (img_size - margin, pos), (0, 0, 0), 1)
        cv2.line(img, (pos, margin), (pos, img_size - margin), (0, 0, 0), 1)

    stone_radius = max(int(step * 0.42), 3)

    for row, col in (black_stones or []):
        cx = int(margin + col * step)
        cy = int(margin + row * step)
        cv2.circle(img, (cx, cy), stone_radius, (20, 20, 20), -1)

    for row, col in (white_stones or []):
        cx = int(margin + col * step)
        cy = int(margin + row * step)
        cv2.circle(img, (cx, cy), stone_radius, (230, 230, 230), -1)
        cv2.circle(img, (cx, cy), stone_radius, (0, 0, 0), 1)

    return img


@pytest.fixture
def empty_9x9():
    return make_board_image(board_size=9)


@pytest.fixture
def board_with_stones():
    return make_board_image(
        board_size=9,
        black_stones=[(2, 2), (4, 4), (6, 6)],
        white_stones=[(2, 6), (6, 2)],
    )

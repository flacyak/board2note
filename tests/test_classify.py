"""Tests for Layer 3 — Stone Classifier (HSV baseline)."""

import numpy as np
import cv2
import pytest
from board2note.classify import classify_hsv


def _solid_patch(bgr: tuple[int, int, int], size: int = 32) -> np.ndarray:
    return np.full((size, size, 3), bgr, dtype=np.uint8)


class TestHSVClassifier:
    def test_black_stone(self):
        patch = _solid_patch((20, 20, 20))
        labels, confs = classify_hsv(np.array([patch]))
        assert labels[0] == "black"
        assert confs[0] > 0.5

    def test_white_stone(self):
        patch = _solid_patch((220, 220, 220))
        labels, confs = classify_hsv(np.array([patch]))
        assert labels[0] == "white"
        assert confs[0] > 0.5

    def test_empty_intersection(self):
        # Wood colour: mid-brightness, slightly saturated orange/tan
        patch = _solid_patch((80, 150, 190))
        labels, confs = classify_hsv(np.array([patch]))
        assert labels[0] == "empty"

    def test_batch(self):
        patches = np.array([
            _solid_patch((15, 15, 15)),    # black
            _solid_patch((210, 210, 210)), # white
            _solid_patch((80, 150, 190)),  # empty
        ])
        labels, confs = classify_hsv(patches)
        assert labels == ["black", "white", "empty"]
        assert all(0.0 <= c <= 1.0 for c in confs)

    def test_output_length(self):
        patches = np.zeros((81, 32, 32, 3), dtype=np.uint8)
        labels, confs = classify_hsv(patches)
        assert len(labels) == 81
        assert len(confs) == 81

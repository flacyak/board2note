"""Full board2note pipeline: image → position encoding."""

from dataclasses import dataclass
from typing import Literal, Optional

import cv2
import numpy as np

from .localize import localize
from .extract import extract_patches
from .classify import classify, Stone
from .encode import encode, _GTP_COLS

CONFIDENCE_THRESHOLD = 0.6


@dataclass
class ProcessResult:
    board_size: int
    labels: list[Stone]
    confidences: list[float]
    low_confidence: list[tuple[int, int, float]]  # (row, col, conf)
    sgf: str
    gtp: str
    ascii: str


def process_image(
    img: np.ndarray,
    hint_size: Optional[int] = None,
    model=None,
    conf_threshold: float = CONFIDENCE_THRESHOLD,
) -> ProcessResult:
    loc = localize(img, hint_size)
    patches = extract_patches(loc.rectified, loc.intersections, loc.board_size)
    labels, confs = classify(patches, model)

    low_conf: list[tuple[int, int, float]] = []
    for i, (label, conf) in enumerate(zip(labels, confs)):
        if label != "empty" and conf < conf_threshold:
            row, col = divmod(i, loc.board_size)
            low_conf.append((row, col, conf))

    return ProcessResult(
        board_size=loc.board_size,
        labels=labels,
        confidences=confs,
        low_confidence=low_conf,
        sgf=encode(labels, loc.board_size, "sgf"),
        gtp=encode(labels, loc.board_size, "gtp"),
        ascii=encode(labels, loc.board_size, "ascii"),
    )


def process_path(
    path: str,
    hint_size: Optional[int] = None,
    model=None,
    conf_threshold: float = CONFIDENCE_THRESHOLD,
) -> ProcessResult:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return process_image(img, hint_size, model, conf_threshold)

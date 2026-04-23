"""Layer 3 — Stone Classifier.

Input:  (N*N, H, W, 3) uint8 patch array
Output: parallel lists of Stone labels and confidence scores

Baseline: HSV thresholding — no model required.
ONNX:     pass an onnxruntime.InferenceSession as *model* for learned inference.
          Expected input: (N, 3, H, W) float32 in [0, 1].
          Expected output: (N, 3) logits for classes [black, white, empty].
"""

from typing import Literal
import cv2
import numpy as np

Stone = Literal["black", "white", "empty"]
_LABEL_MAP = {0: "black", 1: "white", 2: "empty"}


# ---------------------------------------------------------------------------
# HSV baseline
# ---------------------------------------------------------------------------

def _classify_patch_hsv(patch: np.ndarray) -> tuple[Stone, float]:
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    _, s, v = cv2.split(hsv)
    mean_v = float(v.mean())
    mean_s = float(s.mean())

    if mean_v < 80:
        conf = 1.0 - mean_v / 80.0
        return "black", min(conf, 1.0)
    if mean_v > 170 and mean_s < 60:
        conf = min((mean_v - 170) / 85.0, 1.0)
        return "white", conf
    return "empty", 0.7


def classify_hsv(patches: np.ndarray) -> tuple[list[Stone], list[float]]:
    labels: list[Stone] = []
    confs: list[float] = []
    for patch in patches:
        label, conf = _classify_patch_hsv(patch)
        labels.append(label)
        confs.append(conf)
    return labels, confs


# ---------------------------------------------------------------------------
# ONNX inference
# ---------------------------------------------------------------------------

def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def _classify_onnx(patches: np.ndarray, model) -> tuple[list[Stone], list[float]]:
    inp = patches.astype(np.float32) / 255.0
    inp = inp.transpose(0, 3, 1, 2)  # NHWC → NCHW
    input_name = model.get_inputs()[0].name
    logits = model.run(None, {input_name: inp})[0]  # (N, 3)
    probs = _softmax(logits)
    preds = probs.argmax(axis=1)
    labels: list[Stone] = [_LABEL_MAP[int(p)] for p in preds]
    confs = [float(probs[i, preds[i]]) for i in range(len(preds))]
    return labels, confs


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def classify(patches: np.ndarray, model=None) -> tuple[list[Stone], list[float]]:
    """Classify patches; use ONNX *model* if provided, else HSV baseline."""
    if model is not None:
        return _classify_onnx(patches, model)
    return classify_hsv(patches)

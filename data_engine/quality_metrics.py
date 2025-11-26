"""Frame quality metrics used to filter/score dataset frames.

Implements simple, interpretable metrics such as sharpness and brightness.
Extend with learned metrics or composite heuristics as needed.
"""
from __future__ import annotations

from typing import Dict
import numpy as np
import cv2


def sharpness_metric(image: np.ndarray) -> float:
    """Return variance of Laplacian as a sharpness proxy.

    Higher -> sharper.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def brightness_metric(image: np.ndarray) -> float:
    """Return mean brightness (0-255).

    Use this to filter over/under-exposed frames.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return float(hsv[..., 2].mean())


def is_blurry(image: np.ndarray, thresh: float = 100.0) -> bool:
    """Heuristic to mark a frame as blurry based on sharpness metric.

    TODO: tune `thresh` for your camera/resolution.
    """
    return sharpness_metric(image) < thresh


def frame_quality(image: np.ndarray) -> Dict[str, float]:
    """Return a dict of quality metrics for a single frame."""
    return {"sharpness": sharpness_metric(image), "brightness": brightness_metric(image)}


if __name__ == "__main__":
    # lightweight smoke test
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m data_engine.quality_metrics <image>")
    else:
        img = cv2.imread(sys.argv[1])
        print(frame_quality(img))

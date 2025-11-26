"""Vehicle detection module using YOLOv8 (boilerplate).

This module provides a thin wrapper around a YOLOv8-style detector. It is
intended as a starting point for research/experiments — replace the model
loading and postprocessing with the appropriate code for your chosen library.

Python: 3.10+
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple
import logging
import argparse

import numpy as np

try:
    from ultralytics import YOLO  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    YOLO = None  # type: ignore

import cv2

logger = logging.getLogger(__name__)


class YOLOv8Detector:
    """Simple YOLOv8 detector wrapper.

    Attributes:
        model_path: Path to a saved YOLOv8 model or a model name string.
    """

    def __init__(self, model_path: str = "yolov8n.pt") -> None:
        """Initialize and load the YOLO model.

        TODO: Update `model_path` to point to your trained weights or change
        loading logic to integrate with your detection framework.
        """
        self.model_path = model_path
        if YOLO is None:
            logger.warning("`ultralytics` package not found. Detector will be a stub.")
            self._model = None
        else:
            self._model = YOLO(model_path)

    def detect(self, image: np.ndarray, conf_thres: float = 0.25) -> List[Dict[str, Any]]:
        """Run detection on a single image.

        Args:
            image: BGR image as produced by OpenCV.
            conf_thres: Confidence threshold for filtering detections. TODO tune.

        Returns:
            A list of detections; each detection is a dict with keys
            `bbox` (x1,y1,x2,y2), `score`, `class`, and optionally `mask`.
        """
        if self._model is None:
            return []

        # ultralytics YOLO accepts numpy images in RGB
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self._model.predict(img_rgb, conf=conf_thres, verbose=False)
        detections: List[Dict[str, Any]] = []
        for res in results:
            # depending on ultralytics API version, results.boxes exists
            boxes = getattr(res, "boxes", None)
            if boxes is None:
                continue
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                score = float(box.conf[0])
                cls = int(box.cls[0])
                detections.append({"bbox": (x1, y1, x2, y2), "score": score, "class": cls})
        return detections


def draw_detections(image: np.ndarray, dets: List[Dict[str, Any]]) -> np.ndarray:
    """Overlay detections on `image` and return annotated image.

    This is a convenience helper for quick visual debugging.
    """
    out = image.copy()
    for d in dets:
        x1, y1, x2, y2 = map(int, d["bbox"])
        label = f"{d.get('class', '?')}:{d.get('score', 0):.2f}"
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(out, label, (x1, max(10, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return out


def main() -> None:
    """CLI: run detector on an image or video stream.

    Example:
        python -m perception.vehicle_detection_yolov8 --source 0
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="0", help="Video file path or camera index (default 0)")
    parser.add_argument("--weights", type=str, default="yolov8n.pt", help="Path to YOLO weights (TODO customize)")
    args = parser.parse_args()

    # cast '0' to int for webcam
    source: str | int = int(args.source) if args.source.isdigit() else args.source

    cap = cv2.VideoCapture(source)
    detector = YOLOv8Detector(args.weights)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        dets = detector.detect(frame)
        out = draw_detections(frame, dets)
        cv2.imshow("YOLOv8 Detection", out)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

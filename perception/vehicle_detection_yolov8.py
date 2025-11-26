"""Vehicle detection using YOLOv8 (ultralytics) — implemented for the nano model.

This module provides a `VehicleDetector` class which wraps a YOLOv8 model
and convenience functions to run inference on images and videos. The
implementation uses the `ultralytics` package; if not available a friendly
warning is shown and the detector returns no detections.

Hyperparameters (top-of-file):
- MODEL_PATH: default weights file name (YOLOv8n)
- DEVICE: compute device string for PyTorch/ultralytics
- CONF_THRES: score threshold for detections
- IOU_THRES: (optional) NMS iou threshold when running inference

"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple
import logging
import argparse
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------- Hyperparameters ----------------------------
MODEL_PATH: str = "yolov8n.pt"  # TODO: point to your trained weights
DEVICE: str = "cpu"  # e.g., 'cpu' or 'cuda'
CONF_THRES: float = 0.35
IOU_THRES: float = 0.45
# -------------------------------------------------------------------------


try:
    from ultralytics import YOLO  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    YOLO = None  # type: ignore


class VehicleDetector:
    """Thin wrapper for a YOLOv8 model to detect vehicles.

    Args:
        model_path: pathlib.Path-like to YOLOv8 weights (str).
        device: device string for inference ("cpu" or "cuda").
    """

    def __init__(self, model_path: str = MODEL_PATH, device: str = DEVICE, conf_thres: float = CONF_THRES) -> None:
        self.model_path = str(model_path)
        self.device = device
        self.conf_thres = conf_thres
        if YOLO is None:
            logger.warning("ultralytics package not available. VehicleDetector will be a no-op.")
            self.model = None
            self.names = {}
        else:
            self.model = YOLO(self.model_path)
            try:
                self.model.to(self.device)
            except Exception:
                # depending on ultralytics version or device string this might not be necessary
                pass
            self.names = getattr(self.model, "names", {}) or {}

    def detect(self, frame: np.ndarray, conf_thres: float | None = None) -> List[Dict[str, Any]]:
        """Detect vehicles in a BGR OpenCV frame.

        Returns a list of dicts where each dict contains:
            - bbox: (x1, y1, x2, y2) float coordinates
            - conf: confidence score
            - class_id: numeric class id
            - class_name: human readable class name
        """
        if self.model is None:
            return []
        conf = conf_thres if conf_thres is not None else self.conf_thres
        # ultralytics expects RGB inputs
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # run inference
        results = self.model(img, conf=conf, iou=IOU_THRES, verbose=False)
        detections: List[Dict[str, Any]] = []
        for res in results:
            # 'boxes' may be different depending on library version
            boxes = getattr(res, "boxes", None)
            if boxes is None:
                continue
            for box in boxes:
                try:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf_score = float(box.conf[0])
                    class_id = int(box.cls[0])
                except Exception:
                    # fallback on object fields if types differ
                    xyxy = box.xyxy.tolist()[0]
                    x1, y1, x2, y2 = map(float, xyxy)
                    conf_score = float(box.conf.tolist()[0])
                    class_id = int(box.cls.tolist()[0])
                class_name = self.names.get(class_id, str(class_id))
                detections.append({"bbox": (x1, y1, x2, y2), "conf": conf_score, "class_id": class_id, "class_name": class_name})
        return detections


def draw_detections(frame: np.ndarray, detections: List[Dict[str, Any]], box_color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
    """Draw bounding boxes and labels on `frame` and return the annotated image.
    """
    out = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = map(int, det["bbox"])
        conf = det.get("conf", 0.0)
        class_name = det.get("class_name", str(det.get("class_id", "?")))
        label = f"{class_name} {conf:.2f}"
        cv2.rectangle(out, (x1, y1), (x2, y2), box_color, 2)
        cv2.putText(out, label, (x1, max(10, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)
    return out


def run_video(input_path: str | int, output_path: str, detector: VehicleDetector | None = None, show: bool = False) -> None:
    """Run vehicle detection on a video file or camera and write an annotated video.

    Args:
        input_path: path to input video file or camera index (0, 1, ...).
        output_path: path to output annotated video file.
        detector: Optional pre-instantiated VehicleDetector. If None a default is created.
        show: whether to display the frames while processing (blocks until q pressed).
    """
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open input {input_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
    if detector is None:
        detector = VehicleDetector()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        dets = detector.detect(frame)
        annotated = draw_detections(frame, dets)
        out.write(annotated)
        if show:
            cv2.imshow("Detections", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    cap.release()
    out.release()
    if show:
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="0", help="Video file path or camera index (default: 0)")
    parser.add_argument("--output", type=str, default="out.mp4", help="Output annotated video path")
    parser.add_argument("--weights", type=str, default=MODEL_PATH, help="YOLO weights path (TODO)")
    parser.add_argument("--conf", type=float, default=CONF_THRES, help="Detection confidence threshold")
    parser.add_argument("--show", action="store_true", help="Display frames while processing")
    args = parser.parse_args()
    source = int(args.source) if str(args.source).isdigit() else args.source
    detector = VehicleDetector(model_path=args.weights, conf_thres=args.conf)
    run_video(source, args.output, detector=detector, show=args.show)


if __name__ == "__main__":
    main()
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

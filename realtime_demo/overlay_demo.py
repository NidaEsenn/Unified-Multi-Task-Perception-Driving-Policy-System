"""Realtime demo that overlays perception outputs on video frames.

This demo loads simple detection and segmentation components and overlays
their outputs on a camera or video stream for quick visualization.
"""
from __future__ import annotations

from typing import Any
import argparse
import logging

import cv2
import numpy as np

from perception.vehicle_detection_yolov8 import YOLOv8Detector, draw_detections
from perception.lane_unet import LaneSegmentationModel as LaneSegmentationModel
from perception.drivable_area_unet import DrivableAreaSegmentationModel as DrivableAreaSegmentationModel
from utils.config import CONFIG

logger = logging.getLogger(__name__)


def overlay_masks(frame: np.ndarray, mask: np.ndarray, color: tuple[int, int, int] = (0, 0, 255), alpha: float = 0.4) -> np.ndarray:
    """Overlay a single-channel mask on `frame` using `color` and `alpha` blending."""
    vis = frame.copy()
    mask_rgb = np.zeros_like(frame, dtype=np.uint8)
    mask_rgb[..., 0] = (mask * 255).astype(np.uint8)
    blended = cv2.addWeighted(vis, 1.0, mask_rgb, alpha, 0)
    return blended


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="0", help="Camera index or video file (TODO)")
    parser.add_argument("--weights-det", type=str, default=f"{CONFIG.CHECKPOINT_DIR}/yolov8n.pt", help="Detector weights (TODO)")
    parser.add_argument("--weights-lane", type=str, default=f"{CONFIG.CHECKPOINT_DIR}/lane_unet.pth", help="Lane model weights (TODO)")
    parser.add_argument("--weights-drivable", type=str, default=f"{CONFIG.CHECKPOINT_DIR}/drivable_unet.pth", help="Drivable model weights (TODO)")
    args = parser.parse_args()

    src: str | int = int(args.source) if str(args.source).isdigit() else args.source
    cap = cv2.VideoCapture(src)
    detector = YOLOv8Detector(args.weights_det)
    lane_model = LaneSegmentationModel(model_path=args.weights_lane)
    drivable_model = DrivableAreaSegmentationModel(model_path=args.weights_drivable)
    # TODO: load weights into lane_model and drivable_model if provided

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        dets = detector.detect(frame)
        lane_mask = predict_lane(lane_model, frame)
        drivable_mask = predict_drivable(drivable_model, frame)

        out = draw_detections(frame, dets)
        out = overlay_masks(out, drivable_mask, color=(0, 255, 0), alpha=0.25)
        out = overlay_masks(out, lane_mask, color=(0, 0, 255), alpha=0.4)

        cv2.imshow("Realtime Overlay Demo", out)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

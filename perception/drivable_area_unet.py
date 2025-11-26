"""Drivable area segmentation UNet implementation and utilities.

Mirrors `lane_unet.py` but dedicated to drivable-area segmentation.
Keep hyperparameters near the top for easy customization.
"""
from __future__ import annotations

from typing import Tuple
import argparse
import logging
from pathlib import Path

import numpy as np
import cv2
import torch
import torch.nn as nn

from perception.lane_unet import UNet  # reuse the UNet implementation

logger = logging.getLogger(__name__)

# ------------------------- Hyperparameters ------------------------------
MODEL_PATH: str = ""  # TODO: path to your model
DEVICE: str = "cpu"
INPUT_SIZE: Tuple[int, int] = (256, 256)
THRESHOLD: float = 0.5
# -----------------------------------------------------------------------


class DrivableAreaSegmentationModel:
    """Wrapper for the UNet that performs drivable area segmentation."""

    def __init__(self, model_path: str = MODEL_PATH, device: str = DEVICE) -> None:
        self.model = UNet(in_ch=3, out_ch=1, features=32)
        self.device = torch.device(device)
        self.model.to(self.device)
        if model_path and Path(model_path).exists():
            state = torch.load(model_path, map_location=self.device)
            if isinstance(state, dict) and "state_dict" in state:
                self.model.load_state_dict(state["state_dict"])
            else:
                try:
                    self.model.load_state_dict(state)
                except Exception as e:
                    logger.warning("Failed to load checkpoint for drivable model: %s", e)

    def _preprocess(self, image: np.ndarray) -> torch.Tensor:
        h, w = INPUT_SIZE
        resized = cv2.resize(image, (w, h))
        tensor = torch.from_numpy(resized).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        return tensor.to(self.device)

    def _postprocess(self, mask: torch.Tensor, orig_size: Tuple[int, int]) -> np.ndarray:
        mask_np = mask.squeeze().cpu().numpy()
        mask_resized = cv2.resize(mask_np, (orig_size[1], orig_size[0]))
        return mask_resized

    def infer_on_image(self, image: np.ndarray, threshold: float = THRESHOLD) -> Tuple[np.ndarray, np.ndarray]:
        self.model.eval()
        tensor = self._preprocess(image)
        with torch.no_grad():
            out = self.model(tensor)
        mask = self._postprocess(out, image.shape[:2])
        binary = (mask >= threshold).astype("uint8")
        overlay = image.copy()
        green = np.zeros_like(overlay)
        green[..., 1] = (binary * 255).astype("uint8")
        overlay = cv2.addWeighted(overlay, 0.6, green, 0.4, 0)
        return mask, overlay

    def infer_on_video(self, input_path: str | int, output_path: str, threshold: float = THRESHOLD, show: bool = False) -> None:
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open input {input_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            _, overlay = self.infer_on_image(frame, threshold=threshold)
            out.write(overlay)
            if show:
                cv2.imshow("Drivable Area", overlay)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        cap.release()
        out.release()
        if show:
            cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="0", help="Video path or webcam (default 0)")
    parser.add_argument("--output", type=str, default="drivable_out.mp4", help="Output annotated video path")
    parser.add_argument("--weights", type=str, default=MODEL_PATH, help="Model checkpoint (TODO)")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    src = int(args.source) if args.source.isdigit() else args.source
    model = DrivableAreaSegmentationModel(model_path=args.weights)
    model.infer_on_video(src, args.output, show=args.show)


if __name__ == "__main__":
    main()

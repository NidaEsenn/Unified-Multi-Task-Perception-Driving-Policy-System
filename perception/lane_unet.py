"""Lane segmentation UNet implementation and utilities.

This file provides a minimal UNet (PyTorch) for binary segmentation and a
wrapper `LaneSegmentationModel` that offers simple inference utilities for
images and videos.

Hyperparameters (near top):
- MODEL_PATH: default checkpoint path
- DEVICE: 'cpu' or 'cuda'
- INPUT_SIZE: (H,W) used to resize images for model input
- THRESHOLD: segmentation threshold for binary mask
"""
from __future__ import annotations

from typing import Tuple, Optional
import argparse
import logging
from pathlib import Path

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ------------------------- Hyperparameters ------------------------------
MODEL_PATH: str = ""  # TODO: point to a trained checkpoint
DEVICE: str = "cpu"
INPUT_SIZE: Tuple[int, int] = (256, 256)
THRESHOLD: float = 0.5
# -----------------------------------------------------------------------


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNet(nn.Module):
    """A lightweight UNet for binary segmentation."""

    def __init__(self, in_ch: int = 3, out_ch: int = 1, features: int = 32) -> None:
        super().__init__()
        self.down1 = DoubleConv(in_ch, features)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(features, features * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = DoubleConv(features * 2, features * 4)
        self.pool3 = nn.MaxPool2d(2)
        self.bridge = DoubleConv(features * 4, features * 8)
        self.up3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.up_conv3 = DoubleConv(features * 8, features * 4)
        self.up2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.up_conv2 = DoubleConv(features * 4, features * 2)
        self.up1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.up_conv1 = DoubleConv(features * 2, features)
        self.final = nn.Conv2d(features, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d1 = self.down1(x)
        p1 = self.pool1(d1)
        d2 = self.down2(p1)
        p2 = self.pool2(d2)
        d3 = self.down3(p2)
        p3 = self.pool3(d3)
        b = self.bridge(p3)
        u3 = self.up3(b)
        cat3 = torch.cat([u3, d3], dim=1)
        uc3 = self.up_conv3(cat3)
        u2 = self.up2(uc3)
        cat2 = torch.cat([u2, d2], dim=1)
        uc2 = self.up_conv2(cat2)
        u1 = self.up1(uc2)
        cat1 = torch.cat([u1, d1], dim=1)
        uc1 = self.up_conv1(cat1)
        out = self.final(uc1)
        return torch.sigmoid(out)


class LaneSegmentationModel:
    """Wrapper for the UNet lane segmentation model providing load/infer.

    Methods:
        - __init__: constructs model and optionally loads weights
        - infer_on_image: returns binary mask and overlay
        - infer_on_video: process a video, write annotated output
    """

    def __init__(self, model_path: str = MODEL_PATH, device: str = DEVICE) -> None:
        self.model = UNet(in_ch=3, out_ch=1, features=32)
        self.device = torch.device(device)
        self.model.to(self.device)
        if model_path and Path(model_path).exists():
            checkpoint = torch.load(model_path, map_location=self.device)
            if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["state_dict"])
            else:
                try:
                    self.model.load_state_dict(checkpoint)
                except Exception as e:
                    logger.warning("Failed to load state_dict directly: %s", e)

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
        """Run inference on a single image and return (mask, overlay).

        mask: float array HxW in [0,1]
        overlay: image with mask blended on top
        """
        self.model.eval()
        tensor = self._preprocess(image)
        with torch.no_grad():
            out = self.model(tensor)
        mask = self._postprocess(out, image.shape[:2])
        # binary mask
        binary = (mask >= threshold).astype("uint8")
        overlay = image.copy()
        red = np.zeros_like(overlay)
        red[..., 2] = (binary * 255).astype("uint8")
        overlay = cv2.addWeighted(overlay, 0.7, red, 0.3, 0)
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
                cv2.imshow("Lane Segmentation", overlay)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        cap.release()
        out.release()
        if show:
            cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="0", help="Video path or webcam (default 0)")
    parser.add_argument("--output", type=str, default="lane_out.mp4", help="Output annotated video path")
    parser.add_argument("--weights", type=str, default=MODEL_PATH, help="Model checkpoint (TODO)")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    src = int(args.source) if args.source.isdigit() else args.source
    model = LaneSegmentationModel(model_path=args.weights)
    model.infer_on_video(src, args.output, show=args.show)


if __name__ == "__main__":
    main()

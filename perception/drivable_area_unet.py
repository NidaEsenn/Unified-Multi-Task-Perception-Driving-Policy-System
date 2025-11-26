"""Drivable area segmentation using a UNet-style model (boilerplate).

This mirrors `lane_unet.py` but is intended for full-area segmentation.
Use this as a starting point to plug in training, datasets and real models.
"""
from __future__ import annotations

from typing import Any
import logging
import argparse

import torch
import torch.nn as nn
import numpy as np

logger = logging.getLogger(__name__)


class DrivableUNet(nn.Module):
    """Placeholder UNet for drivable-area segmentation."""

    def __init__(self, in_channels: int = 3, out_channels: int = 1) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(in_channels, 32, 3, padding=1), nn.ReLU(), nn.Conv2d(32, out_channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return torch.sigmoid(self.net(x))


def build_model() -> nn.Module:
    """Return a drivable-area segmentation model.

    TODO: support configurable backbones and pretrained encoders.
    """
    return DrivableUNet()


def predict(model: nn.Module, image: np.ndarray) -> np.ndarray:
    """Predict a drivable-area mask for `image`.

    Returns an HxW mask in [0,1].
    """
    model.eval()
    tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    with torch.no_grad():
        out = model(tensor)
    mask = out.squeeze(0).squeeze(0).cpu().numpy()
    return mask


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="", help="Path to weights (TODO)")
    args = parser.parse_args()
    model = build_model()
    if args.weights:
        state = torch.load(args.weights, map_location="cpu")
        model.load_state_dict(state)
    logger.info("Drivable area UNet ready. Use predict() programmatically.")


if __name__ == "__main__":
    main()

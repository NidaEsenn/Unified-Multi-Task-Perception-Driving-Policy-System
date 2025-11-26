"""Lane segmentation model (UNet) boilerplate.

This file contains a lightweight UNet model scaffold, a factory function to
construct the model, and a convenience `predict` wrapper.

Python: 3.10+
"""
from __future__ import annotations

from typing import Any
import argparse
import logging

import torch
import torch.nn as nn
import numpy as np

logger = logging.getLogger(__name__)


class SimpleUNet(nn.Module):
    """A minimal UNet-like segmentation backbone used as a placeholder.

    Replace this with a proper UNet implementation or import from a
    segmentation library for production experiments.
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 1, features: int = 32) -> None:
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv2d(in_channels, features, 3, padding=1), nn.ReLU())
        self.decoder = nn.Sequential(nn.Conv2d(features, out_channels, 3, padding=1), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def build_model(in_channels: int = 3, out_channels: int = 1) -> nn.Module:
    """Factory to return a UNet-like model.

    TODO: Replace with a full UNet implementation and expose hyperparameters.
    """
    return SimpleUNet(in_channels=in_channels, out_channels=out_channels)


def predict(model: nn.Module, image: np.ndarray) -> np.ndarray:
    """Run a forward pass and return a segmentation mask in numpy format.

    Args:
        model: A PyTorch segmentation model.
        image: HxWxC uint8 BGR image (OpenCV format).

    Returns:
        mask: HxW float32 mask in range [0,1].
    """
    model.eval()
    tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    with torch.no_grad():
        out = model(tensor)
    mask = out.squeeze(0).squeeze(0).cpu().numpy()
    return mask


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="", help="Path to model weights (TODO)")
    args = parser.parse_args()
    model = build_model()
    if args.weights:
        # TODO: implement robust checkpoint loading
        state = torch.load(args.weights, map_location="cpu")
        model.load_state_dict(state)
    logger.info("Lane UNet ready. Use predict() from other scripts.")


if __name__ == "__main__":
    main()

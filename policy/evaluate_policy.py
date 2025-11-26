"""Evaluation utilities for policy models.

Provides a simple evaluation loop and metrics. Extend with domain-specific
metrics such as lateral error, collision rate, or closed-loop rollout tests.
"""
from __future__ import annotations

from typing import Dict, Any
import argparse
import logging

import torch
from torch.utils.data import DataLoader
import numpy as np

from policy.convlstm_model import ConvLSTMPolicy
from utils.datasets import FrameDataset

logger = logging.getLogger(__name__)


def evaluate(model: torch.nn.Module, loader: DataLoader, device: str = "cpu") -> Dict[str, float]:
    """Run a simple evaluation and return scalar metrics.

    TODO: add application-specific metrics and richer diagnostic outputs.
    """
    model.to(device)
    model.eval()
    losses = []
    mse = torch.nn.MSELoss()
    with torch.no_grad():
        for batch in loader:
            inputs = batch["frames"].to(device)
            targets = batch.get("targets", torch.zeros(inputs.shape[0], 2, device=device))
            outputs = model(inputs)
            losses.append(float(mse(outputs, targets)))
    return {"mse": float(np.mean(losses))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data/val", help="Path to validation dataset (TODO)")
    parser.add_argument("--weights", type=str, default="", help="Path to trained weights (TODO)")
    args = parser.parse_args()

    dataset = FrameDataset(root_dir=args.data_dir)
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=4)
    model = ConvLSTMPolicy()
    if args.weights:
        state = torch.load(args.weights, map_location="cpu")
        model.load_state_dict(state)
    metrics = evaluate(model, loader)
    logger.info("Evaluation metrics: %s", metrics)


if __name__ == "__main__":
    main()

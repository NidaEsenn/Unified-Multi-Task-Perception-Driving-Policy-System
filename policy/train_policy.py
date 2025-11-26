"""Training script for policy models (boilerplate).

This script provides a simple training loop and hooks to integrate your
dataset and model. Customize dataset paths, hyperparameters, and augmentations.
"""
from __future__ import annotations

from typing import Any
import argparse
import logging
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from policy.convlstm_model import ConvLSTMPolicy
from utils.datasets import FrameDataset

logger = logging.getLogger(__name__)


def train(
    train_loader: DataLoader,
    model: nn.Module,
    epochs: int = 10,
    lr: float = 1e-3,
    device: str = "cpu",
) -> None:
    """Basic training loop.

    TODO: Replace with a more complete training/eval loop, checkpointing,
    mixed precision, and logging to TensorBoard/Weights & Biases.
    """
    model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    model.train()
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        t0 = time.time()
        for batch in train_loader:
            # TODO: adapt to your dataset return signature
            inputs = batch["frames"].to(device)
            targets = batch.get("targets", torch.zeros(inputs.shape[0], 2, device=device))
            optim.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optim.step()
            epoch_loss += loss.item()
        logger.info("Epoch %d loss=%.4f time=%.1fs", epoch, epoch_loss / len(train_loader), time.time() - t0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data/train", help="Path to training dataset (TODO)")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs (TODO tune)")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size (TODO tune)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    args = parser.parse_args()

    dataset = FrameDataset(root_dir=args.data_dir)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    model = ConvLSTMPolicy()
    train(loader, model, epochs=args.epochs, lr=args.lr)


if __name__ == "__main__":
    main()

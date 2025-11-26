"""Training script for policy models (boilerplate).

This script provides a simple training loop and hooks to integrate your
dataset and model. Customize dataset paths, hyperparameters, and augmentations.
"""
from __future__ import annotations

from typing import Any
from pathlib import Path
import argparse
import logging
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from policy.convlstm_model import ConvLSTMPolicy
from utils.datasets import DrivingFramesDataset, FrameDataset
from utils.config import CONFIG, STEERING_DATASET_DIR, POLICY_CHECKPOINT_DIR

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
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    model.train()
    best_loss = float("inf")
    checkpoint_dir = Path(POLICY_CHECKPOINT_DIR)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "policy_best.pt"
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        t0 = time.time()
        for batch in train_loader:
            # TODO: adapt to your dataset return signature
            inputs = batch["frames"].to(device)
            # dataset stores `target`; default to zero if missing
            targets = batch.get("target", torch.zeros(inputs.shape[0], device=device))
            # ensure targets are on the correct device
            if isinstance(targets, torch.Tensor):
                targets = targets.to(device)
            else:
                targets = torch.tensor(targets, device=device)
            if targets.dim() == 1:
                targets = targets.unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_avg = epoch_loss / len(train_loader)
        logger.info("Epoch %d loss=%.6f time=%.1fs", epoch, epoch_avg, time.time() - t0)
        # checkpoint if improved
        if epoch_avg < best_loss:
            best_loss = epoch_avg
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "best_loss": best_loss}, checkpoint_path)
            logger.info("Saved best checkpoint to %s with loss %.6f", checkpoint_path, best_loss)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default=str(STEERING_DATASET_DIR), help="Path to steering dataset (Udacity) (TODO)")
    parser.add_argument("--seq-len", type=int, default=3, help="Sequence length for policy frames")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs (TODO tune)")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size (TODO tune)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--num-workers", type=int, default=CONFIG.NUM_WORKERS, help="Number of DataLoader workers")
    args = parser.parse_args()

    num_workers = int(args.num_workers) if hasattr(args, 'num_workers') and args.num_workers is not None else CONFIG.NUM_WORKERS
    pin_memory = True if (CONFIG.DEVICE == "cuda") else False
    dataset = DrivingFramesDataset(csv_path=Path(args.data_dir) / "driving_log.csv", seq_len=args.seq_len, augment=True, resize=CONFIG.POLICY_FRAME_SIZE)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    model = ConvLSTMPolicy()
    train(loader, model, epochs=args.epochs, lr=args.lr, device=CONFIG.DEVICE)


if __name__ == "__main__":
    main()

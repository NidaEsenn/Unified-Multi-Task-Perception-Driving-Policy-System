#!/usr/bin/env python3
"""Small end-to-end training smoke test for ConvLSTM policy.

This script verifies the `DrivingFramesDataset`, runs a short training loop
for a few epochs, and computes evaluation metrics (MSE, MAE and steering
smoothness) to ensure end-to-end pipeline functions. It saves the best
checkpoint to `POLICY_CHECKPOINT_DIR/policy_best.pt`.

Usage:
    python scripts/run_small_training.py
"""
from __future__ import annotations

from pathlib import Path
import time
import argparse
import sys
import logging
import random

import torch
from torch.utils.data import DataLoader, random_split

import sys
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))
from utils.datasets import DrivingFramesDataset
from utils.config import STEERING_DATASET_DIR, POLICY_CHECKPOINT_DIR, CONFIG
from policy.train_policy import train
from policy.convlstm_model import ConvLSTMPolicy
from policy.evaluate_policy import evaluate


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path(STEERING_DATASET_DIR))
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=3)
    parser.add_argument("--num-workers", type=int, default=CONFIG.NUM_WORKERS)
    parser.add_argument("--resize", type=int, nargs=2, default=list(CONFIG.POLICY_FRAME_SIZE), help="Resize frames as H W")
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args(argv or sys.argv[1:])

    # instantiate dataset
    ds = DrivingFramesDataset(csv_path=args.data_dir / "driving_log.csv", seq_len=args.seq_len, augment=True, resize=tuple(args.resize))
    print(f"Dataset size: {len(ds)}")
    if len(ds) <= 10:
        print("Dataset too small for smoke test (>10 required). Aborting.")
        return 1

    sample = ds[0]
    frames = sample["frames"]
    target = sample["target"] if "target" in sample else None
    print("Sample frames type:", type(frames))
    if isinstance(frames, torch.Tensor):
        print("frames shape:", frames.shape)
        # expect [T, C, H, W]
        assert frames.dim() == 4 and frames.shape[0] == args.seq_len
    else:
        print("frames shape (numpy/list):", getattr(frames, 'shape', type(frames)))
    print("Sample steering:", target)

    # split dataset to train/val
    n = len(ds)
    val_len = max(1, int(n * 0.2))
    train_len = n - val_len
    train_ds, val_ds = random_split(ds, [train_len, val_len], generator=torch.Generator().manual_seed(42))
    print(f"Train size: {len(train_ds)}  Val size: {len(val_ds)}")

    num_workers = int(args.num_workers)
    pin_memory = True if (str(CONFIG.DEVICE) == "cuda") else False
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    # build model and run training
    model = ConvLSTMPolicy(in_channels=3, hidden_dim=32, num_outputs=1)
    t0 = time.time()
    train(train_loader, model, epochs=args.epochs, lr=args.lr, device=CONFIG.DEVICE)
    total_time = time.time() - t0
    print(f"Total training time: {total_time:.1f}s")

    # load best checkpoint
    ckpt_path = Path(POLICY_CHECKPOINT_DIR) / "policy_best.pt"
    if ckpt_path.exists():
        ckpt = torch.load(str(ckpt_path), map_location=CONFIG.DEVICE)
        model.load_state_dict(ckpt.get("model_state_dict", ckpt))
        print("Loaded best checkpoint:", ckpt_path)
    else:
        print("No checkpoint found at", ckpt_path)

    metrics = evaluate(model, val_loader, device=CONFIG.DEVICE)
    print("Evaluation metrics:")
    for k, v in metrics.items():
        print(f" - {k}: {v}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

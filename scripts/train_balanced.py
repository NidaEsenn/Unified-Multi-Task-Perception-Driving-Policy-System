#!/usr/bin/env python3
"""Balanced training script using WeightedRandomSampler based on steering bins.

Usage:
    python scripts/train_balanced.py --epochs 8 --batch-size 16 --num-workers 0

This script will:
 - Compute a WeightedRandomSampler on binned steering labels to counter class imbalance
 - Evaluate baseline checkpoint MSE on a validation subset
 - Train for `--epochs` using the balanced sampler
 - Save the best checkpoint as `checkpoints/policy/policy_balanced_best.pt`
 - Evaluate and print final validation statistics (MSE)
"""
from __future__ import annotations

from pathlib import Path
import argparse
import time
import csv
import math
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler, random_split

# repository imports
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))
from utils.config import CONFIG, STEERING_DATASET_DIR
from utils.datasets import DrivingFramesDataset
from policy.convlstm_model import ConvLSTMPolicy


def compute_sample_weights(csv_path: Path, num_bins: int = 11) -> tuple[list, np.ndarray]:
    """Return per-sample weight list and corresponding labels array from CSV path."""
    vals = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            try:
                v = float(row[3]) if len(row) > 3 else 0.0
            except Exception:
                v = 0.0
            vals.append(v)
    labels = np.array(vals)
    bins = np.linspace(-1.0, 1.0, num_bins + 1)
    bidx = np.digitize(labels, bins) - 1
    bidx = np.clip(bidx, 0, num_bins - 1)
    freq = np.bincount(bidx, minlength=num_bins)
    # prevent division by zero
    freq = freq + 1e-6
    inv_freq = 1.0 / freq
    inv_freq = inv_freq / inv_freq.sum() * num_bins
    sample_weights = inv_freq[bidx]
    return sample_weights.tolist(), labels


def evaluate_model_mse(model: torch.nn.Module, dataset: DrivingFramesDataset, indices: list[int], device: torch.device) -> float:
    mse_fn = torch.nn.MSELoss(reduction="mean")
    model.eval()
    with torch.no_grad():
        total = 0.0
        for idx in indices:
            sample = dataset[idx]
            frames = sample["frames"].unsqueeze(0).to(device)
            target = sample["target"].to(device)
            out = model(frames).squeeze()
            total += float(mse_fn(out.unsqueeze(0), target.unsqueeze(0)))
    return total / len(indices)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path(STEERING_DATASET_DIR))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args(argv or sys.argv[1:])

    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print("Device:", device)

    csv_path = args.data_dir / "driving_log.csv"
    if not csv_path.exists():
        raise SystemExit("driving_log.csv not found in dataset dir")

    # Create val subset indices for quick validation (stable random subset)
    full_ds_for_weights = DrivingFramesDataset(csv_path=csv_path, seq_len=args.seq_len, augment=False, resize=CONFIG.POLICY_FRAME_SIZE)
    num_samples = len(full_ds_for_weights)
    print(f"Dataset size: {num_samples}")

    # compute sample weights for weighted sampling
    sample_weights, labels = compute_sample_weights(csv_path, num_bins=11)

    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=num_samples, replacement=True)

    # train/val split
    val_len = max(1, int(num_samples * 0.2))
    train_len = num_samples - val_len
    # use deterministic split
    generator = torch.Generator().manual_seed(42)
    train_ds, val_ds = random_split(full_ds_for_weights, [train_len, val_len], generator=generator)

    # train and val loaders
    pin_memory = True if (str(CONFIG.DEVICE) == "cuda") else False
    # We will use the sampler for train loader; DataLoader with random_split returns subset which still indexes into the same dataset
    # To use the sampler, we pass the full train_ds indices mapping; however using WeightedRandomSampler with subset indices is a bit complex
    # Easiest approach: create a DataLoader using the original dataset but pass the sampler and a custom batch_size
    # then use the same `train_len` number of samples (sampler will sample across full dataset)
    train_loader = DataLoader(full_ds_for_weights, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin_memory)

    # Baseline model: load existing checkpoint if present
    ckpt_candidates = [Path("models/policy/policy_best.pt"), Path("checkpoints/policy/policy_best.pt")]
    ckpt = None
    for p in ckpt_candidates:
        if p.exists():
            ckpt = p
            break

    baseline_val_mse = None
    if ckpt is not None:
        print("Found baseline checkpoint:", ckpt)
        baseline_state = torch.load(ckpt, map_location=device)
        baseline_state = baseline_state.get("model_state_dict", baseline_state)
        model_baseline = ConvLSTMPolicy().to(device)
        model_baseline.load_state_dict(baseline_state)
        # evaluate baseline on a random subset for speed
        n_val = min(1024, len(val_ds))
        indices = np.random.choice(len(val_ds), size=n_val, replace=False).tolist()
        baseline_val_mse = evaluate_model_mse(model_baseline, val_ds, indices, device)
        print(f"Baseline val MSE (subset {n_val}): {baseline_val_mse:.6f}")
    else:
        print("No baseline checkpoint found; skipping baseline evaluation")

    # Start training from scratch
    model = ConvLSTMPolicy().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.MSELoss()

    best_val = float("inf")
    best_state = None

    indices_for_eval = np.random.choice(len(val_ds), size=min(1024, len(val_ds)), replace=False).tolist()

    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        batch_count = 0
        t0 = time.time()
        for batch in train_loader:
            inputs = batch["frames"].to(device)
            targets = batch.get("target", torch.zeros(inputs.shape[0], device=device))
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
            epoch_loss += float(loss.detach().cpu().item())
            batch_count += 1
        epoch_avg = epoch_loss / (batch_count if batch_count > 0 else 1)

        # Validation subset MSE
        val_mse = evaluate_model_mse(model, val_ds, indices_for_eval, device)
        if val_mse < best_val:
            best_val = val_mse
            best_state = model.state_dict()
            # save best
            ckpt_dir = Path('checkpoints/policy')
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save({"epoch": epoch, "model_state_dict": best_state, "best_loss": best_val}, ckpt_dir / 'policy_balanced_best.pt')

        print(f"Epoch {epoch}/{args.epochs} train_loss={epoch_avg:.6f} val_mse={val_mse:.6f} time={(time.time()-t0):.1f}s")

    total_time = time.time() - start_time
    print(f"Training complete, elapsed {total_time:.1f}s best_val_mse={best_val:.6f}")

    # Final evaluation on the same subset
    final_val_mse = evaluate_model_mse(model, val_ds, indices_for_eval, device)
    print(f"Final val MSE (subset {len(indices_for_eval)}): {final_val_mse:.6f}")

    if baseline_val_mse is not None:
        print(f"Baseline val MSE: {baseline_val_mse:.6f}  -> Balanced training val MSE: {final_val_mse:.6f}  (improvement {baseline_val_mse - final_val_mse:+.6f})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

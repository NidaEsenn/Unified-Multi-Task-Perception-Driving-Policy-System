"""Evaluation utilities for policy models.

Provides a simple evaluation loop and metrics. Extend with domain-specific
metrics such as lateral error, collision rate, or closed-loop rollout tests.
"""
from __future__ import annotations

from typing import Dict, Any, List
import argparse
import logging

import torch
from torch.utils.data import DataLoader
import numpy as np

from pathlib import Path
from policy.convlstm_model import ConvLSTMPolicy
from utils.datasets import DrivingFramesDataset, FrameDataset
from utils.config import CONFIG, STEERING_DATASET_DIR

logger = logging.getLogger(__name__)


def evaluate(model: torch.nn.Module, loader: DataLoader, device: str = "cpu") -> Dict[str, float]:
    """Run a simple evaluation and return scalar metrics.

    TODO: add application-specific metrics and richer diagnostic outputs.
    """
    model.to(device)
    model.eval()
    losses: List[float] = []
    maes: List[float] = []
    all_targets: List[float] = []
    all_preds: List[float] = []
    mse = torch.nn.MSELoss()
    with torch.no_grad():
        for batch in loader:
            inputs = batch["frames"].to(device)
            targets = batch.get("target", torch.zeros(inputs.shape[0], device=device))
            if isinstance(targets, torch.Tensor):
                targets = targets.to(device)
            else:
                targets = torch.tensor(targets, device=device)
            if targets.dim() == 2 and targets.shape[1] == 1:
                targets = targets.squeeze(1)
            outputs = model(inputs)
            # outputs shape (B, 1) or (B,) -> unify
            if outputs.dim() == 2 and outputs.shape[1] == 1:
                preds = outputs.squeeze(1).cpu().numpy()
            else:
                preds = outputs.cpu().numpy().squeeze()
            targs = targets.cpu().numpy().squeeze()
            losses.append(float(mse(torch.from_numpy(preds), torch.from_numpy(targs))))
            maes.append(float(torch.nn.functional.l1_loss(torch.from_numpy(preds), torch.from_numpy(targs)).item()))
            all_targets.extend(targs.tolist())
            all_preds.extend(preds.tolist())
    mse_val = float(np.mean(losses)) if losses else float("nan")
    mae_val = float(np.mean(maes)) if maes else float("nan")
    # steering smoothness metric: mean absolute difference between consecutive target values
    target_smoothness = float(np.mean(np.abs(np.diff(np.array(all_targets))))) if len(all_targets) > 1 else float("nan")
    pred_smoothness = float(np.mean(np.abs(np.diff(np.array(all_preds))))) if len(all_preds) > 1 else float("nan")
    return {"mse": mse_val, "mae": mae_val, "target_smoothness": target_smoothness, "pred_smoothness": pred_smoothness}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default=str(STEERING_DATASET_DIR), help="Path to validation dataset (Udacity steering) (TODO)")
    parser.add_argument("--seq-len", type=int, default=3, help="Sequence length for evaluation")
    parser.add_argument("--batch-size", type=int, default=8, help="Evaluation batch size")
    parser.add_argument("--num-workers", type=int, default=CONFIG.NUM_WORKERS, help="DataLoader workers")
    parser.add_argument("--resize", type=int, nargs=2, default=list(CONFIG.POLICY_FRAME_SIZE), help="Resize frames as H W for eval")
    parser.add_argument("--weights", type=str, default="", help="Path to trained weights (TODO)")
    args = parser.parse_args()

    # choose DrivingFramesDataset if sequence-based policy evaluation is needed
    dataset = DrivingFramesDataset(csv_path=Path(args.data_dir) / "driving_log.csv", seq_len=args.seq_len, augment=False, resize=tuple(args.resize))
    pin_memory = True if (str(CONFIG.DEVICE) != "cpu") else False
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin_memory)
    model = ConvLSTMPolicy()
    if args.weights:
        state = torch.load(args.weights, map_location="cpu")
        model.load_state_dict(state)
    metrics = evaluate(model, loader)
    logger.info("Evaluation metrics: %s", metrics)


if __name__ == "__main__":
    main()

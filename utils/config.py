"""Configuration helpers and lightweight AppConfig dataclass.

This module defines a small `AppConfig` dataclass and utilities to load
configuration from YAML files or environment variables. For research projects
we recommend adding validation (pydantic) and secure defaults.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple
from pathlib import Path
import os
import yaml


@dataclass
class Config:
    """Minimal configuration container.

    Add fields as your project grows (dataset paths, training hyperparameters,
    model names, etc.).
    """
    # Paths
    RAW_VIDEO_DIR: str = "data/raw_videos"
    FRAMES_DIR: str = "data/frames"
    CURATED_DIR: str = "data/curated"
    CHECKPOINT_DIR: str = "models"
    LOGS_DIR: str = "logs"

    # Training hyperparameters
    BATCH_SIZE: int = 8
    LR: float = 1e-3
    EPOCHS: int = 10

    # Video / frame defaults
    FPS: int = 30
    FRAME_WIDTH: int = 256
    FRAME_HEIGHT: int = 256
    FRAME_SIZE: Tuple[int, int] = (256, 256)
    # Policy training defaults (smaller for faster experimentation)
    POLICY_FRAME_HEIGHT: int = 96
    POLICY_FRAME_WIDTH: int = 192
    POLICY_FRAME_SIZE: Tuple[int, int] = (96, 192)
    NUM_WORKERS: int = 2

    # Runtime
    DEVICE: str = "cpu"


# module-level config instance: use `from utils.config import CONFIG`
CONFIG = Config(
    RAW_VIDEO_DIR=os.environ.get("RAW_VIDEO_DIR", "data/raw_videos"),
    FRAMES_DIR=os.environ.get("FRAMES_DIR", "data/frames"),
    CURATED_DIR=os.environ.get("CURATED_DIR", "data/curated"),
    CHECKPOINT_DIR=os.environ.get("CHECKPOINT_DIR", "models"),
    LOGS_DIR=os.environ.get("LOGS_DIR", "logs"),
    BATCH_SIZE=int(os.environ.get("BATCH_SIZE", 8)),
    LR=float(os.environ.get("LR", 1e-3)),
    EPOCHS=int(os.environ.get("EPOCHS", 10)),
    FPS=int(os.environ.get("FPS", 30)),
    FRAME_WIDTH=int(os.environ.get("FRAME_WIDTH", 256)),
    FRAME_HEIGHT=int(os.environ.get("FRAME_HEIGHT", 256)),
    FRAME_SIZE=(int(os.environ.get("FRAME_HEIGHT", 256)), int(os.environ.get("FRAME_WIDTH", 256))),
    DEVICE=os.environ.get("DEVICE", "cpu"),
    POLICY_FRAME_HEIGHT=int(os.environ.get("POLICY_FRAME_HEIGHT", 96)),
    POLICY_FRAME_WIDTH=int(os.environ.get("POLICY_FRAME_WIDTH", 192)),
    POLICY_FRAME_SIZE=(int(os.environ.get("POLICY_FRAME_HEIGHT", 96)), int(os.environ.get("POLICY_FRAME_WIDTH", 192))),
    NUM_WORKERS=int(os.environ.get("NUM_WORKERS", 2)),
)

# Project root and dataset/checkpoint constants
ROOT_DIR = Path(__file__).resolve().parents[1]
STEERING_DATASET_DIR = ROOT_DIR / "data" / "steering_dataset"
POLICY_CHECKPOINT_DIR = ROOT_DIR / "checkpoints" / "policy"
POLICY_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# Dynamic device detection: prefer MPS on Apple Silicon, then CUDA, else CPU
try:
    import torch as _torch  # local alias
    if _torch.backends.mps.is_available():
        CONFIG.DEVICE = "mps"
    elif _torch.cuda.is_available():
        CONFIG.DEVICE = "cuda"
    else:
        CONFIG.DEVICE = "cpu"
except Exception:
    # torch not available; keep default from environment or 'cpu'
    pass


def load_config(path: str | Path) -> Config:
    """Load YAML config and return an `AppConfig` instance.

    TODO: switch to `pydantic` or add schema validation for large projects.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    # Merge with defaults: allow missing keys so only provided values override
    cfg_kwargs: dict[str, Any] = {k: v for k, v in raw.items() if v is not None}
    cfg = Config(**{**CONFIG.__dict__, **cfg_kwargs})
    return cfg


if __name__ == "__main__":
    # trivial smoke test: creates a default config if no file provided
    import sys

    if len(sys.argv) > 1:
        cfg = load_config(sys.argv[1])
        print(cfg)
    else:
        print(CONFIG)

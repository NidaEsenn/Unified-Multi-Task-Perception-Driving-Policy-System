"""Configuration helpers and lightweight AppConfig dataclass.

This module defines a small `AppConfig` dataclass and utilities to load
configuration from YAML files or environment variables. For research projects
we recommend adding validation (pydantic) and secure defaults.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from pathlib import Path
import yaml


@dataclass
class AppConfig:
    """Minimal configuration container.

    Add fields as your project grows (dataset paths, training hyperparameters,
    model names, etc.).
    """
    data_root: str = "data"
    train_dir: str = "data/train"
    val_dir: str = "data/val"
    model_dir: str = "models"
    device: str = "cpu"


def load_config(path: str | Path) -> AppConfig:
    """Load YAML config and return an `AppConfig` instance.

    TODO: switch to `pydantic` or add schema validation for large projects.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return AppConfig(**(raw or {}))


if __name__ == "__main__":
    # trivial smoke test: creates a default config if no file provided
    import sys

    if len(sys.argv) > 1:
        cfg = load_config(sys.argv[1])
        print(cfg)
    else:
        print(AppConfig())

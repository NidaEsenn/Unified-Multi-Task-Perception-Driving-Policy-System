"""Dataset utilities used by training and evaluation scripts.

Contains a lightweight `FrameDataset` to load frames and optional targets.
Customize transforms, annotation formats, and batching to fit your pipeline.
"""
from __future__ import annotations

from typing import Optional, Callable, Dict, Any
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset


class FrameDataset(Dataset):
    """Dataset that loads frames from a directory.

    Expected directory layout: `root_dir/*.jpg` or `root_dir/*/*.jpg`.
    TODO: support annotation files (JSON/CSV) and segmentation masks.
    """

    def __init__(self, root_dir: str | Path, transform: Optional[Callable] = None) -> None:
        self.root_dir = Path(root_dir)
        self.paths = sorted([p for p in self.root_dir.rglob("*.jpg")])
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        else:
            # convert to tensor [C,H,W] float32 [0,1]
            img = torch.from_numpy((np.array(img).astype("float32") / 255.0).transpose(2, 0, 1))
        # TODO: load targets/annotations and return them as `targets`
        return {"frames": img, "path": str(path)}


if __name__ == "__main__":
    # quick smoke test
    import sys, numpy as np

    if len(sys.argv) < 2:
        print("Usage: python -m utils.datasets <root_dir>")
    else:
        ds = FrameDataset(sys.argv[1])
        print("Found", len(ds), "images")

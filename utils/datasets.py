"""Dataset utilities used by training and evaluation scripts.

Provides `DrivingFramesDataset` which loads frames and optional labels and
supports both single-frame usage (perception) and sequence mode (policy).
Customizable and lightweight augmentations are included for data variability.
"""
from __future__ import annotations

from typing import Optional, Callable, Dict, Any, List, Tuple, Union
from pathlib import Path
import csv
import os
import random
import math

import numpy as np
import cv2
try:
    import torch
    from torch.utils.data import Dataset
except Exception:
    # allow lightweight imports in environments without torch; tests will skip heavy checks
    torch = None
    Dataset = object  # type: ignore

from utils.config import CONFIG, STEERING_DATASET_DIR


def _read_labels_csv(csv_path: str | Path) -> Dict[str, float]:
    labels: Dict[str, float] = {}
    with open(str(csv_path), "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        # Expect fields: filename, steering
        for row in reader:
            filename = row.get("filename") or row.get("file") or row.get("frame")
            if not filename:
                continue
            try:
                steering = float(row.get("steering", 0.0))
            except Exception:
                steering = 0.0
            labels[filename] = steering
    return labels


def _random_brightness(img: np.ndarray, low: float = 0.7, high: float = 1.3) -> np.ndarray:
    factor = random.uniform(low, high)
    img = img.astype("float32") * factor
    img = np.clip(img, 0, 255).astype("uint8")
    return img


def _random_horizontal_shift(img: np.ndarray, max_shift: int = 16) -> np.ndarray:
    shift = random.randint(-max_shift, max_shift)
    if shift == 0:
        return img
    h, w = img.shape[:2]
    M = np.float32([[1, 0, shift], [0, 1, 0]])
    shifted = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    return shifted


def _random_horizontal_flip(img: np.ndarray) -> np.ndarray:
    return cv2.flip(img, 1)


class DrivingFramesDataset(Dataset):
    """Dataset for driving frames with optional steering labels.

    Supports sequence mode for policy learning (returns series of frames + label)
    and single-frame mode for perception tasks.

    Args:
        root_dir: folder with collected frames (images) — expected to contain
            image files, e.g. JPEG/PNG.
        labels_csv: optional CSV file mapping filenames to numeric labels (steering).
        sequence_length: how many frames to include in a single example when
            in sequence mode (policy). sequence_length=1 behaves like single-frame mode.
        transform: optional callable applied to each frame (after augmentations).
        augment: whether to apply random augmentations (brightness, shift, flip).
        step: frame step between items of sequence; for step>1 sequences skip frames between each element.
    """

    def __init__(
        self,
        root_dir: Optional[Union[str, Path]] = None,
        labels_csv: Optional[Union[str, Path]] = None,
        sequence_length: int = 1,
        transform: Optional[Callable[[np.ndarray], torch.Tensor]] = None,
        augment: bool = True,
        step: int = 1,
    ) -> None:
        # default to STEERING_DATASET_DIR provided by CONFIG
        self.root_dir = Path(root_dir) if root_dir else Path(STEERING_DATASET_DIR)
        # default CSV if not provided
        self.labels_csv = Path(labels_csv) if labels_csv else Path(STEERING_DATASET_DIR) / "driving_log.csv"
        # Read labels and center filenames from CSV if present
        self.labels: Dict[str, float] = {}
        self.paths: List[str] = []
        if self.labels_csv.exists():
            # Parse CSV to extract center image filenames and steering values
            with open(self.labels_csv, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    center_path = row.get("center") or row.get("center_image") or row.get("image")
                    if not center_path:
                        # fallback: try filename column
                        center_path = row.get("filename") or row.get("file")
                    if not center_path:
                        continue
                    basename = os.path.basename(center_path)
                    steering_val = 0.0
                    try:
                        steering_val = float(row.get("steering", 0.0))
                    except Exception:
                        steering_val = 0.0
                    self.labels[basename] = steering_val
                    img_path = self.root_dir / "IMG" / basename
                    if img_path.exists():
                        self.paths.append(str(img_path))
        else:
            # fallback to scanning IMG folder
            img_dir = self.root_dir / "IMG"
            if img_dir.exists():
                self.paths = sorted([str(p) for p in img_dir.rglob("*.jpg")])
            else:
                self.paths = sorted([str(p) for p in self.root_dir.rglob("*.jpg")])
        self.sequence_length = int(sequence_length)
        self.step = int(step)
        self.transform = transform
        self.augment = bool(augment)
        # labels already loaded if CSV present

        # number of valid starting indices for sequences
        if self.sequence_length <= 1:
            self._length = len(self.paths)
        else:
            self._length = max(0, len(self.paths) - (self.sequence_length - 1) * self.step)

    def __len__(self) -> int:
        return self._length

    def _load_frame(self, path: str) -> np.ndarray:
        # load as BGR OpenCV then convert to RGB for consistency
        img = cv2.imread(path)
        if img is None:
            raise RuntimeError(f"Failed to read image: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def _apply_augmentations(self, frames: List[np.ndarray]) -> Tuple[List[np.ndarray], bool, int]:
        """Apply consistent augmentations across the sequence of frames.

        Returns: frames, flipped_flag, brightness_factor (approx as int) or shift
        """
        flipped = False
        # decide augmentations
        if not self.augment:
            return frames, flipped, 0
        if random.random() < 0.5:
            # brightness
            factor = random.uniform(0.8, 1.2)
            frames = [np.clip(frame.astype("float32") * factor, 0, 255).astype("uint8") for frame in frames]
        if random.random() < 0.5:
            frames = [ _random_horizontal_shift(frame, max_shift=16) for frame in frames ]
        if random.random() < 0.5:
            frames = [ _random_horizontal_flip(frame) for frame in frames ]
            flipped = True
        return frames, flipped, 0

    def _to_tensor(self, frames: List[np.ndarray]) -> torch.Tensor:
        # frames: List[H,W,C] uint8
        # convert to numpy floats and to Tensor shape (T,C,H,W)
        arr = np.stack(frames, axis=0)  # (T,H,W,C)
        arr = arr.astype("float32") / 255.0
        arr = arr.transpose(0, 3, 1, 2)  # (T,C,H,W)
        return torch.from_numpy(arr)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self.sequence_length <= 1:
            path = self.paths[idx]
            img = self._load_frame(path)
            frames = [img]
            frames, flipped, _ = self._apply_augmentations(frames)
            if self.transform:
                frame_t = self.transform(frames[0])
            else:
                frame_t = torch.from_numpy(frames[0].astype("float32").transpose(2, 0, 1) / 255.0)
            out: Dict[str, Any] = {"frames": frame_t, "path": path}
            # add label if available
            filename = Path(path).name
            if filename in self.labels:
                out["target"] = float(self.labels[filename])
            return out
        # sequence mode
        start = idx
        indices = [start + i * self.step for i in range(self.sequence_length)]
        frames = [self._load_frame(self.paths[i]) for i in indices]
        frames, flipped, _ = self._apply_augmentations(frames)
        if self.transform:
            frames_t = [self.transform(f) for f in frames]
            # assume transform returns torch.Tensor (C,H,W)
            if torch is None:
                # If torch missing return list or numpy stack for light testing
                frames_t = np.stack([t.numpy() if hasattr(t, 'numpy') else np.asarray(t) for t in frames_t], axis=0)
            else:
                frames_t = torch.stack(frames_t, dim=0)  # (T,C,H,W)
        else:
            if torch is None:
                frames_t = np.stack(frames, axis=0).astype("float32") / 255.0
                frames_t = frames_t.transpose(0, 3, 1, 2)
            else:
                frames_t = self._to_tensor(frames)
        # label: choose label of last frame in sequence if available
        last_filename = Path(self.paths[indices[-1]]).name
        label = float(self.labels.get(last_filename, 0.0))
        if flipped:
            # flip steering sign
            label = -label
        return {"frames": frames_t, "target": torch.tensor(label, dtype=torch.float32), "path": self.paths[indices[-1]]}


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m utils.datasets <frames_dir> [labels.csv]")
    else:
        root = sys.argv[1]
        csv_path = sys.argv[2] if len(sys.argv) > 2 else None
        ds = DrivingFramesDataset(root, labels_csv=csv_path, sequence_length=4, augment=True)
        print("Dataset size:", len(ds))
        item = ds[0]
        print("Sample keys:", list(item.keys()))

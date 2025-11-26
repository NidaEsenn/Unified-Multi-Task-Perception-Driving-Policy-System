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
    """Udacity-styled DrivingFramesDataset loader.

    - Expects `driving_log.csv` with columns: center,left,right,steering,...
    - Default CSV: `STEERING_DATASET_DIR / 'driving_log.csv'` and images in
      `STEERING_DATASET_DIR / 'IMG'`.
    - Uses only the `center` column. Stores samples as (img_path, steering).
    - Supports sequence mode with `seq_len`; sequences are composed of T consecutive
      frames ending at `idx` (clamped to start at 0), and the label is the steering
      of the last frame in the sequence.
    - Augmentations: random brightness and random horizontal flip; when flipped,
      steering is inverted.

    Args:
        csv_path: explicit path to a driving_log CSV; if None uses
            `STEERING_DATASET_DIR / 'driving_log.csv'`.
        seq_len: number of frames in a sequence (T). seq_len=1 returns a single frame and label.
        augment: whether to apply augmentations.
        transform: optional callable (PIL/np->tensor) applied to each frame after augment.
    """
    def __init__(
        self,
        csv_path: Optional[Union[str, Path]] = None,
        seq_len: int = 1,
        augment: bool = True,
        transform: Optional[Callable[[np.ndarray], torch.Tensor]] = None,
        step: int = 1,
        resize: Optional[Tuple[int, int]] = None,
    ) -> None:
        # Setup default CSV and image directory paths
        self.csv_path = Path(csv_path) if csv_path else Path(STEERING_DATASET_DIR) / "driving_log.csv"
        self.img_dir = Path(STEERING_DATASET_DIR) / "IMG"
        self.seq_len = int(seq_len)
        self.step = int(step)
        self.augment = bool(augment)
        self.transform = transform
        # resize: (H, W) to which frames will be resized using cv2
        from utils.config import CONFIG as _CONFIG
        default_size = tuple(getattr(_CONFIG, "POLICY_FRAME_SIZE", (96, 192)))
        self.resize = tuple(resize) if resize else default_size

        # samples holds (Path, steering) pairs
        self.samples: List[Tuple[Path, float]] = []
        if self.csv_path.exists():
            with open(self.csv_path, "r", encoding="utf-8") as f:
                # CSV may or may not include header; using csv.reader is robust
                reader = csv.reader(f)
                for row in reader:
                    if not row:
                        continue
                    center_path = row[0]
                    if not center_path:
                        continue
                    center_path_norm = str(center_path).replace("\\", "/").strip()
                    basename = os.path.basename(center_path_norm)
                    img_path = self.img_dir / basename
                    if not img_path.exists():
                        # skip missing files
                        continue
                    try:
                        steering_val = float(row[3]) if len(row) > 3 else 0.0
                    except Exception:
                        steering_val = 0.0
                    self.samples.append((img_path, steering_val))
        else:
            # If CSV doesn't exist, scan IMG folder and set default steering = 0.0
                if self.img_dir.exists():
                    for p in sorted(self.img_dir.rglob("*.jpg")):
                        self.samples.append((p, 0.0))
        # padding conventions
        self.sequence_length = int(self.seq_len)

        # number of valid starting indices for sequences
        # dataset length equals number of sample entries
        self._length = len(self.samples)

    def __len__(self) -> int:
        return self._length

    def _load_frame(self, path: str) -> np.ndarray:
        # load as BGR OpenCV then convert to RGB for consistency
        img = cv2.imread(path)
        if img is None:
            raise RuntimeError(f"Failed to read image: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # optionally resize for faster training
        if self.resize:
            h, w = self.resize
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
        return img

    def _apply_augmentations(self, frames: List[np.ndarray]) -> Tuple[List[np.ndarray], bool]:
        """Apply consistent augmentations across the sequence of frames.

        Returns: (frames, flipped_flag)
        """
        flipped = False
        # decide augmentations
        if not self.augment:
            return frames, flipped
        # reduce augmentation rates for fast smoke-test
        if random.random() < 0.3:
            # brightness
            factor = random.uniform(0.8, 1.2)
            frames = [np.clip(frame.astype("float32") * factor, 0, 255).astype("uint8") for frame in frames]
        if random.random() < 0.3:
            frames = [ _random_horizontal_shift(frame, max_shift=16) for frame in frames ]
        if random.random() < 0.25:
            frames = [ _random_horizontal_flip(frame) for frame in frames ]
            flipped = True
        return frames, flipped

    def _to_tensor(self, frames: List[np.ndarray]) -> torch.Tensor:
        # frames: List[H,W,C] uint8
        # convert to numpy floats and to Tensor shape (T,C,H,W)
        arr = np.stack(frames, axis=0)  # (T,H,W,C)
        arr = arr.astype("float32") / 255.0
        arr = arr.transpose(0, 3, 1, 2)  # (T,C,H,W)
        return torch.from_numpy(arr)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # clamp idx
        idx_clamped = max(0, min(idx, self._length - 1))
        if self.seq_len <= 1:
            img_path, steering = self.samples[idx_clamped]
            img = self._load_frame(str(img_path))
            frames = [img]
            frames, flipped = self._apply_augmentations(frames)
            if self.transform:
                frame_t = self.transform(frames[0])
            else:
                frame_t = torch.from_numpy(frames[0].astype("float32").transpose(2, 0, 1) / 255.0)
            if flipped:
                steering = -steering
            return {"frames": frame_t, "target": torch.tensor(steering, dtype=torch.float32), "path": str(img_path)}
        # sequence mode
        # sequence mode: build indices as a window ending at idx
        end = idx_clamped
        start = max(0, end - self.seq_len + 1)
        indices = list(range(start, end + 1))
        # if not enough frames at the start, pad by repeating the first frame in the sequence
        if len(indices) < self.seq_len:
            pad_needed = self.seq_len - len(indices)
            indices = [indices[0]] * pad_needed + indices
        frames = [self._load_frame(str(self.samples[i][0])) for i in indices]
        frames, flipped = self._apply_augmentations(frames)
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
        label = float(self.samples[indices[-1]][1])
        if flipped:
            # flip steering sign
            label = -label
        return {"frames": frames_t, "target": torch.tensor(label, dtype=torch.float32), "path": str(self.samples[indices[-1]][0])}


class FrameDataset(Dataset):
    """Compatibility wrapper: single-frame dataset loader for perception tasks.

    Returns dict with 'frames' (C,H,W tensor) and 'path'. If labels are present
    in the steering CSV, returns 'target'.
    """
    def __init__(self, root_dir: str | Path, transform: Optional[Callable] = None) -> None:
        self.root_dir = Path(root_dir)
        img_dir = self.root_dir / "IMG" if (self.root_dir / "IMG").exists() else self.root_dir
        self.paths = sorted([str(p) for p in img_dir.rglob("*.jpg")])
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        path = self.paths[idx]
        img = cv2.imread(path)
        if img is None:
            raise RuntimeError(f"Failed to read image: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            return {"frames": self.transform(img), "path": path}
        # default to tensor [C,H,W]
        if torch is None:
            arr = (img.astype("float32") / 255.0).transpose(2, 0, 1)
            return {"frames": arr, "path": path}
        else:
            tensor = torch.from_numpy(img.astype("float32").transpose(2, 0, 1) / 255.0)
            return {"frames": tensor, "path": path}


if __name__ == "__main__":
    import sys
    import argparse
    parser = argparse.ArgumentParser(description="Sanity check for DrivingFramesDataset")
    parser.add_argument("--csv", type=Path, default=Path(STEERING_DATASET_DIR) / "driving_log.csv")
    parser.add_argument("--seq-len", type=int, default=4)
    args = parser.parse_args(sys.argv[1:])
    print("CSV:", args.csv)
    ds = DrivingFramesDataset(csv_path=args.csv, seq_len=args.seq_len, augment=False)
    print("Dataset size:", len(ds))
    if len(ds) > 0:
        item = ds[min(0, len(ds) - 1)]
        print("Sample path:", item["path"]) 
        frames = item["frames"]
        print("Frames type:", type(frames))
        try:
            import torch
            if isinstance(frames, torch.Tensor):
                print("Frames shape:", frames.shape)
        except Exception:
            pass
        print("Sample steering:", float(item["target"]))

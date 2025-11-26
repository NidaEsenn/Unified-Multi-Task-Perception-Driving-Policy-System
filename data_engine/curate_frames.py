"""Utilities to extract and curate frames from video sources.

Includes helpers to extract frames, sample them, and save to disk for
downstream annotation or training pipelines.
"""
from __future__ import annotations

from typing import Iterable, List
import argparse
import os
from pathlib import Path

import cv2
import numpy as np


def extract_frames(video_path: str, out_dir: str, stride: int = 30) -> List[str]:
    """Extract frames from `video_path` every `stride` frames and save JPEGs.

    Args:
        video_path: Path to video file.
        out_dir: Directory to save extracted frames.
        stride: Save one frame every `stride` frames. TODO tune.

    Returns:
        List of saved filenames.
    """
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    saved: List[str] = []
    idx = 0
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % stride == 0:
            fname = Path(out_dir) / f"frame_{idx:06d}.jpg"
            cv2.imwrite(str(fname), frame)
            saved.append(str(fname))
            idx += 1
        frame_idx += 1
    cap.release()
    return saved


def sample_frames(frames: Iterable[str], max_samples: int = 1000) -> List[str]:
    """Sample up to `max_samples` frames evenly from `frames` list.

    Useful for building a balanced initial dataset.
    """
    frames = list(frames)
    n = len(frames)
    if n <= max_samples:
        return frames
    idxs = np.linspace(0, n - 1, max_samples, dtype=int)
    return [frames[i] for i in idxs]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True, help="Path to video file")
    parser.add_argument("--out", type=str, default="data/frames", help="Output frames dir (TODO customize)")
    parser.add_argument("--stride", type=int, default=30, help="Frame stride (TODO tune)")
    args = parser.parse_args()
    saved = extract_frames(args.video, args.out, args.stride)
    print(f"Saved {len(saved)} frames to {args.out}")


if __name__ == "__main__":
    main()

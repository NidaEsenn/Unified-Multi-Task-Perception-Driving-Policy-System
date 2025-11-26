#!/usr/bin/env python3
"""Create the steering dataset directories and provide download instructions.

This script ensures the dataset directories exist for steering/behavioral
cloning tasks and writes a `download_notes.txt` file in the dataset folder
containing clear instructions for obtaining the Udacity Self-Driving Car
Simulator behavioral cloning dataset (driving_log.csv + IMG folder).

Usage:
    python scripts/setup_steering_dataset.py

Options:
    --base-dir  -- base output directory for the steering dataset
    --raw-videos -- separate location for original/raw videos
    --overwrite  -- overwrite existing download_notes.txt if present

This is a pure helper script — it does NOT download any dataset itself it
helps you prepare a local layout and tells you where to place files.
"""
from __future__ import annotations

from pathlib import Path
import argparse
import textwrap
import datetime
import sys
from typing import Optional


DOWNLOAD_LINK = "https://github.com/udacity/self-driving-car-sim"
DEFAULT_BASE_DIR = Path("data/steering_dataset")
DEFAULT_RAW_VIDEOS = Path("data/raw_videos")


def render_notes(base_dir: Path, raw_videos_dir: Path) -> str:
    """Return the content for download_notes.txt.

    Provide essential information to the user on how to obtain, place, and
    prepare the dataset for use with the ConvLSTM/Policy training.
    """
    return textwrap.dedent(
        f"""
        Udacity Self-Driving Car Simulator Behavioral Cloning Dataset
        ==========================================================

        Dataset link: {DOWNLOAD_LINK}

        Expected folder structure for this project (place these under `{base_dir}`):

        {base_dir}/
          driving_log.csv
          IMG/
            center_2016_01_01_13_00_00_000.jpg
            center_2016_01_01_13_00_00_001.jpg
            ...

        What to download / how to obtain the dataset:
        - The repository above contains the Udacity simulator and the behavioral cloning dataset
          (sample driving logs + IMG/ folder). Please download or clone the dataset files
          and place the `driving_log.csv` and `IMG/` folder in the `{base_dir}` directory.

        Using extracted video data:
        - If you have raw simulator videos, place them under: `{raw_videos_dir}`.
        - Use the `data_engine/curate_frames.py` script (or your own extractor) to
          sample frames and save into `{base_dir}` as images.

        How to use this dataset in the project:
        - The ConvLSTM driving policy training expects a curated set of frames
          (images) and optionally a `driving_log.csv` that maps images to steering
          angle targets. If you keep the dataset layout as above you can run
          `scripts/curate_frames.py` (not present by default) or a simple loader to
          convert the CSV into a dataset used by `DrivingFramesDataset` (utils/datasets.py).
        - Make sure you point `utils/config.py` `CONFIG.FRAMES_DIR` or `CONFIG.CURATED_DIR`
          to this `{base_dir}` folder so scripts pick it up by default.

        Reminder:
        - The behavioral cloning dataset is required for ConvLSTM driving policy training.
        - This script does not host or download the dataset — you will need to obtain it
          manually and place files according to the expected structure.

        Generated: {datetime.datetime.utcnow().isoformat()} UTC
        """
    )


def create_dirs(base_dir: Path, raw_videos_dir: Path) -> None:
    """Create the dataset and raw video directories if they don't exist."""
    base_dir.mkdir(parents=True, exist_ok=True)
    raw_videos_dir.mkdir(parents=True, exist_ok=True)


def write_download_notes(base_dir: Path, raw_videos_dir: Path, overwrite: bool = False) -> Path:
    """Write the download notes file into the dataset folder with clear instructions.

    Returns the path to the notes file.
    """
    notes_path = base_dir / "download_notes.txt"
    if notes_path.exists() and not overwrite:
        return notes_path
    content = render_notes(base_dir, raw_videos_dir)
    notes_path.write_text(content, encoding="utf-8")
    return notes_path


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare steering dataset folders and notes")
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE_DIR, help="Folder for steering dataset")
    parser.add_argument("--raw-videos", type=Path, default=DEFAULT_RAW_VIDEOS, help="Folder to store raw videos")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing download_notes.txt if present")
    args = parser.parse_args(argv or sys.argv[1:])

    base = args.base_dir
    raw = args.raw_videos
    create_dirs(base, raw)
    notes = write_download_notes(base, raw, overwrite=args.overwrite)

    print("\nDataset setup complete:")
    print(f" - Steering dataset folder: {base.resolve()}")
    print(f" - Raw videos folder: {raw.resolve()}")
    print(f" - Download notes written to: {notes.resolve()}")
    print("")
    print("Next steps:")
    print(" 1) Download or prepare Udacity simulator dataset and place `driving_log.csv` and `IMG/` under the steering dataset folder.")
    print(" 2) If starting from videos, place them under raw videos folder and use a frame extraction tool to generate images.")
    print(" 3) Ensure utils/config.py references this dataset via CONFIG.FRAMES_DIR or CONFIG.CURATED_DIR.")
    print(" 4) Use the `DrivingFramesDataset` to load frames and steering labels for training the ConvLSTM policy.")
    print("")
    print("For further guidance, see the README and the data_engine/curate_frames.py script.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

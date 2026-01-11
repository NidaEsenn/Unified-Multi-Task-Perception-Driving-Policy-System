"""Script to prepare and organize test dataset for benchmarking.

This script helps prepare synthetic test data when real test data is not available.
It creates sample images, ground truth annotations, and steering labels for testing
the complete pipeline.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple
import json
import numpy as np
import cv2
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_synthetic_image(height: int = 480, width: int = 640) -> np.ndarray:
    """Create a synthetic test image with basic road-like features.
    
    Args:
        height: Image height
        width: Image width
        
    Returns:
        BGR image array
    """
    # Create base image with road-like gradient
    img = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Gray road
    img[:, :] = [100, 100, 100]
    
    # Add some noise for texture
    noise = np.random.randint(-20, 20, (height, width, 3), dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # Add lane markers (white lines)
    # Left lane
    cv2.line(img, (width // 4, height), (width // 3, height // 2), (255, 255, 255), 3)
    # Right lane
    cv2.line(img, (3 * width // 4, height), (2 * width // 3, height // 2), (255, 255, 255), 3)
    
    # Add some random vehicle-like rectangles
    num_vehicles = np.random.randint(0, 4)
    for _ in range(num_vehicles):
        x = np.random.randint(width // 4, 3 * width // 4)
        y = np.random.randint(height // 3, 2 * height // 3)
        w = np.random.randint(40, 80)
        h = np.random.randint(30, 60)
        color = tuple(np.random.randint(0, 255, 3).tolist())
        cv2.rectangle(img, (x, y), (x + w, y + h), color, -1)
    
    return img


def create_synthetic_lane_mask(height: int = 480, width: int = 640) -> np.ndarray:
    """Create a synthetic lane segmentation mask.
    
    Args:
        height: Mask height
        width: Mask width
        
    Returns:
        Binary mask array (0 or 1)
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # Draw lane regions
    # Left lane
    pts_left = np.array([
        [width // 4, height],
        [width // 3, height // 2],
        [width // 3 + 10, height // 2],
        [width // 4 + 10, height]
    ], np.int32)
    cv2.fillPoly(mask, [pts_left], 1)
    
    # Right lane
    pts_right = np.array([
        [3 * width // 4, height],
        [2 * width // 3, height // 2],
        [2 * width // 3 + 10, height // 2],
        [3 * width // 4 + 10, height]
    ], np.int32)
    cv2.fillPoly(mask, [pts_right], 1)
    
    return mask


def create_synthetic_drivable_mask(height: int = 480, width: int = 640) -> np.ndarray:
    """Create a synthetic drivable area mask.
    
    Args:
        height: Mask height
        width: Mask width
        
    Returns:
        Binary mask array (0 or 1)
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # Drivable area is roughly the lower 2/3 of the image, tapered
    pts = np.array([
        [0, height],
        [width, height],
        [2 * width // 3, height // 2],
        [width // 3, height // 2]
    ], np.int32)
    cv2.fillPoly(mask, [pts], 1)
    
    return mask


def create_synthetic_detections(width: int = 640, height: int = 480) -> List[Dict[str, Any]]:
    """Create synthetic ground truth detections.
    
    Args:
        width: Image width
        height: Image height
        
    Returns:
        List of detection dictionaries
    """
    detections = []
    num_vehicles = np.random.randint(0, 4)
    
    for _ in range(num_vehicles):
        x = np.random.randint(width // 4, 3 * width // 4)
        y = np.random.randint(height // 3, 2 * height // 3)
        w = np.random.randint(40, 80)
        h = np.random.randint(30, 60)
        
        detections.append({
            'bbox': [x, y, x + w, y + h],
            'class_id': np.random.choice([2, 5, 7]),  # car, bus, truck
            'class_name': np.random.choice(['car', 'bus', 'truck'])
        })
    
    return detections


def create_test_dataset(
    output_dir: Path,
    num_samples: int = 100,
    img_height: int = 480,
    img_width: int = 640
) -> None:
    """Create a synthetic test dataset.
    
    Args:
        output_dir: Directory to save the test dataset
        num_samples: Number of test samples to create
        img_height: Image height
        img_width: Image width
    """
    logger.info(f"Creating synthetic test dataset with {num_samples} samples")
    
    # Create directory structure
    images_dir = output_dir / "images"
    lane_masks_dir = output_dir / "lane_masks"
    drivable_masks_dir = output_dir / "drivable_masks"
    annotations_dir = output_dir / "annotations"
    
    for d in [images_dir, lane_masks_dir, drivable_masks_dir, annotations_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    # Create synthetic samples
    metadata = []
    
    for i in tqdm(range(num_samples), desc="Generating samples"):
        # Create synthetic data
        img = create_synthetic_image(img_height, img_width)
        lane_mask = create_synthetic_lane_mask(img_height, img_width)
        drivable_mask = create_synthetic_drivable_mask(img_height, img_width)
        detections = create_synthetic_detections(img_width, img_height)
        steering_angle = np.random.uniform(-1.0, 1.0)  # Normalized steering
        
        # Save files
        img_path = images_dir / f"frame_{i:06d}.png"
        lane_path = lane_masks_dir / f"lane_{i:06d}.png"
        drivable_path = drivable_masks_dir / f"drivable_{i:06d}.png"
        ann_path = annotations_dir / f"ann_{i:06d}.json"
        
        cv2.imwrite(str(img_path), img)
        cv2.imwrite(str(lane_path), lane_mask * 255)
        cv2.imwrite(str(drivable_path), drivable_mask * 255)
        
        # Save annotations
        annotation = {
            'image_path': str(img_path.relative_to(output_dir)),
            'lane_mask_path': str(lane_path.relative_to(output_dir)),
            'drivable_mask_path': str(drivable_path.relative_to(output_dir)),
            'detections': detections,
            'steering_angle': float(steering_angle),
            'image_width': img_width,
            'image_height': img_height
        }
        
        with open(ann_path, 'w') as f:
            json.dump(annotation, f, indent=2)
        
        metadata.append({
            'frame_id': i,
            'annotation_path': str(ann_path.relative_to(output_dir))
        })
    
    # Save dataset metadata
    metadata_path = output_dir / "dataset_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump({
            'num_samples': num_samples,
            'image_height': img_height,
            'image_width': img_width,
            'metadata': metadata
        }, f, indent=2)
    
    logger.info(f"Test dataset created at: {output_dir}")
    logger.info(f"Dataset metadata saved to: {metadata_path}")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Prepare test dataset for benchmarking")
    parser.add_argument("--output-dir", type=str, default="data/test_dataset",
                       help="Output directory for test dataset")
    parser.add_argument("--num-samples", type=int, default=100,
                       help="Number of test samples to create")
    parser.add_argument("--height", type=int, default=480,
                       help="Image height")
    parser.add_argument("--width", type=int, default=640,
                       help="Image width")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    create_test_dataset(output_dir, args.num_samples, args.height, args.width)
    
    logger.info("Test data collection complete!")


if __name__ == "__main__":
    main()

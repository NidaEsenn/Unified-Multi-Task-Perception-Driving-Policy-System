"""Main comprehensive benchmarking script for the entire pipeline.

This script runs complete benchmarks across all system components:
- Object detection (YOLOv8)
- Segmentation (Lane + Drivable Area U-Net)
- Driving policy (ConvLSTM)
- System-level performance

It generates comprehensive metrics, visualizations, and auto-generates documentation.
"""
from __future__ import annotations

import argparse
import json
import logging
import platform
import subprocess
import time
from pathlib import Path
from typing import Dict, Any
import sys

import psutil
import torch
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.utils.report_generator import generate_performance_metrics_report
from scripts.utils.visualization import (
    plot_metrics_comparison,
    plot_latency_breakdown,
    plot_fps_vs_resolution
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_hardware_info() -> Dict[str, str]:
    """Detect and return hardware configuration.
    
    Returns:
        Dictionary with hardware information
    """
    info = {}
    
    # CPU
    try:
        import cpuinfo
        cpu_info = cpuinfo.get_cpu_info()
        info['CPU'] = cpu_info.get('brand_raw', 'Unknown')
    except Exception:
        info['CPU'] = platform.processor() or 'Unknown'
    
    # GPU
    if torch.cuda.is_available():
        info['GPU'] = torch.cuda.get_device_name(0)
        info['CUDA Version'] = torch.version.cuda or 'Unknown'
    else:
        info['GPU'] = 'CPU only'
        info['CUDA Version'] = 'N/A'
    
    # RAM
    mem = psutil.virtual_memory()
    info['RAM'] = f"{mem.total / (1024**3):.1f} GB"
    
    # PyTorch version
    info['PyTorch Version'] = torch.__version__
    
    # OS
    info['OS'] = f"{platform.system()} {platform.release()}"
    
    return info


def run_detection_benchmark(
    test_data_dir: str,
    weights: str,
    device: str,
    output_file: str
) -> Dict[str, Any]:
    """Run detection benchmark script.
    
    Args:
        test_data_dir: Path to test dataset
        weights: Path to YOLOv8 weights
        device: Device to use
        output_file: Output JSON file path
        
    Returns:
        Detection metrics dictionary
    """
    logger.info("Running detection benchmark...")
    
    cmd = [
        sys.executable,
        "scripts/benchmark_detection.py",
        "--test-data", test_data_dir,
        "--weights", weights,
        "--device", device,
        "--output", output_file
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            logger.warning(f"Detection benchmark failed: {result.stderr}")
        
        # Load results
        if Path(output_file).exists():
            with open(output_file, 'r') as f:
                return json.load(f)
    except Exception as e:
        logger.error(f"Error running detection benchmark: {e}")
    
    return {}


def run_segmentation_benchmark(
    test_data_dir: str,
    lane_weights: str,
    drivable_weights: str,
    device: str,
    output_file: str
) -> Dict[str, Any]:
    """Run segmentation benchmark script.
    
    Args:
        test_data_dir: Path to test dataset
        lane_weights: Path to lane segmentation weights
        drivable_weights: Path to drivable area segmentation weights
        device: Device to use
        output_file: Output JSON file path
        
    Returns:
        Segmentation metrics dictionary
    """
    logger.info("Running segmentation benchmark...")
    
    cmd = [
        sys.executable,
        "scripts/benchmark_segmentation.py",
        "--test-data", test_data_dir,
        "--lane-weights", lane_weights,
        "--drivable-weights", drivable_weights,
        "--device", device,
        "--output", output_file
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            logger.warning(f"Segmentation benchmark failed: {result.stderr}")
        
        # Load results
        if Path(output_file).exists():
            with open(output_file, 'r') as f:
                return json.load(f)
    except Exception as e:
        logger.error(f"Error running segmentation benchmark: {e}")
    
    return {}


def run_policy_benchmark(
    test_data_dir: str,
    weights: str,
    device: str,
    output_file: str,
    sequence_length: int = 1
) -> Dict[str, Any]:
    """Run policy benchmark script.
    
    Args:
        test_data_dir: Path to test dataset
        weights: Path to policy weights
        device: Device to use
        output_file: Output JSON file path
        sequence_length: Sequence length for temporal model
        
    Returns:
        Policy metrics dictionary
    """
    logger.info("Running policy benchmark...")
    
    cmd = [
        sys.executable,
        "scripts/benchmark_policy.py",
        "--test-data", test_data_dir,
        "--weights", weights,
        "--device", device,
        "--output", output_file,
        "--sequence-length", str(sequence_length)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            logger.warning(f"Policy benchmark failed: {result.stderr}")
        
        # Load results
        if Path(output_file).exists():
            with open(output_file, 'r') as f:
                return json.load(f)
    except Exception as e:
        logger.error(f"Error running policy benchmark: {e}")
    
    return {}


def calculate_system_metrics(
    detection_metrics: Dict[str, Any],
    segmentation_metrics: Dict[str, Any],
    policy_metrics: Dict[str, Any]
) -> Dict[str, Any]:
    """Calculate system-level metrics from component metrics.
    
    Args:
        detection_metrics: Detection results
        segmentation_metrics: Segmentation results
        policy_metrics: Policy results
        
    Returns:
        System-level metrics dictionary
    """
    # Calculate total latency
    det_latency = detection_metrics.get('inference_time_ms', 0)
    lane_latency = segmentation_metrics.get('lane', {}).get('inference_time_ms', 0)
    drivable_latency = segmentation_metrics.get('drivable_area', {}).get('inference_time_ms', 0)
    policy_latency = policy_metrics.get('inference_time_ms', 0)
    
    # Assume segmentation models run in parallel (take max)
    seg_latency = max(lane_latency, drivable_latency)
    
    total_latency = det_latency + seg_latency + policy_latency
    system_fps = 1000.0 / total_latency if total_latency > 0 else 0
    
    components = {
        'Detection': {
            'latency_ms': det_latency,
            'memory_mb': 0  # Placeholder
        },
        'Lane Segmentation': {
            'latency_ms': lane_latency,
            'memory_mb': 0
        },
        'Drivable Area Segmentation': {
            'latency_ms': drivable_latency,
            'memory_mb': 0
        },
        'Policy': {
            'latency_ms': policy_latency,
            'memory_mb': 0
        }
    }
    
    return {
        'total_latency_ms': total_latency,
        'fps': system_fps,
        'components': components,
        'resolution_tradeoffs': {
            '640x480': {
                'fps': system_fps,
                'detection_map': detection_metrics.get('mAP@0.5', 0),
                'segmentation_iou': segmentation_metrics.get('lane', {}).get('iou', 0)
            }
        }
    }


def generate_visualizations(
    metrics: Dict[str, Any],
    figures_dir: Path
) -> None:
    """Generate all visualization plots.
    
    Args:
        metrics: Complete metrics dictionary
        figures_dir: Directory to save figures
    """
    logger.info("Generating visualizations...")
    
    metrics_dir = figures_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    
    # Detection metrics comparison
    if 'detection' in metrics:
        det_metrics = {
            'mAP@0.5': metrics['detection'].get('mAP@0.5', 0),
            'Precision': metrics['detection'].get('precision', 0),
            'Recall': metrics['detection'].get('recall', 0),
            'F1 Score': metrics['detection'].get('f1', 0)
        }
        plot_metrics_comparison(
            det_metrics,
            "Object Detection Performance",
            metrics_dir / "detection_metrics.png",
            ylabel="Score"
        )
    
    # Segmentation metrics comparison
    if 'segmentation' in metrics:
        seg_metrics = {}
        if 'lane' in metrics['segmentation']:
            seg_metrics['Lane IoU'] = metrics['segmentation']['lane'].get('iou', 0)
            seg_metrics['Lane Dice'] = metrics['segmentation']['lane'].get('dice', 0)
        if 'drivable_area' in metrics['segmentation']:
            seg_metrics['Drivable IoU'] = metrics['segmentation']['drivable_area'].get('iou', 0)
            seg_metrics['Drivable Dice'] = metrics['segmentation']['drivable_area'].get('dice', 0)
        
        if seg_metrics:
            plot_metrics_comparison(
                seg_metrics,
                "Segmentation Performance",
                metrics_dir / "segmentation_metrics.png",
                ylabel="Score"
            )
    
    # System latency breakdown
    if 'system' in metrics and 'components' in metrics['system']:
        latencies = {
            name: comp['latency_ms']
            for name, comp in metrics['system']['components'].items()
        }
        plot_latency_breakdown(
            latencies,
            metrics_dir / "latency_breakdown.png",
            title="System Latency Breakdown"
        )
    
    logger.info(f"Visualizations saved to: {metrics_dir}")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Comprehensive benchmark suite")
    parser.add_argument("--test-data", type=str, default="data/test_dataset",
                       help="Path to test dataset directory")
    parser.add_argument("--detection-weights", type=str, default="yolov8n.pt",
                       help="Path to YOLOv8 weights")
    parser.add_argument("--lane-weights", type=str, default="",
                       help="Path to lane segmentation weights")
    parser.add_argument("--drivable-weights", type=str, default="",
                       help="Path to drivable area segmentation weights")
    parser.add_argument("--policy-weights", type=str, default="",
                       help="Path to policy weights")
    parser.add_argument("--device", type=str, default="cpu",
                       help="Device to use (cpu or cuda)")
    parser.add_argument("--sequence-length", type=int, default=1,
                       help="Sequence length for policy")
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("Starting Comprehensive Benchmark Suite")
    logger.info("=" * 60)
    
    # Detect hardware
    hardware_info = get_hardware_info()
    logger.info("\nHardware Configuration:")
    for key, value in hardware_info.items():
        logger.info(f"  {key}: {value}")
    
    # Create output directories
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    figures_dir = Path("docs/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # Ensure test data exists
    test_dir = Path(args.test_data)
    if not (test_dir / "dataset_metadata.json").exists():
        logger.info("Test dataset not found. Creating synthetic test data...")
        from scripts.collect_test_data import create_test_dataset
        create_test_dataset(test_dir, num_samples=100)
    
    # Run individual benchmarks
    detection_metrics = run_detection_benchmark(
        args.test_data,
        args.detection_weights,
        args.device,
        str(results_dir / "detection_metrics.json")
    )
    
    segmentation_metrics = run_segmentation_benchmark(
        args.test_data,
        args.lane_weights,
        args.drivable_weights,
        args.device,
        str(results_dir / "segmentation_metrics.json")
    )
    
    policy_metrics = run_policy_benchmark(
        args.test_data,
        args.policy_weights,
        args.device,
        str(results_dir / "policy_metrics.json"),
        args.sequence_length
    )
    
    # Calculate system-level metrics
    system_metrics = calculate_system_metrics(
        detection_metrics,
        segmentation_metrics,
        policy_metrics
    )
    
    # Combine all metrics
    all_metrics = {
        'detection': detection_metrics,
        'segmentation': segmentation_metrics,
        'policy': policy_metrics,
        'system': system_metrics,
        'hardware': hardware_info
    }
    
    # Save combined metrics
    metrics_path = results_dir / "metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    
    logger.info(f"\nAll metrics saved to: {metrics_path}")
    
    # Generate visualizations
    generate_visualizations(all_metrics, figures_dir)
    
    # Generate documentation report
    docs_dir = Path("docs")
    docs_dir.mkdir(parents=True, exist_ok=True)
    
    report_path = docs_dir / "performance_metrics.md"
    generate_performance_metrics_report(
        all_metrics,
        report_path,
        hardware_info
    )
    
    logger.info(f"Performance report generated: {report_path}")
    
    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("Benchmark Summary")
    logger.info("=" * 60)
    
    if detection_metrics:
        logger.info(f"\n[Detection]")
        logger.info(f"  mAP@0.5: {detection_metrics.get('mAP@0.5', 0):.3f}")
        logger.info(f"  Precision: {detection_metrics.get('precision', 0):.3f}")
        logger.info(f"  FPS: {detection_metrics.get('fps', 0):.1f}")
    
    if segmentation_metrics:
        logger.info(f"\n[Segmentation]")
        if 'lane' in segmentation_metrics:
            logger.info(f"  Lane IoU: {segmentation_metrics['lane'].get('iou', 0):.3f}")
        if 'drivable_area' in segmentation_metrics:
            logger.info(f"  Drivable IoU: {segmentation_metrics['drivable_area'].get('iou', 0):.3f}")
    
    if policy_metrics:
        logger.info(f"\n[Policy]")
        logger.info(f"  RMSE: {policy_metrics.get('rmse', 0):.4f}")
        logger.info(f"  MAE: {policy_metrics.get('mae', 0):.4f}")
        logger.info(f"  Correlation: {policy_metrics.get('correlation', 0):.3f}")
    
    logger.info(f"\n[System]")
    logger.info(f"  Total Latency: {system_metrics.get('total_latency_ms', 0):.1f} ms")
    logger.info(f"  System FPS: {system_metrics.get('fps', 0):.1f}")
    
    logger.info("\n" + "=" * 60)
    logger.info("Benchmark Complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

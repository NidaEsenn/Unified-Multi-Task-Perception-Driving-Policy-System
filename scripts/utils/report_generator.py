"""Helper module for auto-generating markdown documentation reports.

This module provides utilities for generating structured markdown documentation from
benchmark results, including:
- Performance metrics reports
- Ablation study reports
- Failure analysis reports
- System profiling reports
"""
from __future__ import annotations

from typing import Dict, List, Any, Optional
from pathlib import Path
import json
from datetime import datetime


def generate_table(headers: List[str], rows: List[List[Any]]) -> str:
    """Generate a markdown table.
    
    Args:
        headers: List of column headers
        rows: List of rows (each row is a list of values)
        
    Returns:
        Markdown table string
    """
    lines = []
    
    # Header row
    lines.append("| " + " | ".join(str(h) for h in headers) + " |")
    
    # Separator row
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    
    # Data rows
    for row in rows:
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    
    return "\n".join(lines)


def format_metric_value(value: Any, precision: int = 3) -> str:
    """Format a metric value for display.
    
    Args:
        value: Metric value (float, int, or string)
        precision: Decimal precision for floats
        
    Returns:
        Formatted string
    """
    if isinstance(value, float):
        return f"{value:.{precision}f}"
    elif isinstance(value, int):
        return str(value)
    else:
        return str(value)


def generate_performance_metrics_report(
    metrics: Dict[str, Any],
    output_path: Path,
    hardware_info: Optional[Dict[str, str]] = None
) -> None:
    """Generate performance metrics markdown report.
    
    Args:
        metrics: Dictionary containing all metrics
        output_path: Path to save the markdown file
        hardware_info: Optional hardware information
    """
    lines = []
    
    lines.append("# Performance Metrics Report")
    lines.append("")
    lines.append(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
    lines.append("")
    
    # Hardware info
    if hardware_info:
        lines.append("## Hardware Configuration")
        lines.append("")
        for key, value in hardware_info.items():
            lines.append(f"- **{key}**: {value}")
        lines.append("")
    
    # Detection metrics
    if 'detection' in metrics:
        lines.append("## 1. Object Detection Performance (YOLOv8)")
        lines.append("")
        lines.append("### Metrics on Test Set")
        lines.append("")
        
        det_metrics = metrics['detection']
        headers = ["Metric", "Value", "Hardware"]
        hw = hardware_info.get('GPU', 'N/A') if hardware_info else 'N/A'
        rows = [
            ["mAP@0.5", format_metric_value(det_metrics.get('mAP@0.5', 0)), hw],
            ["mAP@0.75", format_metric_value(det_metrics.get('mAP@0.75', 0)), ""],
            ["Precision", format_metric_value(det_metrics.get('precision', 0)), ""],
            ["Recall", format_metric_value(det_metrics.get('recall', 0)), ""],
            ["Inference", f"{format_metric_value(det_metrics.get('inference_time_ms', 0), 1)} ms", ""],
            ["FPS", format_metric_value(det_metrics.get('fps', 0), 1), ""]
        ]
        lines.append(generate_table(headers, rows))
        lines.append("")
        
        # Per-class performance
        if 'per_class' in det_metrics:
            lines.append("### Per-Class Performance")
            lines.append("")
            pc_headers = ["Class", "AP", "Precision", "Recall"]
            pc_rows = []
            for class_name, class_metrics in det_metrics['per_class'].items():
                pc_rows.append([
                    class_name,
                    format_metric_value(class_metrics.get('ap', 0)),
                    format_metric_value(class_metrics.get('precision', 0)),
                    format_metric_value(class_metrics.get('recall', 0))
                ])
            lines.append(generate_table(pc_headers, pc_rows))
            lines.append("")
    
    # Segmentation metrics
    if 'segmentation' in metrics:
        lines.append("## 2. Segmentation Performance (U-Net)")
        lines.append("")
        
        seg_metrics = metrics['segmentation']
        
        # Lane segmentation
        if 'lane' in seg_metrics:
            lines.append("### Lane Segmentation")
            lines.append("")
            headers = ["Metric", "Value"]
            rows = [
                ["IoU", format_metric_value(seg_metrics['lane'].get('iou', 0))],
                ["Dice Coefficient", format_metric_value(seg_metrics['lane'].get('dice', 0))],
                ["Pixel Accuracy", format_metric_value(seg_metrics['lane'].get('pixel_acc', 0))],
                ["Inference", f"{format_metric_value(seg_metrics['lane'].get('inference_time_ms', 0), 1)} ms"]
            ]
            lines.append(generate_table(headers, rows))
            lines.append("")
        
        # Drivable area segmentation
        if 'drivable_area' in seg_metrics:
            lines.append("### Drivable Area Segmentation")
            lines.append("")
            headers = ["Metric", "Value"]
            rows = [
                ["IoU", format_metric_value(seg_metrics['drivable_area'].get('iou', 0))],
                ["Dice Coefficient", format_metric_value(seg_metrics['drivable_area'].get('dice', 0))],
                ["Pixel Accuracy", format_metric_value(seg_metrics['drivable_area'].get('pixel_acc', 0))],
                ["Inference", f"{format_metric_value(seg_metrics['drivable_area'].get('inference_time_ms', 0), 1)} ms"]
            ]
            lines.append(generate_table(headers, rows))
            lines.append("")
    
    # Policy metrics
    if 'policy' in metrics:
        lines.append("## 3. Driving Policy Performance (ConvLSTM)")
        lines.append("")
        lines.append("### Steering Prediction Metrics")
        lines.append("")
        
        pol_metrics = metrics['policy']
        headers = ["Metric", "Value"]
        rows = [
            ["RMSE", format_metric_value(pol_metrics.get('rmse', 0))],
            ["MAE", format_metric_value(pol_metrics.get('mae', 0))],
            ["Correlation", format_metric_value(pol_metrics.get('correlation', 0))],
            ["Inference", f"{format_metric_value(pol_metrics.get('inference_time_ms', 0), 1)} ms"]
        ]
        lines.append(generate_table(headers, rows))
        lines.append("")
    
    # System-level metrics
    if 'system' in metrics:
        lines.append("## 4. System-Level Performance")
        lines.append("")
        
        sys_metrics = metrics['system']
        headers = ["Component", "Latency (ms)", "Memory (MB)", "% of Total Time"]
        rows = []
        total_time = sys_metrics.get('total_latency_ms', 0)
        
        for component, comp_metrics in sys_metrics.get('components', {}).items():
            latency = comp_metrics.get('latency_ms', 0)
            memory = comp_metrics.get('memory_mb', 0)
            pct = (latency / total_time * 100) if total_time > 0 else 0
            rows.append([component, format_metric_value(latency, 1), 
                        format_metric_value(memory, 1), format_metric_value(pct, 1)])
        
        lines.append(generate_table(headers, rows))
        lines.append("")
        lines.append(f"**Total System Latency**: {format_metric_value(total_time, 1)} ms")
        lines.append("")
        lines.append(f"**System FPS**: {format_metric_value(sys_metrics.get('fps', 0), 1)}")
        lines.append("")
        
        # Resolution trade-offs
        if 'resolution_tradeoffs' in sys_metrics:
            lines.append("### Resolution vs Performance Trade-offs")
            lines.append("")
            headers = ["Resolution", "FPS", "Detection mAP", "Segmentation IoU"]
            rows = []
            for res, res_metrics in sys_metrics['resolution_tradeoffs'].items():
                rows.append([
                    res,
                    format_metric_value(res_metrics.get('fps', 0), 1),
                    format_metric_value(res_metrics.get('detection_map', 0)),
                    format_metric_value(res_metrics.get('segmentation_iou', 0))
                ])
            lines.append(generate_table(headers, rows))
            lines.append("")
    
    # Write to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))


def generate_ablation_report(
    ablation_results: Dict[str, Any],
    output_path: Path
) -> None:
    """Generate ablation studies markdown report.
    
    Args:
        ablation_results: Dictionary containing ablation study results
        output_path: Path to save the markdown file
    """
    lines = []
    
    lines.append("# Ablation Studies Report")
    lines.append("")
    lines.append(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
    lines.append("")
    
    # Multi-task vs single-task
    if 'multi_task_comparison' in ablation_results:
        lines.append("## Ablation 1: Multi-Task Learning vs Single-Task Models")
        lines.append("")
        lines.append("**Hypothesis**: Shared encoder reduces computation while maintaining accuracy")
        lines.append("")
        lines.append("### Results")
        lines.append("")
        
        results = ablation_results['multi_task_comparison']
        headers = ["Configuration", "Lane IoU", "Det mAP", "Total Latency", "GPU Mem"]
        rows = []
        for config_name, config_metrics in results.items():
            rows.append([
                config_name,
                format_metric_value(config_metrics.get('lane_iou', 0)),
                format_metric_value(config_metrics.get('detection_map', 0)),
                f"{format_metric_value(config_metrics.get('latency_ms', 0), 1)} ms",
                f"{format_metric_value(config_metrics.get('memory_gb', 0), 2)} GB"
            ])
        lines.append(generate_table(headers, rows))
        lines.append("")
        lines.append("**Analysis**: " + results.get('analysis', 'See metrics above for comparison'))
        lines.append("")
    
    # Temporal modeling
    if 'temporal_modeling' in ablation_results:
        lines.append("## Ablation 2: Temporal Modeling Impact")
        lines.append("")
        lines.append("**Hypothesis**: ConvLSTM temporal modeling improves prediction stability")
        lines.append("")
        lines.append("### Results")
        lines.append("")
        
        results = ablation_results['temporal_modeling']
        headers = ["Model Type", "Steering RMSE", "Steering MAE", "Latency", "Smoothness"]
        rows = []
        for model_name, model_metrics in results.items():
            rows.append([
                model_name,
                format_metric_value(model_metrics.get('rmse', 0)),
                format_metric_value(model_metrics.get('mae', 0)),
                f"{format_metric_value(model_metrics.get('latency_ms', 0), 1)} ms",
                format_metric_value(model_metrics.get('smoothness', 0))
            ])
        lines.append(generate_table(headers, rows))
        lines.append("")
        lines.append("**Analysis**: " + results.get('analysis', 'See metrics above for comparison'))
        lines.append("")
    
    # Sequence length sensitivity
    if 'sequence_length' in ablation_results:
        lines.append("## Ablation 3: Sequence Length Sensitivity")
        lines.append("")
        lines.append("**Hypothesis**: Longer sequences capture more temporal context")
        lines.append("")
        lines.append("### Results")
        lines.append("")
        
        results = ablation_results['sequence_length']
        headers = ["Sequence Length", "Steering RMSE", "Latency", "Memory"]
        rows = []
        for seq_len, seq_metrics in results.items():
            rows.append([
                seq_len,
                format_metric_value(seq_metrics.get('rmse', 0)),
                f"{format_metric_value(seq_metrics.get('latency_ms', 0), 1)} ms",
                f"{format_metric_value(seq_metrics.get('memory_mb', 0), 1)} MB"
            ])
        lines.append(generate_table(headers, rows))
        lines.append("")
        lines.append("**Analysis**: " + results.get('analysis', 'See metrics above for comparison'))
        lines.append("")
    
    # Architecture comparison
    if 'architecture_comparison' in ablation_results:
        lines.append("## Ablation 4: Architecture Comparisons")
        lines.append("")
        lines.append("**Hypothesis**: Different backbones trade off accuracy vs speed")
        lines.append("")
        lines.append("### Results")
        lines.append("")
        
        results = ablation_results['architecture_comparison']
        headers = ["Backbone", "Detection mAP", "Seg IoU", "Latency", "Params (M)"]
        rows = []
        for arch_name, arch_metrics in results.items():
            rows.append([
                arch_name,
                format_metric_value(arch_metrics.get('detection_map', 0)),
                format_metric_value(arch_metrics.get('segmentation_iou', 0)),
                f"{format_metric_value(arch_metrics.get('latency_ms', 0), 1)} ms",
                format_metric_value(arch_metrics.get('params_millions', 0), 2)
            ])
        lines.append(generate_table(headers, rows))
        lines.append("")
        lines.append("**Analysis**: " + results.get('analysis', 'See metrics above for comparison'))
        lines.append("")
    
    # Write to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))


def generate_failure_analysis_report(
    failure_data: Dict[str, Any],
    output_path: Path
) -> None:
    """Generate failure analysis markdown report.
    
    Args:
        failure_data: Dictionary containing failure analysis results
        output_path: Path to save the markdown file
    """
    lines = []
    
    lines.append("# Failure Analysis Report")
    lines.append("")
    lines.append(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
    lines.append("")
    
    lines.append("## Overview")
    lines.append("")
    lines.append(f"Total samples analyzed: {failure_data.get('total_samples', 0)}")
    lines.append(f"Failed samples: {failure_data.get('failed_samples', 0)} ({failure_data.get('failure_rate', 0):.1f}%)")
    lines.append("")
    
    # Failure modes
    if 'failure_modes' in failure_data:
        for idx, (mode_name, mode_data) in enumerate(failure_data['failure_modes'].items(), 1):
            lines.append(f"## Failure Mode {idx}: {mode_name}")
            lines.append("")
            lines.append(f"**Description**: {mode_data.get('description', 'N/A')}")
            lines.append("")
            
            # Example images
            if 'examples' in mode_data:
                lines.append("**Examples**:")
                lines.append("")
                for example_path in mode_data['examples'][:3]:  # Show up to 3 examples
                    lines.append(f"![{mode_name}]({example_path})")
                    lines.append("")
            
            # Metrics
            lines.append("**Metrics**:")
            lines.append("")
            for metric_name, metric_value in mode_data.get('metrics', {}).items():
                lines.append(f"- {metric_name}: {format_metric_value(metric_value)}")
            lines.append("")
            lines.append(f"- Affected samples: {mode_data.get('affected_samples', 0)} ({mode_data.get('percentage', 0):.1f}%)")
            lines.append("")
            
            # Root cause
            lines.append("**Root Cause**:")
            lines.append("")
            for cause in mode_data.get('root_causes', []):
                lines.append(f"- {cause}")
            lines.append("")
            
            # Mitigation
            lines.append("**Mitigation Strategy**:")
            lines.append("")
            for strategy in mode_data.get('mitigation_strategies', []):
                lines.append(f"- {strategy}")
            lines.append("")
            lines.append("---")
            lines.append("")
    
    # Write to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))


def generate_profiling_report(
    profiling_data: Dict[str, Any],
    output_path: Path
) -> None:
    """Generate system profiling markdown report.
    
    Args:
        profiling_data: Dictionary containing profiling results
        output_path: Path to save the markdown file
    """
    lines = []
    
    lines.append("# System Profiling Report")
    lines.append("")
    lines.append(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
    lines.append("")
    
    # Hardware configuration
    if 'hardware' in profiling_data:
        lines.append("## Hardware Configuration")
        lines.append("")
        hw = profiling_data['hardware']
        for key, value in hw.items():
            lines.append(f"- **{key}**: {value}")
        lines.append("")
    
    # Per-module profiling
    if 'modules' in profiling_data:
        lines.append("## Per-Module Profiling")
        lines.append("")
        
        for module_name, module_data in profiling_data['modules'].items():
            lines.append(f"### {module_name}")
            lines.append("")
            
            headers = ["Component", "Time (ms)", "% Total"]
            rows = []
            total_time = module_data.get('total_time_ms', 0)
            
            for comp_name, comp_time in module_data.get('components', {}).items():
                pct = (comp_time / total_time * 100) if total_time > 0 else 0
                rows.append([comp_name, format_metric_value(comp_time, 2), 
                           format_metric_value(pct, 1) + "%"])
            
            lines.append(generate_table(headers, rows))
            lines.append("")
    
    # Memory profile
    if 'memory' in profiling_data:
        lines.append("## Memory Profile")
        lines.append("")
        
        headers = ["Module", "Weights (MB)", "Activations (MB)", "Peak (MB)"]
        rows = []
        for module_name, mem_data in profiling_data['memory'].items():
            rows.append([
                module_name,
                format_metric_value(mem_data.get('weights_mb', 0), 1),
                format_metric_value(mem_data.get('activations_mb', 0), 1),
                format_metric_value(mem_data.get('peak_mb', 0), 1)
            ])
        
        lines.append(generate_table(headers, rows))
        lines.append("")
    
    # Batch size sensitivity
    if 'batch_size_analysis' in profiling_data:
        lines.append("## Batch Size Sensitivity")
        lines.append("")
        
        headers = ["Batch Size", "Latency (ms)", "Throughput (samples/s)", "Memory (GB)"]
        rows = []
        for bs, bs_data in profiling_data['batch_size_analysis'].items():
            rows.append([
                bs,
                format_metric_value(bs_data.get('latency_ms', 0), 1),
                format_metric_value(bs_data.get('throughput', 0), 1),
                format_metric_value(bs_data.get('memory_gb', 0), 2)
            ])
        
        lines.append(generate_table(headers, rows))
        lines.append("")
    
    # Bottleneck analysis
    if 'bottleneck' in profiling_data:
        lines.append("## Bottleneck Analysis")
        lines.append("")
        lines.append(f"**Finding**: {profiling_data['bottleneck'].get('finding', 'N/A')}")
        lines.append("")
        lines.append("**Optimization Opportunities**:")
        lines.append("")
        for opp in profiling_data['bottleneck'].get('opportunities', []):
            lines.append(f"- {opp}")
        lines.append("")
    
    # Write to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))

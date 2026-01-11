"""Helper module for generating plots and visualizations.

This module provides utilities for creating publication-quality plots for:
- Performance metrics comparison
- Ablation study results
- Failure mode analysis
- System profiling breakdowns
"""
from __future__ import annotations

from typing import Dict, List, Any, Optional
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set publication-quality defaults
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 9


def plot_metrics_comparison(
    metrics: Dict[str, float],
    title: str,
    output_path: Path,
    ylabel: str = "Score"
) -> None:
    """Create a bar chart comparing different metrics.
    
    Args:
        metrics: Dictionary of metric names and values
        title: Plot title
        output_path: Path to save the figure
        ylabel: Y-axis label
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    names = list(metrics.keys())
    values = list(metrics.values())
    
    bars = ax.bar(names, values, color=sns.color_palette("husl", len(names)))
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim([0, max(values) * 1.2 if values else 1.0])
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=9)
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_confusion_matrix(
    matrix: np.ndarray,
    class_names: List[str],
    output_path: Path,
    title: str = "Confusion Matrix"
) -> None:
    """Create a confusion matrix heatmap.
    
    Args:
        matrix: Confusion matrix (N, N)
        class_names: List of class names
        output_path: Path to save the figure
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                ax=ax, cbar_kws={'label': 'Count'})
    
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(title)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_latency_breakdown(
    components: Dict[str, float],
    output_path: Path,
    title: str = "Latency Breakdown"
) -> None:
    """Create a pie chart showing latency breakdown by component.
    
    Args:
        components: Dictionary of component names and latencies (ms)
        output_path: Path to save the figure
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    names = list(components.keys())
    values = list(components.values())
    colors = sns.color_palette("husl", len(names))
    
    def autopct_format(pct):
        return f'{pct:.1f}%' if pct > 5 else ''
    
    wedges, texts, autotexts = ax.pie(values, labels=names, autopct=autopct_format,
                                        colors=colors, startangle=90)
    
    # Add legend with actual values
    legend_labels = [f'{name}: {value:.2f} ms' for name, value in zip(names, values)]
    ax.legend(legend_labels, loc='center left', bbox_to_anchor=(1, 0, 0.5, 1))
    
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()


def plot_ablation_comparison(
    ablation_results: List[Dict[str, Any]],
    metric_name: str,
    output_path: Path,
    title: str = "Ablation Study Results"
) -> None:
    """Create a grouped bar chart comparing ablation study results.
    
    Args:
        ablation_results: List of dicts with 'name' and metric values
        metric_name: Name of the metric to plot
        output_path: Path to save the figure
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    names = [r['name'] for r in ablation_results]
    values = [r.get(metric_name, 0) for r in ablation_results]
    
    bars = ax.bar(names, values, color=sns.color_palette("Set2", len(names)))
    ax.set_ylabel(metric_name)
    ax.set_title(title)
    ax.set_ylim([0, max(values) * 1.2 if values else 1.0])
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_distribution(
    data: np.ndarray,
    output_path: Path,
    title: str = "Data Distribution",
    xlabel: str = "Value",
    bins: int = 50
) -> None:
    """Create a histogram showing data distribution.
    
    Args:
        data: Array of values
        output_path: Path to save the figure
        title: Plot title
        xlabel: X-axis label
        bins: Number of histogram bins
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.hist(data, bins=bins, color=sns.color_palette()[0], alpha=0.7, edgecolor='black')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Frequency')
    ax.set_title(title)
    
    # Add statistics
    mean_val = np.mean(data)
    std_val = np.std(data)
    ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
    ax.axvline(mean_val + std_val, color='orange', linestyle=':', linewidth=1.5, 
               label=f'Std: {std_val:.2f}')
    ax.axvline(mean_val - std_val, color='orange', linestyle=':', linewidth=1.5)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_temporal_prediction(
    timestamps: np.ndarray,
    predictions: np.ndarray,
    ground_truth: Optional[np.ndarray],
    output_path: Path,
    title: str = "Temporal Predictions"
) -> None:
    """Create a line plot showing temporal predictions vs ground truth.
    
    Args:
        timestamps: Array of time indices
        predictions: Predicted values
        ground_truth: Ground truth values (optional)
        output_path: Path to save the figure
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(timestamps, predictions, label='Predictions', linewidth=2, alpha=0.8)
    
    if ground_truth is not None:
        ax.plot(timestamps, ground_truth, label='Ground Truth', 
                linewidth=2, alpha=0.8, linestyle='--')
    
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Steering Angle')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_fps_vs_resolution(
    resolutions: List[str],
    fps_values: List[float],
    output_path: Path,
    title: str = "FPS vs Resolution"
) -> None:
    """Create a line plot showing FPS at different resolutions.
    
    Args:
        resolutions: List of resolution strings (e.g., "640x480")
        fps_values: Corresponding FPS values
        output_path: Path to save the figure
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = range(len(resolutions))
    ax.plot(x, fps_values, marker='o', linewidth=2, markersize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(resolutions, rotation=45, ha='right')
    ax.set_xlabel('Resolution')
    ax.set_ylabel('FPS')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    # Add value labels
    for i, (res, fps) in enumerate(zip(resolutions, fps_values)):
        ax.text(i, fps + max(fps_values) * 0.02, f'{fps:.1f}',
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_failure_distribution(
    failure_types: Dict[str, int],
    output_path: Path,
    title: str = "Failure Mode Distribution"
) -> None:
    """Create a pie chart showing distribution of failure types.
    
    Args:
        failure_types: Dictionary of failure type names and counts
        output_path: Path to save the figure
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    names = list(failure_types.keys())
    values = list(failure_types.values())
    colors = sns.color_palette("Set3", len(names))
    
    wedges, texts, autotexts = ax.pie(values, labels=names, autopct='%1.1f%%',
                                        colors=colors, startangle=90)
    
    # Make percentage text more readable
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_weight('bold')
    
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_memory_usage(
    modules: Dict[str, float],
    output_path: Path,
    title: str = "Memory Usage by Module"
) -> None:
    """Create a stacked bar chart showing memory usage breakdown.
    
    Args:
        modules: Dictionary of module names and memory usage (GB)
        output_path: Path to save the figure
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    names = list(modules.keys())
    values = list(modules.values())
    colors = sns.color_palette("viridis", len(names))
    
    bars = ax.bar(names, values, color=colors)
    ax.set_ylabel('Memory Usage (GB)')
    ax.set_title(title)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f} GB',
                ha='center', va='bottom')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

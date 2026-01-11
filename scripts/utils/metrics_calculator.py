"""Helper module for computing metrics across detection, segmentation, and policy tasks.

This module provides utilities for calculating:
- Object detection metrics (mAP, precision, recall, per-class AP)
- Segmentation metrics (IoU, Dice coefficient, pixel accuracy)
- Policy metrics (RMSE, MAE, steering prediction accuracy)
"""
from __future__ import annotations

from typing import Dict, List, Any, Tuple
import numpy as np
from pathlib import Path


def calculate_iou(pred_mask: np.ndarray, gt_mask: np.ndarray, threshold: float = 0.5) -> float:
    """Calculate Intersection over Union for binary segmentation masks.
    
    Args:
        pred_mask: Predicted mask (H, W) with values in [0, 1]
        gt_mask: Ground truth mask (H, W) with binary values {0, 1}
        threshold: Threshold to binarize prediction
        
    Returns:
        IoU score between 0 and 1
    """
    pred_binary = (pred_mask >= threshold).astype(np.uint8)
    gt_binary = gt_mask.astype(np.uint8)
    
    intersection = np.logical_and(pred_binary, gt_binary).sum()
    union = np.logical_or(pred_binary, gt_binary).sum()
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return float(intersection / union)


def calculate_dice(pred_mask: np.ndarray, gt_mask: np.ndarray, threshold: float = 0.5) -> float:
    """Calculate Dice coefficient for binary segmentation masks.
    
    Args:
        pred_mask: Predicted mask (H, W) with values in [0, 1]
        gt_mask: Ground truth mask (H, W) with binary values {0, 1}
        threshold: Threshold to binarize prediction
        
    Returns:
        Dice coefficient between 0 and 1
    """
    pred_binary = (pred_mask >= threshold).astype(np.uint8)
    gt_binary = gt_mask.astype(np.uint8)
    
    intersection = np.logical_and(pred_binary, gt_binary).sum()
    total = pred_binary.sum() + gt_binary.sum()
    
    if total == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return float(2.0 * intersection / total)


def calculate_pixel_accuracy(pred_mask: np.ndarray, gt_mask: np.ndarray, threshold: float = 0.5) -> float:
    """Calculate pixel-wise accuracy for binary segmentation masks.
    
    Args:
        pred_mask: Predicted mask (H, W) with values in [0, 1]
        gt_mask: Ground truth mask (H, W) with binary values {0, 1}
        threshold: Threshold to binarize prediction
        
    Returns:
        Pixel accuracy between 0 and 1
    """
    pred_binary = (pred_mask >= threshold).astype(np.uint8)
    gt_binary = gt_mask.astype(np.uint8)
    
    correct = (pred_binary == gt_binary).sum()
    total = pred_binary.size
    
    return float(correct / total)


def calculate_detection_metrics(
    predictions: List[Dict[str, Any]],
    ground_truths: List[Dict[str, Any]],
    iou_threshold: float = 0.5,
    conf_threshold: float = 0.5
) -> Dict[str, float]:
    """Calculate detection metrics (precision, recall, F1).
    
    Args:
        predictions: List of detection dicts with 'bbox', 'conf', 'class_id'
        ground_truths: List of ground truth dicts with 'bbox', 'class_id'
        iou_threshold: IoU threshold for matching detections
        conf_threshold: Confidence threshold for predictions
        
    Returns:
        Dictionary with precision, recall, f1 scores
    """
    # Filter predictions by confidence
    filtered_preds = [p for p in predictions if p.get('conf', 0) >= conf_threshold]
    
    if len(filtered_preds) == 0:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    
    if len(ground_truths) == 0:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    
    # Simple matching based on IoU
    tp = 0
    matched_gts = set()
    
    for pred in filtered_preds:
        best_iou = 0.0
        best_gt_idx = -1
        
        for gt_idx, gt in enumerate(ground_truths):
            if gt_idx in matched_gts:
                continue
            
            # Calculate bbox IoU
            pred_box = pred['bbox']
            gt_box = gt['bbox']
            iou = calculate_bbox_iou(pred_box, gt_box)
            
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        
        if best_iou >= iou_threshold and best_gt_idx >= 0:
            tp += 1
            matched_gts.add(best_gt_idx)
    
    fp = len(filtered_preds) - tp
    fn = len(ground_truths) - tp
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'tp': tp,
        'fp': fp,
        'fn': fn
    }


def calculate_bbox_iou(box1: Tuple[float, float, float, float], 
                       box2: Tuple[float, float, float, float]) -> float:
    """Calculate IoU between two bounding boxes.
    
    Args:
        box1: (x1, y1, x2, y2) coordinates
        box2: (x1, y1, x2, y2) coordinates
        
    Returns:
        IoU score between 0 and 1
    """
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Calculate intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i < x1_i or y2_i < y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    
    # Calculate union
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    if union == 0:
        return 0.0
    
    return float(intersection / union)


def calculate_steering_metrics(predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    """Calculate policy metrics for steering angle prediction.
    
    Args:
        predictions: Predicted steering angles (N,)
        targets: Ground truth steering angles (N,)
        
    Returns:
        Dictionary with RMSE, MAE, and correlation
    """
    predictions = np.array(predictions).flatten()
    targets = np.array(targets).flatten()
    
    # Root Mean Squared Error
    rmse = np.sqrt(np.mean((predictions - targets) ** 2))
    
    # Mean Absolute Error
    mae = np.mean(np.abs(predictions - targets))
    
    # Correlation coefficient
    if len(predictions) > 1:
        correlation = np.corrcoef(predictions, targets)[0, 1]
    else:
        correlation = 0.0
    
    return {
        'rmse': float(rmse),
        'mae': float(mae),
        'correlation': float(correlation)
    }


def calculate_map(
    predictions_per_image: List[List[Dict[str, Any]]],
    ground_truths_per_image: List[List[Dict[str, Any]]],
    iou_threshold: float = 0.5,
    num_classes: int = 80
) -> Dict[str, float]:
    """Calculate mean Average Precision (simplified version).
    
    Args:
        predictions_per_image: List of prediction lists per image
        ground_truths_per_image: List of ground truth lists per image
        iou_threshold: IoU threshold for matching
        num_classes: Number of object classes
        
    Returns:
        Dictionary with mAP and per-class AP
    """
    # Simplified mAP calculation
    # For production, use torchmetrics or pycocotools
    
    class_aps = []
    
    for class_id in range(num_classes):
        # Collect all predictions and GTs for this class
        class_preds = []
        class_gts = []
        
        for img_idx, (preds, gts) in enumerate(zip(predictions_per_image, ground_truths_per_image)):
            for pred in preds:
                if pred.get('class_id') == class_id:
                    class_preds.append({**pred, 'image_id': img_idx})
            
            for gt in gts:
                if gt.get('class_id') == class_id:
                    class_gts.append({**gt, 'image_id': img_idx})
        
        if len(class_gts) == 0:
            continue
        
        # Sort predictions by confidence
        class_preds.sort(key=lambda x: x.get('conf', 0), reverse=True)
        
        # Calculate TP/FP for this class
        tp = np.zeros(len(class_preds))
        fp = np.zeros(len(class_preds))
        matched_gts = set()
        
        for pred_idx, pred in enumerate(class_preds):
            img_id = pred['image_id']
            best_iou = 0.0
            best_gt_idx = -1
            
            for gt_idx, gt in enumerate(class_gts):
                if gt['image_id'] != img_id:
                    continue
                
                gt_key = (img_id, gt_idx)
                if gt_key in matched_gts:
                    continue
                
                iou = calculate_bbox_iou(pred['bbox'], gt['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            if best_iou >= iou_threshold and best_gt_idx >= 0:
                tp[pred_idx] = 1
                matched_gts.add((img_id, best_gt_idx))
            else:
                fp[pred_idx] = 1
        
        # Calculate precision and recall
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        recalls = tp_cumsum / len(class_gts)
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-10)
        
        # Calculate AP using 11-point interpolation
        ap = 0.0
        for t in np.linspace(0, 1, 11):
            if np.sum(recalls >= t) == 0:
                p = 0
            else:
                p = np.max(precisions[recalls >= t])
            ap += p / 11
        
        class_aps.append(ap)
    
    map_score = np.mean(class_aps) if class_aps else 0.0
    
    return {
        'mAP': float(map_score),
        'num_classes_evaluated': len(class_aps)
    }

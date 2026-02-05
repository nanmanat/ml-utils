"""
Metrics tracking and aggregation.

Provides utilities for tracking, aggregating, and computing
evaluation metrics for ML experiments.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union

import torch


@dataclass
class MetricValue:
    """Container for a single metric value with metadata."""
    
    name: str
    value: float
    step: int
    epoch: Optional[int] = None
    timestamp: Optional[str] = None


class MetricAggregator:
    """
    Aggregator for computing running statistics of metrics.
    
    Supports mean, sum, min, max, and custom aggregation functions.
    """
    
    def __init__(self, reduction: str = "mean") -> None:
        """
        Initialize aggregator.
        
        Args:
            reduction: Aggregation method ('mean', 'sum', 'min', 'max').
        """
        self.reduction = reduction
        self._values: List[float] = []
        self._weights: List[float] = []
    
    def update(self, value: float, weight: float = 1.0) -> None:
        """Add a value to the aggregator."""
        self._values.append(value)
        self._weights.append(weight)
    
    def compute(self) -> float:
        """Compute the aggregated value."""
        if not self._values:
            return 0.0
        
        if self.reduction == "mean":
            total_weight = sum(self._weights)
            if total_weight == 0:
                return 0.0
            return sum(v * w for v, w in zip(self._values, self._weights)) / total_weight
        
        elif self.reduction == "sum":
            return sum(self._values)
        
        elif self.reduction == "min":
            return min(self._values)
        
        elif self.reduction == "max":
            return max(self._values)
        
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")
    
    def reset(self) -> None:
        """Reset the aggregator."""
        self._values.clear()
        self._weights.clear()
    
    @property
    def count(self) -> int:
        """Get number of values."""
        return len(self._values)


class MetricsTracker:
    """
    Comprehensive metrics tracking for ML experiments.
    
    Tracks metrics across epochs and steps with support for
    aggregation, history, and best value tracking.
    """
    
    def __init__(self) -> None:
        self._aggregators: Dict[str, MetricAggregator] = {}
        self._history: Dict[str, List[MetricValue]] = defaultdict(list)
        self._best: Dict[str, MetricValue] = {}
        self._best_mode: Dict[str, str] = {}  # 'min' or 'max'
        self._current_epoch = 0
        self._current_step = 0
    
    def register_metric(
        self,
        name: str,
        reduction: str = "mean",
        best_mode: Optional[str] = None,
    ) -> None:
        """
        Register a metric for tracking.
        
        Args:
            name: Metric name.
            reduction: Aggregation method.
            best_mode: 'min' or 'max' for best tracking.
        """
        self._aggregators[name] = MetricAggregator(reduction)
        if best_mode:
            self._best_mode[name] = best_mode
    
    def update(
        self,
        metrics: Dict[str, float],
        weight: float = 1.0,
    ) -> None:
        """
        Update metrics with new values.
        
        Args:
            metrics: Dictionary of metric values.
            weight: Weight for weighted averaging.
        """
        for name, value in metrics.items():
            if name not in self._aggregators:
                self.register_metric(name)
            self._aggregators[name].update(value, weight)
    
    def compute(self) -> Dict[str, float]:
        """Compute aggregated metrics."""
        return {name: agg.compute() for name, agg in self._aggregators.items()}
    
    def reset(self) -> None:
        """Reset all aggregators."""
        for aggregator in self._aggregators.values():
            aggregator.reset()
    
    def step(self, step: Optional[int] = None) -> Dict[str, float]:
        """
        Complete a step and record metrics.
        
        Args:
            step: Step number (auto-increments if None).
            
        Returns:
            Computed metrics for this step.
        """
        if step is not None:
            self._current_step = step
        else:
            self._current_step += 1
        
        metrics = self.compute()
        
        # Record in history
        for name, value in metrics.items():
            metric_value = MetricValue(
                name=name,
                value=value,
                step=self._current_step,
                epoch=self._current_epoch,
            )
            self._history[name].append(metric_value)
            
            # Update best
            if name in self._best_mode:
                self._update_best(name, metric_value)
        
        return metrics
    
    def epoch(self, epoch: Optional[int] = None) -> Dict[str, float]:
        """
        Complete an epoch and record metrics.
        
        Args:
            epoch: Epoch number (auto-increments if None).
            
        Returns:
            Computed metrics for this epoch.
        """
        if epoch is not None:
            self._current_epoch = epoch
        else:
            self._current_epoch += 1
        
        metrics = self.step(self._current_step)
        self.reset()
        
        return metrics
    
    def _update_best(self, name: str, value: MetricValue) -> bool:
        """Update best value if improved."""
        mode = self._best_mode[name]
        current_best = self._best.get(name)
        
        if current_best is None:
            self._best[name] = value
            return True
        
        is_better = (
            (mode == "min" and value.value < current_best.value) or
            (mode == "max" and value.value > current_best.value)
        )
        
        if is_better:
            self._best[name] = value
            return True
        
        return False
    
    def get_best(self, name: str) -> Optional[MetricValue]:
        """Get best value for a metric."""
        return self._best.get(name)
    
    def get_history(self, name: str) -> List[MetricValue]:
        """Get history for a metric."""
        return self._history.get(name, [])
    
    def get_last(self, name: str) -> Optional[float]:
        """Get last recorded value for a metric."""
        history = self._history.get(name, [])
        return history[-1].value if history else None
    
    def summary(self) -> Dict[str, Dict[str, Any]]:
        """Get summary of all tracked metrics."""
        summary = {}
        
        for name in self._history.keys():
            history = self._history[name]
            values = [m.value for m in history]
            
            summary[name] = {
                "count": len(values),
                "last": values[-1] if values else None,
                "mean": sum(values) / len(values) if values else None,
                "min": min(values) if values else None,
                "max": max(values) if values else None,
            }
            
            if name in self._best:
                summary[name]["best"] = self._best[name].value
                summary[name]["best_step"] = self._best[name].step
        
        return summary
    
    def to_dict(self) -> Dict[str, List[float]]:
        """Export history as dictionary of lists."""
        return {
            name: [m.value for m in history]
            for name, history in self._history.items()
        }


class ClassificationMetrics:
    """
    Metrics computation for classification tasks.
    """
    
    @staticmethod
    def accuracy(
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> float:
        """Compute accuracy."""
        correct = (predictions == targets).sum()
        total = targets.numel()
        return (correct / total).item()
    
    @staticmethod
    def top_k_accuracy(
        probabilities: torch.Tensor,
        targets: torch.Tensor,
        k: int = 5,
    ) -> float:
        """Compute top-k accuracy."""
        top_k_preds = probabilities.topk(k, dim=1).indices
        correct = (top_k_preds == targets.unsqueeze(1)).any(dim=1).sum()
        total = targets.numel()
        return (correct / total).item()
    
    @staticmethod
    def precision_recall_f1(
        predictions: torch.Tensor,
        targets: torch.Tensor,
        num_classes: int,
        average: str = "macro",
    ) -> Dict[str, float]:
        """Compute precision, recall, and F1 score."""
        precisions = []
        recalls = []
        
        for cls in range(num_classes):
            tp = ((predictions == cls) & (targets == cls)).sum().float()
            fp = ((predictions == cls) & (targets != cls)).sum().float()
            fn = ((predictions != cls) & (targets == cls)).sum().float()
            
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            
            precisions.append(precision.item())
            recalls.append(recall.item())
        
        if average == "macro":
            avg_precision = sum(precisions) / len(precisions)
            avg_recall = sum(recalls) / len(recalls)
        else:
            # Weighted average (by class frequency)
            class_counts = torch.bincount(targets, minlength=num_classes).float()
            total = class_counts.sum()
            weights = class_counts / total
            
            avg_precision = sum(p * w.item() for p, w in zip(precisions, weights))
            avg_recall = sum(r * w.item() for r, w in zip(recalls, weights))
        
        if avg_precision + avg_recall > 0:
            f1 = 2 * avg_precision * avg_recall / (avg_precision + avg_recall)
        else:
            f1 = 0.0
        
        return {
            "precision": avg_precision,
            "recall": avg_recall,
            "f1": f1,
        }
    
    @staticmethod
    def confusion_matrix(
        predictions: torch.Tensor,
        targets: torch.Tensor,
        num_classes: int,
    ) -> torch.Tensor:
        """Compute confusion matrix."""
        matrix = torch.zeros(num_classes, num_classes, dtype=torch.long)
        
        for pred, target in zip(predictions, targets):
            matrix[target.item(), pred.item()] += 1
        
        return matrix


class DetectionMetrics:
    """
    Metrics computation for object detection tasks.
    """
    
    @staticmethod
    def compute_iou(
        box1: torch.Tensor,
        box2: torch.Tensor,
    ) -> torch.Tensor:
        """Compute IoU between boxes."""
        # box format: [x1, y1, x2, y2]
        x1 = torch.max(box1[:, 0], box2[:, 0])
        y1 = torch.max(box1[:, 1], box2[:, 1])
        x2 = torch.min(box1[:, 2], box2[:, 2])
        y2 = torch.min(box1[:, 3], box2[:, 3])
        
        intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
        
        area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
        area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
        
        union = area1 + area2 - intersection
        
        return intersection / (union + 1e-8)
    
    @staticmethod
    def average_precision(
        precisions: List[float],
        recalls: List[float],
        use_11_point: bool = True,
    ) -> float:
        """Compute average precision from precision-recall curve."""
        if use_11_point:
            # 11-point interpolation
            ap = 0.0
            for t in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
                prec_at_recall = max(
                    [p for p, r in zip(precisions, recalls) if r >= t],
                    default=0.0,
                )
                ap += prec_at_recall
            return ap / 11.0
        else:
            # All-point interpolation
            mrec = [0.0] + recalls + [1.0]
            mpre = [0.0] + precisions + [0.0]
            
            for i in range(len(mpre) - 2, -1, -1):
                mpre[i] = max(mpre[i], mpre[i + 1])
            
            ap = 0.0
            for i in range(1, len(mrec)):
                if mrec[i] != mrec[i - 1]:
                    ap += (mrec[i] - mrec[i - 1]) * mpre[i]
            
            return ap

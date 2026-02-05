"""
Object detection task implementation.

Provides training, evaluation, and inference logic for
object detection models with mAP evaluation.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.cuda.amp import autocast

from mlcli.core.config import ExperimentConfig
from mlcli.core.registry import register_task
from mlcli.tasks.base import BaseTask


@register_task("object_detection")
class DetectionTask(BaseTask):
    """
    Task implementation for object detection.
    
    Supports various detection models with mAP, IoU-based evaluation.
    """
    
    def __init__(
        self,
        config: ExperimentConfig,
        model: Optional[nn.Module] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__(config, model, device)
        
        self.num_classes = config.model.num_classes
        self.iou_threshold = 0.5
        self.conf_threshold = 0.05
        self.nms_threshold = 0.5
    
    def build_criterion(self) -> nn.Module:
        """
        Build detection loss.
        
        Note: Most detection models have built-in loss computation.
        This returns an identity module as a placeholder.
        """
        return nn.Identity()
    
    def train_step(
        self,
        batch: Tuple[torch.Tensor, List[Dict[str, torch.Tensor]]],
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None,
        scaler: Optional[torch.cuda.amp.GradScaler] = None,
    ) -> Dict[str, float]:
        """Execute a single training step."""
        images, targets = batch
        images = images.to(self.device)
        targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
        
        optimizer.zero_grad()
        
        use_amp = scaler is not None and self.device.type == "cuda"
        
        with autocast(enabled=use_amp):
            # Detection models return losses during training
            loss_dict = self.model(images, targets)
            
            if isinstance(loss_dict, dict):
                losses = sum(loss for loss in loss_dict.values())
            else:
                # Handle models that return loss directly
                losses = loss_dict
                loss_dict = {"loss": losses}
        
        if scaler is not None:
            scaler.scale(losses).backward()
            
            if self.config.training.gradient_clip:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.training.gradient_clip,
                )
            
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            
            if self.config.training.gradient_clip:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.training.gradient_clip,
                )
            
            optimizer.step()
        
        return {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in loss_dict.items()}
    
    def eval_step(
        self,
        batch: Tuple[torch.Tensor, List[Dict[str, torch.Tensor]]],
    ) -> Dict[str, Any]:
        """Execute a single evaluation step."""
        images, targets = batch
        images = images.to(self.device)
        
        with autocast(enabled=self.config.training.mixed_precision and self.device.type == "cuda"):
            predictions = self.model(images)
        
        # Move predictions to CPU
        predictions_cpu = []
        for pred in predictions:
            predictions_cpu.append({
                k: v.cpu() if isinstance(v, torch.Tensor) else v
                for k, v in pred.items()
            })
        
        # Move targets to CPU
        targets_cpu = []
        for target in targets:
            targets_cpu.append({
                k: v.cpu() if isinstance(v, torch.Tensor) else v
                for k, v in target.items()
            })
        
        return {
            "predictions": predictions_cpu,
            "targets": targets_cpu,
        }
    
    def compute_metrics(
        self,
        outputs: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        """Compute detection metrics (mAP, IoU)."""
        all_predictions = []
        all_targets = []
        
        for output in outputs:
            all_predictions.extend(output["predictions"])
            all_targets.extend(output["targets"])
        
        # Compute mAP
        ap_per_class = self._compute_ap_per_class(all_predictions, all_targets)
        mAP = sum(ap_per_class.values()) / len(ap_per_class) if ap_per_class else 0.0
        
        # Compute mAP at different IoU thresholds
        mAP50 = self._compute_map_at_iou(all_predictions, all_targets, iou_threshold=0.5)
        mAP75 = self._compute_map_at_iou(all_predictions, all_targets, iou_threshold=0.75)
        
        # Compute average IoU
        avg_iou = self._compute_average_iou(all_predictions, all_targets)
        
        return {
            "mAP": mAP,
            "mAP50": mAP50,
            "mAP75": mAP75,
            "avg_iou": avg_iou,
        }
    
    def _compute_ap_per_class(
        self,
        predictions: List[Dict[str, torch.Tensor]],
        targets: List[Dict[str, torch.Tensor]],
    ) -> Dict[int, float]:
        """Compute average precision per class."""
        ap_per_class = {}
        
        # Collect all detections and ground truths per class
        class_detections: Dict[int, List] = {}
        class_gt: Dict[int, int] = {}
        
        for pred, target in zip(predictions, targets):
            # Ground truth
            for label in target.get("labels", []):
                label_int = label.item() if isinstance(label, torch.Tensor) else label
                class_gt[label_int] = class_gt.get(label_int, 0) + 1
            
            # Predictions
            boxes = pred.get("boxes", torch.empty(0, 4))
            scores = pred.get("scores", torch.empty(0))
            labels = pred.get("labels", torch.empty(0))
            
            for i in range(len(boxes)):
                label = labels[i].item() if isinstance(labels[i], torch.Tensor) else labels[i]
                score = scores[i].item() if isinstance(scores[i], torch.Tensor) else scores[i]
                
                if label not in class_detections:
                    class_detections[label] = []
                
                # Check if detection is TP or FP
                is_tp = self._is_true_positive(boxes[i], label, target)
                class_detections[label].append((score, is_tp))
        
        # Compute AP for each class
        for cls_id in set(class_gt.keys()) | set(class_detections.keys()):
            detections = class_detections.get(cls_id, [])
            n_gt = class_gt.get(cls_id, 0)
            
            if n_gt == 0:
                ap_per_class[cls_id] = 0.0
                continue
            
            # Sort by score
            detections.sort(key=lambda x: x[0], reverse=True)
            
            # Compute precision-recall curve
            tp = 0
            fp = 0
            precisions = []
            recalls = []
            
            for score, is_tp in detections:
                if is_tp:
                    tp += 1
                else:
                    fp += 1
                
                precision = tp / (tp + fp)
                recall = tp / n_gt
                
                precisions.append(precision)
                recalls.append(recall)
            
            # Compute AP using 11-point interpolation
            ap = self._compute_ap_11_point(precisions, recalls)
            ap_per_class[cls_id] = ap
        
        return ap_per_class
    
    def _is_true_positive(
        self,
        pred_box: torch.Tensor,
        pred_label: int,
        target: Dict[str, torch.Tensor],
    ) -> bool:
        """Check if prediction is a true positive."""
        gt_boxes = target.get("boxes", torch.empty(0, 4))
        gt_labels = target.get("labels", torch.empty(0))
        
        for i in range(len(gt_boxes)):
            if gt_labels[i].item() == pred_label:
                iou = self._compute_iou(pred_box.unsqueeze(0), gt_boxes[i].unsqueeze(0))
                if iou >= self.iou_threshold:
                    return True
        
        return False
    
    def _compute_iou(
        self,
        boxes1: torch.Tensor,
        boxes2: torch.Tensor,
    ) -> float:
        """Compute IoU between two boxes."""
        # boxes format: [x1, y1, x2, y2]
        x1 = torch.max(boxes1[:, 0], boxes2[:, 0])
        y1 = torch.max(boxes1[:, 1], boxes2[:, 1])
        x2 = torch.min(boxes1[:, 2], boxes2[:, 2])
        y2 = torch.min(boxes1[:, 3], boxes2[:, 3])
        
        intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
        
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        
        union = area1 + area2 - intersection
        
        iou = intersection / (union + 1e-8)
        return iou.item()
    
    def _compute_ap_11_point(
        self,
        precisions: List[float],
        recalls: List[float],
    ) -> float:
        """Compute AP using 11-point interpolation."""
        ap = 0.0
        
        for t in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            precision_at_recall = 0.0
            for p, r in zip(precisions, recalls):
                if r >= t:
                    precision_at_recall = max(precision_at_recall, p)
            ap += precision_at_recall
        
        return ap / 11.0
    
    def _compute_map_at_iou(
        self,
        predictions: List[Dict[str, torch.Tensor]],
        targets: List[Dict[str, torch.Tensor]],
        iou_threshold: float,
    ) -> float:
        """Compute mAP at specific IoU threshold."""
        old_threshold = self.iou_threshold
        self.iou_threshold = iou_threshold
        
        ap_per_class = self._compute_ap_per_class(predictions, targets)
        mAP = sum(ap_per_class.values()) / len(ap_per_class) if ap_per_class else 0.0
        
        self.iou_threshold = old_threshold
        return mAP
    
    def _compute_average_iou(
        self,
        predictions: List[Dict[str, torch.Tensor]],
        targets: List[Dict[str, torch.Tensor]],
    ) -> float:
        """Compute average IoU between matched predictions and targets."""
        ious = []
        
        for pred, target in zip(predictions, targets):
            pred_boxes = pred.get("boxes", torch.empty(0, 4))
            pred_labels = pred.get("labels", torch.empty(0))
            gt_boxes = target.get("boxes", torch.empty(0, 4))
            gt_labels = target.get("labels", torch.empty(0))
            
            for i in range(len(pred_boxes)):
                best_iou = 0.0
                pred_label = pred_labels[i].item() if isinstance(pred_labels[i], torch.Tensor) else pred_labels[i]
                
                for j in range(len(gt_boxes)):
                    gt_label = gt_labels[j].item() if isinstance(gt_labels[j], torch.Tensor) else gt_labels[j]
                    
                    if pred_label == gt_label:
                        iou = self._compute_iou(
                            pred_boxes[i].unsqueeze(0),
                            gt_boxes[j].unsqueeze(0),
                        )
                        best_iou = max(best_iou, iou)
                
                if best_iou > 0:
                    ious.append(best_iou)
        
        return sum(ious) / len(ious) if ious else 0.0
    
    def predict(
        self,
        inputs: torch.Tensor,
    ) -> List[Dict[str, torch.Tensor]]:
        """Run inference on inputs."""
        self.model.eval()
        inputs = inputs.to(self.device)
        
        with torch.no_grad():
            with autocast(enabled=self.config.training.mixed_precision and self.device.type == "cuda"):
                predictions = self.model(inputs)
        
        # Apply confidence threshold
        filtered_predictions = []
        for pred in predictions:
            scores = pred.get("scores", torch.empty(0))
            mask = scores >= self.conf_threshold
            
            filtered_pred = {
                "boxes": pred["boxes"][mask],
                "scores": scores[mask],
                "labels": pred["labels"][mask],
            }
            filtered_predictions.append(filtered_pred)
        
        return filtered_predictions
    
    def predict_with_nms(
        self,
        inputs: torch.Tensor,
        conf_threshold: float = 0.5,
        nms_threshold: float = 0.5,
    ) -> List[Dict[str, torch.Tensor]]:
        """Run inference with NMS."""
        from torchvision.ops import nms
        
        predictions = self.predict(inputs)
        
        results = []
        for pred in predictions:
            boxes = pred["boxes"]
            scores = pred["scores"]
            labels = pred["labels"]
            
            # Filter by confidence
            mask = scores >= conf_threshold
            boxes = boxes[mask]
            scores = scores[mask]
            labels = labels[mask]
            
            # Apply NMS per class
            keep_indices = []
            for cls_id in labels.unique():
                cls_mask = labels == cls_id
                cls_boxes = boxes[cls_mask]
                cls_scores = scores[cls_mask]
                
                keep = nms(cls_boxes, cls_scores, nms_threshold)
                indices = torch.where(cls_mask)[0][keep]
                keep_indices.append(indices)
            
            if keep_indices:
                keep = torch.cat(keep_indices)
                results.append({
                    "boxes": boxes[keep],
                    "scores": scores[keep],
                    "labels": labels[keep],
                })
            else:
                results.append({
                    "boxes": torch.empty(0, 4),
                    "scores": torch.empty(0),
                    "labels": torch.empty(0, dtype=torch.long),
                })
        
        return results
    
    def get_primary_metric(self) -> Tuple[str, str]:
        """Get primary metric for model selection."""
        return ("mAP50", "max")

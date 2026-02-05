"""
Classification task implementation.

Provides training, evaluation, and inference logic for
image classification models.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

from mlcli.core.config import ExperimentConfig
from mlcli.core.registry import register_task
from mlcli.tasks.base import BaseTask


@register_task("classification")
class ClassificationTask(BaseTask):
    """
    Task implementation for image classification.
    
    Supports multi-class classification with cross-entropy loss,
    label smoothing, and various evaluation metrics.
    """
    
    def __init__(
        self,
        config: ExperimentConfig,
        model: Optional[nn.Module] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__(config, model, device)
        
        self.num_classes = config.model.num_classes
        self.label_smoothing = config.training.label_smoothing
        self.use_mixup = config.training.training.get("mixup_alpha", 0) > 0 if hasattr(config.training, "training") else False
    
    def build_criterion(self) -> nn.Module:
        """Build cross-entropy loss with optional label smoothing."""
        return nn.CrossEntropyLoss(
            label_smoothing=self.label_smoothing,
        )
    
    def train_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None,
        scaler: Optional[torch.cuda.amp.GradScaler] = None,
    ) -> Dict[str, float]:
        """Execute a single training step."""
        images, targets = self.move_batch_to_device(batch)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass with optional mixed precision
        use_amp = scaler is not None and self.device.type == "cuda"
        
        with autocast(enabled=use_amp):
            outputs = self.model(images)
            loss = self.criterion(outputs, targets)
        
        # Backward pass
        if scaler is not None:
            scaler.scale(loss).backward()
            
            # Gradient clipping
            if self.config.training.gradient_clip:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.training.gradient_clip,
                )
            
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            
            if self.config.training.gradient_clip:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.training.gradient_clip,
                )
            
            optimizer.step()
        
        # Compute training accuracy
        with torch.no_grad():
            preds = outputs.argmax(dim=1)
            acc = (preds == targets).float().mean()
        
        return {
            "loss": loss.item(),
            "accuracy": acc.item(),
        }
    
    def eval_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
    ) -> Dict[str, Any]:
        """Execute a single evaluation step."""
        images, targets = self.move_batch_to_device(batch)
        
        with autocast(enabled=self.config.training.mixed_precision and self.device.type == "cuda"):
            outputs = self.model(images)
            loss = self.criterion(outputs, targets)
        
        probs = F.softmax(outputs, dim=1)
        preds = outputs.argmax(dim=1)
        
        return {
            "loss": loss.item(),
            "predictions": preds.cpu(),
            "probabilities": probs.cpu(),
            "targets": targets.cpu(),
        }
    
    def compute_metrics(
        self,
        outputs: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        """Compute classification metrics."""
        all_preds = torch.cat([o["predictions"] for o in outputs])
        all_targets = torch.cat([o["targets"] for o in outputs])
        all_probs = torch.cat([o["probabilities"] for o in outputs])
        total_loss = sum(o["loss"] for o in outputs) / len(outputs)
        
        # Accuracy
        accuracy = (all_preds == all_targets).float().mean().item()
        
        # Top-5 accuracy (if applicable)
        if self.num_classes >= 5:
            top5_preds = all_probs.topk(5, dim=1).indices
            top5_correct = (top5_preds == all_targets.unsqueeze(1)).any(dim=1)
            top5_accuracy = top5_correct.float().mean().item()
        else:
            top5_accuracy = accuracy
        
        # Per-class metrics
        precision, recall, f1 = self._compute_precision_recall_f1(all_preds, all_targets)
        
        return {
            "loss": total_loss,
            "accuracy": accuracy,
            "top5_accuracy": top5_accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    
    def _compute_precision_recall_f1(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> Tuple[float, float, float]:
        """Compute macro-averaged precision, recall, and F1."""
        precisions = []
        recalls = []
        
        for cls in range(self.num_classes):
            tp = ((predictions == cls) & (targets == cls)).sum().float()
            fp = ((predictions == cls) & (targets != cls)).sum().float()
            fn = ((predictions != cls) & (targets == cls)).sum().float()
            
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            
            precisions.append(precision.item())
            recalls.append(recall.item())
        
        macro_precision = sum(precisions) / len(precisions)
        macro_recall = sum(recalls) / len(recalls)
        
        if macro_precision + macro_recall > 0:
            macro_f1 = 2 * macro_precision * macro_recall / (macro_precision + macro_recall)
        else:
            macro_f1 = 0.0
        
        return macro_precision, macro_recall, macro_f1
    
    def predict(
        self,
        inputs: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Run inference on inputs."""
        self.model.eval()
        inputs = inputs.to(self.device)
        
        with torch.no_grad():
            with autocast(enabled=self.config.training.mixed_precision and self.device.type == "cuda"):
                outputs = self.model(inputs)
        
        probs = F.softmax(outputs, dim=1)
        preds = outputs.argmax(dim=1)
        confidences = probs.max(dim=1).values
        
        return {
            "predictions": preds,
            "probabilities": probs,
            "confidences": confidences,
        }
    
    def predict_with_labels(
        self,
        inputs: torch.Tensor,
        class_names: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Run inference and return labeled predictions."""
        results = self.predict(inputs)
        
        predictions = []
        for i in range(len(inputs)):
            pred_idx = results["predictions"][i].item()
            pred = {
                "class_id": pred_idx,
                "confidence": results["confidences"][i].item(),
                "probabilities": results["probabilities"][i].tolist(),
            }
            
            if class_names is not None:
                pred["class_name"] = class_names[pred_idx]
            
            predictions.append(pred)
        
        return predictions
    
    def get_primary_metric(self) -> Tuple[str, str]:
        """Get primary metric for model selection."""
        return ("accuracy", "max")

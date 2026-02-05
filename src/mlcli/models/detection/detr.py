"""
DETR (DEtection TRansformer) models.

Provides DETR object detection using transformers library.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from mlcli.core.config import ModelFamily, TaskType
from mlcli.core.registry import register_model
from mlcli.models.base import DetectionModel

# Try to import transformers for DETR
try:
    from transformers import DetrForObjectDetection, DetrImageProcessor
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class DETR(DetectionModel):
    """
    DETR (DEtection TRansformer) object detection model.
    
    End-to-end object detection with transformers using
    the HuggingFace transformers library.
    """
    
    VARIANTS = {
        "detr_resnet50": "facebook/detr-resnet-50",
        "detr_resnet101": "facebook/detr-resnet-101",
        "detr_resnet50_dc5": "facebook/detr-resnet-50-dc5",
    }
    
    def __init__(
        self,
        variant: str = "detr_resnet50",
        num_classes: int = 91,
        pretrained: bool = True,
        num_queries: int = 100,
        **kwargs: Any,
    ) -> None:
        """
        Initialize DETR model.
        
        Args:
            variant: Model variant name.
            num_classes: Number of detection classes.
            pretrained: Use pretrained weights.
            num_queries: Number of object queries.
        """
        super().__init__(num_classes, pretrained, **kwargs)
        
        self._box_format = "cxcywh"  # DETR uses center format
        self.num_queries = num_queries
        
        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError(
                "DETR requires transformers library. "
                "Install with: pip install transformers"
            )
        
        if variant not in self.VARIANTS:
            raise ValueError(f"Unknown variant: {variant}")
        
        model_name = self.VARIANTS[variant]
        
        if pretrained:
            self.model = DetrForObjectDetection.from_pretrained(model_name)
            self.processor = DetrImageProcessor.from_pretrained(model_name)
        else:
            from transformers import DetrConfig
            config = DetrConfig(
                num_labels=num_classes,
                num_queries=num_queries,
            )
            self.model = DetrForObjectDetection(config)
            self.processor = DetrImageProcessor()
        
        # Update number of classes if different
        if num_classes != 91 and pretrained:
            self._update_num_classes(num_classes)
    
    def _update_num_classes(self, num_classes: int) -> None:
        """Update model for different number of classes."""
        # Replace classification head
        hidden_dim = self.model.config.d_model
        self.model.class_labels_classifier = nn.Linear(hidden_dim, num_classes + 1)
    
    def forward(
        self,
        images: torch.Tensor,
        targets: Optional[List[Dict[str, torch.Tensor]]] = None,
    ) -> Union[Dict[str, torch.Tensor], List[Dict[str, torch.Tensor]]]:
        """
        Forward pass.
        
        Args:
            images: Input tensor of shape (B, C, H, W).
            targets: Optional list of target dicts for training.
            
        Returns:
            Training: Dict with losses.
            Inference: List of dicts with boxes, scores, labels.
        """
        if self.training and targets is not None:
            return self._forward_train(images, targets)
        else:
            return self._forward_inference(images)
    
    def _forward_train(
        self,
        images: torch.Tensor,
        targets: List[Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        """Training forward pass with loss computation."""
        # Convert targets to DETR format
        labels = []
        for target in targets:
            label = {
                "class_labels": target["labels"],
                "boxes": self._convert_boxes_to_cxcywh(target["boxes"], images.shape[-2:]),
            }
            labels.append(label)
        
        outputs = self.model(pixel_values=images, labels=labels)
        
        return {
            "loss": outputs.loss,
            "loss_ce": outputs.loss_dict.get("loss_ce", torch.tensor(0.0)),
            "loss_bbox": outputs.loss_dict.get("loss_bbox", torch.tensor(0.0)),
            "loss_giou": outputs.loss_dict.get("loss_giou", torch.tensor(0.0)),
        }
    
    def _forward_inference(
        self,
        images: torch.Tensor,
    ) -> List[Dict[str, torch.Tensor]]:
        """Inference forward pass."""
        outputs = self.model(pixel_values=images)
        
        # Get predictions
        logits = outputs.logits  # (batch_size, num_queries, num_classes + 1)
        boxes = outputs.pred_boxes  # (batch_size, num_queries, 4)
        
        # Convert to detection format
        batch_size = images.shape[0]
        image_size = images.shape[-2:]
        
        results = []
        for i in range(batch_size):
            # Get probabilities (excluding no-object class)
            probs = F.softmax(logits[i], dim=-1)
            scores, labels = probs[..., :-1].max(dim=-1)
            
            # Convert boxes from cxcywh normalized to xyxy absolute
            pred_boxes = self._convert_boxes_from_cxcywh(boxes[i], image_size)
            
            results.append({
                "boxes": pred_boxes,
                "scores": scores,
                "labels": labels,
            })
        
        return results
    
    def _convert_boxes_to_cxcywh(
        self,
        boxes: torch.Tensor,
        image_size: tuple[int, int],
    ) -> torch.Tensor:
        """Convert boxes from xyxy to normalized cxcywh."""
        h, w = image_size
        x1, y1, x2, y2 = boxes.unbind(-1)
        
        cx = ((x1 + x2) / 2) / w
        cy = ((y1 + y2) / 2) / h
        bw = (x2 - x1) / w
        bh = (y2 - y1) / h
        
        return torch.stack([cx, cy, bw, bh], dim=-1)
    
    def _convert_boxes_from_cxcywh(
        self,
        boxes: torch.Tensor,
        image_size: tuple[int, int],
    ) -> torch.Tensor:
        """Convert boxes from normalized cxcywh to xyxy."""
        h, w = image_size
        cx, cy, bw, bh = boxes.unbind(-1)
        
        x1 = (cx - bw / 2) * w
        y1 = (cy - bh / 2) * h
        x2 = (cx + bw / 2) * w
        y2 = (cy + bh / 2) * h
        
        return torch.stack([x1, y1, x2, y2], dim=-1)
    
    def postprocess(
        self,
        outputs: List[Dict[str, torch.Tensor]],
        conf_threshold: float = 0.5,
        nms_threshold: float = 0.5,
    ) -> List[Dict[str, torch.Tensor]]:
        """Post-process model outputs."""
        from torchvision.ops import nms
        
        results = []
        for output in outputs:
            scores = output["scores"]
            mask = scores >= conf_threshold
            
            if mask.sum() == 0:
                results.append({
                    "boxes": torch.empty(0, 4, device=scores.device),
                    "scores": torch.empty(0, device=scores.device),
                    "labels": torch.empty(0, dtype=torch.long, device=scores.device),
                })
                continue
            
            boxes = output["boxes"][mask]
            scores = scores[mask]
            labels = output["labels"][mask]
            
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
                    "boxes": torch.empty(0, 4, device=scores.device),
                    "scores": torch.empty(0, device=scores.device),
                    "labels": torch.empty(0, dtype=torch.long, device=scores.device),
                })
        
        return results
    
    def freeze_backbone(self) -> None:
        """Freeze backbone parameters."""
        for param in self.model.model.backbone.parameters():
            param.requires_grad = False


# Register DETR variants
for variant_name in DETR.VARIANTS.keys():
    @register_model(
        name=variant_name,
        task_type=TaskType.OBJECT_DETECTION,
        family=ModelFamily.TRANSFORMER,
    )
    class _DETRVariant(DETR):
        _variant = variant_name
        
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(variant=self._variant, **kwargs)
    
    _DETRVariant.__name__ = f"DETR_{variant_name}"
    _DETRVariant.__qualname__ = f"DETR_{variant_name}"
    globals()[f"DETR_{variant_name}"] = _DETRVariant

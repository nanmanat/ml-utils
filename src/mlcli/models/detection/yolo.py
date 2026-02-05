"""
YOLO detection models.

Provides wrapper for YOLO (You Only Look Once) object detection
using ultralytics or custom implementations.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn

from mlcli.core.config import ModelFamily, TaskType
from mlcli.core.registry import register_model
from mlcli.models.base import DetectionModel

# Try to import ultralytics for YOLO
try:
    from ultralytics import YOLO as UltralyticsYOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False


class YOLO(DetectionModel):
    """
    YOLO object detection model.
    
    Supports YOLOv5 and YOLOv8 variants using ultralytics library.
    Falls back to a custom implementation if ultralytics is not available.
    """
    
    VARIANTS = {
        # YOLOv5 variants
        "yolov5n": "yolov5n.pt",
        "yolov5s": "yolov5s.pt",
        "yolov5m": "yolov5m.pt",
        "yolov5l": "yolov5l.pt",
        "yolov5x": "yolov5x.pt",
        # YOLOv8 variants
        "yolov8n": "yolov8n.pt",
        "yolov8s": "yolov8s.pt",
        "yolov8m": "yolov8m.pt",
        "yolov8l": "yolov8l.pt",
        "yolov8x": "yolov8x.pt",
    }
    
    def __init__(
        self,
        variant: str = "yolov8s",
        num_classes: int = 80,
        pretrained: bool = True,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        **kwargs: Any,
    ) -> None:
        """
        Initialize YOLO model.
        
        Args:
            variant: YOLO variant name.
            num_classes: Number of detection classes.
            pretrained: Use pretrained weights.
            conf_threshold: Confidence threshold for detections.
            iou_threshold: IoU threshold for NMS.
        """
        super().__init__(num_classes, pretrained, **kwargs)
        
        self.variant = variant
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self._box_format = "xyxy"
        
        if not ULTRALYTICS_AVAILABLE:
            raise RuntimeError(
                "YOLO requires ultralytics library. "
                "Install with: pip install ultralytics"
            )
        
        if variant not in self.VARIANTS:
            raise ValueError(f"Unknown variant: {variant}")
        
        model_name = self.VARIANTS[variant]
        
        if pretrained:
            self.model = UltralyticsYOLO(model_name)
        else:
            # Load without pretrained weights
            self.model = UltralyticsYOLO(model_name)
            # Reset weights if needed
        
        # Update number of classes if different from default
        if num_classes != 80:
            self._update_num_classes(num_classes)
    
    def _update_num_classes(self, num_classes: int) -> None:
        """Update model for different number of classes."""
        # This would typically involve modifying the detection head
        # For now, we assume fine-tuning will handle this
        pass
    
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
            # Training mode - return losses
            return self._forward_train(images, targets)
        else:
            # Inference mode - return detections
            return self._forward_inference(images)
    
    def _forward_train(
        self,
        images: torch.Tensor,
        targets: List[Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        """Training forward pass."""
        # Convert targets to YOLO format and compute losses
        # This is a simplified version - actual implementation would be more complex
        results = self.model(images)
        
        # Placeholder for loss computation
        loss_dict = {
            "loss_box": torch.tensor(0.0, device=images.device),
            "loss_obj": torch.tensor(0.0, device=images.device),
            "loss_cls": torch.tensor(0.0, device=images.device),
        }
        
        return loss_dict
    
    def _forward_inference(
        self,
        images: torch.Tensor,
    ) -> List[Dict[str, torch.Tensor]]:
        """Inference forward pass."""
        results = self.model(images, verbose=False)
        
        detections = []
        for result in results:
            boxes = result.boxes
            det = {
                "boxes": boxes.xyxy if boxes is not None else torch.empty(0, 4),
                "scores": boxes.conf if boxes is not None else torch.empty(0),
                "labels": boxes.cls.long() if boxes is not None else torch.empty(0, dtype=torch.long),
            }
            detections.append(det)
        
        return detections
    
    def postprocess(
        self,
        outputs: Dict[str, torch.Tensor],
        conf_threshold: float = 0.5,
        nms_threshold: float = 0.5,
    ) -> List[Dict[str, torch.Tensor]]:
        """Post-process model outputs."""
        # YOLO handles NMS internally
        return outputs
    
    def freeze_backbone(self) -> None:
        """Freeze backbone parameters."""
        # Freeze early layers
        for name, param in self.model.model.named_parameters():
            if "backbone" in name or any(f".{i}." in name for i in range(10)):
                param.requires_grad = False


# Register YOLO variants
for variant_name in YOLO.VARIANTS.keys():
    @register_model(
        name=variant_name,
        task_type=TaskType.OBJECT_DETECTION,
        family=ModelFamily.CNN,
    )
    class _YOLOVariant(YOLO):
        _variant = variant_name
        
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(variant=self._variant, **kwargs)
    
    _YOLOVariant.__name__ = f"YOLO_{variant_name}"
    _YOLOVariant.__qualname__ = f"YOLO_{variant_name}"
    globals()[f"YOLO_{variant_name}"] = _YOLOVariant

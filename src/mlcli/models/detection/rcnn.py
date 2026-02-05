"""
Faster R-CNN detection models.

Provides Faster R-CNN object detection using torchvision.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    fasterrcnn_resnet50_fpn_v2,
    fasterrcnn_mobilenet_v3_large_fpn,
    fasterrcnn_mobilenet_v3_large_320_fpn,
    FasterRCNN_ResNet50_FPN_Weights,
    FasterRCNN_ResNet50_FPN_V2_Weights,
    FasterRCNN_MobileNet_V3_Large_FPN_Weights,
    FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from mlcli.core.config import ModelFamily, TaskType
from mlcli.core.registry import register_model
from mlcli.models.base import DetectionModel


class FasterRCNN(DetectionModel):
    """
    Faster R-CNN object detection model.
    
    Supports ResNet-50 and MobileNetV3 backbones.
    """
    
    VARIANTS = {
        "fasterrcnn_resnet50": (
            fasterrcnn_resnet50_fpn,
            FasterRCNN_ResNet50_FPN_Weights.DEFAULT,
        ),
        "fasterrcnn_resnet50_v2": (
            fasterrcnn_resnet50_fpn_v2,
            FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT,
        ),
        "fasterrcnn_mobilenet": (
            fasterrcnn_mobilenet_v3_large_fpn,
            FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT,
        ),
        "fasterrcnn_mobilenet_320": (
            fasterrcnn_mobilenet_v3_large_320_fpn,
            FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT,
        ),
    }
    
    def __init__(
        self,
        variant: str = "fasterrcnn_resnet50",
        num_classes: int = 91,
        pretrained: bool = True,
        pretrained_backbone: bool = True,
        trainable_backbone_layers: int = 3,
        min_size: int = 800,
        max_size: int = 1333,
        **kwargs: Any,
    ) -> None:
        """
        Initialize Faster R-CNN model.
        
        Args:
            variant: Model variant name.
            num_classes: Number of detection classes (including background).
            pretrained: Use pretrained COCO weights.
            pretrained_backbone: Use pretrained backbone.
            trainable_backbone_layers: Number of trainable backbone layers.
            min_size: Minimum image size.
            max_size: Maximum image size.
        """
        super().__init__(num_classes, pretrained, **kwargs)
        
        self._box_format = "xyxy"
        
        if variant not in self.VARIANTS:
            raise ValueError(f"Unknown variant: {variant}")
        
        model_fn, weights = self.VARIANTS[variant]
        
        # Create model
        if pretrained:
            self.model = model_fn(
                weights=weights,
                trainable_backbone_layers=trainable_backbone_layers,
                min_size=min_size,
                max_size=max_size,
            )
        else:
            self.model = model_fn(
                weights=None,
                weights_backbone=pretrained_backbone,
                trainable_backbone_layers=trainable_backbone_layers,
                min_size=min_size,
                max_size=max_size,
            )
        
        # Replace box predictor for custom number of classes
        if num_classes != 91:
            in_features = self.model.roi_heads.box_predictor.cls_score.in_features
            self.model.roi_heads.box_predictor = FastRCNNPredictor(
                in_features, num_classes
            )
    
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
        # Convert tensor to list of images (required by torchvision)
        if isinstance(images, torch.Tensor):
            images = [img for img in images]
        
        if self.training:
            if targets is None:
                raise ValueError("Targets required for training")
            return self.model(images, targets)
        else:
            return self.model(images)
    
    def postprocess(
        self,
        outputs: List[Dict[str, torch.Tensor]],
        conf_threshold: float = 0.5,
        nms_threshold: float = 0.5,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Post-process model outputs.
        
        Args:
            outputs: Raw model outputs.
            conf_threshold: Confidence threshold.
            nms_threshold: Not used (NMS done internally).
            
        Returns:
            Filtered detection dicts.
        """
        results = []
        for output in outputs:
            mask = output["scores"] >= conf_threshold
            results.append({
                "boxes": output["boxes"][mask],
                "scores": output["scores"][mask],
                "labels": output["labels"][mask],
            })
        return results
    
    def freeze_backbone(self) -> None:
        """Freeze backbone parameters."""
        for param in self.model.backbone.parameters():
            param.requires_grad = False


# Register Faster R-CNN variants
for variant_name in FasterRCNN.VARIANTS.keys():
    @register_model(
        name=variant_name,
        task_type=TaskType.OBJECT_DETECTION,
        family=ModelFamily.CNN,
    )
    class _FasterRCNNVariant(FasterRCNN):
        _variant = variant_name
        
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(variant=self._variant, **kwargs)
    
    _FasterRCNNVariant.__name__ = f"FasterRCNN_{variant_name}"
    _FasterRCNNVariant.__qualname__ = f"FasterRCNN_{variant_name}"
    globals()[f"FasterRCNN_{variant_name}"] = _FasterRCNNVariant

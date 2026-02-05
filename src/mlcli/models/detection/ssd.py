"""
SSD (Single Shot MultiBox Detector) models.

Provides SSD object detection using torchvision.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
from torchvision.models.detection import (
    ssd300_vgg16,
    ssdlite320_mobilenet_v3_large,
    SSD300_VGG16_Weights,
    SSDLite320_MobileNet_V3_Large_Weights,
)
from torchvision.models.detection.ssd import SSDHead

from mlcli.core.config import ModelFamily, TaskType
from mlcli.core.registry import register_model
from mlcli.models.base import DetectionModel


class SSD(DetectionModel):
    """
    SSD (Single Shot MultiBox Detector) object detection model.
    
    Supports SSD300 with VGG16 backbone and SSDLite with MobileNetV3.
    """
    
    VARIANTS = {
        "ssd300": (
            ssd300_vgg16,
            SSD300_VGG16_Weights.DEFAULT,
            300,
        ),
        "ssd512": (
            ssd300_vgg16,  # Use same model, resize input
            SSD300_VGG16_Weights.DEFAULT,
            512,
        ),
        "ssdlite320": (
            ssdlite320_mobilenet_v3_large,
            SSDLite320_MobileNet_V3_Large_Weights.DEFAULT,
            320,
        ),
    }
    
    def __init__(
        self,
        variant: str = "ssd300",
        num_classes: int = 91,
        pretrained: bool = True,
        **kwargs: Any,
    ) -> None:
        """
        Initialize SSD model.
        
        Args:
            variant: Model variant name.
            num_classes: Number of detection classes.
            pretrained: Use pretrained weights.
        """
        super().__init__(num_classes, pretrained, **kwargs)
        
        self._box_format = "xyxy"
        
        if variant not in self.VARIANTS:
            raise ValueError(f"Unknown variant: {variant}")
        
        model_fn, weights, self.input_size = self.VARIANTS[variant]
        
        if pretrained:
            self.model = model_fn(weights=weights)
        else:
            self.model = model_fn(weights=None)
        
        # Replace head for custom number of classes
        if num_classes != 91:
            self._replace_head(num_classes)
    
    def _replace_head(self, num_classes: int) -> None:
        """Replace detection head for custom number of classes."""
        # Get anchor configuration from existing head
        anchor_generator = self.model.anchor_generator
        num_anchors = anchor_generator.num_anchors_per_location()
        
        # Get in_channels from backbone
        in_channels = [512, 1024, 512, 256, 256, 256]  # Default for VGG backbone
        
        # Create new head
        self.model.head = SSDHead(
            in_channels=in_channels,
            num_anchors=num_anchors,
            num_classes=num_classes,
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
        # Convert tensor to list of images
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
        """Post-process model outputs."""
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


# Register SSD variants
for variant_name in SSD.VARIANTS.keys():
    @register_model(
        name=variant_name,
        task_type=TaskType.OBJECT_DETECTION,
        family=ModelFamily.CNN,
    )
    class _SSDVariant(SSD):
        _variant = variant_name
        
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(variant=self._variant, **kwargs)
    
    _SSDVariant.__name__ = f"SSD_{variant_name}"
    _SSDVariant.__qualname__ = f"SSD_{variant_name}"
    globals()[f"SSD_{variant_name}"] = _SSDVariant

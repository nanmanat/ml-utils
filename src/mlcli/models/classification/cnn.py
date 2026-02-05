"""
CNN-based classification models.

Implements ResNet, EfficientNet, and MobileNet families
with automatic registration to the model registry.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import (
    ResNet18_Weights,
    ResNet34_Weights,
    ResNet50_Weights,
    ResNet101_Weights,
    ResNet152_Weights,
    EfficientNet_B0_Weights,
    EfficientNet_B1_Weights,
    EfficientNet_B2_Weights,
    EfficientNet_B3_Weights,
    EfficientNet_B4_Weights,
    EfficientNet_B5_Weights,
    EfficientNet_B6_Weights,
    EfficientNet_B7_Weights,
    MobileNet_V2_Weights,
    MobileNet_V3_Small_Weights,
    MobileNet_V3_Large_Weights,
)

from mlcli.core.config import ModelFamily, TaskType
from mlcli.core.registry import register_model, MODEL_REGISTRY
from mlcli.models.base import ClassificationModel


class ResNet(ClassificationModel):
    """
    ResNet model family for image classification.
    
    Supports ResNet18, ResNet34, ResNet50, ResNet101, and ResNet152.
    """
    
    VARIANTS = {
        "resnet18": (models.resnet18, ResNet18_Weights.DEFAULT, 512),
        "resnet34": (models.resnet34, ResNet34_Weights.DEFAULT, 512),
        "resnet50": (models.resnet50, ResNet50_Weights.DEFAULT, 2048),
        "resnet101": (models.resnet101, ResNet101_Weights.DEFAULT, 2048),
        "resnet152": (models.resnet152, ResNet152_Weights.DEFAULT, 2048),
    }
    
    def __init__(
        self,
        variant: str = "resnet50",
        num_classes: int = 1000,
        pretrained: bool = True,
        dropout: float = 0.0,
        **kwargs: Any,
    ) -> None:
        """
        Initialize ResNet model.
        
        Args:
            variant: ResNet variant name.
            num_classes: Number of output classes.
            pretrained: Use pretrained ImageNet weights.
            dropout: Dropout rate before classifier.
        """
        super().__init__(num_classes, pretrained, dropout, **kwargs)
        
        if variant not in self.VARIANTS:
            raise ValueError(f"Unknown variant: {variant}. Choose from {list(self.VARIANTS.keys())}")
        
        model_fn, weights, feature_dim = self.VARIANTS[variant]
        self._set_feature_dim(feature_dim)
        
        # Load pretrained model
        if pretrained:
            self.backbone = model_fn(weights=weights)
        else:
            self.backbone = model_fn(weights=None)
        
        # Replace classifier
        self.backbone.fc = nn.Identity()
        
        # Build classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(feature_dim, num_classes),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        features = self.backbone(x)
        return self.classifier(features)
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before classifier."""
        return self.backbone(x)
    
    def freeze_backbone(self) -> None:
        """Freeze backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def get_parameter_groups(
        self,
        lr: float,
        weight_decay: float,
        lr_mult_backbone: float = 0.1,
    ) -> List[Dict[str, Any]]:
        """Get parameter groups with different LRs."""
        return [
            {"params": self.backbone.parameters(), "lr": lr * lr_mult_backbone, "weight_decay": weight_decay},
            {"params": self.classifier.parameters(), "lr": lr, "weight_decay": weight_decay},
        ]


def _create_resnet_variant(variant: str):
    """Factory function to create ResNet variant class with proper closure."""
    class ResNetVariant(ResNet):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(variant=variant, **kwargs)
    
    ResNetVariant.__name__ = f"ResNet_{variant}"
    ResNetVariant.__qualname__ = f"ResNet_{variant}"
    return ResNetVariant


# Register ResNet variants
for _variant_name in ResNet.VARIANTS.keys():
    _cls = _create_resnet_variant(_variant_name)
    MODEL_REGISTRY.register_class(
        _cls,
        name=_variant_name,
        task_type=TaskType.CLASSIFICATION,
        family=ModelFamily.CNN,
    )
    globals()[f"ResNet_{_variant_name}"] = _cls


class EfficientNet(ClassificationModel):
    """
    EfficientNet model family for image classification.
    
    Supports EfficientNet-B0 through B7.
    """
    
    VARIANTS = {
        "efficientnet_b0": (models.efficientnet_b0, EfficientNet_B0_Weights.DEFAULT, 1280),
        "efficientnet_b1": (models.efficientnet_b1, EfficientNet_B1_Weights.DEFAULT, 1280),
        "efficientnet_b2": (models.efficientnet_b2, EfficientNet_B2_Weights.DEFAULT, 1408),
        "efficientnet_b3": (models.efficientnet_b3, EfficientNet_B3_Weights.DEFAULT, 1536),
        "efficientnet_b4": (models.efficientnet_b4, EfficientNet_B4_Weights.DEFAULT, 1792),
        "efficientnet_b5": (models.efficientnet_b5, EfficientNet_B5_Weights.DEFAULT, 2048),
        "efficientnet_b6": (models.efficientnet_b6, EfficientNet_B6_Weights.DEFAULT, 2304),
        "efficientnet_b7": (models.efficientnet_b7, EfficientNet_B7_Weights.DEFAULT, 2560),
    }
    
    def __init__(
        self,
        variant: str = "efficientnet_b0",
        num_classes: int = 1000,
        pretrained: bool = True,
        dropout: float = 0.2,
        **kwargs: Any,
    ) -> None:
        """Initialize EfficientNet model."""
        super().__init__(num_classes, pretrained, dropout, **kwargs)
        
        if variant not in self.VARIANTS:
            raise ValueError(f"Unknown variant: {variant}")
        
        model_fn, weights, feature_dim = self.VARIANTS[variant]
        self._set_feature_dim(feature_dim)
        
        if pretrained:
            self.backbone = model_fn(weights=weights)
        else:
            self.backbone = model_fn(weights=None)
        
        # Replace classifier
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(feature_dim, num_classes),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.backbone(x)
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before classifier."""
        x = self.backbone.features(x)
        x = self.backbone.avgpool(x)
        return torch.flatten(x, 1)
    
    def freeze_backbone(self) -> None:
        """Freeze backbone parameters."""
        for param in self.backbone.features.parameters():
            param.requires_grad = False


def _create_efficientnet_variant(variant: str):
    """Factory function to create EfficientNet variant class with proper closure."""
    class EfficientNetVariant(EfficientNet):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(variant=variant, **kwargs)
    
    EfficientNetVariant.__name__ = f"EfficientNet_{variant}"
    EfficientNetVariant.__qualname__ = f"EfficientNet_{variant}"
    return EfficientNetVariant


# Register EfficientNet variants
for _variant_name in EfficientNet.VARIANTS.keys():
    _cls = _create_efficientnet_variant(_variant_name)
    MODEL_REGISTRY.register_class(
        _cls,
        name=_variant_name,
        task_type=TaskType.CLASSIFICATION,
        family=ModelFamily.CNN,
    )
    globals()[f"EfficientNet_{_variant_name}"] = _cls


class MobileNet(ClassificationModel):
    """
    MobileNet model family for efficient image classification.
    
    Supports MobileNetV2 and MobileNetV3 (small/large).
    """
    
    VARIANTS = {
        "mobilenet_v2": (models.mobilenet_v2, MobileNet_V2_Weights.DEFAULT, 1280),
        "mobilenet_v3_small": (models.mobilenet_v3_small, MobileNet_V3_Small_Weights.DEFAULT, 576),
        "mobilenet_v3_large": (models.mobilenet_v3_large, MobileNet_V3_Large_Weights.DEFAULT, 960),
    }
    
    def __init__(
        self,
        variant: str = "mobilenet_v2",
        num_classes: int = 1000,
        pretrained: bool = True,
        dropout: float = 0.2,
        **kwargs: Any,
    ) -> None:
        """Initialize MobileNet model."""
        super().__init__(num_classes, pretrained, dropout, **kwargs)
        
        if variant not in self.VARIANTS:
            raise ValueError(f"Unknown variant: {variant}")
        
        model_fn, weights, feature_dim = self.VARIANTS[variant]
        self._set_feature_dim(feature_dim)
        
        if pretrained:
            self.backbone = model_fn(weights=weights)
        else:
            self.backbone = model_fn(weights=None)
        
        # Replace classifier
        if "v3" in variant:
            self.backbone.classifier[-1] = nn.Linear(feature_dim, num_classes)
        else:
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(feature_dim, num_classes),
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.backbone(x)
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before classifier."""
        x = self.backbone.features(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        return torch.flatten(x, 1)
    
    def freeze_backbone(self) -> None:
        """Freeze backbone parameters."""
        for param in self.backbone.features.parameters():
            param.requires_grad = False


def _create_mobilenet_variant(variant: str):
    """Factory function to create MobileNet variant class with proper closure."""
    class MobileNetVariant(MobileNet):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(variant=variant, **kwargs)
    
    MobileNetVariant.__name__ = f"MobileNet_{variant}"
    MobileNetVariant.__qualname__ = f"MobileNet_{variant}"
    return MobileNetVariant


# Register MobileNet variants
for _variant_name in MobileNet.VARIANTS.keys():
    _cls = _create_mobilenet_variant(_variant_name)
    MODEL_REGISTRY.register_class(
        _cls,
        name=_variant_name,
        task_type=TaskType.CLASSIFICATION,
        family=ModelFamily.CNN,
    )
    globals()[f"MobileNet_{_variant_name}"] = _cls

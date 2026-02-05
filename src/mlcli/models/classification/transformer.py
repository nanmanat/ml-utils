"""
Transformer-based classification models.

Implements Vision Transformer (ViT) and Swin Transformer
with automatic registration to the model registry.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from mlcli.core.config import ModelFamily, TaskType
from mlcli.core.registry import register_model, MODEL_REGISTRY
from mlcli.models.base import ClassificationModel

# Try to import timm for transformer models
try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False

# Fallback to torchvision for basic ViT support
try:
    from torchvision.models import (
        vit_b_16,
        vit_b_32,
        vit_l_16,
        ViT_B_16_Weights,
        ViT_B_32_Weights,
        ViT_L_16_Weights,
        swin_t,
        swin_s,
        swin_b,
        Swin_T_Weights,
        Swin_S_Weights,
        Swin_B_Weights,
    )
    TORCHVISION_VIT_AVAILABLE = True
except ImportError:
    TORCHVISION_VIT_AVAILABLE = False


class VisionTransformer(ClassificationModel):
    """
    Vision Transformer (ViT) for image classification.
    
    Supports various ViT configurations using either timm or torchvision.
    """
    
    # Torchvision variants
    TORCHVISION_VARIANTS = {
        "vit_base_patch16": (vit_b_16, ViT_B_16_Weights.DEFAULT, 768) if TORCHVISION_VIT_AVAILABLE else None,
        "vit_base_patch32": (vit_b_32, ViT_B_32_Weights.DEFAULT, 768) if TORCHVISION_VIT_AVAILABLE else None,
        "vit_large_patch16": (vit_l_16, ViT_L_16_Weights.DEFAULT, 1024) if TORCHVISION_VIT_AVAILABLE else None,
    }
    
    # Timm variants (more comprehensive)
    TIMM_VARIANTS = {
        "vit_base_patch16": ("vit_base_patch16_224", 768),
        "vit_base_patch32": ("vit_base_patch32_224", 768),
        "vit_large_patch16": ("vit_large_patch16_224", 1024),
        "vit_small_patch16": ("vit_small_patch16_224", 384),
        "vit_tiny_patch16": ("vit_tiny_patch16_224", 192),
    }
    
    def __init__(
        self,
        variant: str = "vit_base_patch16",
        num_classes: int = 1000,
        pretrained: bool = True,
        dropout: float = 0.0,
        use_timm: bool = True,
        **kwargs: Any,
    ) -> None:
        """
        Initialize Vision Transformer model.
        
        Args:
            variant: ViT variant name.
            num_classes: Number of output classes.
            pretrained: Use pretrained weights.
            dropout: Dropout rate.
            use_timm: Prefer timm library if available.
        """
        super().__init__(num_classes, pretrained, dropout, **kwargs)
        
        self.use_timm = use_timm and TIMM_AVAILABLE
        
        if self.use_timm:
            self._init_timm_model(variant, num_classes, pretrained, dropout)
        elif TORCHVISION_VIT_AVAILABLE and variant in self.TORCHVISION_VARIANTS:
            self._init_torchvision_model(variant, num_classes, pretrained, dropout)
        else:
            raise RuntimeError(
                f"Cannot create ViT model '{variant}'. "
                f"Install timm: pip install timm"
            )
    
    def _init_timm_model(
        self,
        variant: str,
        num_classes: int,
        pretrained: bool,
        dropout: float,
    ) -> None:
        """Initialize using timm library."""
        if variant not in self.TIMM_VARIANTS:
            # Try direct timm model name
            timm_name = variant
            feature_dim = 768  # Default, will be updated
        else:
            timm_name, feature_dim = self.TIMM_VARIANTS[variant]
        
        self.backbone = timm.create_model(
            timm_name,
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=dropout,
        )
        
        # Get actual feature dimension
        if hasattr(self.backbone, "num_features"):
            feature_dim = self.backbone.num_features
        
        self._set_feature_dim(feature_dim)
    
    def _init_torchvision_model(
        self,
        variant: str,
        num_classes: int,
        pretrained: bool,
        dropout: float,
    ) -> None:
        """Initialize using torchvision."""
        model_fn, weights, feature_dim = self.TORCHVISION_VARIANTS[variant]
        self._set_feature_dim(feature_dim)
        
        if pretrained:
            self.backbone = model_fn(weights=weights)
        else:
            self.backbone = model_fn(weights=None)
        
        # Replace head
        self.backbone.heads = nn.Sequential(
            nn.Dropout(p=dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(feature_dim, num_classes),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.backbone(x)
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before classifier."""
        if self.use_timm:
            return self.backbone.forward_features(x)
        else:
            # Torchvision ViT
            x = self.backbone._process_input(x)
            n = x.shape[0]
            batch_class_token = self.backbone.class_token.expand(n, -1, -1)
            x = torch.cat([batch_class_token, x], dim=1)
            x = self.backbone.encoder(x)
            return x[:, 0]
    
    def freeze_backbone(self) -> None:
        """Freeze backbone except classifier."""
        for name, param in self.backbone.named_parameters():
            if "head" not in name and "classifier" not in name:
                param.requires_grad = False


class SwinTransformer(ClassificationModel):
    """
    Swin Transformer for image classification.
    
    Supports Swin-Tiny, Swin-Small, Swin-Base, and Swin-Large.
    """
    
    TORCHVISION_VARIANTS = {
        "swin_tiny": (swin_t, Swin_T_Weights.DEFAULT, 768) if TORCHVISION_VIT_AVAILABLE else None,
        "swin_small": (swin_s, Swin_S_Weights.DEFAULT, 768) if TORCHVISION_VIT_AVAILABLE else None,
        "swin_base": (swin_b, Swin_B_Weights.DEFAULT, 1024) if TORCHVISION_VIT_AVAILABLE else None,
    }
    
    TIMM_VARIANTS = {
        "swin_tiny": ("swin_tiny_patch4_window7_224", 768),
        "swin_small": ("swin_small_patch4_window7_224", 768),
        "swin_base": ("swin_base_patch4_window7_224", 1024),
        "swin_large": ("swin_large_patch4_window7_224", 1536),
    }
    
    def __init__(
        self,
        variant: str = "swin_tiny",
        num_classes: int = 1000,
        pretrained: bool = True,
        dropout: float = 0.0,
        use_timm: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize Swin Transformer model."""
        super().__init__(num_classes, pretrained, dropout, **kwargs)
        
        self.use_timm = use_timm and TIMM_AVAILABLE
        
        if self.use_timm and variant in self.TIMM_VARIANTS:
            self._init_timm_model(variant, num_classes, pretrained, dropout)
        elif TORCHVISION_VIT_AVAILABLE and variant in self.TORCHVISION_VARIANTS:
            self._init_torchvision_model(variant, num_classes, pretrained, dropout)
        else:
            raise RuntimeError(
                f"Cannot create Swin model '{variant}'. "
                f"Install timm: pip install timm"
            )
    
    def _init_timm_model(
        self,
        variant: str,
        num_classes: int,
        pretrained: bool,
        dropout: float,
    ) -> None:
        """Initialize using timm library."""
        timm_name, feature_dim = self.TIMM_VARIANTS[variant]
        self._set_feature_dim(feature_dim)
        
        self.backbone = timm.create_model(
            timm_name,
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=dropout,
        )
    
    def _init_torchvision_model(
        self,
        variant: str,
        num_classes: int,
        pretrained: bool,
        dropout: float,
    ) -> None:
        """Initialize using torchvision."""
        model_fn, weights, feature_dim = self.TORCHVISION_VARIANTS[variant]
        self._set_feature_dim(feature_dim)
        
        if pretrained:
            self.backbone = model_fn(weights=weights)
        else:
            self.backbone = model_fn(weights=None)
        
        # Replace head
        self.backbone.head = nn.Sequential(
            nn.Dropout(p=dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(feature_dim, num_classes),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.backbone(x)
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before classifier."""
        if self.use_timm:
            return self.backbone.forward_features(x)
        else:
            x = self.backbone.features(x)
            x = self.backbone.norm(x)
            x = self.backbone.permute(x)
            x = self.backbone.avgpool(x)
            return self.backbone.flatten(x)
    
    def freeze_backbone(self) -> None:
        """Freeze backbone except classifier."""
        for name, param in self.backbone.named_parameters():
            if "head" not in name and "classifier" not in name:
                param.requires_grad = False


def _create_vit_variant(variant: str):
    """Factory function to create ViT variant class with proper closure."""
    class ViTVariant(VisionTransformer):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(variant=variant, **kwargs)
    
    ViTVariant.__name__ = f"ViT_{variant}"
    ViTVariant.__qualname__ = f"ViT_{variant}"
    return ViTVariant


# Register ViT variants
for _variant_name in ["vit_base_patch16", "vit_base_patch32", "vit_large_patch16"]:
    _cls = _create_vit_variant(_variant_name)
    MODEL_REGISTRY.register_class(
        _cls,
        name=_variant_name,
        task_type=TaskType.CLASSIFICATION,
        family=ModelFamily.TRANSFORMER,
    )
    globals()[f"ViT_{_variant_name}"] = _cls


def _create_swin_variant(variant: str):
    """Factory function to create Swin variant class with proper closure."""
    class SwinVariant(SwinTransformer):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(variant=variant, **kwargs)
    
    SwinVariant.__name__ = f"Swin_{variant}"
    SwinVariant.__qualname__ = f"Swin_{variant}"
    return SwinVariant


# Register Swin variants
for _variant_name in ["swin_tiny", "swin_small", "swin_base", "swin_large"]:
    _cls = _create_swin_variant(_variant_name)
    MODEL_REGISTRY.register_class(
        _cls,
        name=_variant_name,
        task_type=TaskType.CLASSIFICATION,
        family=ModelFamily.TRANSFORMER,
    )
    globals()[f"Swin_{_variant_name}"] = _cls

"""
Base model interface and utilities.

Provides abstract base class for all models with common
functionality like weight initialization, feature extraction,
and checkpoint handling.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn


class BaseModel(nn.Module, ABC):
    """
    Abstract base class for all models.
    
    Provides common interface and utilities for classification
    and detection models.
    """
    
    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,
        dropout: float = 0.0,
        **kwargs: Any,
    ) -> None:
        """
        Initialize base model.
        
        Args:
            num_classes: Number of output classes.
            pretrained: Whether to use pretrained weights.
            dropout: Dropout rate.
            **kwargs: Additional arguments.
        """
        super().__init__()
        self.num_classes = num_classes
        self._pretrained = pretrained
        self._dropout = dropout
        self._feature_dim: Optional[int] = None
    
    @property
    def feature_dim(self) -> int:
        """Get feature dimension before final classifier."""
        if self._feature_dim is None:
            raise RuntimeError("Feature dimension not set. Call _set_feature_dim() in subclass.")
        return self._feature_dim
    
    def _set_feature_dim(self, dim: int) -> None:
        """Set the feature dimension."""
        self._feature_dim = dim
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            x: Input tensor.
            
        Returns:
            Model output (logits for classification, dict for detection).
        """
        pass
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features before the final classifier.
        
        Args:
            x: Input tensor.
            
        Returns:
            Feature tensor.
        """
        raise NotImplementedError("Subclass must implement extract_features()")
    
    def freeze_backbone(self) -> None:
        """Freeze backbone parameters for transfer learning."""
        raise NotImplementedError("Subclass must implement freeze_backbone()")
    
    def unfreeze_backbone(self) -> None:
        """Unfreeze backbone parameters."""
        for param in self.parameters():
            param.requires_grad = True
    
    def get_trainable_params(self) -> List[nn.Parameter]:
        """Get list of trainable parameters."""
        return [p for p in self.parameters() if p.requires_grad]
    
    def count_parameters(self, trainable_only: bool = True) -> int:
        """
        Count model parameters.
        
        Args:
            trainable_only: Only count trainable parameters.
            
        Returns:
            Number of parameters.
        """
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())
    
    def get_num_parameters(self, trainable_only: bool = False) -> int:
        """
        Get number of model parameters.
        
        Alias for count_parameters with different default.
        
        Args:
            trainable_only: Only count trainable parameters.
            
        Returns:
            Number of parameters.
        """
        return self.count_parameters(trainable_only=trainable_only)
    
    def get_parameter_groups(
        self,
        lr: float,
        weight_decay: float,
        lr_mult_backbone: float = 0.1,
    ) -> List[Dict[str, Any]]:
        """
        Get parameter groups with different learning rates.
        
        Useful for fine-tuning with lower LR on backbone.
        
        Args:
            lr: Base learning rate.
            weight_decay: Weight decay.
            lr_mult_backbone: LR multiplier for backbone.
            
        Returns:
            List of parameter group dicts.
        """
        # Default implementation - can be overridden by subclasses
        return [{"params": self.parameters(), "lr": lr, "weight_decay": weight_decay}]
    
    @classmethod
    def init_weights(cls, module: nn.Module) -> None:
        """
        Initialize module weights.
        
        Args:
            module: Module to initialize.
        """
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)


class ClassificationModel(BaseModel):
    """Base class for classification models."""
    
    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,
        dropout: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(num_classes, pretrained, dropout, **kwargs)
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for classification.
        
        Args:
            x: Input tensor of shape (B, C, H, W).
            
        Returns:
            Logits tensor of shape (B, num_classes).
        """
        pass


class DetectionModel(BaseModel):
    """Base class for object detection models."""
    
    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(num_classes, pretrained, **kwargs)
        self._box_format: str = "xyxy"  # or "xywh", "cxcywh"
    
    @property
    def box_format(self) -> str:
        """Get bounding box format."""
        return self._box_format
    
    @abstractmethod
    def forward(
        self,
        images: torch.Tensor,
        targets: Optional[List[Dict[str, torch.Tensor]]] = None,
    ) -> Union[Dict[str, torch.Tensor], List[Dict[str, torch.Tensor]]]:
        """
        Forward pass for detection.
        
        Args:
            images: Input tensor of shape (B, C, H, W).
            targets: Optional list of target dicts for training.
            
        Returns:
            Training: Dict with losses.
            Inference: List of dicts with boxes, scores, labels.
        """
        pass
    
    def postprocess(
        self,
        outputs: Dict[str, torch.Tensor],
        conf_threshold: float = 0.5,
        nms_threshold: float = 0.5,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Post-process model outputs.
        
        Args:
            outputs: Raw model outputs.
            conf_threshold: Confidence threshold.
            nms_threshold: NMS IoU threshold.
            
        Returns:
            List of detection dicts per image.
        """
        raise NotImplementedError("Subclass must implement postprocess()")

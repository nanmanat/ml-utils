"""
Data transforms and augmentation.

Provides configurable transforms for classification and detection
with support for standard and custom augmentation strategies.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from torchvision import transforms as T

from mlcli.core.config import TaskType


class AugmentationStrategy(str, Enum):
    """Predefined augmentation strategies."""
    
    NONE = "none"
    MINIMAL = "minimal"
    STANDARD = "standard"
    AGGRESSIVE = "aggressive"
    AUTOAUGMENT = "autoaugment"
    RANDAUGMENT = "randaugment"
    TRIVIALAUGMENT = "trivialaugment"


@dataclass
class TransformConfig:
    """Configuration for data transforms."""
    
    input_size: Tuple[int, int] = (224, 224)
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    interpolation: str = "bilinear"
    
    # Training augmentations
    random_crop: bool = True
    random_flip: bool = True
    color_jitter: bool = True
    random_rotation: float = 0.0
    random_erasing: bool = False
    mixup_alpha: float = 0.0
    cutmix_alpha: float = 0.0
    
    # AutoAugment settings
    auto_augment: Optional[str] = None  # "imagenet", "cifar10", "svhn"
    rand_augment_n: int = 2
    rand_augment_m: int = 9


def get_interpolation_mode(name: str) -> T.InterpolationMode:
    """Get interpolation mode from string."""
    modes = {
        "nearest": T.InterpolationMode.NEAREST,
        "bilinear": T.InterpolationMode.BILINEAR,
        "bicubic": T.InterpolationMode.BICUBIC,
        "lanczos": T.InterpolationMode.LANCZOS,
    }
    return modes.get(name.lower(), T.InterpolationMode.BILINEAR)


def get_classification_train_transform(
    input_size: Tuple[int, int] = (224, 224),
    strategy: Union[str, AugmentationStrategy] = AugmentationStrategy.STANDARD,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    **kwargs: Any,
) -> Callable:
    """
    Get training transforms for classification.
    
    Args:
        input_size: Target image size.
        strategy: Augmentation strategy.
        mean: Normalization mean.
        std: Normalization std.
        
    Returns:
        Transform function.
    """
    if isinstance(strategy, str):
        strategy = AugmentationStrategy(strategy.lower())
    
    transforms_list = []
    
    # Base resize/crop
    if strategy == AugmentationStrategy.NONE:
        transforms_list.append(T.Resize(input_size))
    else:
        transforms_list.append(T.RandomResizedCrop(input_size, scale=(0.08, 1.0)))
    
    # Strategy-specific augmentations
    if strategy == AugmentationStrategy.MINIMAL:
        transforms_list.append(T.RandomHorizontalFlip())
    
    elif strategy == AugmentationStrategy.STANDARD:
        transforms_list.extend([
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        ])
    
    elif strategy == AugmentationStrategy.AGGRESSIVE:
        transforms_list.extend([
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(p=0.1),
            T.RandomRotation(15),
            T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.15),
            T.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        ])
    
    elif strategy == AugmentationStrategy.AUTOAUGMENT:
        transforms_list.extend([
            T.RandomHorizontalFlip(),
            T.AutoAugment(policy=T.AutoAugmentPolicy.IMAGENET),
        ])
    
    elif strategy == AugmentationStrategy.RANDAUGMENT:
        transforms_list.extend([
            T.RandomHorizontalFlip(),
            T.RandAugment(
                num_ops=kwargs.get("rand_augment_n", 2),
                magnitude=kwargs.get("rand_augment_m", 9),
            ),
        ])
    
    elif strategy == AugmentationStrategy.TRIVIALAUGMENT:
        transforms_list.extend([
            T.RandomHorizontalFlip(),
            T.TrivialAugmentWide(),
        ])
    
    # Convert to tensor and normalize
    transforms_list.extend([
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])
    
    # Random erasing (after ToTensor)
    if strategy in [AugmentationStrategy.AGGRESSIVE, AugmentationStrategy.RANDAUGMENT]:
        transforms_list.append(T.RandomErasing(p=0.25))
    
    return T.Compose(transforms_list)


def get_classification_val_transform(
    input_size: Tuple[int, int] = (224, 224),
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    center_crop: bool = True,
    crop_ratio: float = 0.875,  # 224/256
    **kwargs: Any,
) -> Callable:
    """
    Get validation/test transforms for classification.
    
    Args:
        input_size: Target image size.
        mean: Normalization mean.
        std: Normalization std.
        center_crop: Whether to center crop.
        crop_ratio: Ratio for center crop.
        
    Returns:
        Transform function.
    """
    transforms_list = []
    
    if center_crop:
        resize_size = tuple(int(s / crop_ratio) for s in input_size)
        transforms_list.extend([
            T.Resize(resize_size),
            T.CenterCrop(input_size),
        ])
    else:
        transforms_list.append(T.Resize(input_size))
    
    transforms_list.extend([
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])
    
    return T.Compose(transforms_list)


def get_detection_train_transform(
    input_size: Tuple[int, int] = (640, 640),
    strategy: Union[str, AugmentationStrategy] = AugmentationStrategy.STANDARD,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    **kwargs: Any,
) -> Callable:
    """
    Get training transforms for object detection.
    
    Note: Detection transforms need to handle both images and bounding boxes.
    This returns a basic transform - use specialized detection transforms
    for production.
    
    Args:
        input_size: Target image size.
        strategy: Augmentation strategy.
        mean: Normalization mean.
        std: Normalization std.
        
    Returns:
        Transform function.
    """
    # For detection, we use simpler transforms as boxes need special handling
    transforms_list = [
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ]
    
    return T.Compose(transforms_list)


def get_detection_val_transform(
    input_size: Tuple[int, int] = (640, 640),
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    **kwargs: Any,
) -> Callable:
    """
    Get validation transforms for object detection.
    
    Args:
        input_size: Target image size.
        mean: Normalization mean.
        std: Normalization std.
        
    Returns:
        Transform function.
    """
    transforms_list = [
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ]
    
    return T.Compose(transforms_list)


def get_transforms(
    task_type: TaskType,
    input_size: Tuple[int, int] = (224, 224),
    strategy: Union[str, AugmentationStrategy] = AugmentationStrategy.STANDARD,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    **kwargs: Any,
) -> Tuple[Callable, Callable]:
    """
    Get train and validation transforms for a task type.
    
    Args:
        task_type: Type of ML task.
        input_size: Target image size.
        strategy: Augmentation strategy.
        mean: Normalization mean.
        std: Normalization std.
        
    Returns:
        Tuple of (train_transform, val_transform).
    """
    if task_type == TaskType.CLASSIFICATION:
        train_transform = get_classification_train_transform(
            input_size=input_size,
            strategy=strategy,
            mean=mean,
            std=std,
            **kwargs,
        )
        val_transform = get_classification_val_transform(
            input_size=input_size,
            mean=mean,
            std=std,
            **kwargs,
        )
    elif task_type == TaskType.OBJECT_DETECTION:
        train_transform = get_detection_train_transform(
            input_size=input_size,
            strategy=strategy,
            mean=mean,
            std=std,
            **kwargs,
        )
        val_transform = get_detection_val_transform(
            input_size=input_size,
            mean=mean,
            std=std,
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown task type: {task_type}")
    
    return train_transform, val_transform


class Mixup:
    """Mixup augmentation for training."""
    
    def __init__(self, alpha: float = 1.0, num_classes: int = 1000) -> None:
        self.alpha = alpha
        self.num_classes = num_classes
    
    def __call__(
        self,
        images: torch.Tensor,
        targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply mixup to batch."""
        if self.alpha <= 0:
            return images, targets
        
        lam = torch.distributions.Beta(self.alpha, self.alpha).sample()
        batch_size = images.size(0)
        index = torch.randperm(batch_size, device=images.device)
        
        mixed_images = lam * images + (1 - lam) * images[index]
        
        # One-hot encode targets
        targets_onehot = torch.zeros(
            batch_size, self.num_classes, device=targets.device
        )
        targets_onehot.scatter_(1, targets.unsqueeze(1), 1)
        
        targets_shuffled = targets_onehot[index]
        mixed_targets = lam * targets_onehot + (1 - lam) * targets_shuffled
        
        return mixed_images, mixed_targets


class CutMix:
    """CutMix augmentation for training."""
    
    def __init__(self, alpha: float = 1.0, num_classes: int = 1000) -> None:
        self.alpha = alpha
        self.num_classes = num_classes
    
    def __call__(
        self,
        images: torch.Tensor,
        targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply cutmix to batch."""
        if self.alpha <= 0:
            return images, targets
        
        lam = torch.distributions.Beta(self.alpha, self.alpha).sample()
        batch_size = images.size(0)
        index = torch.randperm(batch_size, device=images.device)
        
        _, _, h, w = images.shape
        
        # Get random box
        cut_rat = torch.sqrt(1 - lam)
        cut_w = int(w * cut_rat)
        cut_h = int(h * cut_rat)
        
        cx = torch.randint(0, w, (1,)).item()
        cy = torch.randint(0, h, (1,)).item()
        
        x1 = max(0, cx - cut_w // 2)
        y1 = max(0, cy - cut_h // 2)
        x2 = min(w, cx + cut_w // 2)
        y2 = min(h, cy + cut_h // 2)
        
        # Apply cutmix
        mixed_images = images.clone()
        mixed_images[:, :, y1:y2, x1:x2] = images[index, :, y1:y2, x1:x2]
        
        # Adjust lambda based on actual box size
        lam = 1 - ((x2 - x1) * (y2 - y1) / (w * h))
        
        # One-hot encode targets
        targets_onehot = torch.zeros(
            batch_size, self.num_classes, device=targets.device
        )
        targets_onehot.scatter_(1, targets.unsqueeze(1), 1)
        
        targets_shuffled = targets_onehot[index]
        mixed_targets = lam * targets_onehot + (1 - lam) * targets_shuffled
        
        return mixed_images, mixed_targets

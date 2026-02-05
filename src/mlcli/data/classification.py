"""
Classification datasets.

Provides dataset implementations for image classification tasks.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from PIL import Image
from torch.utils.data import Dataset

from mlcli.core.registry import register_dataset
from mlcli.data.base import BaseDataset, DatasetInfo


@register_dataset("imagefolder", aliases=["folder", "image_folder"])
class ImageFolderDataset(BaseDataset):
    """
    Dataset for loading images from folder structure.
    
    Expected structure:
        root/
            class1/
                img1.jpg
                img2.jpg
            class2/
                img3.jpg
                img4.jpg
    """
    
    EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}
    
    def __init__(
        self,
        root: Union[str, Path],
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        cache_data: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize ImageFolder dataset."""
        self.samples: List[Tuple[Path, int]] = []
        self.classes: List[str] = []
        self.class_to_idx: Dict[str, int] = {}
        
        super().__init__(root, split, transform, target_transform, cache_data, **kwargs)
    
    def _load_dataset(self) -> None:
        """Load dataset from folder structure."""
        split_dir = self.root / self.split
        
        # Fallback to root if split directory doesn't exist
        if not split_dir.exists():
            split_dir = self.root
        
        # Get classes from subdirectories
        self.classes = sorted([
            d.name for d in split_dir.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        ])
        
        if not self.classes:
            raise ValueError(f"No class directories found in {split_dir}")
        
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Load all image paths
        self.samples = []
        for class_name in self.classes:
            class_dir = split_dir / class_name
            class_idx = self.class_to_idx[class_name]
            
            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() in self.EXTENSIONS:
                    self.samples.append((img_path, class_idx))
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        """Get sample by index."""
        img_path, target = self.samples[index]
        
        def load_fn():
            return Image.open(img_path).convert("RGB")
        
        image = self._get_cached(index, load_fn)
        
        if self.transform is not None:
            image = self.transform(image)
        
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return image, target
    
    @property
    def num_classes(self) -> int:
        return len(self.classes)
    
    @property
    def class_names(self) -> List[str]:
        return self.classes
    
    def get_sample_weights(self) -> torch.Tensor:
        """Get sample weights for weighted sampling."""
        class_counts = Counter(label for _, label in self.samples)
        total = len(self.samples)
        
        weights = []
        for _, label in self.samples:
            weight = total / (len(class_counts) * class_counts[label])
            weights.append(weight)
        
        return torch.tensor(weights, dtype=torch.float)
    
    def get_class_weights(self) -> torch.Tensor:
        """Get class weights for loss weighting."""
        class_counts = Counter(label for _, label in self.samples)
        total = len(self.samples)
        
        weights = []
        for i in range(len(self.classes)):
            count = class_counts.get(i, 1)
            weight = total / (len(self.classes) * count)
            weights.append(weight)
        
        return torch.tensor(weights, dtype=torch.float)
    
    def get_info(self) -> DatasetInfo:
        """Get dataset information."""
        class_distribution = Counter(label for _, label in self.samples)
        return DatasetInfo(
            name=f"ImageFolder-{self.root.name}",
            num_samples=len(self.samples),
            num_classes=len(self.classes),
            class_names=self.classes,
            class_distribution={
                self.classes[k]: v for k, v in class_distribution.items()
            },
        )


@register_dataset("classification", aliases=["clf"])
class ClassificationDataset(ImageFolderDataset):
    """Generic classification dataset with additional features."""
    
    def __init__(
        self,
        root: Union[str, Path],
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        cache_data: bool = False,
        metadata_file: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize classification dataset.
        
        Args:
            root: Root directory.
            split: Dataset split.
            transform: Input transform.
            target_transform: Target transform.
            cache_data: Cache data in memory.
            metadata_file: Optional JSON file with class mappings.
        """
        self.metadata_file = metadata_file
        self.metadata: Dict[str, Any] = {}
        
        super().__init__(root, split, transform, target_transform, cache_data, **kwargs)
    
    def _load_dataset(self) -> None:
        """Load dataset with optional metadata."""
        # Load metadata if provided
        if self.metadata_file:
            metadata_path = self.root / self.metadata_file
            if metadata_path.exists():
                with open(metadata_path) as f:
                    self.metadata = json.load(f)
                
                # Use metadata for class mappings
                if "classes" in self.metadata:
                    self.classes = self.metadata["classes"]
                    self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Fall back to folder-based loading
        if not self.classes:
            super()._load_dataset()
        else:
            # Load samples with predefined classes
            split_dir = self.root / self.split
            if not split_dir.exists():
                split_dir = self.root
            
            self.samples = []
            for class_name in self.classes:
                class_dir = split_dir / class_name
                if not class_dir.exists():
                    continue
                
                class_idx = self.class_to_idx[class_name]
                for img_path in class_dir.iterdir():
                    if img_path.suffix.lower() in self.EXTENSIONS:
                        self.samples.append((img_path, class_idx))


@register_dataset("cifar10")
class CIFAR10Dataset(BaseDataset):
    """CIFAR-10 dataset wrapper."""
    
    def __init__(
        self,
        root: Union[str, Path],
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = True,
        **kwargs: Any,
    ) -> None:
        from torchvision.datasets import CIFAR10
        
        self._dataset = CIFAR10(
            root=root,
            train=(split == "train"),
            transform=transform,
            target_transform=target_transform,
            download=download,
        )
        
        self.classes = self._dataset.classes
        self.class_to_idx = self._dataset.class_to_idx
    
    def _load_dataset(self) -> None:
        pass  # Handled by CIFAR10
    
    def __len__(self) -> int:
        return len(self._dataset)
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        return self._dataset[index]
    
    @property
    def num_classes(self) -> int:
        return 10
    
    @property
    def class_names(self) -> List[str]:
        return self.classes


@register_dataset("imagenet", aliases=["ilsvrc"])
class ImageNetDataset(ImageFolderDataset):
    """ImageNet (ILSVRC) dataset."""
    
    def __init__(
        self,
        root: Union[str, Path],
        split: str = "train",
        transform: Optional[Callable] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(root, split, transform, **kwargs)
    
    @property
    def num_classes(self) -> int:
        return 1000

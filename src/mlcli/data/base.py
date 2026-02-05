"""
Base dataset interface and utilities.

Provides abstract base class for all datasets with common
functionality for loading, caching, and preprocessing.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset


class BaseDataset(Dataset, ABC):
    """
    Abstract base class for all datasets.
    
    Provides common interface and utilities for classification
    and detection datasets.
    """
    
    def __init__(
        self,
        root: Union[str, Path],
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        cache_data: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        Initialize dataset.
        
        Args:
            root: Root directory of the dataset.
            split: Dataset split (train, val, test).
            transform: Transform to apply to inputs.
            target_transform: Transform to apply to targets.
            cache_data: Whether to cache data in memory.
        """
        super().__init__()
        self.root = Path(root).expanduser().resolve()
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.cache_data = cache_data
        
        # Data cache
        self._cache: Dict[int, Any] = {}
        
        # Validate root exists
        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root not found: {self.root}")
        
        # Load dataset
        self._load_dataset()
    
    @abstractmethod
    def _load_dataset(self) -> None:
        """Load dataset metadata and file paths."""
        pass
    
    @abstractmethod
    def __len__(self) -> int:
        """Return number of samples."""
        pass
    
    @abstractmethod
    def __getitem__(self, index: int) -> Any:
        """Get sample by index."""
        pass
    
    def _get_cached(self, index: int, load_fn: Callable) -> Any:
        """Get item from cache or load it."""
        if self.cache_data:
            if index not in self._cache:
                self._cache[index] = load_fn()
            return self._cache[index]
        return load_fn()
    
    def clear_cache(self) -> None:
        """Clear data cache."""
        self._cache.clear()
    
    @property
    def num_classes(self) -> int:
        """Return number of classes."""
        raise NotImplementedError
    
    @property
    def class_names(self) -> List[str]:
        """Return list of class names."""
        raise NotImplementedError
    
    def get_num_classes(self) -> int:
        """Get number of classes (method form)."""
        return self.num_classes
    
    def get_class_names(self) -> List[str]:
        """Get class names (method form)."""
        return self.class_names
    
    def get_sample_weights(self) -> torch.Tensor:
        """
        Get sample weights for weighted sampling.
        
        Useful for handling class imbalance.
        
        Returns:
            Tensor of sample weights.
        """
        raise NotImplementedError
    
    def get_class_weights(self) -> torch.Tensor:
        """
        Get class weights for loss weighting.
        
        Returns:
            Tensor of class weights.
        """
        raise NotImplementedError
    
    def split_dataset(
        self,
        val_ratio: float = 0.2,
        seed: int = 42,
    ) -> Tuple["BaseDataset", "BaseDataset"]:
        """
        Split dataset into train and validation sets.
        
        Args:
            val_ratio: Fraction of data for validation.
            seed: Random seed.
            
        Returns:
            Tuple of (train_dataset, val_dataset).
        """
        from torch.utils.data import random_split
        
        total_size = len(self)
        val_size = int(total_size * val_ratio)
        train_size = total_size - val_size
        
        generator = torch.Generator().manual_seed(seed)
        train_dataset, val_dataset = random_split(
            self,
            [train_size, val_size],
            generator=generator,
        )
        
        return train_dataset, val_dataset
    
    def create_dataloader(
        self,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 4,
        pin_memory: bool = True,
        drop_last: bool = False,
        **kwargs: Any,
    ) -> "torch.utils.data.DataLoader":
        """
        Create a DataLoader for this dataset.
        
        Args:
            batch_size: Batch size.
            shuffle: Whether to shuffle data.
            num_workers: Number of data loading workers.
            pin_memory: Pin memory for faster GPU transfer.
            drop_last: Drop last incomplete batch.
            **kwargs: Additional DataLoader arguments.
            
        Returns:
            DataLoader instance.
        """
        from torch.utils.data import DataLoader
        
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            **kwargs,
        )


class DatasetInfo:
    """Container for dataset metadata and statistics."""
    
    def __init__(
        self,
        name: str,
        num_samples: int,
        num_classes: int,
        class_names: List[str],
        class_distribution: Optional[Dict[str, int]] = None,
        image_size: Optional[Tuple[int, int]] = None,
        mean: Optional[Tuple[float, float, float]] = None,
        std: Optional[Tuple[float, float, float]] = None,
    ) -> None:
        self.name = name
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.class_names = class_names
        self.class_distribution = class_distribution or {}
        self.image_size = image_size
        self.mean = mean or (0.485, 0.456, 0.406)  # ImageNet defaults
        self.std = std or (0.229, 0.224, 0.225)
    
    def __repr__(self) -> str:
        return (
            f"DatasetInfo(name='{self.name}', "
            f"samples={self.num_samples}, "
            f"classes={self.num_classes})"
        )

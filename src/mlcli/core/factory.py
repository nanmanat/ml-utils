"""
Factory pattern for creating models, datasets, and tasks.

Provides a unified interface for instantiating components from
configuration with automatic registration lookup.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Type, TypeVar

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from mlcli.core.config import (
    DatasetConfig,
    ExperimentConfig,
    ModelConfig,
    ModelFamily,
    TaskType,
)
from mlcli.core.registry import (
    DATASET_REGISTRY,
    MODEL_REGISTRY,
    TASK_REGISTRY,
    Registry,
)

T = TypeVar("T")


class FactoryError(Exception):
    """Exception raised for factory-related errors."""
    pass


class BaseFactory(ABC):
    """Abstract base class for factories."""
    
    @classmethod
    @abstractmethod
    def create(cls, config: Any, **kwargs: Any) -> Any:
        """Create an instance from configuration."""
        pass
    
    @classmethod
    @abstractmethod
    def get_registry(cls) -> Registry:
        """Get the registry for this factory."""
        pass


class ModelFactory(BaseFactory):
    """
    Factory for creating model instances.
    
    Supports creating models from configuration with automatic
    architecture lookup and weight initialization.
    """
    
    @classmethod
    def get_registry(cls) -> Registry:
        """Get the model registry."""
        return MODEL_REGISTRY
    
    @classmethod
    def create(
        cls,
        config: ModelConfig,
        device: Optional[torch.device] = None,
        **kwargs: Any,
    ) -> nn.Module:
        """
        Create a model instance from configuration.
        
        Args:
            config: Model configuration.
            device: Target device.
            **kwargs: Additional arguments passed to model constructor.
            
        Returns:
            Instantiated model.
            
        Raises:
            FactoryError: If model architecture not found.
        """
        try:
            model_cls = MODEL_REGISTRY.get(config.architecture)
        except Exception as e:
            raise FactoryError(
                f"Failed to find model '{config.architecture}': {e}"
            ) from e
        
        # Build model kwargs from config
        model_kwargs = {
            "num_classes": config.num_classes,
            "pretrained": config.pretrained,
            **config.custom_params,
            **kwargs,
        }
        
        try:
            model = model_cls(**model_kwargs)
        except TypeError as e:
            # Try without pretrained if not supported
            model_kwargs.pop("pretrained", None)
            model = model_cls(**model_kwargs)
        
        if device is not None:
            model = model.to(device)
        
        return model
    
    @classmethod
    def create_from_checkpoint(
        cls,
        checkpoint_path: str,
        config: Optional[ModelConfig] = None,
        device: Optional[torch.device] = None,
        strict: bool = True,
    ) -> nn.Module:
        """
        Create a model and load weights from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file.
            config: Model configuration (loaded from checkpoint if None).
            device: Target device.
            strict: Strict state dict loading.
            
        Returns:
            Model with loaded weights.
        """
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        
        if config is None:
            if "config" not in checkpoint:
                raise FactoryError(
                    "No config in checkpoint and no config provided."
                )
            config = ModelConfig.model_validate(checkpoint["config"]["model"])
        
        model = cls.create(config, device=None)
        
        state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict"))
        if state_dict is None:
            raise FactoryError("No state_dict found in checkpoint.")
        
        model.load_state_dict(state_dict, strict=strict)
        
        if device is not None:
            model = model.to(device)
        
        return model
    
    @classmethod
    def list_available(
        cls,
        task_type: Optional[TaskType] = None,
        family: Optional[ModelFamily] = None,
    ) -> list[str]:
        """
        List available model architectures.
        
        Args:
            task_type: Filter by task type.
            family: Filter by model family.
            
        Returns:
            List of available architecture names.
        """
        all_models = MODEL_REGISTRY.list_registered()
        
        if task_type is None and family is None:
            return all_models
        
        filtered = []
        for name in all_models:
            metadata = MODEL_REGISTRY.get_metadata(name)
            if task_type is not None and metadata.get("task_type") != task_type:
                continue
            if family is not None and metadata.get("family") != family:
                continue
            filtered.append(name)
        
        return filtered


class DatasetFactory(BaseFactory):
    """
    Factory for creating dataset instances.
    
    Supports creating datasets from configuration with automatic
    transform application and split handling.
    """
    
    @classmethod
    def get_registry(cls) -> Registry:
        """Get the dataset registry."""
        return DATASET_REGISTRY
    
    @classmethod
    def create(
        cls,
        config: DatasetConfig,
        split: str = "train",
        transform: Optional[Any] = None,
        **kwargs: Any,
    ) -> Dataset:
        """
        Create a dataset instance from configuration.
        
        Args:
            config: Dataset configuration.
            split: Dataset split to load.
            transform: Transform to apply.
            **kwargs: Additional arguments.
            
        Returns:
            Dataset instance.
        """
        try:
            dataset_cls = DATASET_REGISTRY.get(config.name)
        except Exception as e:
            raise FactoryError(
                f"Failed to find dataset '{config.name}': {e}"
            ) from e
        
        dataset_kwargs = {
            "root": config.root_path,
            "split": split,
            "transform": transform,
            **kwargs,
        }
        
        try:
            return dataset_cls(**dataset_kwargs)
        except TypeError:
            # Try alternate signature
            dataset_kwargs.pop("split", None)
            train = split == "train"
            return dataset_cls(train=train, **dataset_kwargs)
    
    @classmethod
    def create_dataloaders(
        cls,
        config: DatasetConfig,
        train_transform: Optional[Any] = None,
        val_transform: Optional[Any] = None,
        **kwargs: Any,
    ) -> tuple:
        """
        Create train and validation dataloaders.
        
        Args:
            config: Dataset configuration.
            train_transform: Training transforms.
            val_transform: Validation transforms.
            **kwargs: Additional arguments.
            
        Returns:
            Tuple of (train_loader, val_loader).
        """
        from torch.utils.data import DataLoader
        
        train_dataset = cls.create(
            config,
            split=config.train_split,
            transform=train_transform,
            **kwargs,
        )
        
        val_dataset = cls.create(
            config,
            split=config.val_split,
            transform=val_transform,
            **kwargs,
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            drop_last=True,
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
        )
        
        return train_loader, val_loader
    
    @classmethod
    def list_available(cls) -> list[str]:
        """List available datasets."""
        return DATASET_REGISTRY.list_registered()


class TaskFactory(BaseFactory):
    """
    Factory for creating task instances.
    
    Tasks encapsulate training/evaluation logic specific to
    different ML task types (classification, detection, etc.).
    """
    
    @classmethod
    def get_registry(cls) -> Registry:
        """Get the task registry."""
        return TASK_REGISTRY
    
    @classmethod
    def create(
        cls,
        task_type: TaskType,
        config: ExperimentConfig,
        **kwargs: Any,
    ) -> Any:
        """
        Create a task instance from configuration.
        
        Args:
            task_type: Type of ML task.
            config: Experiment configuration.
            **kwargs: Additional arguments.
            
        Returns:
            Task instance.
        """
        try:
            task_cls = TASK_REGISTRY.get(task_type.value)
        except Exception as e:
            raise FactoryError(
                f"Failed to find task '{task_type.value}': {e}"
            ) from e
        
        return task_cls(config, **kwargs)
    
    @classmethod
    def list_available(cls) -> list[str]:
        """List available tasks."""
        return TASK_REGISTRY.list_registered()


class ComponentFactory:
    """
    Unified factory for creating all experiment components.
    
    Provides a single entry point for creating models, datasets,
    tasks, optimizers, and schedulers from experiment configuration.
    """
    
    @classmethod
    def create_experiment(
        cls,
        config: ExperimentConfig,
    ) -> Dict[str, Any]:
        """
        Create all components for an experiment.
        
        Args:
            config: Complete experiment configuration.
            
        Returns:
            Dictionary containing model, dataloaders, task, optimizer, scheduler.
        """
        from mlcli.training.optimizer import create_optimizer, create_scheduler
        from mlcli.data.transforms import get_transforms
        
        # Set reproducibility
        config.set_seed()
        device = config.get_device()
        
        # Create model
        model = ModelFactory.create(config.model, device=device)
        
        # Create transforms
        train_transform, val_transform = get_transforms(
            config.model.task_type,
            config.model.input_size,
            config.dataset.augmentation,
        )
        
        # Create dataloaders
        train_loader, val_loader = DatasetFactory.create_dataloaders(
            config.dataset,
            train_transform=train_transform,
            val_transform=val_transform,
        )
        
        # Create optimizer and scheduler
        optimizer = create_optimizer(model, config.training)
        scheduler = create_scheduler(optimizer, config.training, len(train_loader))
        
        # Create task
        task = TaskFactory.create(config.model.task_type, config)
        
        return {
            "model": model,
            "train_loader": train_loader,
            "val_loader": val_loader,
            "optimizer": optimizer,
            "scheduler": scheduler,
            "task": task,
            "device": device,
            "config": config,
        }

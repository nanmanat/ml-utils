"""
MLCLI - Production-grade ML CLI tool for training, evaluation, and inference.

Supports image classification and object detection with CNN and Transformer models.

Features:
    - Multiple model architectures (ResNet, EfficientNet, ViT, YOLO, etc.)
    - Modular plugin-based architecture
    - Comprehensive training with callbacks and checkpointing
    - Flexible logging (console, file, TensorBoard)
    - Reproducible experiments

Example:
    >>> from mlcli.core.factory import ModelFactory
    >>> from mlcli.core.config import ModelConfig
    >>> 
    >>> config = ModelConfig(
    ...     task_type="classification",
    ...     architecture="resnet50",
    ...     num_classes=10,
    ... )
    >>> model = ModelFactory.create(config)
"""

__version__ = "1.0.0"
__author__ = "ML Team"

from mlcli.core.config import (
    ExperimentConfig,
    ModelConfig,
    TrainingConfig,
    DatasetConfig,
)
from mlcli.core.registry import (
    Registry,
    MODEL_REGISTRY,
    DATASET_REGISTRY,
    TASK_REGISTRY,
)
from mlcli.core.factory import ModelFactory, DatasetFactory, TaskFactory

__all__ = [
    # Version
    "__version__",
    "__author__",
    # Config
    "ExperimentConfig",
    "ModelConfig",
    "TrainingConfig",
    "DatasetConfig",
    # Registry
    "Registry",
    "MODEL_REGISTRY",
    "DATASET_REGISTRY",
    "TASK_REGISTRY",
    # Factory
    "ModelFactory",
    "DatasetFactory",
    "TaskFactory",
]

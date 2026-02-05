"""
Configuration management for MLCLI.

Provides Pydantic-based configuration classes for experiments, models, training,
and datasets with validation, serialization, and reproducibility support.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


class TaskType(str, Enum):
    """Supported task types."""
    
    CLASSIFICATION = "classification"
    OBJECT_DETECTION = "object_detection"


class ModelFamily(str, Enum):
    """Model architecture families."""
    
    CNN = "cnn"
    TRANSFORMER = "transformer"


class DeviceType(str, Enum):
    """Compute device types."""
    
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"
    AUTO = "auto"


class OptimizerType(str, Enum):
    """Supported optimizer types."""
    
    SGD = "sgd"
    ADAM = "adam"
    ADAMW = "adamw"
    RMSPROP = "rmsprop"


class SchedulerType(str, Enum):
    """Supported learning rate scheduler types."""
    
    STEP = "step"
    COSINE = "cosine"
    EXPONENTIAL = "exponential"
    PLATEAU = "plateau"
    WARMUP_COSINE = "warmup_cosine"
    NONE = "none"


class ModelConfig(BaseModel):
    """Configuration for model architecture."""
    
    task_type: TaskType = Field(..., description="Type of ML task")
    family: ModelFamily = Field(..., description="Model architecture family")
    architecture: str = Field(..., description="Specific model architecture name")
    pretrained: bool = Field(default=True, description="Use pretrained weights")
    num_classes: int = Field(default=1000, ge=1, description="Number of output classes")
    input_size: tuple[int, int] = Field(default=(224, 224), description="Input image size")
    dropout: float = Field(default=0.0, ge=0.0, le=1.0, description="Dropout rate")
    custom_params: Dict[str, Any] = Field(default_factory=dict, description="Architecture-specific params")
    
    @field_validator("architecture")
    @classmethod
    def validate_architecture(cls, v: str) -> str:
        """Validate architecture name."""
        valid_architectures = {
            # CNN Classification
            "resnet18", "resnet34", "resnet50", "resnet101", "resnet152",
            "efficientnet_b0", "efficientnet_b1", "efficientnet_b2", "efficientnet_b3",
            "efficientnet_b4", "efficientnet_b5", "efficientnet_b6", "efficientnet_b7",
            "mobilenet_v2", "mobilenet_v3_small", "mobilenet_v3_large",
            # Transformer Classification
            "vit_base_patch16", "vit_large_patch16", "vit_base_patch32",
            "swin_tiny", "swin_small", "swin_base", "swin_large",
            # Object Detection
            "yolov5s", "yolov5m", "yolov5l", "yolov5x",
            "yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x",
            "ssd300", "ssd512",
            "fasterrcnn_resnet50", "fasterrcnn_resnet101", "fasterrcnn_mobilenet",
            "detr_resnet50", "detr_resnet101",
        }
        v_lower = v.lower()
        if v_lower not in valid_architectures:
            # Allow custom architectures with a warning-level validation
            pass
        return v_lower
    
    def get_config_hash(self) -> str:
        """Generate a hash for this configuration."""
        config_str = json.dumps(self.model_dump(), sort_keys=True, default=str)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]


class DatasetConfig(BaseModel):
    """Configuration for dataset."""
    
    name: str = Field(..., description="Dataset name or identifier")
    root_path: Path = Field(..., description="Root path to dataset")
    train_split: str = Field(default="train", description="Training split name")
    val_split: str = Field(default="val", description="Validation split name")
    test_split: Optional[str] = Field(default="test", description="Test split name")
    batch_size: int = Field(default=32, ge=1, description="Batch size")
    num_workers: int = Field(default=4, ge=0, description="DataLoader workers")
    pin_memory: bool = Field(default=True, description="Pin memory for faster GPU transfer")
    augmentation: str = Field(default="standard", description="Augmentation strategy")
    cache_data: bool = Field(default=False, description="Cache dataset in memory")
    
    @field_validator("root_path", mode="before")
    @classmethod
    def validate_path(cls, v: Union[str, Path]) -> Path:
        """Convert string to Path."""
        return Path(v).expanduser().resolve()


class TrainingConfig(BaseModel):
    """Configuration for training hyperparameters."""
    
    epochs: int = Field(default=100, ge=1, description="Number of training epochs")
    learning_rate: float = Field(default=1e-3, gt=0, description="Initial learning rate")
    weight_decay: float = Field(default=1e-4, ge=0, description="Weight decay for regularization")
    momentum: float = Field(default=0.9, ge=0, le=1, description="Momentum for SGD")
    optimizer: OptimizerType = Field(default=OptimizerType.ADAMW, description="Optimizer type")
    scheduler: SchedulerType = Field(default=SchedulerType.COSINE, description="LR scheduler type")
    warmup_epochs: int = Field(default=5, ge=0, description="Warmup epochs")
    warmup_lr: float = Field(default=1e-6, ge=0, description="Initial warmup learning rate")
    min_lr: float = Field(default=1e-6, ge=0, description="Minimum learning rate")
    gradient_clip: Optional[float] = Field(default=1.0, ge=0, description="Gradient clipping value")
    accumulation_steps: int = Field(default=1, ge=1, description="Gradient accumulation steps")
    mixed_precision: bool = Field(default=True, description="Use automatic mixed precision")
    early_stopping_patience: Optional[int] = Field(default=10, ge=1, description="Early stopping patience")
    label_smoothing: float = Field(default=0.0, ge=0, le=1, description="Label smoothing factor")
    
    # Scheduler-specific params
    scheduler_params: Dict[str, Any] = Field(default_factory=dict, description="Scheduler params")


class CheckpointConfig(BaseModel):
    """Configuration for checkpointing."""
    
    save_dir: Path = Field(default=Path("./checkpoints"), description="Checkpoint directory")
    save_frequency: int = Field(default=1, ge=1, description="Save every N epochs")
    save_best_only: bool = Field(default=False, description="Only save best model")
    save_last: bool = Field(default=True, description="Always save last checkpoint")
    max_checkpoints: int = Field(default=5, ge=1, description="Maximum checkpoints to keep")
    resume_from: Optional[Path] = Field(default=None, description="Resume from checkpoint")
    
    @field_validator("save_dir", mode="before")
    @classmethod
    def validate_save_dir(cls, v: Union[str, Path]) -> Path:
        """Convert string to Path and create directory."""
        path = Path(v).expanduser().resolve()
        return path


class LoggingConfig(BaseModel):
    """Configuration for logging and experiment tracking."""
    
    log_dir: Path = Field(default=Path("./logs"), description="Log directory")
    experiment_name: Optional[str] = Field(default=None, description="Experiment name")
    log_level: str = Field(default="INFO", description="Logging level")
    log_frequency: int = Field(default=100, ge=1, description="Log every N iterations")
    use_tensorboard: bool = Field(default=True, description="Enable TensorBoard logging")
    use_json_logs: bool = Field(default=True, description="Save JSON logs")
    log_gradients: bool = Field(default=False, description="Log gradient histograms")
    log_weights: bool = Field(default=False, description="Log weight histograms")
    
    @field_validator("log_dir", mode="before")
    @classmethod
    def validate_log_dir(cls, v: Union[str, Path]) -> Path:
        """Convert string to Path."""
        return Path(v).expanduser().resolve()


class ExperimentConfig(BaseModel):
    """Complete experiment configuration."""
    
    # Core configs
    model: ModelConfig
    dataset: DatasetConfig
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    checkpoint: CheckpointConfig = Field(default_factory=CheckpointConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    
    # Experiment metadata
    experiment_id: Optional[str] = Field(default=None, description="Unique experiment ID")
    description: str = Field(default="", description="Experiment description")
    tags: List[str] = Field(default_factory=list, description="Experiment tags")
    
    # Reproducibility
    seed: int = Field(default=42, description="Random seed for reproducibility")
    deterministic: bool = Field(default=True, description="Enable deterministic mode")
    
    # Device
    device: DeviceType = Field(default=DeviceType.AUTO, description="Compute device")
    
    @model_validator(mode="after")
    def set_experiment_id(self) -> "ExperimentConfig":
        """Generate experiment ID if not provided."""
        if self.experiment_id is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            config_hash = self.model.get_config_hash()
            self.experiment_id = f"{self.model.architecture}_{timestamp}_{config_hash}"
        return self
    
    def get_device(self) -> torch.device:
        """Get the appropriate torch device."""
        if self.device == DeviceType.AUTO:
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        return torch.device(self.device.value)
    
    def set_seed(self) -> None:
        """Set random seeds for reproducibility."""
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        if self.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    def save(self, path: Union[str, Path]) -> None:
        """Save configuration to YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.model_dump(mode="json"), f, default_flow_style=False, sort_keys=False)
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> "ExperimentConfig":
        """Load configuration from YAML file."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return self.model_dump(mode="json")


class Config:
    """
    Static configuration manager for global settings.
    
    Provides centralized access to paths, environment settings,
    and default configurations.
    """
    
    # Default paths
    PROJECT_ROOT: Path = Path(__file__).parent.parent.parent.parent
    DEFAULT_CONFIG_DIR: Path = PROJECT_ROOT / "configs"
    DEFAULT_CHECKPOINT_DIR: Path = PROJECT_ROOT / "checkpoints"
    DEFAULT_LOG_DIR: Path = PROJECT_ROOT / "logs"
    DEFAULT_DATA_DIR: Path = PROJECT_ROOT / "data"
    
    # Environment variable overrides
    @classmethod
    def get_data_dir(cls) -> Path:
        """Get data directory from environment or default."""
        return Path(os.environ.get("MLCLI_DATA_DIR", cls.DEFAULT_DATA_DIR))
    
    @classmethod
    def get_checkpoint_dir(cls) -> Path:
        """Get checkpoint directory from environment or default."""
        return Path(os.environ.get("MLCLI_CHECKPOINT_DIR", cls.DEFAULT_CHECKPOINT_DIR))
    
    @classmethod
    def get_log_dir(cls) -> Path:
        """Get log directory from environment or default."""
        return Path(os.environ.get("MLCLI_LOG_DIR", cls.DEFAULT_LOG_DIR))
    
    @classmethod
    def ensure_dirs(cls) -> None:
        """Create default directories if they don't exist."""
        for dir_path in [
            cls.get_data_dir(),
            cls.get_checkpoint_dir(),
            cls.get_log_dir(),
            cls.DEFAULT_CONFIG_DIR,
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)

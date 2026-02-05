"""
Optimizer and scheduler creation utilities.

Provides factory functions for creating optimizers and learning rate
schedulers from configuration.
"""

from __future__ import annotations

from typing import Any, Dict, Iterator, Optional, Union

import torch
import torch.nn as nn
from torch.optim import SGD, Adam, AdamW, RMSprop
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    ExponentialLR,
    MultiStepLR,
    ReduceLROnPlateau,
    StepLR,
    LRScheduler,
)

from mlcli.core.config import OptimizerType, SchedulerType, TrainingConfig


def create_optimizer(
    model: Union[nn.Module, Iterator[nn.Parameter]],
    config: TrainingConfig,
    parameter_groups: Optional[list] = None,
) -> torch.optim.Optimizer:
    """
    Create optimizer from configuration.
    
    Args:
        model: Model or parameter iterator.
        config: Training configuration.
        parameter_groups: Optional custom parameter groups.
        
    Returns:
        Configured optimizer.
    """
    # Get parameters
    if parameter_groups is not None:
        params = parameter_groups
    elif isinstance(model, nn.Module):
        params = model.parameters()
    else:
        params = model
    
    # Build optimizer kwargs
    optimizer_kwargs: Dict[str, Any] = {
        "lr": config.learning_rate,
        "weight_decay": config.weight_decay,
    }
    
    if config.optimizer == OptimizerType.SGD:
        optimizer_kwargs["momentum"] = config.momentum
        optimizer_kwargs["nesterov"] = True
        return SGD(params, **optimizer_kwargs)
    
    elif config.optimizer == OptimizerType.ADAM:
        optimizer_kwargs["betas"] = (0.9, 0.999)
        optimizer_kwargs["eps"] = 1e-8
        return Adam(params, **optimizer_kwargs)
    
    elif config.optimizer == OptimizerType.ADAMW:
        optimizer_kwargs["betas"] = (0.9, 0.999)
        optimizer_kwargs["eps"] = 1e-8
        return AdamW(params, **optimizer_kwargs)
    
    elif config.optimizer == OptimizerType.RMSPROP:
        optimizer_kwargs["momentum"] = config.momentum
        optimizer_kwargs["alpha"] = 0.99
        return RMSprop(params, **optimizer_kwargs)
    
    else:
        raise ValueError(f"Unknown optimizer type: {config.optimizer}")


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    config: TrainingConfig,
    steps_per_epoch: int,
) -> Optional[LRScheduler]:
    """
    Create learning rate scheduler from configuration.
    
    Args:
        optimizer: Optimizer to schedule.
        config: Training configuration.
        steps_per_epoch: Number of steps per epoch.
        
    Returns:
        Configured scheduler or None.
    """
    if config.scheduler == SchedulerType.NONE:
        return None
    
    total_steps = config.epochs * steps_per_epoch
    warmup_steps = config.warmup_epochs * steps_per_epoch
    
    scheduler_params = config.scheduler_params or {}
    
    if config.scheduler == SchedulerType.STEP:
        step_size = scheduler_params.get("step_size", 30)
        gamma = scheduler_params.get("gamma", 0.1)
        return StepLR(optimizer, step_size=step_size, gamma=gamma)
    
    elif config.scheduler == SchedulerType.COSINE:
        return CosineAnnealingLR(
            optimizer,
            T_max=config.epochs - config.warmup_epochs,
            eta_min=config.min_lr,
        )
    
    elif config.scheduler == SchedulerType.EXPONENTIAL:
        gamma = scheduler_params.get("gamma", 0.95)
        return ExponentialLR(optimizer, gamma=gamma)
    
    elif config.scheduler == SchedulerType.PLATEAU:
        return ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=scheduler_params.get("factor", 0.1),
            patience=scheduler_params.get("patience", 10),
            min_lr=config.min_lr,
        )
    
    elif config.scheduler == SchedulerType.WARMUP_COSINE:
        return WarmupCosineScheduler(
            optimizer,
            warmup_epochs=config.warmup_epochs,
            total_epochs=config.epochs,
            warmup_lr=config.warmup_lr,
            min_lr=config.min_lr,
        )
    
    else:
        raise ValueError(f"Unknown scheduler type: {config.scheduler}")


class WarmupCosineScheduler(LRScheduler):
    """
    Cosine annealing with linear warmup.
    
    Learning rate schedule:
    - Linear warmup from warmup_lr to base_lr
    - Cosine annealing from base_lr to min_lr
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        warmup_lr: float = 1e-6,
        min_lr: float = 1e-6,
        last_epoch: int = -1,
    ) -> None:
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.warmup_lr = warmup_lr
        self.min_lr = min_lr
        
        # Store base learning rates
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self) -> list[float]:
        """Get current learning rates."""
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            alpha = self.last_epoch / max(1, self.warmup_epochs)
            return [
                self.warmup_lr + alpha * (base_lr - self.warmup_lr)
                for base_lr in self.base_lrs
            ]
        else:
            # Cosine annealing
            import math
            progress = (self.last_epoch - self.warmup_epochs) / max(
                1, self.total_epochs - self.warmup_epochs
            )
            return [
                self.min_lr + 0.5 * (base_lr - self.min_lr) * (1 + math.cos(math.pi * progress))
                for base_lr in self.base_lrs
            ]


class LayerDecayScheduler:
    """
    Layer-wise learning rate decay for fine-tuning.
    
    Applies exponentially decaying learning rates to earlier layers.
    """
    
    def __init__(
        self,
        model: nn.Module,
        base_lr: float,
        layer_decay: float = 0.75,
        num_layers: int = 12,
    ) -> None:
        self.model = model
        self.base_lr = base_lr
        self.layer_decay = layer_decay
        self.num_layers = num_layers
    
    def get_parameter_groups(self) -> list[Dict[str, Any]]:
        """Get parameter groups with layer-wise learning rates."""
        param_groups = []
        
        # Get layer names and assign decay
        layer_scales = {}
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            
            # Determine layer depth from name
            layer_id = self._get_layer_id(name)
            scale = self.layer_decay ** (self.num_layers - layer_id)
            layer_scales[name] = scale
        
        # Group parameters by scale
        scale_to_params: Dict[float, list] = {}
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            
            scale = layer_scales.get(name, 1.0)
            if scale not in scale_to_params:
                scale_to_params[scale] = []
            scale_to_params[scale].append(param)
        
        # Create parameter groups
        for scale, params in scale_to_params.items():
            param_groups.append({
                "params": params,
                "lr": self.base_lr * scale,
            })
        
        return param_groups
    
    def _get_layer_id(self, name: str) -> int:
        """Determine layer ID from parameter name."""
        # Simple heuristic - can be customized per architecture
        if "embed" in name or "patch" in name:
            return 0
        elif "blocks" in name or "layers" in name:
            # Extract layer number
            import re
            match = re.search(r"(?:blocks|layers)\.(\d+)", name)
            if match:
                return int(match.group(1)) + 1
        elif "head" in name or "classifier" in name:
            return self.num_layers
        return self.num_layers // 2

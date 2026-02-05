"""
Base task interface.

Provides abstract base class for ML tasks with common
functionality for training, evaluation, and inference.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from mlcli.core.config import ExperimentConfig


class BaseTask(ABC):
    """
    Abstract base class for ML tasks.
    
    Encapsulates task-specific logic for training, evaluation,
    and inference, providing a unified interface across different
    task types.
    """
    
    def __init__(
        self,
        config: ExperimentConfig,
        model: Optional[nn.Module] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        """
        Initialize task.
        
        Args:
            config: Experiment configuration.
            model: Optional pre-created model.
            device: Compute device.
        """
        self.config = config
        self.device = device or config.get_device()
        self._model = model
        self._criterion: Optional[nn.Module] = None
        self._metrics: Dict[str, Any] = {}
    
    @property
    def model(self) -> nn.Module:
        """Get the model."""
        if self._model is None:
            raise RuntimeError("Model not set. Call set_model() first.")
        return self._model
    
    def set_model(self, model: nn.Module) -> None:
        """Set the model."""
        self._model = model.to(self.device)
    
    @property
    def criterion(self) -> nn.Module:
        """Get the loss criterion."""
        if self._criterion is None:
            self._criterion = self.build_criterion()
        return self._criterion
    
    @abstractmethod
    def build_criterion(self) -> nn.Module:
        """Build the loss criterion for this task."""
        pass
    
    @abstractmethod
    def train_step(
        self,
        batch: Any,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None,
        scaler: Optional[torch.cuda.amp.GradScaler] = None,
    ) -> Dict[str, float]:
        """
        Execute a single training step.
        
        Args:
            batch: Input batch.
            optimizer: Optimizer.
            scheduler: Optional LR scheduler.
            scaler: Optional gradient scaler for mixed precision.
            
        Returns:
            Dictionary of loss values.
        """
        pass
    
    @abstractmethod
    def eval_step(
        self,
        batch: Any,
    ) -> Dict[str, Any]:
        """
        Execute a single evaluation step.
        
        Args:
            batch: Input batch.
            
        Returns:
            Dictionary of predictions and targets.
        """
        pass
    
    @abstractmethod
    def compute_metrics(
        self,
        outputs: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        """
        Compute evaluation metrics from accumulated outputs.
        
        Args:
            outputs: List of outputs from eval_step.
            
        Returns:
            Dictionary of metric values.
        """
        pass
    
    @abstractmethod
    def predict(
        self,
        inputs: torch.Tensor,
    ) -> Any:
        """
        Run inference on inputs.
        
        Args:
            inputs: Input tensor.
            
        Returns:
            Model predictions.
        """
        pass
    
    def train_epoch(
        self,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None,
        scaler: Optional[torch.cuda.amp.GradScaler] = None,
        progress_callback: Optional[Callable] = None,
    ) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            dataloader: Training dataloader.
            optimizer: Optimizer.
            scheduler: Optional LR scheduler.
            scaler: Optional gradient scaler.
            progress_callback: Optional callback for progress updates.
            
        Returns:
            Dictionary of average losses.
        """
        self.model.train()
        
        epoch_losses: Dict[str, List[float]] = {}
        
        for batch_idx, batch in enumerate(dataloader):
            losses = self.train_step(batch, optimizer, scheduler, scaler)
            
            for key, value in losses.items():
                if key not in epoch_losses:
                    epoch_losses[key] = []
                epoch_losses[key].append(value)
            
            if progress_callback is not None:
                progress_callback(batch_idx, len(dataloader), losses)
        
        # Average losses
        return {key: sum(values) / len(values) for key, values in epoch_losses.items()}
    
    @torch.no_grad()
    def evaluate(
        self,
        dataloader: DataLoader,
        progress_callback: Optional[Callable] = None,
    ) -> Dict[str, float]:
        """
        Evaluate model on dataloader.
        
        Args:
            dataloader: Evaluation dataloader.
            progress_callback: Optional callback for progress updates.
            
        Returns:
            Dictionary of evaluation metrics.
        """
        self.model.eval()
        
        all_outputs: List[Dict[str, Any]] = []
        
        for batch_idx, batch in enumerate(dataloader):
            outputs = self.eval_step(batch)
            all_outputs.append(outputs)
            
            if progress_callback is not None:
                progress_callback(batch_idx, len(dataloader))
        
        return self.compute_metrics(all_outputs)
    
    def get_primary_metric(self) -> Tuple[str, str]:
        """
        Get the primary metric for this task.
        
        Returns:
            Tuple of (metric_name, mode) where mode is 'min' or 'max'.
        """
        return ("loss", "min")
    
    def move_batch_to_device(self, batch: Any) -> Any:
        """Move batch tensors to device."""
        if isinstance(batch, torch.Tensor):
            return batch.to(self.device)
        elif isinstance(batch, (list, tuple)):
            return type(batch)(self.move_batch_to_device(item) for item in batch)
        elif isinstance(batch, dict):
            return {key: self.move_batch_to_device(value) for key, value in batch.items()}
        return batch

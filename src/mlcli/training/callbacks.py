"""
Training callbacks for extensible training loop.

Provides callback interface and implementations for common
training behaviors like early stopping, checkpointing, and logging.
"""

from __future__ import annotations

from abc import ABC
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import torch.nn as nn

if TYPE_CHECKING:
    from mlcli.training.trainer import Trainer


class Callback(ABC):
    """
    Base callback class for training events.
    
    Callbacks can be used to extend training behavior
    without modifying the core training loop.
    """
    
    def set_trainer(self, trainer: "Trainer") -> None:
        """Set reference to trainer."""
        self.trainer = trainer
    
    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """Called at the beginning of training."""
        pass
    
    def on_train_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """Called at the end of training."""
        pass
    
    def on_epoch_begin(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Called at the beginning of each epoch."""
        pass
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Called at the end of each epoch."""
        pass
    
    def on_batch_begin(self, batch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Called at the beginning of each batch."""
        pass
    
    def on_batch_end(self, batch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Called at the end of each batch."""
        pass
    
    def on_validation_begin(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """Called at the beginning of validation."""
        pass
    
    def on_validation_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """Called at the end of validation."""
        pass


class CallbackList:
    """Container for managing multiple callbacks."""
    
    def __init__(self, callbacks: Optional[List[Callback]] = None) -> None:
        self.callbacks = callbacks or []
    
    def append(self, callback: Callback) -> None:
        """Add a callback."""
        self.callbacks.append(callback)
    
    def set_trainer(self, trainer: "Trainer") -> None:
        """Set trainer for all callbacks."""
        for callback in self.callbacks:
            callback.set_trainer(trainer)
    
    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None) -> None:
        for callback in self.callbacks:
            callback.on_train_begin(logs)
    
    def on_train_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
        for callback in self.callbacks:
            callback.on_train_end(logs)
    
    def on_epoch_begin(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch, logs)
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, logs)
    
    def on_batch_begin(self, batch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        for callback in self.callbacks:
            callback.on_batch_begin(batch, logs)
    
    def on_batch_end(self, batch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        for callback in self.callbacks:
            callback.on_batch_end(batch, logs)
    
    def on_validation_begin(self, logs: Optional[Dict[str, Any]] = None) -> None:
        for callback in self.callbacks:
            callback.on_validation_begin(logs)
    
    def on_validation_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
        for callback in self.callbacks:
            callback.on_validation_end(logs)


class EarlyStoppingCallback(Callback):
    """
    Early stopping callback.
    
    Stops training when a monitored metric stops improving.
    """
    
    def __init__(
        self,
        monitor: str = "val_loss",
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = "min",
        restore_best_weights: bool = True,
    ) -> None:
        """
        Initialize early stopping.
        
        Args:
            monitor: Metric to monitor.
            patience: Number of epochs with no improvement.
            min_delta: Minimum change to qualify as improvement.
            mode: 'min' or 'max'.
            restore_best_weights: Restore best weights on stop.
        """
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        
        self.best_value: Optional[float] = None
        self.best_weights: Optional[Dict] = None
        self.wait = 0
        self.stopped_epoch = 0
        self.stop_training = False
    
    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None) -> None:
        self.wait = 0
        self.best_value = None
        self.stop_training = False
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        logs = logs or {}
        current_value = logs.get(self.monitor)
        
        if current_value is None:
            return
        
        if self._is_improvement(current_value):
            self.best_value = current_value
            self.wait = 0
            
            if self.restore_best_weights:
                self.best_weights = {
                    k: v.cpu().clone() for k, v in self.trainer.model.state_dict().items()
                }
        else:
            self.wait += 1
            
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.stop_training = True
                self.trainer.stop_training = True
                
                if self.restore_best_weights and self.best_weights is not None:
                    self.trainer.model.load_state_dict(self.best_weights)
    
    def _is_improvement(self, current: float) -> bool:
        if self.best_value is None:
            return True
        
        if self.mode == "min":
            return current < self.best_value - self.min_delta
        else:
            return current > self.best_value + self.min_delta
    
    def on_train_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
        if self.stopped_epoch > 0:
            print(f"Early stopping triggered at epoch {self.stopped_epoch}")


class ModelCheckpointCallback(Callback):
    """
    Model checkpoint callback.
    
    Saves model checkpoints during training.
    """
    
    def __init__(
        self,
        checkpoint_manager: Any,
        monitor: str = "val_loss",
        save_freq: int = 1,
    ) -> None:
        """
        Initialize checkpoint callback.
        
        Args:
            checkpoint_manager: CheckpointManager instance.
            monitor: Metric to monitor for best model.
            save_freq: Save every N epochs.
        """
        self.checkpoint_manager = checkpoint_manager
        self.monitor = monitor
        self.save_freq = save_freq
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        logs = logs or {}
        
        if epoch % self.save_freq == 0:
            self.checkpoint_manager.save(
                model=self.trainer.model,
                optimizer=self.trainer.optimizer,
                epoch=epoch,
                step=self.trainer.global_step,
                metrics=logs,
                scheduler=self.trainer.scheduler,
                scaler=self.trainer.scaler,
                config=self.trainer.config,
            )


class LearningRateCallback(Callback):
    """
    Learning rate monitoring and scheduling callback.
    """
    
    def __init__(self, log_freq: int = 1) -> None:
        self.log_freq = log_freq
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        logs = logs or {}
        
        # Get current learning rate
        if self.trainer.optimizer:
            lr = self.trainer.optimizer.param_groups[0]["lr"]
            logs["learning_rate"] = lr
        
        # Step scheduler if epoch-based
        if self.trainer.scheduler is not None:
            from torch.optim.lr_scheduler import ReduceLROnPlateau
            
            if isinstance(self.trainer.scheduler, ReduceLROnPlateau):
                monitor_value = logs.get("val_loss", logs.get("loss", 0))
                self.trainer.scheduler.step(monitor_value)
            else:
                self.trainer.scheduler.step()


class ProgressCallback(Callback):
    """
    Training progress display callback using rich.
    """
    
    def __init__(self, show_progress: bool = True) -> None:
        self.show_progress = show_progress
        self.progress = None
        self.epoch_task = None
        self.batch_task = None
    
    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None) -> None:
        if self.show_progress:
            from rich.progress import Progress, TextColumn, BarColumn, TimeRemainingColumn
            
            self.progress = Progress(
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeRemainingColumn(),
            )
            self.progress.start()
    
    def on_train_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
        if self.progress:
            self.progress.stop()
    
    def on_epoch_begin(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        if self.progress and hasattr(self.trainer, "config"):
            total_epochs = self.trainer.config.training.epochs
            self.epoch_task = self.progress.add_task(
                f"Epoch {epoch}/{total_epochs}",
                total=100,
            )
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        if self.progress and self.epoch_task is not None:
            self.progress.update(self.epoch_task, completed=100)
            self.progress.remove_task(self.epoch_task)
    
    def on_batch_end(self, batch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        if self.progress and self.epoch_task is not None:
            total_batches = getattr(self.trainer, "total_batches", 100)
            progress = (batch / total_batches) * 100
            self.progress.update(self.epoch_task, completed=progress)


class MetricsHistoryCallback(Callback):
    """
    Callback to track metrics history.
    """
    
    def __init__(self) -> None:
        self.history: Dict[str, List[float]] = {}
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        logs = logs or {}
        for key, value in logs.items():
            if isinstance(value, (int, float)):
                if key not in self.history:
                    self.history[key] = []
                self.history[key].append(value)
    
    def get_history(self) -> Dict[str, List[float]]:
        return self.history


class GradientClippingCallback(Callback):
    """
    Gradient clipping callback.
    """
    
    def __init__(self, max_norm: float = 1.0, norm_type: float = 2.0) -> None:
        self.max_norm = max_norm
        self.norm_type = norm_type
    
    def on_batch_end(self, batch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        import torch
        
        if hasattr(self.trainer, "model"):
            torch.nn.utils.clip_grad_norm_(
                self.trainer.model.parameters(),
                self.max_norm,
                self.norm_type,
            )

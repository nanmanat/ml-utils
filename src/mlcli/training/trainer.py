"""
Main trainer class for training ML models.

Provides a unified training interface with support for
mixed precision, distributed training, callbacks, and logging.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from mlcli.core.config import ExperimentConfig
from mlcli.core.factory import ModelFactory, DatasetFactory, TaskFactory
from mlcli.data.transforms import get_transforms
from mlcli.logging import ExperimentLogger
from mlcli.tasks.base import BaseTask
from mlcli.training.callbacks import Callback, CallbackList
from mlcli.training.checkpoint import CheckpointManager
from mlcli.training.optimizer import create_optimizer, create_scheduler


class Trainer:
    """
    Main trainer class for training ML models.
    
    Handles the complete training loop with support for:
    - Mixed precision training
    - Gradient accumulation
    - Learning rate scheduling
    - Checkpointing
    - Callbacks
    - Logging
    """
    
    def __init__(
        self,
        config: ExperimentConfig,
        model: Optional[nn.Module] = None,
        task: Optional[BaseTask] = None,
        train_loader: Optional[DataLoader] = None,
        val_loader: Optional[DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        callbacks: Optional[List[Callback]] = None,
        logger: Optional[ExperimentLogger] = None,
    ) -> None:
        """
        Initialize trainer.
        
        Args:
            config: Experiment configuration.
            model: Optional pre-created model.
            task: Optional pre-created task.
            train_loader: Optional training dataloader.
            val_loader: Optional validation dataloader.
            optimizer: Optional optimizer.
            scheduler: Optional LR scheduler.
            callbacks: List of callbacks.
            logger: Experiment logger.
        """
        self.config = config
        self.device = config.get_device()
        
        # Set reproducibility
        config.set_seed()
        
        # Initialize model
        if model is not None:
            self.model = model.to(self.device)
        else:
            self.model = ModelFactory.create(config.model, device=self.device)
        
        # Initialize task
        if task is not None:
            self.task = task
            self.task.set_model(self.model)
        else:
            self.task = TaskFactory.create(config.model.task_type, config)
            self.task.set_model(self.model)
        
        # Initialize dataloaders
        if train_loader is not None:
            self.train_loader = train_loader
            self.val_loader = val_loader
        else:
            self._setup_dataloaders()
        
        # Initialize optimizer and scheduler
        if optimizer is not None:
            self.optimizer = optimizer
        else:
            self.optimizer = create_optimizer(self.model, config.training)
        
        if scheduler is not None:
            self.scheduler = scheduler
        else:
            self.scheduler = create_scheduler(
                self.optimizer,
                config.training,
                len(self.train_loader),
            )
        
        # Initialize mixed precision
        self.scaler: Optional[GradScaler] = None
        if config.training.mixed_precision and self.device.type == "cuda":
            self.scaler = GradScaler()
        
        # Initialize callbacks
        self.callbacks = CallbackList(callbacks or [])
        self.callbacks.set_trainer(self)
        
        # Initialize logger
        self.logger = logger or ExperimentLogger(
            log_dir=config.logging.log_dir,
            experiment_name=config.experiment_id,
            config=config,
        )
        
        # Initialize checkpoint manager
        self.checkpoint_manager = CheckpointManager(
            save_dir=config.checkpoint.save_dir,
            config=config.checkpoint,
            experiment_id=config.experiment_id,
            metric_name=self.task.get_primary_metric()[0],
            metric_mode=self.task.get_primary_metric()[1],
        )
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.total_batches = len(self.train_loader) if self.train_loader else 0
        self.stop_training = False
        self.best_metric: Optional[float] = None
        
        # Resume from checkpoint if specified
        if config.checkpoint.resume_from:
            self._resume_from_checkpoint()
    
    def _setup_dataloaders(self) -> None:
        """Set up training and validation dataloaders."""
        train_transform, val_transform = get_transforms(
            self.config.model.task_type,
            self.config.model.input_size,
            self.config.dataset.augmentation,
        )
        
        self.train_loader, self.val_loader = DatasetFactory.create_dataloaders(
            self.config.dataset,
            train_transform=train_transform,
            val_transform=val_transform,
        )
    
    def _resume_from_checkpoint(self) -> None:
        """Resume training from checkpoint."""
        checkpoint = self.checkpoint_manager.load(
            checkpoint_path=self.config.checkpoint.resume_from,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
        )
        
        self.current_epoch = checkpoint.get("epoch", 0) + 1
        self.global_step = checkpoint.get("step", 0)
        self.best_metric = checkpoint.get("metric_value")
        
        self.logger.log(f"Resumed from epoch {self.current_epoch}")
    
    def train(self) -> Dict[str, Any]:
        """
        Run the complete training loop.
        
        Returns:
            Dictionary with training results.
        """
        self.logger.log("Starting training...")
        self.logger.log(f"Device: {self.device}")
        self.logger.log(f"Model: {self.config.model.architecture}")
        self.logger.log(f"Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        start_time = time.time()
        history: Dict[str, List[float]] = {}
        
        self.callbacks.on_train_begin({"config": self.config})
        
        try:
            for epoch in range(self.current_epoch, self.config.training.epochs):
                if self.stop_training:
                    break
                
                self.current_epoch = epoch
                epoch_logs = {"epoch": epoch}
                
                self.callbacks.on_epoch_begin(epoch, epoch_logs)
                
                # Training phase
                train_metrics = self._train_epoch()
                epoch_logs.update({f"train_{k}": v for k, v in train_metrics.items()})
                
                # Validation phase
                if self.val_loader is not None:
                    self.callbacks.on_validation_begin(epoch_logs)
                    val_metrics = self._validate()
                    epoch_logs.update({f"val_{k}": v for k, v in val_metrics.items()})
                    self.callbacks.on_validation_end(epoch_logs)
                
                # Learning rate
                if self.optimizer:
                    epoch_logs["lr"] = self.optimizer.param_groups[0]["lr"]
                
                # Log metrics
                self.logger.log_metrics(epoch_logs, step=epoch)
                
                # Update history
                for key, value in epoch_logs.items():
                    if isinstance(value, (int, float)):
                        if key not in history:
                            history[key] = []
                        history[key].append(value)
                
                # Callbacks
                self.callbacks.on_epoch_end(epoch, epoch_logs)
                
                # Checkpointing
                self.checkpoint_manager.save(
                    model=self.model,
                    optimizer=self.optimizer,
                    epoch=epoch,
                    step=self.global_step,
                    metrics=epoch_logs,
                    scheduler=self.scheduler,
                    scaler=self.scaler,
                    config=self.config,
                )
                
                # Print epoch summary
                self._print_epoch_summary(epoch, epoch_logs)
        
        except Exception as exc:
            self.logger.log(f"Training failed at epoch {self.current_epoch}: {exc}")
            self.checkpoint_manager.save_emergency(
                model=self.model,
                optimizer=self.optimizer,
                epoch=self.current_epoch,
                step=self.global_step,
                error_msg=str(exc),
                scheduler=self.scheduler,
                scaler=self.scaler,
                config=self.config,
            )
            raise
        except KeyboardInterrupt:
            self.logger.log("Training interrupted by user.")

        finally:
            self.callbacks.on_train_end({"history": history})
        
        elapsed_time = time.time() - start_time
        
        # Final results
        results = {
            "epochs_trained": self.current_epoch + 1,
            "best_metric": self.checkpoint_manager.best_metric,
            "total_time": elapsed_time,
            "history": history,
        }
        
        self.logger.log(f"Training completed in {elapsed_time:.2f}s")
        self.logger.log(f"Best {self.task.get_primary_metric()[0]}: {self.checkpoint_manager.best_metric:.4f}")
        
        return results
    
    def _train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        epoch_losses: Dict[str, List[float]] = {}
        
        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.current_epoch}",
            leave=False,
        )
        
        accumulation_steps = self.config.training.accumulation_steps
        
        for batch_idx, batch in enumerate(progress_bar):
            self.callbacks.on_batch_begin(batch_idx)
            
            # Training step
            losses = self.task.train_step(
                batch,
                self.optimizer,
                self.scheduler if batch_idx % accumulation_steps == 0 else None,
                self.scaler,
            )
            
            # Accumulate losses
            for key, value in losses.items():
                if key not in epoch_losses:
                    epoch_losses[key] = []
                epoch_losses[key].append(value)
            
            self.global_step += 1
            
            # Update progress bar
            avg_loss = sum(epoch_losses.get("loss", [0])) / len(epoch_losses.get("loss", [1]))
            progress_bar.set_postfix({"loss": f"{avg_loss:.4f}"})
            
            # Log batch metrics
            if batch_idx % self.config.logging.log_frequency == 0:
                self.logger.log_metrics(losses, step=self.global_step)
            
            self.callbacks.on_batch_end(batch_idx, losses)
        
        # Average epoch losses
        return {key: sum(values) / len(values) for key, values in epoch_losses.items()}
    
    @torch.no_grad()
    def _validate(self) -> Dict[str, float]:
        """Run validation."""
        self.model.eval()
        
        all_outputs = []
        
        progress_bar = tqdm(
            self.val_loader,
            desc="Validating",
            leave=False,
        )
        
        for batch in progress_bar:
            outputs = self.task.eval_step(batch)
            all_outputs.append(outputs)
        
        return self.task.compute_metrics(all_outputs)
    
    def _print_epoch_summary(self, epoch: int, logs: Dict[str, Any]) -> None:
        """Print epoch summary."""
        summary_parts = [f"Epoch {epoch}"]
        
        for key in ["train_loss", "train_accuracy", "val_loss", "val_accuracy", "lr"]:
            if key in logs:
                value = logs[key]
                if isinstance(value, float):
                    summary_parts.append(f"{key}: {value:.4f}")
        
        self.logger.log(" | ".join(summary_parts))
    
    def evaluate(
        self,
        dataloader: Optional[DataLoader] = None,
    ) -> Dict[str, float]:
        """
        Evaluate model on a dataloader.
        
        Args:
            dataloader: Optional dataloader (uses val_loader if None).
            
        Returns:
            Dictionary of metrics.
        """
        loader = dataloader or self.val_loader
        if loader is None:
            raise ValueError("No dataloader provided for evaluation.")
        
        return self.task.evaluate(loader)
    
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
        return self.task.predict(inputs)
    
    def save_model(self, path: Union[str, Path]) -> None:
        """Save model weights only."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)
        self.logger.log(f"Model saved to {path}")
    
    def load_model(self, path: Union[str, Path]) -> None:
        """Load model weights."""
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.logger.log(f"Model loaded from {path}")


def train_from_config(
    config: Union[ExperimentConfig, str, Path],
    callbacks: Optional[List[Callback]] = None,
) -> Dict[str, Any]:
    """
    Train a model from configuration.
    
    Args:
        config: Configuration or path to config file.
        callbacks: Optional callbacks.
        
    Returns:
        Training results.
    """
    if isinstance(config, (str, Path)):
        config = ExperimentConfig.load(config)
    
    trainer = Trainer(config, callbacks=callbacks)
    return trainer.train()

"""
Checkpoint management for training.

Provides utilities for saving, loading, and managing model checkpoints
with support for best model tracking and automatic cleanup.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn

from mlcli.core.config import CheckpointConfig, ExperimentConfig


@dataclass
class CheckpointInfo:
    """Information about a saved checkpoint."""
    
    path: Path
    epoch: int
    step: int
    metric_name: str
    metric_value: float
    timestamp: str
    is_best: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "path": str(self.path),
            "epoch": self.epoch,
            "step": self.step,
            "metric_name": self.metric_name,
            "metric_value": self.metric_value,
            "timestamp": self.timestamp,
            "is_best": self.is_best,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CheckpointInfo":
        """Create from dictionary."""
        return cls(
            path=Path(data["path"]),
            epoch=data["epoch"],
            step=data["step"],
            metric_name=data["metric_name"],
            metric_value=data["metric_value"],
            timestamp=data["timestamp"],
            is_best=data.get("is_best", False),
        )


class CheckpointManager:
    """
    Manager for saving and loading training checkpoints.
    
    Features:
    - Automatic best model tracking
    - Configurable checkpoint retention
    - Experiment state saving
    - Resume from checkpoint support
    """
    
    def __init__(
        self,
        save_dir: Union[str, Path],
        config: Optional[CheckpointConfig] = None,
        experiment_id: Optional[str] = None,
        metric_name: str = "loss",
        metric_mode: str = "min",
    ) -> None:
        """
        Initialize checkpoint manager.
        
        Args:
            save_dir: Directory to save checkpoints.
            config: Checkpoint configuration.
            experiment_id: Unique experiment identifier.
            metric_name: Metric to track for best model.
            metric_mode: 'min' or 'max' for best metric.
        """
        self.save_dir = Path(save_dir)
        self.config = config or CheckpointConfig()
        self.experiment_id = experiment_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.metric_name = metric_name
        self.metric_mode = metric_mode
        
        # Create checkpoint directory
        self.checkpoint_dir = self.save_dir / self.experiment_id
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Track checkpoints
        self.checkpoints: List[CheckpointInfo] = []
        self.best_metric: Optional[float] = None
        self.best_checkpoint: Optional[CheckpointInfo] = None
        
        # Load existing checkpoints if resuming
        self._load_checkpoint_history()
    
    def _load_checkpoint_history(self) -> None:
        """Load checkpoint history from disk."""
        history_path = self.checkpoint_dir / "checkpoint_history.json"
        if history_path.exists():
            with open(history_path) as f:
                data = json.load(f)
            
            self.checkpoints = [
                CheckpointInfo.from_dict(c) for c in data.get("checkpoints", [])
            ]
            self.best_metric = data.get("best_metric")
            
            if data.get("best_checkpoint"):
                self.best_checkpoint = CheckpointInfo.from_dict(data["best_checkpoint"])
    
    def _save_checkpoint_history(self) -> None:
        """Save checkpoint history to disk."""
        history_path = self.checkpoint_dir / "checkpoint_history.json"
        data = {
            "checkpoints": [c.to_dict() for c in self.checkpoints],
            "best_metric": self.best_metric,
            "best_checkpoint": self.best_checkpoint.to_dict() if self.best_checkpoint else None,
        }
        with open(history_path, "w") as f:
            json.dump(data, f, indent=2)
    
    def save(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        step: int,
        metrics: Dict[str, float],
        scheduler: Optional[Any] = None,
        scaler: Optional[torch.cuda.amp.GradScaler] = None,
        config: Optional[ExperimentConfig] = None,
        extra_state: Optional[Dict[str, Any]] = None,
    ) -> Optional[CheckpointInfo]:
        """
        Save a checkpoint.
        
        Args:
            model: Model to save.
            optimizer: Optimizer state.
            epoch: Current epoch.
            step: Current global step.
            metrics: Current metrics.
            scheduler: Optional LR scheduler.
            scaler: Optional gradient scaler.
            config: Optional experiment config.
            extra_state: Additional state to save.
            
        Returns:
            CheckpointInfo if saved, None otherwise.
        """
        # Check if we should save this epoch
        if epoch % self.config.save_frequency != 0 and not self._is_best(metrics):
            if not self.config.save_last or epoch < self.config.save_frequency:
                return None
        
        metric_value = metrics.get(self.metric_name, 0.0)
        is_best = self._is_best(metrics)
        
        # Skip if save_best_only and not best
        if self.config.save_best_only and not is_best:
            return None
        
        # Build checkpoint
        checkpoint = {
            "epoch": epoch,
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "metric_name": self.metric_name,
            "metric_value": metric_value,
        }
        
        if scheduler is not None:
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()
        
        if scaler is not None:
            checkpoint["scaler_state_dict"] = scaler.state_dict()
        
        if config is not None:
            checkpoint["config"] = config.to_dict()
        
        if extra_state is not None:
            checkpoint["extra_state"] = extra_state
        
        # Save checkpoint
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"checkpoint_epoch{epoch:04d}_{timestamp}.pt"
        filepath = self.checkpoint_dir / filename
        
        torch.save(checkpoint, filepath)
        
        # Create checkpoint info
        info = CheckpointInfo(
            path=filepath,
            epoch=epoch,
            step=step,
            metric_name=self.metric_name,
            metric_value=metric_value,
            timestamp=timestamp,
            is_best=is_best,
        )
        
        self.checkpoints.append(info)
        
        # Update best checkpoint
        if is_best:
            self.best_metric = metric_value
            self.best_checkpoint = info
            
            # Save best model separately
            best_path = self.checkpoint_dir / "best_model.pt"
            shutil.copy(filepath, best_path)
        
        # Save last checkpoint
        if self.config.save_last:
            last_path = self.checkpoint_dir / "last_model.pt"
            shutil.copy(filepath, last_path)
        
        # Cleanup old checkpoints
        self._cleanup_checkpoints()
        
        # Save history
        self._save_checkpoint_history()
        
        return info
    
    def _is_best(self, metrics: Dict[str, float]) -> bool:
        """Check if current metrics are the best so far."""
        if self.metric_name not in metrics:
            return False
        
        current_value = metrics[self.metric_name]
        
        if self.best_metric is None:
            return True
        
        if self.metric_mode == "min":
            return current_value < self.best_metric
        else:
            return current_value > self.best_metric
    
    def _cleanup_checkpoints(self) -> None:
        """Remove old checkpoints beyond max_checkpoints."""
        if len(self.checkpoints) <= self.config.max_checkpoints:
            return
        
        # Sort by epoch (keep newest)
        sorted_checkpoints = sorted(self.checkpoints, key=lambda c: c.epoch)
        
        # Keep best and last checkpoints
        to_keep = set()
        if self.best_checkpoint:
            to_keep.add(self.best_checkpoint.path)
        if self.checkpoints:
            to_keep.add(self.checkpoints[-1].path)
        
        # Remove oldest checkpoints
        checkpoints_to_remove = []
        for checkpoint in sorted_checkpoints:
            if checkpoint.path not in to_keep:
                checkpoints_to_remove.append(checkpoint)
        
        while len(self.checkpoints) > self.config.max_checkpoints and checkpoints_to_remove:
            checkpoint = checkpoints_to_remove.pop(0)
            if checkpoint.path.exists():
                checkpoint.path.unlink()
            self.checkpoints.remove(checkpoint)
    
    def load(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        model: Optional[nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        scaler: Optional[torch.cuda.amp.GradScaler] = None,
        load_best: bool = False,
        load_last: bool = False,
        map_location: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Load a checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file.
            model: Model to load state into.
            optimizer: Optimizer to load state into.
            scheduler: Scheduler to load state into.
            scaler: Scaler to load state into.
            load_best: Load best checkpoint.
            load_last: Load last checkpoint.
            map_location: Device mapping for loading.
            
        Returns:
            Loaded checkpoint dictionary.
        """
        # Determine checkpoint path
        if checkpoint_path is None:
            if load_best:
                checkpoint_path = self.checkpoint_dir / "best_model.pt"
            elif load_last:
                checkpoint_path = self.checkpoint_dir / "last_model.pt"
            elif self.config.resume_from:
                checkpoint_path = self.config.resume_from
            else:
                raise ValueError("No checkpoint path specified.")
        
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        
        # Load states
        if model is not None and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        if scheduler is not None and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        if scaler is not None and "scaler_state_dict" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        
        return checkpoint
    
    def get_best_checkpoint(self) -> Optional[CheckpointInfo]:
        """Get info about the best checkpoint."""
        return self.best_checkpoint
    
    def get_last_checkpoint(self) -> Optional[CheckpointInfo]:
        """Get info about the last checkpoint."""
        return self.checkpoints[-1] if self.checkpoints else None
    
    def get_resume_state(self) -> Optional[Dict[str, Any]]:
        """Get state for resuming training."""
        if self.config.resume_from:
            checkpoint_path = self.config.resume_from
        else:
            last_path = self.checkpoint_dir / "last_model.pt"
            if not last_path.exists():
                return None
            checkpoint_path = last_path
        
        return torch.load(checkpoint_path, map_location="cpu")

"""
Experiment logging infrastructure.

Provides unified logging interface with support for console,
file, JSON, and TensorBoard logging.
"""

from __future__ import annotations

import json
import logging
import sys
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from mlcli.core.config import ExperimentConfig


class BaseLogger(ABC):
    """Abstract base class for loggers."""
    
    @abstractmethod
    def log(self, message: str, level: str = "info") -> None:
        """Log a message."""
        pass
    
    @abstractmethod
    def log_metrics(self, metrics: Dict[str, Any], step: int) -> None:
        """Log metrics at a step."""
        pass
    
    @abstractmethod
    def close(self) -> None:
        """Close the logger."""
        pass


class ConsoleLogger(BaseLogger):
    """
    Rich console logger with formatted output.
    """
    
    def __init__(self, verbose: bool = True) -> None:
        self.console = Console()
        self.verbose = verbose
        
        # Set up Python logging with rich handler
        logging.basicConfig(
            level=logging.INFO if verbose else logging.WARNING,
            format="%(message)s",
            handlers=[RichHandler(console=self.console, rich_tracebacks=True)],
        )
        self.logger = logging.getLogger("mlcli")
    
    def log(self, message: str, level: str = "info") -> None:
        """Log a message to console."""
        if not self.verbose and level == "debug":
            return
        
        getattr(self.logger, level.lower(), self.logger.info)(message)
    
    def log_metrics(self, metrics: Dict[str, Any], step: int) -> None:
        """Log metrics to console."""
        if not self.verbose:
            return
        
        # Format metrics as table
        table = Table(title=f"Step {step}")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        for key, value in sorted(metrics.items()):
            if isinstance(value, float):
                table.add_row(key, f"{value:.4f}")
            else:
                table.add_row(key, str(value))
        
        self.console.print(table)
    
    def log_progress(
        self,
        current: int,
        total: int,
        prefix: str = "",
        suffix: str = "",
    ) -> None:
        """Log progress bar."""
        from rich.progress import Progress
        
        with Progress() as progress:
            task = progress.add_task(prefix, total=total)
            progress.update(task, completed=current)
    
    def close(self) -> None:
        """Close console logger."""
        pass


class FileLogger(BaseLogger):
    """
    File logger with structured output.
    """
    
    def __init__(
        self,
        log_dir: Union[str, Path],
        experiment_name: str,
        log_level: str = "INFO",
    ) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.experiment_name = experiment_name
        self.log_file = self.log_dir / f"{experiment_name}.log"
        
        # Set up file handler
        self.logger = logging.getLogger(f"mlcli.{experiment_name}")
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        handler = logging.FileHandler(self.log_file)
        handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        self.logger.addHandler(handler)
    
    def log(self, message: str, level: str = "info") -> None:
        """Log a message to file."""
        getattr(self.logger, level.lower(), self.logger.info)(message)
    
    def log_metrics(self, metrics: Dict[str, Any], step: int) -> None:
        """Log metrics to file."""
        metrics_str = " | ".join(
            f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
            for k, v in sorted(metrics.items())
        )
        self.log(f"Step {step}: {metrics_str}")
    
    def close(self) -> None:
        """Close file logger."""
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)


class JSONLogger(BaseLogger):
    """
    JSON lines logger for structured logging.
    """
    
    def __init__(
        self,
        log_dir: Union[str, Path],
        experiment_name: str,
    ) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.experiment_name = experiment_name
        self.metrics_file = self.log_dir / f"{experiment_name}_metrics.jsonl"
        self.events_file = self.log_dir / f"{experiment_name}_events.jsonl"
        
        # Open file handles
        self._metrics_handle = open(self.metrics_file, "a")
        self._events_handle = open(self.events_file, "a")
    
    def log(self, message: str, level: str = "info") -> None:
        """Log an event to JSON."""
        event = {
            "timestamp": datetime.now().isoformat(),
            "level": level,
            "message": message,
        }
        self._events_handle.write(json.dumps(event) + "\n")
        self._events_handle.flush()
    
    def log_metrics(self, metrics: Dict[str, Any], step: int) -> None:
        """Log metrics to JSON."""
        record = {
            "timestamp": datetime.now().isoformat(),
            "step": step,
            "metrics": metrics,
        }
        self._metrics_handle.write(json.dumps(record, default=str) + "\n")
        self._metrics_handle.flush()
    
    def close(self) -> None:
        """Close JSON logger."""
        self._metrics_handle.close()
        self._events_handle.close()
    
    def load_metrics(self) -> List[Dict[str, Any]]:
        """Load all metrics from file."""
        metrics = []
        if self.metrics_file.exists():
            with open(self.metrics_file) as f:
                for line in f:
                    if line.strip():
                        metrics.append(json.loads(line))
        return metrics


class ExperimentLogger:
    """
    Unified experiment logger combining multiple backends.
    
    Provides a single interface for logging to console, file,
    JSON, and TensorBoard.
    """
    
    def __init__(
        self,
        log_dir: Union[str, Path],
        experiment_name: Optional[str] = None,
        config: Optional[ExperimentConfig] = None,
        use_console: bool = True,
        use_file: bool = True,
        use_json: bool = True,
        use_tensorboard: bool = True,
        verbose: bool = True,
    ) -> None:
        """
        Initialize experiment logger.
        
        Args:
            log_dir: Directory for log files.
            experiment_name: Experiment identifier.
            config: Experiment configuration.
            use_console: Enable console logging.
            use_file: Enable file logging.
            use_json: Enable JSON logging.
            use_tensorboard: Enable TensorBoard logging.
            verbose: Verbose console output.
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.experiment_name = experiment_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = self.log_dir / self.experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize loggers
        self.loggers: List[BaseLogger] = []
        
        if use_console:
            self.console_logger = ConsoleLogger(verbose=verbose)
            self.loggers.append(self.console_logger)
        else:
            self.console_logger = None
        
        if use_file:
            self.file_logger = FileLogger(
                self.experiment_dir,
                self.experiment_name,
            )
            self.loggers.append(self.file_logger)
        else:
            self.file_logger = None
        
        if use_json:
            self.json_logger = JSONLogger(
                self.experiment_dir,
                self.experiment_name,
            )
            self.loggers.append(self.json_logger)
        else:
            self.json_logger = None
        
        if use_tensorboard:
            try:
                from mlcli.logging.tensorboard import TensorBoardLogger
                self.tb_logger = TensorBoardLogger(
                    self.experiment_dir / "tensorboard",
                )
                self.loggers.append(self.tb_logger)
            except ImportError:
                self.tb_logger = None
        else:
            self.tb_logger = None
        
        # Save config
        if config is not None:
            self._save_config(config)
    
    def _save_config(self, config: ExperimentConfig) -> None:
        """Save experiment configuration."""
        config.save(self.experiment_dir / "config.yaml")
        
        # Also save as JSON for easy parsing
        config_path = self.experiment_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(config.to_dict(), f, indent=2, default=str)
    
    def log(self, message: str, level: str = "info") -> None:
        """Log a message to all backends."""
        for logger in self.loggers:
            logger.log(message, level)
    
    def info(self, message: str) -> None:
        """Log info message."""
        self.log(message, "info")
    
    def warning(self, message: str) -> None:
        """Log warning message."""
        self.log(message, "warning")
    
    def error(self, message: str) -> None:
        """Log error message."""
        self.log(message, "error")
    
    def debug(self, message: str) -> None:
        """Log debug message."""
        self.log(message, "debug")
    
    def log_metrics(
        self,
        metrics: Dict[str, Any],
        step: int,
        prefix: str = "",
    ) -> None:
        """Log metrics to all backends."""
        if prefix:
            metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}
        
        for logger in self.loggers:
            logger.log_metrics(metrics, step)
    
    def log_hyperparameters(self, hparams: Dict[str, Any]) -> None:
        """Log hyperparameters."""
        if self.tb_logger:
            self.tb_logger.log_hyperparameters(hparams)
        
        # Save to file
        hparams_path = self.experiment_dir / "hyperparameters.json"
        with open(hparams_path, "w") as f:
            json.dump(hparams, f, indent=2, default=str)
    
    def log_artifact(self, name: str, data: Any) -> None:
        """Log an artifact (model, figure, etc.)."""
        artifact_dir = self.experiment_dir / "artifacts"
        artifact_dir.mkdir(exist_ok=True)
        
        artifact_path = artifact_dir / name
        
        if isinstance(data, dict):
            with open(artifact_path.with_suffix(".json"), "w") as f:
                json.dump(data, f, indent=2, default=str)
        else:
            import pickle
            with open(artifact_path.with_suffix(".pkl"), "wb") as f:
                pickle.dump(data, f)
    
    def get_experiment_dir(self) -> Path:
        """Get experiment directory path."""
        return self.experiment_dir
    
    def close(self) -> None:
        """Close all loggers."""
        for logger in self.loggers:
            logger.close()
    
    def __enter__(self) -> "ExperimentLogger":
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

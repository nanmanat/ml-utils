"""
TensorBoard logging integration.

Provides TensorBoard logging for metrics, images, histograms,
and hyperparameters.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn

from mlcli.logging.logger import BaseLogger


class TensorBoardLogger(BaseLogger):
    """
    TensorBoard logger for experiment tracking.
    
    Logs scalars, images, histograms, and hyperparameters
    to TensorBoard format.
    """
    
    def __init__(
        self,
        log_dir: Union[str, Path],
        comment: str = "",
        flush_secs: int = 120,
    ) -> None:
        """
        Initialize TensorBoard logger.
        
        Args:
            log_dir: Directory for TensorBoard logs.
            comment: Comment suffix for the run.
            flush_secs: Flush interval in seconds.
        """
        from torch.utils.tensorboard import SummaryWriter
        
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.writer = SummaryWriter(
            log_dir=str(self.log_dir),
            comment=comment,
            flush_secs=flush_secs,
        )
        
        self._step = 0
    
    def log(self, message: str, level: str = "info") -> None:
        """Log a text message."""
        self.writer.add_text(
            "logs",
            f"[{level.upper()}] {message}",
            self._step,
        )
    
    def log_metrics(
        self,
        metrics: Dict[str, Any],
        step: Optional[int] = None,
    ) -> None:
        """Log scalar metrics."""
        if step is not None:
            self._step = step
        
        for name, value in metrics.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(name, value, self._step)
            elif isinstance(value, torch.Tensor) and value.numel() == 1:
                self.writer.add_scalar(name, value.item(), self._step)
    
    def log_hyperparameters(
        self,
        hparams: Dict[str, Any],
        metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        """Log hyperparameters."""
        # Flatten nested dicts
        flat_hparams = {}
        for key, value in hparams.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    flat_hparams[f"{key}/{sub_key}"] = sub_value
            else:
                flat_hparams[key] = value
        
        # Convert non-supported types to strings
        for key, value in flat_hparams.items():
            if not isinstance(value, (int, float, str, bool, torch.Tensor)):
                flat_hparams[key] = str(value)
        
        self.writer.add_hparams(
            flat_hparams,
            metrics or {},
        )
    
    def log_image(
        self,
        tag: str,
        image: torch.Tensor,
        step: Optional[int] = None,
        dataformats: str = "CHW",
    ) -> None:
        """
        Log an image.
        
        Args:
            tag: Image tag.
            image: Image tensor.
            step: Step number.
            dataformats: Data format string.
        """
        step = step if step is not None else self._step
        self.writer.add_image(tag, image, step, dataformats=dataformats)
    
    def log_images(
        self,
        tag: str,
        images: torch.Tensor,
        step: Optional[int] = None,
    ) -> None:
        """
        Log a batch of images as a grid.
        
        Args:
            tag: Image tag.
            images: Batch of images (N, C, H, W).
            step: Step number.
        """
        from torchvision.utils import make_grid
        
        step = step if step is not None else self._step
        grid = make_grid(images, normalize=True)
        self.writer.add_image(tag, grid, step)
    
    def log_histogram(
        self,
        tag: str,
        values: torch.Tensor,
        step: Optional[int] = None,
        bins: str = "tensorflow",
    ) -> None:
        """
        Log a histogram.
        
        Args:
            tag: Histogram tag.
            values: Values to histogram.
            step: Step number.
            bins: Binning strategy.
        """
        step = step if step is not None else self._step
        self.writer.add_histogram(tag, values, step, bins=bins)
    
    def log_model_graph(
        self,
        model: nn.Module,
        input_shape: tuple[int, ...],
    ) -> None:
        """
        Log model computation graph.
        
        Args:
            model: Model to visualize.
            input_shape: Input tensor shape.
        """
        dummy_input = torch.zeros(1, *input_shape)
        self.writer.add_graph(model, dummy_input)
    
    def log_gradients(
        self,
        model: nn.Module,
        step: Optional[int] = None,
    ) -> None:
        """Log gradient histograms for all parameters."""
        step = step if step is not None else self._step
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                self.writer.add_histogram(
                    f"gradients/{name}",
                    param.grad,
                    step,
                )
    
    def log_weights(
        self,
        model: nn.Module,
        step: Optional[int] = None,
    ) -> None:
        """Log weight histograms for all parameters."""
        step = step if step is not None else self._step
        
        for name, param in model.named_parameters():
            self.writer.add_histogram(
                f"weights/{name}",
                param.data,
                step,
            )
    
    def log_embedding(
        self,
        tag: str,
        embedding: torch.Tensor,
        metadata: Optional[List[str]] = None,
        label_img: Optional[torch.Tensor] = None,
        step: Optional[int] = None,
    ) -> None:
        """
        Log embedding for visualization.
        
        Args:
            tag: Embedding tag.
            embedding: Embedding tensor (N, D).
            metadata: List of labels for each sample.
            label_img: Optional images for each sample.
            step: Step number.
        """
        step = step if step is not None else self._step
        self.writer.add_embedding(
            embedding,
            metadata=metadata,
            label_img=label_img,
            global_step=step,
            tag=tag,
        )
    
    def log_pr_curve(
        self,
        tag: str,
        labels: torch.Tensor,
        predictions: torch.Tensor,
        step: Optional[int] = None,
    ) -> None:
        """
        Log precision-recall curve.
        
        Args:
            tag: Curve tag.
            labels: Ground truth labels (binary).
            predictions: Prediction probabilities.
            step: Step number.
        """
        step = step if step is not None else self._step
        self.writer.add_pr_curve(tag, labels, predictions, step)
    
    def log_confusion_matrix(
        self,
        tag: str,
        confusion_matrix: torch.Tensor,
        class_names: Optional[List[str]] = None,
        step: Optional[int] = None,
    ) -> None:
        """
        Log confusion matrix as image.
        
        Args:
            tag: Matrix tag.
            confusion_matrix: Confusion matrix tensor.
            class_names: Optional class names.
            step: Step number.
        """
        import matplotlib.pyplot as plt
        import numpy as np
        
        step = step if step is not None else self._step
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 10))
        
        cm = confusion_matrix.numpy() if isinstance(confusion_matrix, torch.Tensor) else confusion_matrix
        
        im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        
        if class_names is not None:
            ax.set(
                xticks=np.arange(cm.shape[1]),
                yticks=np.arange(cm.shape[0]),
                xticklabels=class_names,
                yticklabels=class_names,
                ylabel="True label",
                xlabel="Predicted label",
            )
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add values to cells
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                )
        
        fig.tight_layout()
        
        self.writer.add_figure(tag, fig, step)
        plt.close(fig)
    
    def flush(self) -> None:
        """Flush pending writes."""
        self.writer.flush()
    
    def close(self) -> None:
        """Close the writer."""
        self.writer.close()
    
    def __enter__(self) -> "TensorBoardLogger":
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

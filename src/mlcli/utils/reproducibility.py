"""
Reproducibility utilities for mlcli.

Provides functions for setting random seeds and ensuring
reproducible experiments.
"""

from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility across all libraries.
    
    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Set environment variable for hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)


def make_reproducible(seed: int = 42) -> None:
    """
    Configure PyTorch for fully reproducible training.
    
    Note: This may impact performance due to deterministic algorithms.
    
    Args:
        seed: Random seed value.
    """
    set_seed(seed)
    
    # Enable deterministic algorithms
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Enable deterministic operations (PyTorch 1.8+)
    if hasattr(torch, "use_deterministic_algorithms"):
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            # Some operations don't have deterministic implementations
            pass
    
    # Set environment variables for additional reproducibility
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def enable_benchmark_mode() -> None:
    """
    Enable cuDNN benchmark mode for faster training.
    
    Note: This may reduce reproducibility.
    """
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


class ReproducibilityContext:
    """
    Context manager for reproducible operations.
    
    Temporarily sets seeds and deterministic settings within a context.
    """
    
    def __init__(self, seed: int, deterministic: bool = True) -> None:
        """
        Initialize reproducibility context.
        
        Args:
            seed: Random seed.
            deterministic: Whether to use deterministic algorithms.
        """
        self.seed = seed
        self.deterministic = deterministic
        
        # Save current states
        self._saved_python_seed: Optional[int] = None
        self._saved_numpy_state: Optional[dict] = None
        self._saved_torch_state: Optional[torch.Tensor] = None
        self._saved_cuda_state: Optional[list] = None
        self._saved_cudnn_deterministic: bool = False
        self._saved_cudnn_benchmark: bool = False
    
    def __enter__(self) -> "ReproducibilityContext":
        """Enter reproducibility context."""
        # Save current states
        self._saved_numpy_state = np.random.get_state()
        self._saved_torch_state = torch.get_rng_state()
        
        if torch.cuda.is_available():
            self._saved_cuda_state = [
                torch.cuda.get_rng_state(i)
                for i in range(torch.cuda.device_count())
            ]
        
        self._saved_cudnn_deterministic = torch.backends.cudnn.deterministic
        self._saved_cudnn_benchmark = torch.backends.cudnn.benchmark
        
        # Set new seeds
        if self.deterministic:
            make_reproducible(self.seed)
        else:
            set_seed(self.seed)
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit reproducibility context and restore states."""
        # Restore states
        if self._saved_numpy_state is not None:
            np.random.set_state(self._saved_numpy_state)
        
        if self._saved_torch_state is not None:
            torch.set_rng_state(self._saved_torch_state)
        
        if self._saved_cuda_state is not None:
            for i, state in enumerate(self._saved_cuda_state):
                torch.cuda.set_rng_state(state, i)
        
        torch.backends.cudnn.deterministic = self._saved_cudnn_deterministic
        torch.backends.cudnn.benchmark = self._saved_cudnn_benchmark


def get_worker_init_fn(seed: int):
    """
    Get a worker initialization function for DataLoader.
    
    Ensures each worker has a different but reproducible seed.
    
    Args:
        seed: Base seed value.
    
    Returns:
        Worker init function.
    """
    def worker_init_fn(worker_id: int) -> None:
        worker_seed = seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)
    
    return worker_init_fn


def get_generator(seed: int) -> torch.Generator:
    """
    Create a seeded generator for DataLoader.
    
    Args:
        seed: Random seed.
    
    Returns:
        PyTorch generator.
    """
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator

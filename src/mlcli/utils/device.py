"""
Device management utilities for mlcli.

Provides utilities for device detection, selection, and management.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Union

import torch


def get_device(device: str = "auto") -> torch.device:
    """
    Get the appropriate device for computation.
    
    Args:
        device: Device specification. Options:
            - "auto": Automatically select best available device
            - "cpu": Force CPU
            - "cuda": Use CUDA (first available GPU)
            - "cuda:N": Use specific CUDA device
            - "mps": Use Apple Metal Performance Shaders
    
    Returns:
        torch.device object.
    """
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    
    return torch.device(device)


@dataclass
class GPUInfo:
    """Information about a GPU device."""
    
    index: int
    name: str
    total_memory: int  # bytes
    allocated_memory: int  # bytes
    cached_memory: int  # bytes
    compute_capability: tuple[int, int]
    
    @property
    def free_memory(self) -> int:
        """Get free memory in bytes."""
        return self.total_memory - self.allocated_memory
    
    @property
    def memory_usage_percent(self) -> float:
        """Get memory usage percentage."""
        return (self.allocated_memory / self.total_memory) * 100
    
    def __str__(self) -> str:
        total_gb = self.total_memory / (1024**3)
        free_gb = self.free_memory / (1024**3)
        return (
            f"GPU {self.index}: {self.name} "
            f"({free_gb:.1f}/{total_gb:.1f} GB free, "
            f"CC {self.compute_capability[0]}.{self.compute_capability[1]})"
        )


class DeviceManager:
    """
    Manager for device selection and resource tracking.
    
    Provides utilities for automatic device selection,
    memory management, and multi-GPU support.
    """
    
    def __init__(self, device: str = "auto") -> None:
        """
        Initialize device manager.
        
        Args:
            device: Device specification.
        """
        self.device = get_device(device)
        self._initial_memory: Optional[int] = None
    
    @property
    def is_cuda(self) -> bool:
        """Check if using CUDA."""
        return self.device.type == "cuda"
    
    @property
    def is_mps(self) -> bool:
        """Check if using MPS."""
        return self.device.type == "mps"
    
    @property
    def is_cpu(self) -> bool:
        """Check if using CPU."""
        return self.device.type == "cpu"
    
    @staticmethod
    def get_available_gpus() -> List[GPUInfo]:
        """Get information about all available GPUs."""
        if not torch.cuda.is_available():
            return []
        
        gpus = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            gpus.append(GPUInfo(
                index=i,
                name=props.name,
                total_memory=props.total_memory,
                allocated_memory=torch.cuda.memory_allocated(i),
                cached_memory=torch.cuda.memory_reserved(i),
                compute_capability=(props.major, props.minor),
            ))
        
        return gpus
    
    @staticmethod
    def get_best_gpu() -> Optional[int]:
        """
        Get the GPU index with the most free memory.
        
        Returns:
            GPU index or None if no GPUs available.
        """
        gpus = DeviceManager.get_available_gpus()
        if not gpus:
            return None
        
        return max(gpus, key=lambda g: g.free_memory).index
    
    def to_device(
        self,
        data: Union[torch.Tensor, torch.nn.Module, list, tuple, dict],
    ) -> Union[torch.Tensor, torch.nn.Module, list, tuple, dict]:
        """
        Move data to the managed device.
        
        Args:
            data: Tensor, model, or nested structure of tensors.
        
        Returns:
            Data on the target device.
        """
        if isinstance(data, torch.Tensor):
            return data.to(self.device)
        elif isinstance(data, torch.nn.Module):
            return data.to(self.device)
        elif isinstance(data, (list, tuple)):
            return type(data)(self.to_device(item) for item in data)
        elif isinstance(data, dict):
            return {key: self.to_device(value) for key, value in data.items()}
        else:
            return data
    
    def memory_allocated(self) -> int:
        """Get currently allocated memory in bytes."""
        if self.is_cuda:
            return torch.cuda.memory_allocated(self.device)
        return 0
    
    def memory_cached(self) -> int:
        """Get cached memory in bytes."""
        if self.is_cuda:
            return torch.cuda.memory_reserved(self.device)
        return 0
    
    def empty_cache(self) -> None:
        """Empty the cache to free memory."""
        if self.is_cuda:
            torch.cuda.empty_cache()
        elif self.is_mps:
            torch.mps.empty_cache()
    
    def synchronize(self) -> None:
        """Synchronize device operations."""
        if self.is_cuda:
            torch.cuda.synchronize(self.device)
        elif self.is_mps:
            torch.mps.synchronize()
    
    def start_memory_tracking(self) -> None:
        """Start tracking memory usage."""
        if self.is_cuda:
            torch.cuda.reset_peak_memory_stats(self.device)
            self._initial_memory = self.memory_allocated()
    
    def get_memory_usage(self) -> dict:
        """
        Get memory usage statistics.
        
        Returns:
            Dictionary with memory statistics.
        """
        if not self.is_cuda:
            return {}
        
        return {
            "allocated": self.memory_allocated(),
            "cached": self.memory_cached(),
            "peak": torch.cuda.max_memory_allocated(self.device),
            "initial": self._initial_memory,
        }
    
    def print_memory_summary(self) -> None:
        """Print memory usage summary."""
        if not self.is_cuda:
            print(f"Device: {self.device} (no memory tracking available)")
            return
        
        stats = self.get_memory_usage()
        print(f"\nMemory Summary ({self.device}):")
        print(f"  Allocated: {stats['allocated'] / 1024**2:.1f} MB")
        print(f"  Cached: {stats['cached'] / 1024**2:.1f} MB")
        print(f"  Peak: {stats['peak'] / 1024**2:.1f} MB")
    
    def __str__(self) -> str:
        return f"DeviceManager(device={self.device})"
    
    def __repr__(self) -> str:
        return self.__str__()


class DataParallelWrapper:
    """
    Wrapper for data parallel training across multiple GPUs.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        device_ids: Optional[List[int]] = None,
        output_device: Optional[int] = None,
    ) -> None:
        """
        Initialize data parallel wrapper.
        
        Args:
            model: Model to wrap.
            device_ids: GPU device IDs to use.
            output_device: Device for output.
        """
        self.model = model
        
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            if device_ids is None:
                device_ids = list(range(torch.cuda.device_count()))
            
            self.parallel_model = torch.nn.DataParallel(
                model,
                device_ids=device_ids,
                output_device=output_device,
            )
            self.device = torch.device(f"cuda:{device_ids[0]}")
        else:
            self.parallel_model = model
            self.device = get_device("auto")
        
        self.parallel_model = self.parallel_model.to(self.device)
    
    @property
    def module(self) -> torch.nn.Module:
        """Get the underlying module."""
        if isinstance(self.parallel_model, torch.nn.DataParallel):
            return self.parallel_model.module
        return self.parallel_model
    
    def __call__(self, *args, **kwargs):
        return self.parallel_model(*args, **kwargs)
    
    def parameters(self):
        return self.model.parameters()
    
    def train(self, mode: bool = True):
        self.parallel_model.train(mode)
        return self
    
    def eval(self):
        self.parallel_model.eval()
        return self

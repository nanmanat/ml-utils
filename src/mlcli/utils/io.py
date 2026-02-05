"""
I/O utilities for mlcli.

Provides utilities for loading and saving configurations,
models, and other data.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch


def save_config(
    config: Dict[str, Any],
    path: Union[str, Path],
    format: Optional[str] = None,
) -> None:
    """
    Save configuration to file.
    
    Args:
        config: Configuration dictionary.
        path: Output file path.
        format: File format (yaml, json, toml). Auto-detected from extension if not specified.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    if format is None:
        format = path.suffix.lstrip(".").lower()
    
    if format == "json":
        with open(path, "w") as f:
            json.dump(config, f, indent=2, default=str)
    
    elif format in ("yaml", "yml"):
        import yaml
        with open(path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    elif format == "toml":
        try:
            import tomli_w
            with open(path, "wb") as f:
                tomli_w.dump(config, f)
        except ImportError:
            raise ImportError("tomli_w is required for TOML writing. Install with: pip install tomli_w")
    
    else:
        raise ValueError(f"Unsupported format: {format}")


def load_config(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from file.
    
    Args:
        path: Configuration file path.
    
    Returns:
        Configuration dictionary.
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    
    suffix = path.suffix.lower()
    
    if suffix == ".json":
        with open(path) as f:
            return json.load(f)
    
    elif suffix in (".yaml", ".yml"):
        import yaml
        with open(path) as f:
            return yaml.safe_load(f)
    
    elif suffix == ".toml":
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib
        
        with open(path, "rb") as f:
            return tomllib.load(f)
    
    else:
        raise ValueError(f"Unsupported configuration format: {suffix}")


def save_checkpoint(
    path: Union[str, Path],
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    epoch: int = 0,
    metrics: Optional[Dict[str, float]] = None,
    config: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> None:
    """
    Save a training checkpoint.
    
    Args:
        path: Checkpoint file path.
        model: Model to save.
        optimizer: Optimizer state.
        scheduler: Scheduler state.
        epoch: Current epoch.
        metrics: Training metrics.
        config: Configuration dictionary.
        **kwargs: Additional items to save.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
    }
    
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    
    if metrics is not None:
        checkpoint["metrics"] = metrics
    
    if config is not None:
        checkpoint["config"] = config
    
    checkpoint.update(kwargs)
    
    torch.save(checkpoint, path)


def load_checkpoint(
    path: Union[str, Path],
    model: Optional[torch.nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    device: Optional[Union[str, torch.device]] = None,
    strict: bool = True,
) -> Dict[str, Any]:
    """
    Load a training checkpoint.
    
    Args:
        path: Checkpoint file path.
        model: Model to load weights into.
        optimizer: Optimizer to load state into.
        scheduler: Scheduler to load state into.
        device: Device to load checkpoint to.
        strict: Whether to enforce strict loading.
    
    Returns:
        Checkpoint dictionary with additional data.
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    
    map_location = device if device else "cpu"
    checkpoint = torch.load(path, map_location=map_location)
    
    if model is not None and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
    
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    
    return checkpoint


def export_model(
    model: torch.nn.Module,
    path: Union[str, Path],
    input_shape: tuple,
    format: str = "onnx",
    opset_version: int = 11,
    dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
) -> None:
    """
    Export model to deployment format.
    
    Args:
        model: Model to export.
        path: Output file path.
        input_shape: Model input shape (without batch dimension).
        format: Export format (onnx, torchscript).
        opset_version: ONNX opset version.
        dynamic_axes: Dynamic axes for ONNX export.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    model.eval()
    dummy_input = torch.zeros(1, *input_shape)
    
    if format == "onnx":
        if dynamic_axes is None:
            dynamic_axes = {"input": {0: "batch_size"}, "output": {0: "batch_size"}}
        
        torch.onnx.export(
            model,
            dummy_input,
            path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes=dynamic_axes,
        )
    
    elif format == "torchscript":
        traced_model = torch.jit.trace(model, dummy_input)
        traced_model.save(str(path))
    
    else:
        raise ValueError(f"Unsupported export format: {format}")


def download_file(
    url: str,
    path: Union[str, Path],
    chunk_size: int = 8192,
    show_progress: bool = True,
) -> Path:
    """
    Download a file from URL.
    
    Args:
        url: URL to download from.
        path: Output file path.
        chunk_size: Download chunk size.
        show_progress: Whether to show progress bar.
    
    Returns:
        Downloaded file path.
    """
    import urllib.request
    
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    if show_progress:
        from tqdm import tqdm
        
        with urllib.request.urlopen(url) as response:
            total_size = int(response.headers.get("content-length", 0))
            
            with open(path, "wb") as f:
                with tqdm(total=total_size, unit="B", unit_scale=True, desc=path.name) as pbar:
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        pbar.update(len(chunk))
    else:
        urllib.request.urlretrieve(url, path)
    
    return path


def hash_file(path: Union[str, Path], algorithm: str = "sha256") -> str:
    """
    Compute hash of a file.
    
    Args:
        path: File path.
        algorithm: Hash algorithm (md5, sha1, sha256).
    
    Returns:
        Hex digest of hash.
    """
    import hashlib
    
    path = Path(path)
    hasher = hashlib.new(algorithm)
    
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    
    return hasher.hexdigest()

"""
Main CLI entry point for mlcli.

Provides the main CLI group and common options.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import click

from mlcli import __version__

# Import models to trigger registration
import mlcli.models  # noqa: F401


class Context:
    """CLI context object to pass state between commands."""
    
    def __init__(self) -> None:
        self.verbose: bool = False
        self.config_path: Optional[Path] = None
        self.output_dir: Optional[Path] = None
        self.device: str = "auto"
        self.seed: Optional[int] = None


pass_context = click.make_pass_decorator(Context, ensure=True)


def print_version(ctx: click.Context, param: click.Parameter, value: bool) -> None:
    """Print version and exit."""
    if not value or ctx.resilient_parsing:
        return
    click.echo(f"mlcli version {__version__}")
    ctx.exit()


@click.group()
@click.option(
    "-v", "--verbose",
    is_flag=True,
    default=False,
    help="Enable verbose output.",
)
@click.option(
    "-c", "--config",
    type=click.Path(exists=True, path_type=Path),
    help="Path to configuration file.",
)
@click.option(
    "-o", "--output-dir",
    type=click.Path(path_type=Path),
    help="Output directory for results.",
)
@click.option(
    "-d", "--device",
    type=click.Choice(["auto", "cpu", "cuda", "mps"]),
    default="auto",
    help="Device to use for computation.",
)
@click.option(
    "-s", "--seed",
    type=int,
    default=None,
    help="Random seed for reproducibility.",
)
@click.option(
    "--version",
    is_flag=True,
    callback=print_version,
    expose_value=False,
    is_eager=True,
    help="Show version and exit.",
)
@pass_context
def cli(
    ctx: Context,
    verbose: bool,
    config: Optional[Path],
    output_dir: Optional[Path],
    device: str,
    seed: Optional[int],
) -> None:
    """
    MLCLI - Production-grade Machine Learning CLI Tool.
    
    A modular CLI for training, evaluating, and running inference
    with various ML models for image classification and object detection.
    
    \b
    Supported Model Families:
      - CNN: ResNet, EfficientNet, MobileNet
      - Transformer: ViT, Swin Transformer
      - Detection: YOLO, Faster R-CNN, SSD, DETR
    
    \b
    Examples:
      mlcli train --model resnet50 --dataset ./data --epochs 100
      mlcli evaluate --checkpoint ./model.pt --dataset ./test
      mlcli inference --model ./model.pt --input ./images
    """
    ctx.verbose = verbose
    ctx.config_path = config
    ctx.output_dir = output_dir
    ctx.device = device
    ctx.seed = seed
    
    if verbose:
        click.echo(f"Verbose mode enabled")
        click.echo(f"Device: {device}")
        if config:
            click.echo(f"Config: {config}")
        if output_dir:
            click.echo(f"Output directory: {output_dir}")


# Import and register subcommands
from mlcli.cli.train import train
from mlcli.cli.evaluate import evaluate
from mlcli.cli.inference import inference
from mlcli.cli.interactive import interactive

cli.add_command(train)
cli.add_command(evaluate)
cli.add_command(inference)
cli.add_command(interactive)


@cli.command()
@pass_context
def info(ctx: Context) -> None:
    """Display system and environment information."""
    import torch
    
    click.echo("\n" + "=" * 50)
    click.echo("MLCLI System Information")
    click.echo("=" * 50)
    
    click.echo(f"\nVersion: {__version__}")
    click.echo(f"Python: {sys.version}")
    click.echo(f"PyTorch: {torch.__version__}")
    
    # Device information
    click.echo(f"\nCUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        click.echo(f"CUDA version: {torch.version.cuda}")
        click.echo(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            click.echo(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    
    click.echo(f"MPS available: {torch.backends.mps.is_available()}")
    
    # Available models
    click.echo("\n" + "-" * 50)
    click.echo("Registered Models:")
    click.echo("-" * 50)
    
    from mlcli.core.registry import MODEL_REGISTRY
    for name in sorted(MODEL_REGISTRY.list_registered()):
        click.echo(f"  - {name}")
    
    click.echo("\n")


@cli.command()
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["yaml", "json", "toml"]),
    default="yaml",
    help="Configuration file format.",
)
@click.option(
    "--task",
    type=click.Choice(["classification", "detection"]),
    default="classification",
    help="Task type for the configuration.",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    default=None,
    help="Output file path.",
)
def init(output_format: str, task: str, output: Optional[Path]) -> None:
    """Generate a template configuration file."""
    
    config_template = {
        "experiment": {
            "name": "my_experiment",
            "seed": 42,
            "device": "auto",
            "output_dir": "./outputs",
            "reproducible": True,
        },
        "model": {
            "task_type": task,
            "model_type": "cnn",
            "architecture": "resnet50" if task == "classification" else "yolo",
            "num_classes": 10 if task == "classification" else 80,
            "pretrained": True,
        },
        "training": {
            "epochs": 100,
            "batch_size": 32,
            "learning_rate": 0.001,
            "optimizer": "adamw",
            "scheduler": "cosine",
            "mixed_precision": True,
            "gradient_clip": 1.0,
            "early_stopping_patience": 10,
        },
        "dataset": {
            "train_path": "./data/train",
            "val_path": "./data/val",
            "test_path": "./data/test",
            "num_workers": 4,
        },
        "augmentation": {
            "train": {
                "random_crop": True,
                "horizontal_flip": True,
                "color_jitter": True,
            },
            "val": {
                "center_crop": True,
            },
        },
        "logging": {
            "console": True,
            "file": True,
            "tensorboard": True,
            "log_interval": 100,
        },
    }
    
    if output_format == "yaml":
        import yaml
        content = yaml.dump(config_template, default_flow_style=False, sort_keys=False)
        default_name = "config.yaml"
    elif output_format == "json":
        import json
        content = json.dumps(config_template, indent=2)
        default_name = "config.json"
    else:  # toml
        try:
            import tomli_w
            content = tomli_w.dumps(config_template)
        except ImportError:
            # Fallback to simple TOML generation
            content = _dict_to_toml(config_template)
        default_name = "config.toml"
    
    if output:
        output.write_text(content)
        click.echo(f"Configuration written to: {output}")
    else:
        click.echo(content)


def _dict_to_toml(data: dict, prefix: str = "") -> str:
    """Simple TOML serialization."""
    lines = []
    
    for key, value in data.items():
        if isinstance(value, dict):
            section = f"{prefix}{key}" if not prefix else f"{prefix}.{key}"
            lines.append(f"\n[{section}]")
            for k, v in value.items():
                if isinstance(v, dict):
                    lines.append(_dict_to_toml({k: v}, section))
                else:
                    lines.append(f"{k} = {_toml_value(v)}")
        else:
            lines.append(f"{key} = {_toml_value(value)}")
    
    return "\n".join(lines)


def _toml_value(value) -> str:
    """Convert Python value to TOML string."""
    if isinstance(value, bool):
        return "true" if value else "false"
    elif isinstance(value, str):
        return f'"{value}"'
    elif isinstance(value, (int, float)):
        return str(value)
    else:
        return f'"{value}"'


@cli.command()
@click.argument("model_name")
def describe(model_name: str) -> None:
    """Describe a registered model architecture."""
    from mlcli.core.registry import MODEL_REGISTRY
    
    if model_name not in MODEL_REGISTRY.list_registered():
        click.echo(f"Error: Model '{model_name}' not found.", err=True)
        click.echo("\nAvailable models:")
        for name in sorted(MODEL_REGISTRY.list_registered()):
            click.echo(f"  - {name}")
        sys.exit(1)
    
    model_cls = MODEL_REGISTRY.get(model_name)
    
    click.echo(f"\n{'=' * 50}")
    click.echo(f"Model: {model_name}")
    click.echo(f"{'=' * 50}")
    
    if model_cls.__doc__:
        click.echo(f"\nDescription:\n{model_cls.__doc__}")
    
    click.echo(f"\nClass: {model_cls.__module__}.{model_cls.__name__}")
    
    # Try to get model info
    try:
        import inspect
        sig = inspect.signature(model_cls.__init__)
        click.echo("\nParameters:")
        for name, param in sig.parameters.items():
            if name == "self":
                continue
            default = param.default if param.default != inspect.Parameter.empty else "required"
            click.echo(f"  - {name}: {default}")
    except Exception:
        pass
    
    click.echo("\n")


def main() -> None:
    """Main entry point."""
    cli()


if __name__ == "__main__":
    main()

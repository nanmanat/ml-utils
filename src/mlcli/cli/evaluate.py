"""
Evaluate command for mlcli.

Provides model evaluation functionality on test datasets.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import click

from mlcli.cli.main import Context, pass_context


@click.command()
@click.option(
    "--checkpoint", "-c",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to model checkpoint.",
)
@click.option(
    "--dataset", "-d",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to evaluation dataset.",
)
@click.option(
    "--task",
    type=click.Choice(["classification", "detection"]),
    default=None,
    help="Task type (auto-detected if not specified).",
)
@click.option(
    "--batch-size", "-b",
    type=int,
    default=32,
    help="Evaluation batch size.",
)
@click.option(
    "--num-workers",
    type=int,
    default=4,
    help="Number of data loading workers.",
)
@click.option(
    "--output", "-o",
    type=click.Path(path_type=Path),
    default=None,
    help="Output file for results (JSON).",
)
@click.option(
    "--confusion-matrix/--no-confusion-matrix",
    default=False,
    help="Generate confusion matrix (classification only).",
)
@click.option(
    "--per-class/--no-per-class",
    default=False,
    help="Show per-class metrics.",
)
@click.option(
    "--iou-threshold",
    type=float,
    default=0.5,
    help="IoU threshold for detection (default: 0.5).",
)
@pass_context
def evaluate(
    ctx: Context,
    checkpoint: Path,
    dataset: Path,
    task: Optional[str],
    batch_size: int,
    num_workers: int,
    output: Optional[Path],
    confusion_matrix: bool,
    per_class: bool,
    iou_threshold: float,
) -> None:
    """
    Evaluate a trained model on a dataset.
    
    \b
    Examples:
      mlcli evaluate --checkpoint ./model.pt --dataset ./data/test
      mlcli evaluate -c ./model.pt -d ./test --per-class
      mlcli evaluate -c ./detection.pt -d ./coco/val --iou-threshold 0.75
    """
    import json
    import torch
    
    from mlcli.core.factory import ModelFactory
    from mlcli.core.config import ModelConfig
    from mlcli.logging.metrics import ClassificationMetrics, DetectionMetrics
    
    click.echo("\n" + "=" * 60)
    click.echo("MLCLI Model Evaluation")
    click.echo("=" * 60)
    
    # Set up device
    if ctx.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = ctx.device
    
    click.echo(f"\nDevice: {device}")
    click.echo(f"Checkpoint: {checkpoint}")
    click.echo(f"Dataset: {dataset}")
    
    # Load checkpoint
    click.echo("\nLoading checkpoint...")
    ckpt = torch.load(checkpoint, map_location=device)
    
    # Get model config from checkpoint
    if "model_config" in ckpt:
        model_config = ModelConfig(**ckpt["model_config"])
        task = task or model_config.task_type
    else:
        if task is None:
            click.echo("Error: Task type not found in checkpoint. Please specify --task.", err=True)
            sys.exit(1)
        model_config = None
    
    click.echo(f"Task: {task}")
    
    # Load dataset
    click.echo(f"\nLoading dataset...")
    
    if task == "classification":
        from mlcli.data.classification import ImageFolderDataset
        test_dataset = ImageFolderDataset(
            root=dataset,
            transform=None,
        )
        num_classes = test_dataset.get_num_classes()
        class_names = test_dataset.get_class_names()
    else:
        from mlcli.data.detection import COCODataset
        test_dataset = COCODataset(
            root=dataset,
            annotation_file=dataset / "annotations.json",
            transform=None,
        )
        num_classes = test_dataset.get_num_classes()
        class_names = test_dataset.get_class_names()
    
    click.echo(f"Samples: {len(test_dataset)}")
    click.echo(f"Classes: {num_classes}")
    
    # Create data loader
    test_loader = test_dataset.create_dataloader(
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    
    # Create model
    if model_config:
        model = ModelFactory.create(model_config)
    else:
        # Try to infer model architecture from checkpoint
        click.echo("Warning: Model config not found, attempting to load state dict directly.", err=True)
        raise NotImplementedError("Please provide a checkpoint with model config.")
    
    # Load weights
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    
    click.echo("\n" + "-" * 60)
    click.echo("Running evaluation...")
    click.echo("-" * 60)
    
    # Initialize metrics
    if task == "classification":
        metrics = ClassificationMetrics(num_classes=num_classes)
    else:
        metrics = DetectionMetrics(
            num_classes=num_classes,
            iou_threshold=iou_threshold,
        )
    
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if task == "classification":
                images, labels = batch
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = model(images)
                predictions = torch.argmax(outputs, dim=1)
                
                all_predictions.extend(predictions.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())
                
                metrics.update(predictions, labels)
            else:
                images, targets = batch
                images = [img.to(device) for img in images]
                
                predictions = model(images)
                
                for pred, target in zip(predictions, targets):
                    metrics.update(pred, target)
            
            if ctx.verbose and (batch_idx + 1) % 10 == 0:
                click.echo(f"Processed {batch_idx + 1}/{len(test_loader)} batches")
    
    # Compute final metrics
    results = metrics.compute()
    
    click.echo("\n" + "=" * 60)
    click.echo("Evaluation Results")
    click.echo("=" * 60)
    
    if task == "classification":
        click.echo(f"\nAccuracy: {results['accuracy']:.4f}")
        click.echo(f"Precision: {results['precision']:.4f}")
        click.echo(f"Recall: {results['recall']:.4f}")
        click.echo(f"F1 Score: {results['f1']:.4f}")
        
        if per_class and "per_class" in results:
            click.echo("\n" + "-" * 40)
            click.echo("Per-Class Metrics")
            click.echo("-" * 40)
            
            for i, class_name in enumerate(class_names or range(num_classes)):
                class_metrics = results["per_class"].get(i, {})
                click.echo(f"\n{class_name}:")
                click.echo(f"  Precision: {class_metrics.get('precision', 0):.4f}")
                click.echo(f"  Recall: {class_metrics.get('recall', 0):.4f}")
                click.echo(f"  F1: {class_metrics.get('f1', 0):.4f}")
        
        if confusion_matrix:
            click.echo("\n" + "-" * 40)
            click.echo("Confusion Matrix")
            click.echo("-" * 40)
            
            # Build confusion matrix
            import numpy as np
            cm = np.zeros((num_classes, num_classes), dtype=int)
            for pred, label in zip(all_predictions, all_labels):
                cm[label][pred] += 1
            
            # Print confusion matrix
            header = "     " + "  ".join(f"{i:4d}" for i in range(min(10, num_classes)))
            click.echo(f"\n{header}")
            for i in range(min(10, num_classes)):
                row = "  ".join(f"{cm[i][j]:4d}" for j in range(min(10, num_classes)))
                click.echo(f"{i:4d} {row}")
            
            if num_classes > 10:
                click.echo(f"  ... (showing first 10 classes of {num_classes})")
    
    else:  # detection
        click.echo(f"\nmAP@{iou_threshold}: {results.get('mAP', 0):.4f}")
        click.echo(f"mAP@0.5: {results.get('mAP50', 0):.4f}")
        click.echo(f"mAP@0.75: {results.get('mAP75', 0):.4f}")
        click.echo(f"mAP@[0.5:0.95]: {results.get('mAP50_95', 0):.4f}")
        
        if per_class and "per_class_ap" in results:
            click.echo("\n" + "-" * 40)
            click.echo("Per-Class AP")
            click.echo("-" * 40)
            
            for i, class_name in enumerate(class_names or range(num_classes)):
                ap = results["per_class_ap"].get(i, 0)
                click.echo(f"  {class_name}: {ap:.4f}")
    
    # Save results
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        
        output_data = {
            "checkpoint": str(checkpoint),
            "dataset": str(dataset),
            "task": task,
            "num_samples": len(test_dataset),
            "metrics": results,
        }
        
        with open(output, "w") as f:
            json.dump(output_data, f, indent=2, default=str)
        
        click.echo(f"\nResults saved to: {output}")
    
    click.echo("\n")

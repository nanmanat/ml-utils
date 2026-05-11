"""
Train command for mlcli.

Provides the training pipeline command with all configuration options.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import click

from mlcli.cli.main import Context, pass_context


@click.command()
@click.option(
    "--model",
    "-m",
    type=str,
    required=True,
    help="Model architecture (e.g., resnet50, vit_base, yolo).",
)
@click.option(
    "--dataset",
    "-d",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to training dataset.",
)
@click.option(
    "--val-dataset",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to validation dataset (optional).",
)
@click.option(
    "--task",
    type=click.Choice(["classification", "detection"]),
    default="classification",
    help="Task type.",
)
@click.option(
    "--epochs",
    "-e",
    type=int,
    default=100,
    help="Number of training epochs.",
)
@click.option(
    "--batch-size",
    "-b",
    type=int,
    default=32,
    help="Training batch size.",
)
@click.option(
    "--learning-rate",
    "--lr",
    type=float,
    default=0.001,
    help="Initial learning rate.",
)
@click.option(
    "--optimizer",
    type=click.Choice(["adam", "adamw", "sgd"]),
    default="adamw",
    help="Optimizer to use.",
)
@click.option(
    "--scheduler",
    type=click.Choice(["step", "cosine", "plateau", "warmup_cosine", "none"]),
    default="cosine",
    help="Learning rate scheduler.",
)
@click.option(
    "--weight-decay",
    type=float,
    default=0.01,
    help="Weight decay (L2 regularization).",
)
@click.option(
    "--num-classes",
    type=int,
    default=None,
    help="Number of output classes (auto-detected if not specified).",
)
@click.option(
    "--pretrained/--no-pretrained",
    default=True,
    help="Use pretrained weights.",
)
@click.option(
    "--freeze-backbone/--no-freeze-backbone",
    default=False,
    help="Freeze backbone weights.",
)
@click.option(
    "--mixed-precision/--no-mixed-precision",
    default=True,
    help="Enable mixed precision training.",
)
@click.option(
    "--gradient-clip",
    type=float,
    default=None,
    help="Gradient clipping value.",
)
@click.option(
    "--early-stopping",
    type=int,
    default=None,
    help="Early stopping patience (epochs).",
)
@click.option(
    "--checkpoint-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Directory for saving checkpoints.",
)
@click.option(
    "--resume",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Resume training from checkpoint.",
)
@click.option(
    "--num-workers",
    type=int,
    default=4,
    help="Number of data loading workers.",
)
@click.option(
    "--log-interval",
    type=int,
    default=100,
    help="Logging interval (batches).",
)
@click.option(
    "--tensorboard/--no-tensorboard",
    default=True,
    help="Enable TensorBoard logging.",
)
@click.option(
    "--experiment-name",
    type=str,
    default=None,
    help="Experiment name for logging.",
)
@click.option(
    "--wandb/--no-wandb",
    default=False,
    help="Enable Weights & Biases logging.",
)
@click.option(
    "--wandb-project",
    type=str,
    default=None,
    help="W&B project name.",
)
@pass_context
def train(
    ctx: Context,
    model: str,
    dataset: Path,
    val_dataset: Optional[Path],
    task: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    optimizer: str,
    scheduler: str,
    weight_decay: float,
    num_classes: Optional[int],
    pretrained: bool,
    freeze_backbone: bool,
    mixed_precision: bool,
    gradient_clip: Optional[float],
    early_stopping: Optional[int],
    checkpoint_dir: Optional[Path],
    resume: Optional[Path],
    num_workers: int,
    log_interval: int,
    tensorboard: bool,
    experiment_name: Optional[str],
    wandb: bool,
    wandb_project: Optional[str],
) -> None:
    """
    Train a model on a dataset.

    \b
    Examples:
      mlcli train --model resnet50 --dataset ./data/train --epochs 100
      mlcli train -m vit_base -d ./data --task classification --lr 0.0001
      mlcli train --model yolo --dataset ./coco --task detection
    """
    import torch

    from tqdm import tqdm

    from mlcli.core.config import ModelConfig, TrainingConfig, TaskType, ModelFamily
    from mlcli.core.factory import ModelFactory

    click.echo("\n" + "=" * 60)
    click.echo("MLCLI Training Pipeline")
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

    # Set seed for reproducibility
    seed = ctx.seed or 42
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)

    # Determine output directory
    output_dir = ctx.output_dir or checkpoint_dir or Path("./outputs")
    output_dir = output_dir / (experiment_name or f"{model}_{task}")
    output_dir.mkdir(parents=True, exist_ok=True)

    click.echo(f"Output directory: {output_dir}")

    # Load datasets
    click.echo(f"\nLoading dataset from: {dataset}")

    if task == "classification":
        from mlcli.data.classification import ImageFolderDataset
        from torchvision import transforms

        train_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

        train_dataset = ImageFolderDataset(
            root=dataset,
            transform=train_transform,
        )
        num_classes = num_classes or train_dataset.get_num_classes()

        if val_dataset:
            val_data = ImageFolderDataset(
                root=val_dataset,
                transform=train_transform,
            )
        else:
            val_data = None
    else:
        from mlcli.data.detection import COCODataset

        train_dataset = COCODataset(
            root=dataset,
            annotation_file=dataset / "annotations.json",
            transform=None,
        )
        num_classes = num_classes or train_dataset.get_num_classes()

        if val_dataset:
            val_data = COCODataset(
                root=val_dataset,
                annotation_file=val_dataset / "annotations.json",
                transform=None,
            )
        else:
            val_data = None

    click.echo(f"Training samples: {len(train_dataset)}")
    click.echo(f"Number of classes: {num_classes}")
    if val_data:
        click.echo(f"Validation samples: {len(val_data)}")

    # Determine task type and model family
    model_lower = model.lower()
    is_detection = task == "detection"
    is_transformer = any(x in model_lower for x in ["vit", "swin", "detr"])

    task_type = TaskType.OBJECT_DETECTION if is_detection else TaskType.CLASSIFICATION
    family = ModelFamily.TRANSFORMER if is_transformer else ModelFamily.CNN

    # Create model config
    model_config = ModelConfig(
        task_type=task_type,
        family=family,
        architecture=model,
        num_classes=num_classes,
        pretrained=pretrained,
    )

    # Create model
    click.echo(f"\nCreating model: {model}")
    model_instance = ModelFactory.create(model_config)

    if freeze_backbone:
        model_instance.freeze_backbone()
        click.echo("Backbone frozen")

    # Move model to device
    model_instance = model_instance.to(device)

    total_params = model_instance.get_num_parameters()
    trainable_params = model_instance.get_num_parameters(trainable_only=True)
    click.echo(f"Total parameters: {total_params:,}")
    click.echo(f"Trainable parameters: {trainable_params:,}")

    # Create training config
    training_config = TrainingConfig(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        optimizer=optimizer,
        scheduler=scheduler if scheduler != "none" else None,
        weight_decay=weight_decay,
        mixed_precision=mixed_precision,
        gradient_clip=gradient_clip,
    )

    click.echo(f"\nTraining configuration:")
    click.echo(f"  Epochs: {epochs}")
    click.echo(f"  Batch size: {batch_size}")
    click.echo(f"  Learning rate: {learning_rate}")
    click.echo(f"  Optimizer: {optimizer}")
    click.echo(f"  Scheduler: {scheduler}")
    click.echo(f"  Mixed precision: {mixed_precision}")

    # Create data loaders
    train_loader = train_dataset.create_dataloader(
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    val_loader = None
    if val_data:
        val_loader = val_data.create_dataloader(
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

    # Create optimizer and scheduler
    from mlcli.training.optimizer import create_optimizer, create_scheduler

    optim = create_optimizer(
        model_instance,
        optimizer_name=optimizer,
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    sched = None
    if scheduler and scheduler != "none":
        sched = create_scheduler(
            optim,
            scheduler_name=scheduler,
            epochs=epochs,
        )

    # Resume from checkpoint if specified
    start_epoch = 0
    if resume:
        click.echo(f"\nResuming from checkpoint: {resume}")
        checkpoint = torch.load(resume)
        model_instance.load_state_dict(checkpoint["model_state_dict"])
        optim.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        click.echo(f"Resuming from epoch {start_epoch}")

    # Define loss function
    criterion = torch.nn.CrossEntropyLoss() if task == "classification" else None

    click.echo("\n" + "-" * 60)
    click.echo("Starting training...")
    click.echo("-" * 60 + "\n")

    try:
        # Training loop
        if criterion is None:
            raise NotImplementedError("Detection training path is not implemented in this command.")

        checkpoints_dir = output_dir / "checkpoints"
        checkpoints_dir.mkdir(parents=True, exist_ok=True)

        use_amp = mixed_precision and device == "cuda"
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        best_monitor = float("inf")
        no_improve_epochs = 0

        for epoch_idx in range(start_epoch, epochs):
            model_instance.train()
            running_loss = 0.0
            running_correct = 0
            running_total = 0

            progress = tqdm(
                train_loader,
                desc=f"Epoch {epoch_idx + 1}/{epochs}",
                leave=True,
            )

            for batch_idx, (images, labels) in enumerate(progress):
                images = images.to(device)
                labels = labels.to(device)

                optim.zero_grad()

                if use_amp:
                    with torch.cuda.amp.autocast(enabled=True):
                        outputs = model_instance(images)
                        loss = criterion(outputs, labels)
                    scaler.scale(loss).backward()
                    if gradient_clip is not None:
                        scaler.unscale_(optim)
                        torch.nn.utils.clip_grad_norm_(model_instance.parameters(), gradient_clip)
                    scaler.step(optim)
                    scaler.update()
                else:
                    outputs = model_instance(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    if gradient_clip is not None:
                        torch.nn.utils.clip_grad_norm_(model_instance.parameters(), gradient_clip)
                    optim.step()

                running_loss += loss.item()
                _, predictions = outputs.max(1)
                running_total += labels.size(0)
                running_correct += predictions.eq(labels).sum().item()

                if (batch_idx + 1) % max(1, log_interval) == 0:
                    avg_loss = running_loss / (batch_idx + 1)
                    avg_acc = 100.0 * running_correct / max(1, running_total)
                    progress.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{avg_acc:.2f}%")

            epoch_train_loss = running_loss / max(1, len(train_loader))
            epoch_train_acc = 100.0 * running_correct / max(1, running_total)

            # Validation
            epoch_val_loss = None
            epoch_val_acc = None
            if val_loader is not None:
                model_instance.eval()
                val_loss_sum = 0.0
                val_correct = 0
                val_total = 0
                with torch.no_grad():
                    for images, labels in val_loader:
                        images = images.to(device)
                        labels = labels.to(device)
                        outputs = model_instance(images)
                        val_loss = criterion(outputs, labels)
                        val_loss_sum += val_loss.item()
                        _, predictions = outputs.max(1)
                        val_total += labels.size(0)
                        val_correct += predictions.eq(labels).sum().item()

                epoch_val_loss = val_loss_sum / max(1, len(val_loader))
                epoch_val_acc = 100.0 * val_correct / max(1, val_total)

            # Scheduler step
            if sched is not None:
                from torch.optim.lr_scheduler import ReduceLROnPlateau

                if isinstance(sched, ReduceLROnPlateau):
                    sched.step(epoch_val_loss if epoch_val_loss is not None else epoch_train_loss)
                else:
                    sched.step()

            ckpt_path = checkpoints_dir / f"epoch_{epoch_idx + 1:03d}.pt"
            torch.save(
                {
                    "epoch": epoch_idx,
                    "model_state_dict": model_instance.state_dict(),
                    "optimizer_state_dict": optim.state_dict(),
                    "model_config": model_config.model_dump(),
                    "training_config": training_config.model_dump(),
                },
                ckpt_path,
            )

            msg = (
                f"Epoch {epoch_idx + 1}/{epochs} - "
                f"train_loss: {epoch_train_loss:.4f}, train_acc: {epoch_train_acc:.2f}%"
            )
            if epoch_val_loss is not None and epoch_val_acc is not None:
                msg += f", val_loss: {epoch_val_loss:.4f}, val_acc: {epoch_val_acc:.2f}%"
            click.echo(msg)

            monitor_value = epoch_val_loss if epoch_val_loss is not None else epoch_train_loss
            if monitor_value < best_monitor:
                best_monitor = monitor_value
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1

            if early_stopping and no_improve_epochs >= early_stopping:
                click.echo(f"Early stopping triggered at epoch {epoch_idx + 1}")
                break

        click.echo("\n" + "=" * 60)
        click.echo("Training completed successfully!")
        click.echo("=" * 60)

        # Save final model
        final_path = output_dir / "final_model.pt"
        torch.save(
            {
                "model_state_dict": model_instance.state_dict(),
                "model_config": model_config.model_dump(),
                "training_config": training_config.model_dump(),
            },
            final_path,
        )
        click.echo(f"\nFinal model saved to: {final_path}")

    except KeyboardInterrupt:
        click.echo("\n\nTraining interrupted by user.")
        sys.exit(1)
    except Exception as e:
        click.echo(f"\nError during training: {e}", err=True)
        if ctx.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)
    finally:
        pass

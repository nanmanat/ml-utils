"""
Interactive CLI mode for mlcli.

Provides a REPL-style interface for interactive model exploration,
training, and inference.
"""

from __future__ import annotations

import cmd
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import click

from mlcli.cli.main import Context, pass_context

# Import models to trigger registration
import mlcli.models  # noqa: F401


class ExperimentRun:
    """Container for an experiment run with its configuration and results."""
    
    def __init__(
        self,
        name: str,
        config: Optional[Dict[str, Any]] = None,
        status: str = "created",
    ) -> None:
        self.name = name
        self.config = config or {}
        self.status = status  # created, running, completed, failed
        self.created_at = datetime.now()
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None
        self.results: Dict[str, Any] = {}
        self.history: Dict[str, List[float]] = {}
        self.best_metric: Optional[float] = None
        
        # Multi-model/dataset support
        self.models: List[str] = []
        self.datasets: List[str] = []
        self.configs: List[Dict[str, Any]] = []
        self.run_results: List[Dict[str, Any]] = []  # Results for each combination
        self.completed_run_keys: set = set()

    def save_to_disk(self, path: "Path") -> None:
        import json
        data = {
            "name": self.name,
            "config": self.config,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "results": self.results,
            "history": self.history,
            "best_metric": self.best_metric,
            "models": self.models,
            "datasets": self.datasets,
            "configs": self.configs,
            "run_results": self.run_results,
            "completed_run_keys": list(self.completed_run_keys),
        }
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, default=str))
        tmp.replace(path)

    @classmethod
    def load_from_disk(cls, path: "Path") -> "ExperimentRun":
        import json
        data = json.loads(path.read_text())
        exp = cls(name=data["name"], config=data.get("config", {}), status=data.get("status", "created"))
        exp.created_at = datetime.fromisoformat(data["created_at"]) if data.get("created_at") else exp.created_at
        exp.started_at = datetime.fromisoformat(data["started_at"]) if data.get("started_at") else None
        exp.completed_at = datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None
        exp.results = data.get("results", {})
        exp.history = data.get("history", {})
        exp.best_metric = data.get("best_metric")
        exp.models = data.get("models", [])
        exp.datasets = data.get("datasets", [])
        exp.configs = data.get("configs", [])
        exp.run_results = data.get("run_results", [])
        exp.completed_run_keys = set(data.get("completed_run_keys", []))
        return exp

    def __repr__(self) -> str:
        return f"ExperimentRun(name={self.name!r}, status={self.status!r})"


class InteractiveShell(cmd.Cmd):
    """
    Interactive shell for mlcli.
    
    Provides a REPL interface for exploring models, datasets,
    and running experiments interactively.
    """
    
    intro = """
╔══════════════════════════════════════════════════════════════╗
║                    MLCLI Interactive Mode                     ║
║                                                               ║
║  Type 'help' or '?' for available commands.                  ║
║  Type 'quit' or 'exit' to leave.                             ║
╚══════════════════════════════════════════════════════════════╝
"""
    prompt = "\033[1;36mmlcli>\033[0m "
    
    def __init__(self, ctx: Optional[Context] = None) -> None:
        super().__init__()
        self.ctx = ctx or Context()
        self._model = None
        self._model_name: Optional[str] = None
        self._dataset = None
        self._dataset_path: Optional[Path] = None
        self._device = "auto"
        self._history: List[str] = []
        
        # Multi-model and multi-dataset support
        self._models: Dict[str, Any] = {}  # name -> model instance
        self._datasets: Dict[str, Any] = {}  # name -> dataset instance
        self._configs: Dict[str, Dict[str, Any]] = {}  # name -> config dict
        
        # Experiment runs management
        self._experiments: Dict[str, ExperimentRun] = {}
        self._current_experiment: Optional[str] = None
        
        # Initialize device
        self._setup_device()
    
    def _setup_device(self) -> None:
        """Set up the compute device."""
        import torch
        
        if self._device == "auto":
            if torch.cuda.is_available():
                self._device = "cuda"
            elif torch.backends.mps.is_available():
                self._device = "mps"
            else:
                self._device = "cpu"
        
        self._torch_device = torch.device(self._device)
    
    def _print_success(self, msg: str) -> None:
        """Print success message in green."""
        click.echo(click.style(f"✓ {msg}", fg="green"))
    
    def _print_error(self, msg: str) -> None:
        """Print error message in red."""
        click.echo(click.style(f"✗ {msg}", fg="red"))
    
    def _print_info(self, msg: str) -> None:
        """Print info message in blue."""
        click.echo(click.style(f"ℹ {msg}", fg="blue"))
    
    def _print_warning(self, msg: str) -> None:
        """Print warning message in yellow."""
        click.echo(click.style(f"⚠ {msg}", fg="yellow"))
    
    def _calculate_metrics(
        self, predictions: List[int], targets: List[int]
    ) -> tuple:
        """
        Calculate precision, recall, and F1-score (macro-averaged).
        
        Args:
            predictions: List of predicted class labels
            targets: List of true class labels
            
        Returns:
            Tuple of (precision, recall, f1_score)
        """
        import numpy as np
        
        predictions = np.array(predictions)
        targets = np.array(targets)
        
        # Get unique classes
        classes = np.unique(np.concatenate([predictions, targets]))
        
        precisions = []
        recalls = []
        f1_scores = []
        
        for cls in classes:
            # True positives, false positives, false negatives
            tp = np.sum((predictions == cls) & (targets == cls))
            fp = np.sum((predictions == cls) & (targets != cls))
            fn = np.sum((predictions != cls) & (targets == cls))
            
            # Precision
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            precisions.append(precision)
            
            # Recall
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            recalls.append(recall)
            
            # F1
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            f1_scores.append(f1)
        
        # Macro average
        avg_precision = np.mean(precisions) if precisions else 0.0
        avg_recall = np.mean(recalls) if recalls else 0.0
        avg_f1 = np.mean(f1_scores) if f1_scores else 0.0
        
        return avg_precision, avg_recall, avg_f1
    
    def precmd(self, line: str) -> str:
        """Store command in history."""
        if line.strip():
            self._history.append(line)
        return line
    
    def emptyline(self) -> bool:
        """Do nothing on empty line."""
        return False
    
    def default(self, line: str) -> None:
        """Handle unknown commands."""
        self._print_error(f"Unknown command: {line}")
        click.echo("Type 'help' for available commands.")
    
    # ==================== System Commands ====================
    
    def do_quit(self, arg: str) -> bool:
        """Exit the interactive shell."""
        click.echo("\nGoodbye! 👋")
        return True
    
    def do_exit(self, arg: str) -> bool:
        """Exit the interactive shell."""
        return self.do_quit(arg)
    
    def do_clear(self, arg: str) -> None:
        """Clear the terminal screen."""
        click.clear()
    
    def do_history(self, arg: str) -> None:
        """Show command history."""
        if not self._history:
            self._print_info("No commands in history.")
            return
        
        click.echo("\nCommand History:")
        for i, cmd in enumerate(self._history[-20:], 1):
            click.echo(f"  {i:3d}. {cmd}")
        click.echo()
    
    def do_status(self, arg: str) -> None:
        """Show current session status."""
        click.echo("\n" + "=" * 50)
        click.echo("Session Status")
        click.echo("=" * 50)
        
        click.echo(f"\nDevice: {self._device}")
        
        # Current active model/dataset
        if self._model:
            click.echo(f"Active Model: {self._model_name}")
            params = self._model.get_num_parameters()
            click.echo(f"  Parameters: {params:,}")
        else:
            click.echo("Active Model: Not loaded")
        
        if self._dataset:
            click.echo(f"Active Dataset: {self._dataset_path}")
            click.echo(f"  Samples: {len(self._dataset)}")
        else:
            click.echo("Active Dataset: Not loaded")
        
        # Registered models, datasets, configs
        click.echo(f"\nRegistered Models: {len(self._models)}")
        if self._models:
            click.echo(f"  {', '.join(self._models.keys())}")
        
        click.echo(f"Registered Datasets: {len(self._datasets)}")
        if self._datasets:
            click.echo(f"  {', '.join(self._datasets.keys())}")
        
        click.echo(f"Registered Configs: {len(self._configs)}")
        if self._configs:
            click.echo(f"  {', '.join(self._configs.keys())}")
        
        # Experiment information
        click.echo(f"\nExperiments: {len(self._experiments)}")
        if self._current_experiment:
            exp = self._experiments[self._current_experiment]
            click.echo(f"  Current: {self._current_experiment} [{exp.status}]")
            if exp.models:
                click.echo(f"    Models: {len(exp.models)}")
            if exp.datasets:
                click.echo(f"    Datasets: {len(exp.datasets)}")
            if exp.configs:
                click.echo(f"    Configs: {len(exp.configs)}")
        
        # Count by status
        status_counts = {}
        for exp in self._experiments.values():
            status_counts[exp.status] = status_counts.get(exp.status, 0) + 1
        if status_counts:
            click.echo(f"  Status breakdown: {status_counts}")
        
        click.echo()
    
    # ==================== Device Commands ====================
    
    def do_device(self, arg: str) -> None:
        """
        Get or set the compute device.
        
        Usage:
            device           - Show current device
            device cpu       - Switch to CPU
            device cuda      - Switch to CUDA
            device cuda:0    - Switch to specific GPU
            device mps       - Switch to Apple MPS
        """
        if not arg:
            click.echo(f"Current device: {self._device}")
            
            import torch
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    name = torch.cuda.get_device_name(i)
                    mem = torch.cuda.get_device_properties(i).total_memory / 1e9
                    click.echo(f"  cuda:{i} - {name} ({mem:.1f} GB)")
            
            if torch.backends.mps.is_available():
                click.echo("  mps - Apple Metal")
            
            return
        
        try:
            import torch
            device = torch.device(arg)
            self._device = arg
            self._torch_device = device
            self._print_success(f"Switched to device: {arg}")
            
            # Move model if loaded
            if self._model:
                self._model = self._model.to(device)
                self._print_info("Model moved to new device")
        except Exception as e:
            self._print_error(f"Invalid device: {e}")
    
    # ==================== Model Commands ====================
    
    def do_models(self, arg: str) -> None:
        """
        List available models.
        
        Usage:
            models              - List all models
            models cnn          - List CNN models
            models transformer  - List transformer models
            models detection    - List detection models
        """
        from mlcli.core.registry import MODEL_REGISTRY
        
        all_models = sorted(MODEL_REGISTRY.list_registered())
        
        if not all_models:
            self._print_warning("No models registered.")
            return
        
        filter_type = arg.lower() if arg else None
        
        click.echo("\nAvailable Models:")
        click.echo("-" * 40)
        
        # Group models by type
        cnn_models = [m for m in all_models if any(x in m.lower() for x in ["resnet", "efficientnet", "mobilenet"])]
        transformer_models = [m for m in all_models if any(x in m.lower() for x in ["vit", "swin"])]
        detection_models = [m for m in all_models if any(x in m.lower() for x in ["yolo", "rcnn", "ssd", "detr", "fasterrcnn"])]
        other_models = [m for m in all_models if m not in cnn_models + transformer_models + detection_models]
        
        if filter_type is None or filter_type == "cnn":
            if cnn_models:
                click.echo("\n  CNN Classification:")
                for m in cnn_models:
                    click.echo(f"    • {m}")
        
        if filter_type is None or filter_type == "transformer":
            if transformer_models:
                click.echo("\n  Transformer Classification:")
                for m in transformer_models:
                    click.echo(f"    • {m}")
        
        if filter_type is None or filter_type == "detection":
            if detection_models:
                click.echo("\n  Object Detection:")
                for m in detection_models:
                    click.echo(f"    • {m}")
        
        if filter_type is None and other_models:
            click.echo("\n  Other:")
            for m in other_models:
                click.echo(f"    • {m}")
        
        click.echo()
    
    def do_load_model(self, arg: str) -> None:
        """
        Load a model.
        
        Usage:
            load_model resnet50 --classes 10
            load_model vit_base --classes 100 --pretrained
            load_model ./checkpoint.pt
        """
        if not arg:
            self._print_error("Please specify a model name or checkpoint path.")
            click.echo("Usage: load_model <model_name> [--classes N] [--pretrained]")
            return
        
        parts = arg.split()
        model_name = parts[0]
        
        # Parse options
        num_classes = 1000
        pretrained = True
        
        for i, part in enumerate(parts[1:], 1):
            if part == "--classes" and i < len(parts) - 1:
                try:
                    num_classes = int(parts[i + 1])
                except ValueError:
                    pass
            elif part == "--pretrained":
                pretrained = True
            elif part == "--no-pretrained":
                pretrained = False
        
        try:
            # Check if it's a checkpoint path
            path = Path(model_name)
            if path.exists() and path.suffix in (".pt", ".pth"):
                import torch
                self._print_info(f"Loading checkpoint: {path}")
                checkpoint = torch.load(path, map_location=self._torch_device)
                
                if "model_config" in checkpoint:
                    from mlcli.core.config import ModelConfig
                    from mlcli.core.factory import ModelFactory
                    
                    config = ModelConfig(**checkpoint["model_config"])
                    self._model = ModelFactory.create(config)
                    self._model.load_state_dict(checkpoint["model_state_dict"])
                    self._model_name = config.architecture
                else:
                    self._print_error("Checkpoint missing model_config. Use load_model <name> first.")
                    return
            else:
                # Load from registry
                from mlcli.core.config import ModelConfig, TaskType, ModelFamily
                from mlcli.core.factory import ModelFactory
                
                self._print_info(f"Creating model: {model_name}")
                
                # Determine task type and model family
                model_lower = model_name.lower()
                is_detection = any(x in model_lower for x in ["yolo", "rcnn", "ssd", "detr", "fasterrcnn"])
                is_transformer = any(x in model_lower for x in ["vit", "swin", "detr"])
                
                task_type = TaskType.OBJECT_DETECTION if is_detection else TaskType.CLASSIFICATION
                family = ModelFamily.TRANSFORMER if is_transformer else ModelFamily.CNN
                
                config = ModelConfig(
                    task_type=task_type,
                    family=family,
                    architecture=model_name,
                    num_classes=num_classes,
                    pretrained=pretrained,
                )
                self._model = ModelFactory.create(config)
                self._model_name = model_name
            
            # Move to device and set to eval mode
            self._model = self._model.to(self._torch_device)
            self._model.eval()
            
            params = self._model.get_num_parameters()
            self._print_success(f"Model loaded: {self._model_name}")
            click.echo(f"  Parameters: {params:,}")
            click.echo(f"  Device: {self._device}")
            
        except Exception as e:
            self._print_error(f"Failed to load model: {e}")
    
    def do_model_info(self, arg: str) -> None:
        """Show information about the loaded model."""
        if not self._model:
            self._print_warning("No model loaded. Use 'load_model <name>' first.")
            return
        
        click.echo("\n" + "=" * 50)
        click.echo(f"Model: {self._model_name}")
        click.echo("=" * 50)
        
        total_params = self._model.get_num_parameters()
        trainable_params = self._model.get_num_parameters(trainable_only=True)
        
        click.echo(f"\nTotal parameters: {total_params:,}")
        click.echo(f"Trainable parameters: {trainable_params:,}")
        click.echo(f"Frozen parameters: {total_params - trainable_params:,}")
        click.echo(f"Device: {self._device}")
        
        # Show layer summary
        if arg == "-v" or arg == "--verbose":
            click.echo("\nLayers:")
            for name, module in self._model.named_children():
                num_params = sum(p.numel() for p in module.parameters())
                click.echo(f"  {name}: {type(module).__name__} ({num_params:,} params)")
        
        click.echo()
    
    def do_freeze(self, arg: str) -> None:
        """Freeze model backbone weights."""
        if not self._model:
            self._print_warning("No model loaded.")
            return
        
        self._model.freeze_backbone()
        trainable = self._model.get_num_parameters(trainable_only=True)
        self._print_success(f"Backbone frozen. Trainable params: {trainable:,}")
    
    def do_unfreeze(self, arg: str) -> None:
        """Unfreeze model backbone weights."""
        if not self._model:
            self._print_warning("No model loaded.")
            return
        
        self._model.unfreeze_backbone()
        trainable = self._model.get_num_parameters(trainable_only=True)
        self._print_success(f"Backbone unfrozen. Trainable params: {trainable:,}")
    
    # ==================== Dataset Commands ====================
    
    def do_load_dataset(self, arg: str) -> None:
        """
        Load a dataset.
        
        Usage:
            load_dataset ./data/train
            load_dataset ./data/train --type folder
            load_dataset ./coco --type coco
        """
        if not arg:
            self._print_error("Please specify a dataset path.")
            click.echo("Usage: load_dataset <path> [--type folder|coco|voc]")
            return
        
        parts = arg.split()
        dataset_path = Path(parts[0])
        
        # Parse dataset type
        dataset_type = "folder"
        for i, part in enumerate(parts[1:], 1):
            if part == "--type" and i < len(parts) - 1:
                dataset_type = parts[i + 1]
        
        if not dataset_path.exists():
            self._print_error(f"Path not found: {dataset_path}")
            return
        
        try:
            # Create default transforms for training
            from torchvision import transforms
            default_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])
            
            if dataset_type == "folder":
                from mlcli.data.classification import ImageFolderDataset
                self._dataset = ImageFolderDataset(
                    root=dataset_path,
                    transform=default_transform,
                )
            elif dataset_type == "coco":
                from mlcli.data.detection import COCODataset
                ann_file = dataset_path / "annotations.json"
                self._dataset = COCODataset(root=dataset_path, annotation_file=ann_file)
            elif dataset_type == "voc":
                from mlcli.data.detection import VOCDataset
                self._dataset = VOCDataset(root=dataset_path)
            else:
                self._print_error(f"Unknown dataset type: {dataset_type}")
                return
            
            self._dataset_path = dataset_path
            self._print_success(f"Dataset loaded: {dataset_path}")
            click.echo(f"  Samples: {len(self._dataset)}")
            click.echo(f"  Classes: {self._dataset.get_num_classes()}")
            
        except Exception as e:
            self._print_error(f"Failed to load dataset: {e}")
    
    def do_dataset_info(self, arg: str) -> None:
        """Show information about the loaded dataset."""
        if not self._dataset:
            self._print_warning("No dataset loaded. Use 'load_dataset <path>' first.")
            return
        
        click.echo("\n" + "=" * 50)
        click.echo(f"Dataset: {self._dataset_path}")
        click.echo("=" * 50)
        
        click.echo(f"\nSamples: {len(self._dataset)}")
        click.echo(f"Classes: {self._dataset.get_num_classes()}")
        
        class_names = self._dataset.get_class_names()
        if class_names:
            click.echo("\nClass names:")
            for i, name in enumerate(class_names[:20]):
                click.echo(f"  {i}: {name}")
            if len(class_names) > 20:
                click.echo(f"  ... and {len(class_names) - 20} more")
        
        click.echo()
    
    # ==================== Inference Commands ====================
    
    def do_predict(self, arg: str) -> None:
        """
        Run inference on an image.
        
        Usage:
            predict ./image.jpg
            predict ./image.jpg --top 5
        """
        if not self._model:
            self._print_warning("No model loaded. Use 'load_model <name>' first.")
            return
        
        if not arg:
            self._print_error("Please specify an image path.")
            click.echo("Usage: predict <image_path> [--top N]")
            return
        
        parts = arg.split()
        image_path = Path(parts[0])
        
        # Parse options
        top_k = 5
        for i, part in enumerate(parts[1:], 1):
            if part == "--top" and i < len(parts) - 1:
                try:
                    top_k = int(parts[i + 1])
                except ValueError:
                    pass
        
        if not image_path.exists():
            self._print_error(f"Image not found: {image_path}")
            return
        
        try:
            import torch
            from PIL import Image
            import torchvision.transforms as T
            
            # Load and preprocess image
            image = Image.open(image_path).convert("RGB")
            
            transform = T.Compose([
                T.Resize(256),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            
            input_tensor = transform(image).unsqueeze(0).to(self._torch_device)
            
            # Run inference
            self._model.eval()
            with torch.no_grad():
                outputs = self._model(input_tensor)
            
            # Process outputs
            if isinstance(outputs, torch.Tensor):
                probabilities = torch.softmax(outputs, dim=1)
                top_probs, top_indices = torch.topk(probabilities, min(top_k, probabilities.size(1)))
                
                click.echo(f"\nPredictions for: {image_path.name}")
                click.echo("-" * 40)
                
                for prob, idx in zip(top_probs[0], top_indices[0]):
                    class_name = f"class_{idx.item()}"
                    if self._dataset:
                        names = self._dataset.get_class_names()
                        if names and idx.item() < len(names):
                            class_name = names[idx.item()]
                    
                    bar_width = int(prob.item() * 30)
                    bar = "█" * bar_width + "░" * (30 - bar_width)
                    click.echo(f"  {class_name:20s} {bar} {prob.item():.2%}")
            else:
                # Detection output
                click.echo(f"\nDetections for: {image_path.name}")
                click.echo("-" * 40)
                
                if isinstance(outputs, list) and len(outputs) > 0:
                    output = outputs[0]
                    boxes = output.get("boxes", [])
                    scores = output.get("scores", [])
                    labels = output.get("labels", [])
                    
                    for box, score, label in zip(boxes[:10], scores[:10], labels[:10]):
                        click.echo(f"  Class {label}: {score:.2%} @ {box.tolist()}")
                    
                    if len(boxes) > 10:
                        click.echo(f"  ... and {len(boxes) - 10} more")
            
            click.echo()
            
        except Exception as e:
            self._print_error(f"Prediction failed: {e}")
    
    def do_benchmark(self, arg: str) -> None:
        """
        Benchmark model inference speed.
        
        Usage:
            benchmark              - Run 100 iterations
            benchmark --iters 500  - Run 500 iterations
        """
        if not self._model:
            self._print_warning("No model loaded. Use 'load_model <name>' first.")
            return
        
        import time
        import torch
        
        # Parse options
        iterations = 100
        parts = arg.split() if arg else []
        for i, part in enumerate(parts):
            if part == "--iters" and i < len(parts) - 1:
                try:
                    iterations = int(parts[i + 1])
                except ValueError:
                    pass
        
        click.echo(f"\nBenchmarking {self._model_name} on {self._device}...")
        click.echo(f"Running {iterations} iterations...")
        
        # Create dummy input
        dummy_input = torch.randn(1, 3, 224, 224).to(self._torch_device)
        
        # Warmup
        self._model.eval()
        with torch.no_grad():
            for _ in range(10):
                _ = self._model(dummy_input)
        
        # Synchronize if CUDA
        if self._device.startswith("cuda"):
            torch.cuda.synchronize()
        
        # Benchmark
        start_time = time.perf_counter()
        
        with torch.no_grad():
            for _ in range(iterations):
                _ = self._model(dummy_input)
        
        if self._device.startswith("cuda"):
            torch.cuda.synchronize()
        
        total_time = time.perf_counter() - start_time
        avg_time = total_time / iterations * 1000  # ms
        fps = iterations / total_time
        
        click.echo("\n" + "=" * 40)
        click.echo("Benchmark Results")
        click.echo("=" * 40)
        click.echo(f"  Total time: {total_time:.2f}s")
        click.echo(f"  Avg latency: {avg_time:.2f}ms")
        click.echo(f"  Throughput: {fps:.1f} FPS")
        click.echo()
    
    # ==================== Training Commands ====================
    
    def do_train(self, arg: str) -> None:
        """
        Start training (launches full training pipeline).
        
        Usage:
            train --epochs 10 --lr 0.001
        """
        if not self._model:
            self._print_warning("No model loaded. Use 'load_model <name>' first.")
            return
        
        if not self._dataset:
            self._print_warning("No dataset loaded. Use 'load_dataset <path>' first.")
            return
        
        # Parse options
        epochs = 10
        lr = 0.001
        batch_size = 32
        
        parts = arg.split() if arg else []
        for i, part in enumerate(parts):
            if part == "--epochs" and i < len(parts) - 1:
                try:
                    epochs = int(parts[i + 1])
                except ValueError:
                    pass
            elif part == "--lr" and i < len(parts) - 1:
                try:
                    lr = float(parts[i + 1])
                except ValueError:
                    pass
            elif part == "--batch-size" and i < len(parts) - 1:
                try:
                    batch_size = int(parts[i + 1])
                except ValueError:
                    pass
        
        self._print_info(f"Starting training for {epochs} epochs...")
        click.echo(f"  Learning rate: {lr}")
        click.echo(f"  Batch size: {batch_size}")
        
        try:
            import torch
            import torch.nn as nn
            from tqdm import tqdm
            
            # Create data loader
            train_loader = self._dataset.create_dataloader(
                batch_size=batch_size,
                shuffle=True,
                num_workers=4,
            )
            
            # Create optimizer and criterion
            optimizer = torch.optim.AdamW(self._model.parameters(), lr=lr)
            criterion = nn.CrossEntropyLoss()
            
            # Move model to device
            self._model.to(self._device)
            self._model.train()
            
            # Simple training loop
            for epoch in range(epochs):
                total_loss = 0.0
                correct = 0
                total = 0
                
                pbar = tqdm(
                    train_loader,
                    desc=f"Epoch {epoch + 1}/{epochs}",
                    leave=True,
                )
                
                for batch_idx, (inputs, targets) in enumerate(pbar):
                    inputs = inputs.to(self._device)
                    targets = targets.to(self._device)
                    
                    optimizer.zero_grad()
                    outputs = self._model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
                    
                    # Update progress bar
                    avg_loss = total_loss / (batch_idx + 1)
                    acc = 100.0 * correct / total
                    pbar.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{acc:.2f}%")
                
                # Epoch summary
                avg_loss = total_loss / len(train_loader)
                acc = 100.0 * correct / total
                click.echo(f"  Epoch {epoch + 1}: loss={avg_loss:.4f}, accuracy={acc:.2f}%")
            
            self._print_success("Training completed!")
            
        except KeyboardInterrupt:
            self._print_warning("\nTraining interrupted by user.")
        except Exception as e:
            self._print_error(f"Training failed: {e}")
    
    # ==================== Experiment Run Commands ====================
    
    def do_experiment(self, arg: str) -> None:
        """
        Create a new experiment run.
        
        Usage:
            experiment my_exp                       - Create new experiment
            experiment my_exp --config config.yaml  - Create from config file
            experiment my_exp --model resnet50 --epochs 50
            
        Config file can define multiple models, datasets, and configs for grid search.
        """
        if not arg:
            self._print_error("Please specify an experiment name.")
            click.echo("Usage: experiment <name> [--config <path>] [--model <name>] [--epochs N]")
            return
        
        parts = arg.split()
        exp_name = parts[0]
        
        # Parse options
        config_path = None
        model_name = self._model_name
        epochs = 10
        lr = 0.001
        batch_size = 32
        
        i = 1
        while i < len(parts):
            if parts[i] == "--config" and i + 1 < len(parts):
                config_path = parts[i + 1]
                i += 2
            elif parts[i] == "--model" and i + 1 < len(parts):
                model_name = parts[i + 1]
                i += 2
            elif parts[i] == "--epochs" and i + 1 < len(parts):
                try:
                    epochs = int(parts[i + 1])
                except ValueError:
                    pass
                i += 2
            elif parts[i] == "--lr" and i + 1 < len(parts):
                try:
                    lr = float(parts[i + 1])
                except ValueError:
                    pass
                i += 2
            elif parts[i] == "--batch-size" and i + 1 < len(parts):
                try:
                    batch_size = int(parts[i + 1])
                except ValueError:
                    pass
                i += 2
            else:
                i += 1
        
        try:
            config = {}
            
            # Load from config file if specified
            if config_path:
                path = Path(config_path)
                if not path.exists():
                    self._print_error(f"Config file not found: {config_path}")
                    return
                
                import yaml
                with open(path) as f:
                    config = yaml.safe_load(f)
                self._print_info(f"Loaded config from: {config_path}")
                
                # Create experiment run
                exp = ExperimentRun(name=exp_name, config=config)
                
                # Process multi-model definitions
                if "models" in config:
                    self._load_models_from_config(config["models"], exp)
                
                # Process multi-dataset definitions
                if "datasets" in config:
                    self._load_datasets_from_config(config["datasets"], exp)
                
                # Process multi-config (training configs) definitions
                if "configs" in config or "training_configs" in config:
                    configs_list = config.get("configs", config.get("training_configs", []))
                    self._load_training_configs_from_config(configs_list, exp)
                
                self._experiments[exp_name] = exp
                self._current_experiment = exp_name

                # Set up state path for fail-safety
                _exp_output_dir = Path(self.ctx.output_dir or "./outputs") / exp_name
                _exp_output_dir.mkdir(parents=True, exist_ok=True)
                _state_path = _exp_output_dir / "experiment_state.json"
                exp.state_path = _state_path
                exp.save_to_disk(_state_path)

                # Display summary
                self._print_success(f"Created experiment: {exp_name}")
                if exp.models:
                    click.echo(f"  Models: {len(exp.models)} - {', '.join(exp.models)}")
                if exp.datasets:
                    click.echo(f"  Datasets: {len(exp.datasets)} - {', '.join(exp.datasets)}")
                if exp.configs:
                    config_names = [c.get("_name", "unnamed") for c in exp.configs]
                    click.echo(f"  Configs: {len(exp.configs)} - {', '.join(config_names)}")
                
                # Show grid size if multi
                if exp.models and exp.datasets and exp.configs:
                    total = len(exp.models) * len(exp.datasets) * len(exp.configs)
                    click.echo(f"  Grid Size: {total} runs")
                elif not exp.models and not exp.datasets and not exp.configs:
                    # Single model/training config
                    click.echo(f"  Model: {config.get('model', {}).get('architecture', 'Not set')}")
                    click.echo(f"  Epochs: {config.get('training', {}).get('epochs', 'Not set')}")
            else:
                # Build config from options
                config = {
                    "model": {"architecture": model_name} if model_name else {},
                    "training": {
                        "epochs": epochs,
                        "learning_rate": lr,
                        "batch_size": batch_size,
                    },
                    "dataset": {
                        "path": str(self._dataset_path) if self._dataset_path else None,
                    },
                }
                
                # Create experiment run
                exp = ExperimentRun(name=exp_name, config=config)
                self._experiments[exp_name] = exp
                self._current_experiment = exp_name

                # Set up state path for fail-safety
                _exp_output_dir = Path(self.ctx.output_dir or "./outputs") / exp_name
                _exp_output_dir.mkdir(parents=True, exist_ok=True)
                _state_path = _exp_output_dir / "experiment_state.json"
                exp.state_path = _state_path
                exp.save_to_disk(_state_path)

                self._print_success(f"Created experiment: {exp_name}")
                click.echo(f"  Model: {config.get('model', {}).get('architecture', 'Not set')}")
                click.echo(f"  Epochs: {config.get('training', {}).get('epochs', 'Not set')}")
                click.echo(f"  Learning Rate: {config.get('training', {}).get('learning_rate', 'Not set')}")
            
        except Exception as e:
            self._print_error(f"Failed to create experiment: {e}")
    
    def _load_models_from_config(self, models_config: List[Dict], exp: ExperimentRun) -> None:
        """Load multiple models from config and register them."""
        from mlcli.core.config import ModelConfig, TaskType, ModelFamily
        from mlcli.core.factory import ModelFactory
        
        for model_def in models_config:
            name = model_def.get("name", model_def.get("architecture"))
            architecture = model_def.get("architecture")
            num_classes = model_def.get("num_classes", 1000)
            pretrained = model_def.get("pretrained", True)
            
            if not architecture:
                self._print_warning(f"Skipping model without architecture: {model_def}")
                continue
            
            try:
                # Determine task type and model family
                model_lower = architecture.lower()
                is_detection = any(x in model_lower for x in ["yolo", "rcnn", "ssd", "detr", "fasterrcnn"])
                is_transformer = any(x in model_lower for x in ["vit", "swin", "detr"])
                
                task_type = TaskType.OBJECT_DETECTION if is_detection else TaskType.CLASSIFICATION
                family = ModelFamily.TRANSFORMER if is_transformer else ModelFamily.CNN
                
                config = ModelConfig(
                    task_type=task_type,
                    family=family,
                    architecture=architecture,
                    num_classes=num_classes,
                    pretrained=pretrained,
                )
                model = ModelFactory.create(config)
                model = model.to(self._torch_device)
                model.eval()
                
                self._models[name] = {
                    "model": model,
                    "architecture": architecture,
                    "num_classes": num_classes,
                    "pretrained": pretrained,
                    "config": config,
                }
                
                exp.models.append(name)
                self._print_info(f"  Loaded model: {name} ({architecture})")
                
            except Exception as e:
                self._print_warning(f"  Failed to load model {name}: {e}")
    
    def _load_datasets_from_config(self, datasets_config: List[Dict], exp: ExperimentRun) -> None:
        """Load multiple datasets from config and register them."""
        from torchvision import transforms
        
        default_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
        
        for ds_def in datasets_config:
            name = ds_def.get("name")
            path = ds_def.get("path")
            ds_type = ds_def.get("type", "folder")
            
            if not name or not path:
                self._print_warning(f"Skipping dataset without name or path: {ds_def}")
                continue
            
            dataset_path = Path(path)
            if not dataset_path.exists():
                self._print_warning(f"  Dataset path not found: {path}")
                continue
            
            try:
                if ds_type == "folder":
                    from mlcli.data.classification import ImageFolderDataset
                    dataset = ImageFolderDataset(
                        root=dataset_path,
                        transform=default_transform,
                    )
                elif ds_type == "coco":
                    from mlcli.data.detection import COCODataset
                    ann_file = dataset_path / "annotations.json"
                    dataset = COCODataset(root=dataset_path, annotation_file=ann_file)
                elif ds_type == "voc":
                    from mlcli.data.detection import VOCDataset
                    dataset = VOCDataset(root=dataset_path)
                else:
                    self._print_warning(f"  Unknown dataset type: {ds_type}")
                    continue
                
                self._datasets[name] = {
                    "dataset": dataset,
                    "path": dataset_path,
                    "type": ds_type,
                    "samples": len(dataset),
                    "classes": dataset.get_num_classes(),
                }
                
                exp.datasets.append(name)
                self._print_info(f"  Loaded dataset: {name} ({len(dataset)} samples)")
                
            except Exception as e:
                self._print_warning(f"  Failed to load dataset {name}: {e}")
    
    def _load_training_configs_from_config(self, configs_list: List[Dict], exp: ExperimentRun) -> None:
        """Load multiple training configurations from config."""
        for cfg in configs_list:
            name = cfg.get("name", f"config_{len(exp.configs) + 1}")
            
            training_config = {
                "_name": name,
                "epochs": cfg.get("epochs", 10),
                "learning_rate": cfg.get("learning_rate", cfg.get("lr", 0.001)),
                "batch_size": cfg.get("batch_size", 32),
                "optimizer": cfg.get("optimizer", "adamw"),
                "scheduler": cfg.get("scheduler", "cosine"),
                "mixed_precision": cfg.get("mixed_precision", False),
                "weight_decay": cfg.get("weight_decay", 0.0001),
            }
            
            # Store in global configs registry
            self._configs[name] = training_config
            
            # Add to experiment
            exp.configs.append(training_config)
            self._print_info(f"  Loaded config: {name} (epochs={training_config['epochs']}, lr={training_config['learning_rate']})")
    
    def do_experiments(self, arg: str) -> None:
        """
        List all experiment runs.
        
        Usage:
            experiments              - List all experiments
            experiments --status running  - Filter by status
        """
        if not self._experiments:
            self._print_info("No experiments created yet.")
            click.echo("Use 'experiment <name>' to create one.")
            return
        
        # Parse filter
        status_filter = None
        if arg:
            parts = arg.split()
            for i, part in enumerate(parts):
                if part == "--status" and i + 1 < len(parts):
                    status_filter = parts[i + 1]
        
        click.echo("\n" + "=" * 60)
        click.echo("Experiment Runs")
        click.echo("=" * 60)
        
        for name, exp in self._experiments.items():
            if status_filter and exp.status != status_filter:
                continue
            
            # Status indicator
            status_icons = {
                "created": "○",
                "running": "●",
                "completed": "✓",
                "failed": "✗",
            }
            icon = status_icons.get(exp.status, "?")
            
            # Color based on status
            status_colors = {
                "created": "white",
                "running": "yellow",
                "completed": "green",
                "failed": "red",
            }
            color = status_colors.get(exp.status, "white")
            
            current_marker = " *" if name == self._current_experiment else ""
            
            click.echo(
                click.style(f"  {icon} ", fg=color) +
                f"{name}{current_marker} " +
                click.style(f"[{exp.status}]", fg=color) +
                f" - {exp.created_at.strftime('%Y-%m-%d %H:%M')}"
            )
            
            if exp.best_metric is not None:
                click.echo(f"      Best metric: {exp.best_metric:.4f}")
        
        click.echo()
        if self._current_experiment:
            click.echo(f"Current experiment: {self._current_experiment}")
        click.echo()
    
    def do_experiment_info(self, arg: str) -> None:
        """
        Show detailed information about an experiment.
        
        Usage:
            experiment_info           - Show current experiment
            experiment_info my_exp    - Show specific experiment
        """
        exp_name = arg.strip() if arg else self._current_experiment
        
        if not exp_name:
            self._print_warning("No experiment specified and no current experiment.")
            return
        
        if exp_name not in self._experiments:
            self._print_error(f"Experiment not found: {exp_name}")
            return
        
        exp = self._experiments[exp_name]
        
        click.echo("\n" + "=" * 60)
        click.echo(f"Experiment: {exp.name}")
        click.echo("=" * 60)
        
        click.echo(f"\nStatus: {exp.status}")
        click.echo(f"Created: {exp.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
        
        if exp.started_at:
            click.echo(f"Started: {exp.started_at.strftime('%Y-%m-%d %H:%M:%S')}")
        if exp.completed_at:
            click.echo(f"Completed: {exp.completed_at.strftime('%Y-%m-%d %H:%M:%S')}")
            duration = (exp.completed_at - exp.started_at).total_seconds()
            click.echo(f"Duration: {duration:.1f}s")
        
        # Multi-model/dataset/config info
        if exp.models:
            click.echo(f"\nModels ({len(exp.models)}):")
            for m in exp.models:
                click.echo(f"  • {m}")
        
        if exp.datasets:
            click.echo(f"\nDatasets ({len(exp.datasets)}):")
            for d in exp.datasets:
                click.echo(f"  • {d}")
        
        if exp.configs:
            click.echo(f"\nConfigs ({len(exp.configs)}):")
            for c in exp.configs:
                name = c.get("_name", "unnamed")
                epochs = c.get("epochs", "?")
                lr = c.get("learning_rate", "?")
                click.echo(f"  • {name}: epochs={epochs}, lr={lr}")
        
        # Single config fallback
        if exp.config and not exp.configs:
            click.echo("\nConfiguration:")
            if "model" in exp.config:
                click.echo(f"  Model: {exp.config['model'].get('architecture', 'N/A')}")
            if "training" in exp.config:
                train = exp.config["training"]
                click.echo(f"  Epochs: {train.get('epochs', 'N/A')}")
                click.echo(f"  Learning Rate: {train.get('learning_rate', 'N/A')}")
                click.echo(f"  Batch Size: {train.get('batch_size', 'N/A')}")
        
        if exp.best_metric is not None:
            click.echo(f"\nBest Metric: {exp.best_metric:.4f}")
        
        # Show run results summary if available
        if exp.run_results:
            click.echo(f"\nRun Results: {len(exp.run_results)} runs completed")
            best_run = max(exp.run_results, key=lambda x: x["best_accuracy"])
            click.echo(f"  Best: {best_run['model']} + {best_run['dataset']} ({best_run['best_accuracy']:.2f}%)")
        elif exp.results:
            click.echo("\nResults:")
            for key, value in exp.results.items():
                if isinstance(value, float):
                    click.echo(f"  {key}: {value:.4f}")
                elif not isinstance(value, (list, dict)):
                    click.echo(f"  {key}: {value}")
        
        click.echo()
    
    def do_use_experiment(self, arg: str) -> None:
        """
        Switch to a specific experiment.
        
        Usage:
            use_experiment my_exp
        """
        if not arg:
            self._print_error("Please specify an experiment name.")
            return
        
        exp_name = arg.strip()
        if exp_name not in self._experiments:
            self._print_error(f"Experiment not found: {exp_name}")
            click.echo("Use 'experiments' to list available experiments.")
            return
        
        self._current_experiment = exp_name
        exp = self._experiments[exp_name]
        self._print_success(f"Switched to experiment: {exp_name}")
        
        # Load experiment model if available
        model_arch = exp.config.get("model", {}).get("architecture")
        if model_arch and not self._model:
            self._print_info(f"Experiment uses model: {model_arch}")
            click.echo("Use 'load_model' to load it.")
    
    def do_run_experiment(self, arg: str) -> None:
        """
        Run the current or specified experiment.
        
        Usage:
            run_experiment           - Run current experiment
            run_experiment my_exp    - Run specific experiment
            run_experiment --resume  - Resume interrupted experiment
        """
        parts = arg.split() if arg else []
        
        # Parse arguments
        exp_name = self._current_experiment
        resume = False
        
        for i, part in enumerate(parts):
            if part == "--resume":
                resume = True
            elif not part.startswith("--"):
                exp_name = part
        
        if not exp_name:
            self._print_warning("No experiment specified and no current experiment.")
            click.echo("Use 'experiment <name>' to create one first.")
            return
        
        if exp_name not in self._experiments:
            self._print_error(f"Experiment not found: {exp_name}")
            return
        
        exp = self._experiments[exp_name]
        
        # Validate experiment state
        if exp.status == "running":
            self._print_warning("Experiment is already running.")
            return
        if exp.status == "completed" and not resume:
            self._print_warning("Experiment already completed. Use --resume to continue.")
            return
        
        # Check requirements
        config = exp.config
        
        # Load model if needed
        model_arch = config.get("model", {}).get("architecture")
        if not self._model:
            if model_arch:
                self._print_info(f"Loading model: {model_arch}")
                self.do_load_model(f"{model_arch} --classes {config.get('model', {}).get('num_classes', 1000)}")
            else:
                self._print_error("No model loaded and no model specified in experiment config.")
                return
        
        # Check dataset
        if not self._dataset:
            dataset_path = config.get("dataset", {}).get("path")
            if dataset_path and Path(dataset_path).exists():
                self._print_info(f"Loading dataset: {dataset_path}")
                self.do_load_dataset(dataset_path)
            else:
                self._print_error("No dataset loaded. Use 'load_dataset <path>' first.")
                return
        
        # Get training parameters
        train_config = config.get("training", {})
        epochs = train_config.get("epochs", 10)
        lr = train_config.get("learning_rate", 0.001)
        batch_size = train_config.get("batch_size", 32)
        
        # Update experiment status
        exp.status = "running"
        exp.started_at = datetime.now()
        self._current_experiment = exp_name
        
        click.echo("\n" + "=" * 60)
        click.echo(f"Running Experiment: {exp_name}")
        click.echo("=" * 60)
        click.echo(f"  Model: {self._model_name}")
        click.echo(f"  Epochs: {epochs}")
        click.echo(f"  Learning Rate: {lr}")
        click.echo(f"  Batch Size: {batch_size}")
        click.echo(f"  Device: {self._device}")
        click.echo()
        
        try:
            import torch
            import torch.nn as nn
            from tqdm import tqdm
            
            # Create data loader
            train_loader = self._dataset.create_dataloader(
                batch_size=batch_size,
                shuffle=True,
                num_workers=4,
            )
            
            # Create optimizer and criterion
            optimizer = torch.optim.AdamW(self._model.parameters(), lr=lr)
            criterion = nn.CrossEntropyLoss()
            
            # Learning rate scheduler
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs
            )
            
            # Move model to device
            self._model.to(self._torch_device)
            self._model.train()
            
            # Initialize history
            exp.history = {
                "loss": [], "accuracy": [], "lr": [],
                "precision": [], "recall": [], "f1_score": []
            }
            best_acc = 0.0
            best_f1 = 0.0
            
            # Training loop
            for epoch in range(epochs):
                total_loss = 0.0
                correct = 0
                total = 0
                
                # For precision/recall/F1 calculation
                all_predictions = []
                all_targets = []
                
                pbar = tqdm(
                    train_loader,
                    desc=f"Epoch {epoch + 1}/{epochs}",
                    leave=True,
                )
                
                for batch_idx, (inputs, targets) in enumerate(pbar):
                    inputs = inputs.to(self._torch_device)
                    targets = targets.to(self._torch_device)
                    
                    optimizer.zero_grad()
                    outputs = self._model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
                    
                    # Collect predictions for metrics
                    all_predictions.extend(predicted.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                    
                    # Update progress bar
                    avg_loss = total_loss / (batch_idx + 1)
                    acc = 100.0 * correct / total
                    pbar.set_postfix(
                        loss=f"{avg_loss:.4f}",
                        acc=f"{acc:.2f}%",
                        lr=f"{scheduler.get_last_lr()[0]:.6f}"
                    )
                
                # Update scheduler
                scheduler.step()
                
                # Epoch metrics
                avg_loss = total_loss / len(train_loader)
                acc = 100.0 * correct / total
                current_lr = scheduler.get_last_lr()[0]
                
                # Calculate precision, recall, F1
                precision, recall, f1 = self._calculate_metrics(all_predictions, all_targets)
                
                # Record history
                exp.history["loss"].append(avg_loss)
                exp.history["accuracy"].append(acc)
                exp.history["lr"].append(current_lr)
                exp.history["precision"].append(precision)
                exp.history["recall"].append(recall)
                exp.history["f1_score"].append(f1)
                
                # Track best
                if acc > best_acc:
                    best_acc = acc
                    exp.best_metric = acc
                if f1 > best_f1:
                    best_f1 = f1
                
                click.echo(
                    f"  Epoch {epoch + 1}: "
                    f"loss={avg_loss:.4f}, acc={acc:.2f}%, "
                    f"P={precision:.4f}, R={recall:.4f}, F1={f1:.4f}"
                )
            
            # Experiment completed
            exp.status = "completed"
            exp.completed_at = datetime.now()
            exp.results = {
                "final_loss": exp.history["loss"][-1],
                "final_accuracy": exp.history["accuracy"][-1],
                "best_accuracy": best_acc,
                "final_precision": exp.history["precision"][-1],
                "final_recall": exp.history["recall"][-1],
                "final_f1_score": exp.history["f1_score"][-1],
                "best_f1_score": best_f1,
                "epochs_trained": epochs,
            }
            
            duration = (exp.completed_at - exp.started_at).total_seconds()
            
            click.echo("\n" + "=" * 60)
            click.echo("Experiment Completed!")
            click.echo("=" * 60)
            click.echo(f"  Duration: {duration:.1f}s")
            click.echo(f"  Final Loss: {exp.results['final_loss']:.4f}")
            click.echo(f"  Final Accuracy: {exp.results['final_accuracy']:.2f}%")
            click.echo(f"  Best Accuracy: {exp.results['best_accuracy']:.2f}%")
            click.echo(f"  Final Precision: {exp.results['final_precision']:.4f}")
            click.echo(f"  Final Recall: {exp.results['final_recall']:.4f}")
            click.echo(f"  Final F1-Score: {exp.results['final_f1_score']:.4f}")
            click.echo(f"  Best F1-Score: {exp.results['best_f1_score']:.4f}")
            click.echo()
            
            self._print_success(f"Experiment '{exp_name}' completed successfully!")
            
        except KeyboardInterrupt:
            exp.status = "failed"
            exp.completed_at = datetime.now()
            exp.results["interrupted"] = True
            self._print_warning("\nExperiment interrupted by user.")
        except Exception as e:
            exp.status = "failed"
            exp.completed_at = datetime.now()
            exp.results["error"] = str(e)
            self._print_error(f"Experiment failed: {e}")
    
    def do_compare_experiments(self, arg: str) -> None:
        """
        Compare multiple experiments.
        
        Usage:
            compare_experiments exp1 exp2      - Compare two experiments
            compare_experiments exp1 exp2 exp3 - Compare multiple
        """
        if not arg:
            self._print_error("Please specify experiments to compare.")
            click.echo("Usage: compare_experiments <exp1> <exp2> [exp3...]")
            return
        
        exp_names = arg.split()
        
        if len(exp_names) < 2:
            self._print_error("Need at least two experiments to compare.")
            return
        
        # Validate experiments
        experiments = []
        for name in exp_names:
            if name not in self._experiments:
                self._print_error(f"Experiment not found: {name}")
                return
            experiments.append(self._experiments[name])
        
        click.echo("\n" + "=" * 70)
        click.echo("Experiment Comparison")
        click.echo("=" * 70)
        
        # Header
        header = f"{'Metric':<20}"
        for exp in experiments:
            header += f"{exp.name:<15}"
        click.echo(header)
        click.echo("-" * 70)
        
        # Status
        row = f"{'Status':<20}"
        for exp in experiments:
            row += f"{exp.status:<15}"
        click.echo(row)
        
        # Model
        row = f"{'Model':<20}"
        for exp in experiments:
            arch = exp.config.get("model", {}).get("architecture", "N/A")
            row += f"{arch[:13]:<15}"
        click.echo(row)
        
        # Training params
        row = f"{'Epochs':<20}"
        for exp in experiments:
            epochs = exp.config.get("training", {}).get("epochs", "N/A")
            row += f"{epochs:<15}"
        click.echo(row)
        
        row = f"{'Learning Rate':<20}"
        for exp in experiments:
            lr = exp.config.get("training", {}).get("learning_rate", "N/A")
            if isinstance(lr, float):
                row += f"{lr:<15.6f}"
            else:
                row += f"{lr:<15}"
        click.echo(row)
        
        # Results (if available)
        click.echo("-" * 70)
        
        row = f"{'Best Accuracy':<20}"
        for exp in experiments:
            if exp.best_metric is not None:
                row += f"{exp.best_metric:<15.2f}"
            else:
                row += f"{'N/A':<15}"
        click.echo(row)
        
        row = f"{'Final Loss':<20}"
        for exp in experiments:
            loss = exp.results.get("final_loss")
            if loss is not None:
                row += f"{loss:<15.4f}"
            else:
                row += f"{'N/A':<15}"
        click.echo(row)
        
        click.echo()
    
    def do_delete_experiment(self, arg: str) -> None:
        """
        Delete an experiment.
        
        Usage:
            delete_experiment my_exp
        """
        if not arg:
            self._print_error("Please specify an experiment name.")
            return
        
        exp_name = arg.strip()
        if exp_name not in self._experiments:
            self._print_error(f"Experiment not found: {exp_name}")
            return
        
        exp = self._experiments[exp_name]
        if exp.status == "running":
            self._print_error("Cannot delete a running experiment.")
            return
        
        del self._experiments[exp_name]
        
        if self._current_experiment == exp_name:
            self._current_experiment = None
        
        self._print_success(f"Deleted experiment: {exp_name}")
    
    def do_export_experiment(self, arg: str) -> None:
        """
        Export experiment results to a file.
        
        Usage:
            export_experiment results.json               - Export current experiment
            export_experiment my_exp results.yaml        - Export specific experiment
        """
        if not arg:
            self._print_error("Please specify an output path.")
            click.echo("Usage: export_experiment [exp_name] <output_path>")
            return
        
        parts = arg.split()
        
        if len(parts) == 1:
            exp_name = self._current_experiment
            output_path = Path(parts[0])
        else:
            exp_name = parts[0]
            output_path = Path(parts[1])
        
        if not exp_name:
            self._print_error("No experiment specified and no current experiment.")
            return
        
        if exp_name not in self._experiments:
            self._print_error(f"Experiment not found: {exp_name}")
            return
        
        exp = self._experiments[exp_name]
        
        # Prepare export data
        export_data = {
            "name": exp.name,
            "status": exp.status,
            "created_at": exp.created_at.isoformat(),
            "started_at": exp.started_at.isoformat() if exp.started_at else None,
            "completed_at": exp.completed_at.isoformat() if exp.completed_at else None,
            "config": exp.config,
            "results": exp.results,
            "history": exp.history,
            "best_metric": exp.best_metric,
        }
        
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            if output_path.suffix in (".yaml", ".yml"):
                import yaml
                with open(output_path, "w") as f:
                    yaml.dump(export_data, f, default_flow_style=False)
            else:
                import json
                with open(output_path, "w") as f:
                    json.dump(export_data, f, indent=2, default=str)
            
            self._print_success(f"Exported experiment to: {output_path}")
            
        except Exception as e:
            self._print_error(f"Failed to export: {e}")
    
    def complete_use_experiment(self, text: str, line: str, begidx: int, endidx: int) -> List[str]:
        """Tab completion for use_experiment command."""
        return [name for name in self._experiments if name.startswith(text)]
    
    def complete_experiment_info(self, text: str, line: str, begidx: int, endidx: int) -> List[str]:
        """Tab completion for experiment_info command."""
        return [name for name in self._experiments if name.startswith(text)]
    
    def complete_run_experiment(self, text: str, line: str, begidx: int, endidx: int) -> List[str]:
        """Tab completion for run_experiment command."""
        return [name for name in self._experiments if name.startswith(text)]
    
    def complete_delete_experiment(self, text: str, line: str, begidx: int, endidx: int) -> List[str]:
        """Tab completion for delete_experiment command."""
        return [name for name in self._experiments if name.startswith(text)]
    
    # ==================== Multi-Model/Dataset/Config Commands ====================
    
    def do_add_model(self, arg: str) -> None:
        """
        Add a model to the model registry for experiments.
        
        Usage:
            add_model my_resnet resnet50 --classes 10
            add_model vit vit_base --classes 100 --pretrained
        """
        if not arg:
            self._print_error("Please specify a name and model architecture.")
            click.echo("Usage: add_model <name> <architecture> [--classes N] [--pretrained]")
            return
        
        parts = arg.split()
        if len(parts) < 2:
            self._print_error("Please specify both a name and architecture.")
            return
        
        name = parts[0]
        architecture = parts[1]
        
        # Parse options
        num_classes = 1000
        pretrained = True
        
        for i, part in enumerate(parts[2:], 2):
            if part == "--classes" and i + 1 < len(parts):
                try:
                    num_classes = int(parts[i + 1])
                except ValueError:
                    pass
            elif part == "--pretrained":
                pretrained = True
            elif part == "--no-pretrained":
                pretrained = False
        
        try:
            from mlcli.core.config import ModelConfig, TaskType, ModelFamily
            from mlcli.core.factory import ModelFactory
            
            # Determine task type and model family
            model_lower = architecture.lower()
            is_detection = any(x in model_lower for x in ["yolo", "rcnn", "ssd", "detr", "fasterrcnn"])
            is_transformer = any(x in model_lower for x in ["vit", "swin", "detr"])
            
            task_type = TaskType.OBJECT_DETECTION if is_detection else TaskType.CLASSIFICATION
            family = ModelFamily.TRANSFORMER if is_transformer else ModelFamily.CNN
            
            config = ModelConfig(
                task_type=task_type,
                family=family,
                architecture=architecture,
                num_classes=num_classes,
                pretrained=pretrained,
            )
            model = ModelFactory.create(config)
            model = model.to(self._torch_device)
            model.eval()
            
            self._models[name] = {
                "model": model,
                "architecture": architecture,
                "num_classes": num_classes,
                "pretrained": pretrained,
                "config": config,
            }
            
            params = model.get_num_parameters()
            self._print_success(f"Added model: {name} ({architecture})")
            click.echo(f"  Parameters: {params:,}")
            click.echo(f"  Classes: {num_classes}")
            
        except Exception as e:
            self._print_error(f"Failed to add model: {e}")
    
    def do_list_models(self, arg: str) -> None:
        """
        List all registered models.
        
        Usage:
            list_models
        """
        if not self._models:
            self._print_info("No models registered.")
            click.echo("Use 'add_model <name> <arch>' to add one.")
            return
        
        click.echo("\n" + "=" * 60)
        click.echo("Registered Models")
        click.echo("=" * 60)
        
        for name, info in self._models.items():
            params = info["model"].get_num_parameters()
            click.echo(f"\n  {name}:")
            click.echo(f"    Architecture: {info['architecture']}")
            click.echo(f"    Classes: {info['num_classes']}")
            click.echo(f"    Parameters: {params:,}")
            click.echo(f"    Pretrained: {info['pretrained']}")
        
        click.echo()
    
    def do_remove_model(self, arg: str) -> None:
        """
        Remove a model from the registry.
        
        Usage:
            remove_model my_resnet
        """
        if not arg:
            self._print_error("Please specify a model name.")
            return
        
        name = arg.strip()
        if name not in self._models:
            self._print_error(f"Model not found: {name}")
            return
        
        del self._models[name]
        self._print_success(f"Removed model: {name}")
    
    def do_add_dataset(self, arg: str) -> None:
        """
        Add a dataset to the dataset registry for experiments.
        
        Usage:
            add_dataset train_data ./data/train
            add_dataset val_data ./data/val --type folder
            add_dataset coco_train ./coco --type coco
        """
        if not arg:
            self._print_error("Please specify a name and dataset path.")
            click.echo("Usage: add_dataset <name> <path> [--type folder|coco|voc]")
            return
        
        parts = arg.split()
        if len(parts) < 2:
            self._print_error("Please specify both a name and path.")
            return
        
        name = parts[0]
        dataset_path = Path(parts[1])
        
        # Parse dataset type
        dataset_type = "folder"
        for i, part in enumerate(parts[2:], 2):
            if part == "--type" and i + 1 < len(parts):
                dataset_type = parts[i + 1]
        
        if not dataset_path.exists():
            self._print_error(f"Path not found: {dataset_path}")
            return
        
        try:
            from torchvision import transforms
            default_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])
            
            if dataset_type == "folder":
                from mlcli.data.classification import ImageFolderDataset
                dataset = ImageFolderDataset(
                    root=dataset_path,
                    transform=default_transform,
                )
            elif dataset_type == "coco":
                from mlcli.data.detection import COCODataset
                ann_file = dataset_path / "annotations.json"
                dataset = COCODataset(root=dataset_path, annotation_file=ann_file)
            elif dataset_type == "voc":
                from mlcli.data.detection import VOCDataset
                dataset = VOCDataset(root=dataset_path)
            else:
                self._print_error(f"Unknown dataset type: {dataset_type}")
                return
            
            self._datasets[name] = {
                "dataset": dataset,
                "path": dataset_path,
                "type": dataset_type,
                "samples": len(dataset),
                "classes": dataset.get_num_classes(),
            }
            
            self._print_success(f"Added dataset: {name}")
            click.echo(f"  Path: {dataset_path}")
            click.echo(f"  Samples: {len(dataset)}")
            click.echo(f"  Classes: {dataset.get_num_classes()}")
            
        except Exception as e:
            self._print_error(f"Failed to add dataset: {e}")
    
    def do_list_datasets(self, arg: str) -> None:
        """
        List all registered datasets.
        
        Usage:
            list_datasets
        """
        if not self._datasets:
            self._print_info("No datasets registered.")
            click.echo("Use 'add_dataset <name> <path>' to add one.")
            return
        
        click.echo("\n" + "=" * 60)
        click.echo("Registered Datasets")
        click.echo("=" * 60)
        
        for name, info in self._datasets.items():
            click.echo(f"\n  {name}:")
            click.echo(f"    Path: {info['path']}")
            click.echo(f"    Type: {info['type']}")
            click.echo(f"    Samples: {info['samples']}")
            click.echo(f"    Classes: {info['classes']}")
        
        click.echo()
    
    def do_remove_dataset(self, arg: str) -> None:
        """
        Remove a dataset from the registry.
        
        Usage:
            remove_dataset train_data
        """
        if not arg:
            self._print_error("Please specify a dataset name.")
            return
        
        name = arg.strip()
        if name not in self._datasets:
            self._print_error(f"Dataset not found: {name}")
            return
        
        del self._datasets[name]
        self._print_success(f"Removed dataset: {name}")
    
    def do_add_config(self, arg: str) -> None:
        """
        Add a training configuration for experiments.
        
        Usage:
            add_config fast --epochs 10 --lr 0.01 --batch-size 64
            add_config slow --epochs 100 --lr 0.0001 --batch-size 16
            add_config from_file --file config.yaml
        """
        if not arg:
            self._print_error("Please specify a config name.")
            click.echo("Usage: add_config <name> [--epochs N] [--lr F] [--batch-size N] [--file path]")
            return
        
        parts = arg.split()
        name = parts[0]
        
        # Parse options
        config = {
            "epochs": 10,
            "learning_rate": 0.001,
            "batch_size": 32,
            "optimizer": "adamw",
            "scheduler": "cosine",
            "mixed_precision": False,
        }
        
        config_file = None
        
        i = 1
        while i < len(parts):
            if parts[i] == "--epochs" and i + 1 < len(parts):
                try:
                    config["epochs"] = int(parts[i + 1])
                except ValueError:
                    pass
                i += 2
            elif parts[i] == "--lr" and i + 1 < len(parts):
                try:
                    config["learning_rate"] = float(parts[i + 1])
                except ValueError:
                    pass
                i += 2
            elif parts[i] == "--batch-size" and i + 1 < len(parts):
                try:
                    config["batch_size"] = int(parts[i + 1])
                except ValueError:
                    pass
                i += 2
            elif parts[i] == "--optimizer" and i + 1 < len(parts):
                config["optimizer"] = parts[i + 1]
                i += 2
            elif parts[i] == "--scheduler" and i + 1 < len(parts):
                config["scheduler"] = parts[i + 1]
                i += 2
            elif parts[i] == "--mixed-precision":
                config["mixed_precision"] = True
                i += 1
            elif parts[i] == "--file" and i + 1 < len(parts):
                config_file = parts[i + 1]
                i += 2
            else:
                i += 1
        
        try:
            # Load from file if specified
            if config_file:
                path = Path(config_file)
                if not path.exists():
                    self._print_error(f"Config file not found: {config_file}")
                    return
                
                import yaml
                with open(path) as f:
                    file_config = yaml.safe_load(f)
                
                # Extract training config
                if "training" in file_config:
                    config.update(file_config["training"])
                else:
                    config.update(file_config)
            
            self._configs[name] = config
            
            self._print_success(f"Added config: {name}")
            click.echo(f"  Epochs: {config['epochs']}")
            click.echo(f"  Learning Rate: {config['learning_rate']}")
            click.echo(f"  Batch Size: {config['batch_size']}")
            click.echo(f"  Optimizer: {config['optimizer']}")
            click.echo(f"  Scheduler: {config['scheduler']}")
            
        except Exception as e:
            self._print_error(f"Failed to add config: {e}")
    
    def do_list_configs(self, arg: str) -> None:
        """
        List all registered training configurations.
        
        Usage:
            list_configs
        """
        if not self._configs:
            self._print_info("No configs registered.")
            click.echo("Use 'add_config <name> [options]' to add one.")
            return
        
        click.echo("\n" + "=" * 60)
        click.echo("Registered Configurations")
        click.echo("=" * 60)
        
        for name, config in self._configs.items():
            click.echo(f"\n  {name}:")
            click.echo(f"    Epochs: {config.get('epochs', 'N/A')}")
            click.echo(f"    Learning Rate: {config.get('learning_rate', 'N/A')}")
            click.echo(f"    Batch Size: {config.get('batch_size', 'N/A')}")
            click.echo(f"    Optimizer: {config.get('optimizer', 'N/A')}")
            click.echo(f"    Scheduler: {config.get('scheduler', 'N/A')}")
        
        click.echo()
    
    def do_remove_config(self, arg: str) -> None:
        """
        Remove a configuration from the registry.
        
        Usage:
            remove_config fast
        """
        if not arg:
            self._print_error("Please specify a config name.")
            return
        
        name = arg.strip()
        if name not in self._configs:
            self._print_error(f"Config not found: {name}")
            return
        
        del self._configs[name]
        self._print_success(f"Removed config: {name}")
    
    def do_experiment_add_models(self, arg: str) -> None:
        """
        Add models to the current experiment for multi-model runs.
        
        Usage:
            experiment_add_models model1 model2 model3
            experiment_add_models --all
        """
        if not self._current_experiment:
            self._print_error("No current experiment. Use 'experiment <name>' first.")
            return
        
        exp = self._experiments[self._current_experiment]
        
        if not arg:
            self._print_error("Please specify model names to add.")
            click.echo("Usage: experiment_add_models <model1> [model2] ... or --all")
            return
        
        parts = arg.split()
        
        if parts[0] == "--all":
            if not self._models:
                self._print_error("No models registered. Use 'add_model' first.")
                return
            model_names = list(self._models.keys())
        else:
            model_names = parts
            # Validate models exist
            for name in model_names:
                if name not in self._models:
                    self._print_error(f"Model not found: {name}")
                    return
        
        # Add to experiment
        for name in model_names:
            if name not in exp.models:
                exp.models.append(name)
        
        self._print_success(f"Added {len(model_names)} model(s) to experiment")
        click.echo(f"  Models: {', '.join(exp.models)}")
    
    def do_experiment_add_datasets(self, arg: str) -> None:
        """
        Add datasets to the current experiment for multi-dataset runs.
        
        Usage:
            experiment_add_datasets dataset1 dataset2
            experiment_add_datasets --all
        """
        if not self._current_experiment:
            self._print_error("No current experiment. Use 'experiment <name>' first.")
            return
        
        exp = self._experiments[self._current_experiment]
        
        if not arg:
            self._print_error("Please specify dataset names to add.")
            click.echo("Usage: experiment_add_datasets <dataset1> [dataset2] ... or --all")
            return
        
        parts = arg.split()
        
        if parts[0] == "--all":
            if not self._datasets:
                self._print_error("No datasets registered. Use 'add_dataset' first.")
                return
            dataset_names = list(self._datasets.keys())
        else:
            dataset_names = parts
            # Validate datasets exist
            for name in dataset_names:
                if name not in self._datasets:
                    self._print_error(f"Dataset not found: {name}")
                    return
        
        # Add to experiment
        for name in dataset_names:
            if name not in exp.datasets:
                exp.datasets.append(name)
        
        self._print_success(f"Added {len(dataset_names)} dataset(s) to experiment")
        click.echo(f"  Datasets: {', '.join(exp.datasets)}")
    
    def do_experiment_add_configs(self, arg: str) -> None:
        """
        Add configurations to the current experiment for grid search.
        
        Usage:
            experiment_add_configs config1 config2
            experiment_add_configs --all
        """
        if not self._current_experiment:
            self._print_error("No current experiment. Use 'experiment <name>' first.")
            return
        
        exp = self._experiments[self._current_experiment]
        
        if not arg:
            self._print_error("Please specify config names to add.")
            click.echo("Usage: experiment_add_configs <config1> [config2] ... or --all")
            return
        
        parts = arg.split()
        
        if parts[0] == "--all":
            if not self._configs:
                self._print_error("No configs registered. Use 'add_config' first.")
                return
            config_names = list(self._configs.keys())
        else:
            config_names = parts
            # Validate configs exist
            for name in config_names:
                if name not in self._configs:
                    self._print_error(f"Config not found: {name}")
                    return
        
        # Add to experiment
        for name in config_names:
            config_copy = self._configs[name].copy()
            config_copy["_name"] = name
            if config_copy not in exp.configs:
                exp.configs.append(config_copy)
        
        self._print_success(f"Added {len(config_names)} config(s) to experiment")
        click.echo(f"  Configs: {', '.join(config_names)}")
    
    def do_run_grid_experiment(self, arg: str) -> None:
        """
        Run experiment with all combinations of models, datasets, and configs.

        Usage:
            run_grid_experiment           - Run current experiment grid
            run_grid_experiment my_exp    - Run specific experiment grid
            run_grid_experiment --resume  - Resume interrupted grid experiment
        """
        parts = arg.split() if arg else []
        exp_name = self._current_experiment
        resume = False

        for part in parts:
            if part == "--resume":
                resume = True
            elif not part.startswith("--"):
                exp_name = part

        if not exp_name:
            self._print_error("No experiment specified and no current experiment.")
            return

        if exp_name not in self._experiments:
            self._print_error(f"Experiment not found: {exp_name}")
            return

        exp = self._experiments[exp_name]

        if resume:
            _state_path = Path(self.ctx.output_dir or "./outputs") / exp_name / "experiment_state.json"
            if _state_path.exists():
                exp = ExperimentRun.load_from_disk(_state_path)
                self._experiments[exp_name] = exp
                self._print_info(f"Resumed experiment '{exp_name}': {len(exp.completed_run_keys)} combos already done")
            else:
                self._print_warning("No saved state found, starting fresh")
        
        # Validate experiment has models, datasets, and configs
        models = exp.models if exp.models else ([self._model_name] if self._model_name else [])
        datasets = exp.datasets if exp.datasets else (["default"] if self._dataset else [])
        configs = exp.configs if exp.configs else [exp.config.get("training", {})]
        
        if not models:
            self._print_error("No models specified. Use 'experiment_add_models' or 'add_model'.")
            return
        
        if not datasets:
            self._print_error("No datasets specified. Use 'experiment_add_datasets' or 'add_dataset'.")
            return
        
        # Calculate total runs
        total_runs = len(models) * len(datasets) * len(configs)
        
        click.echo("\n" + "=" * 70)
        click.echo(f"Grid Experiment: {exp_name}")
        click.echo("=" * 70)
        click.echo(f"  Models: {len(models)} - {', '.join(models)}")
        click.echo(f"  Datasets: {len(datasets)} - {', '.join(datasets)}")
        click.echo(f"  Configs: {len(configs)}")
        click.echo(f"  Total Runs: {total_runs}")
        click.echo()
        
        # Confirm
        if total_runs > 1:
            click.echo(click.style("This will run multiple training sessions.", fg="yellow"))
        
        exp.status = "running"
        exp.started_at = datetime.now()
        if not resume:
            exp.run_results = []

        # Determine state path for incremental saves
        _state_path = getattr(exp, "state_path", None)
        if _state_path is None:
            _exp_output_dir = Path(self.ctx.output_dir or "./outputs") / exp_name
            _exp_output_dir.mkdir(parents=True, exist_ok=True)
            _state_path = _exp_output_dir / "experiment_state.json"
            exp.state_path = _state_path

        run_idx = 0
        
        try:
            import torch
            import torch.nn as nn
            from tqdm import tqdm
            
            for model_name in models:
                for dataset_name in datasets:
                    for config in configs:
                        run_idx += 1
                        config_name = config.get("_name", f"config_{run_idx}")

                        _combo_key = f"{model_name}|{dataset_name}|{config_name}"
                        if _combo_key in exp.completed_run_keys:
                            self._print_info(f"Skipping already-completed: {_combo_key}")
                            continue

                        click.echo("\n" + "-" * 60)
                        click.echo(f"Run {run_idx}/{total_runs}: {model_name} + {dataset_name} + {config_name}")
                        click.echo("-" * 60)

                        # Get model
                        if model_name in self._models:
                            model = self._models[model_name]["model"]
                        elif model_name == self._model_name:
                            model = self._model
                        else:
                            self._print_warning(f"Model not found: {model_name}, skipping...")
                            continue
                        
                        # Get dataset
                        if dataset_name in self._datasets:
                            dataset = self._datasets[dataset_name]["dataset"]
                        elif dataset_name == "default":
                            dataset = self._dataset
                        else:
                            self._print_warning(f"Dataset not found: {dataset_name}, skipping...")
                            continue
                        
                        # Get training params
                        epochs = config.get("epochs", 10)
                        lr = config.get("learning_rate", 0.001)
                        batch_size = config.get("batch_size", 32)
                        
                        # Create data loader
                        train_loader = dataset.create_dataloader(
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=4,
                        )
                        
                        # Create optimizer
                        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
                        criterion = nn.CrossEntropyLoss()
                        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                            optimizer, T_max=epochs
                        )
                        
                        # Training
                        model.to(self._torch_device)
                        model.train()
                        
                        run_history = {
                            "loss": [], "accuracy": [],
                            "precision": [], "recall": [], "f1_score": []
                        }
                        best_acc = 0.0
                        best_f1 = 0.0
                        final_precision = 0.0
                        final_recall = 0.0
                        final_f1 = 0.0
                        
                        for epoch in range(epochs):
                            total_loss = 0.0
                            correct = 0
                            total = 0
                            
                            # For precision/recall/F1
                            all_predictions = []
                            all_targets = []
                            
                            pbar = tqdm(
                                train_loader,
                                desc=f"Epoch {epoch + 1}/{epochs}",
                                leave=False,
                            )
                            
                            for batch_idx, (inputs, targets) in enumerate(pbar):
                                inputs = inputs.to(self._torch_device)
                                targets = targets.to(self._torch_device)
                                
                                optimizer.zero_grad()
                                outputs = model(inputs)
                                loss = criterion(outputs, targets)
                                loss.backward()
                                optimizer.step()
                                
                                total_loss += loss.item()
                                _, predicted = outputs.max(1)
                                total += targets.size(0)
                                correct += predicted.eq(targets).sum().item()
                                
                                # Collect for metrics
                                all_predictions.extend(predicted.cpu().numpy())
                                all_targets.extend(targets.cpu().numpy())
                                
                                avg_loss = total_loss / (batch_idx + 1)
                                acc = 100.0 * correct / total
                                pbar.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{acc:.2f}%")
                            
                            scheduler.step()
                            avg_loss = total_loss / len(train_loader)
                            acc = 100.0 * correct / total
                            
                            # Calculate metrics
                            precision, recall, f1 = self._calculate_metrics(all_predictions, all_targets)
                            
                            run_history["loss"].append(avg_loss)
                            run_history["accuracy"].append(acc)
                            run_history["precision"].append(precision)
                            run_history["recall"].append(recall)
                            run_history["f1_score"].append(f1)
                            
                            if acc > best_acc:
                                best_acc = acc
                            if f1 > best_f1:
                                best_f1 = f1
                            
                            final_precision = precision
                            final_recall = recall
                            final_f1 = f1
                        
                        # Store run result
                        run_result = {
                            "model": model_name,
                            "dataset": dataset_name,
                            "config": config_name,
                            "best_accuracy": best_acc,
                            "final_loss": run_history["loss"][-1],
                            "final_accuracy": run_history["accuracy"][-1],
                            "final_precision": final_precision,
                            "final_recall": final_recall,
                            "final_f1_score": final_f1,
                            "best_f1_score": best_f1,
                            "history": run_history,
                        }
                        exp.run_results.append(run_result)
                        exp.completed_run_keys.add(_combo_key)
                        exp.save_to_disk(_state_path)

                        click.echo(f"  ✓ Acc: {best_acc:.2f}%, F1: {best_f1:.4f}")
            
            # Experiment completed
            exp.status = "completed"
            exp.completed_at = datetime.now()
            
            # Find best run by F1-score
            best_run = max(exp.run_results, key=lambda x: x["best_f1_score"])
            exp.best_metric = best_run["best_accuracy"]
            exp.results = {
                "total_runs": total_runs,
                "best_run": best_run,
                "all_runs": exp.run_results,
            }
            
            duration = (exp.completed_at - exp.started_at).total_seconds()
            
            click.echo("\n" + "=" * 70)
            click.echo("Grid Experiment Completed!")
            click.echo("=" * 70)
            click.echo(f"  Total Runs: {total_runs}")
            click.echo(f"  Duration: {duration:.1f}s")
            click.echo(f"\nBest Run (by F1-Score):")
            click.echo(f"  Model: {best_run['model']}")
            click.echo(f"  Dataset: {best_run['dataset']}")
            click.echo(f"  Config: {best_run['config']}")
            click.echo(f"  Accuracy: {best_run['best_accuracy']:.2f}%")
            click.echo(f"  Precision: {best_run['final_precision']:.4f}")
            click.echo(f"  Recall: {best_run['final_recall']:.4f}")
            click.echo(f"  F1-Score: {best_run['best_f1_score']:.4f}")
            click.echo()
            
            self._print_success(f"Grid experiment '{exp_name}' completed!")
            
        except KeyboardInterrupt:
            exp.status = "failed"
            exp.completed_at = datetime.now()
            exp.results["interrupted_at_run"] = run_idx
            self._print_warning("\nExperiment interrupted by user.")
        except Exception as e:
            exp.status = "failed"
            exp.completed_at = datetime.now()
            exp.results["error"] = str(e)
            self._print_error(f"Experiment failed: {e}")
    
    def do_experiment_results(self, arg: str) -> None:
        """
        Show detailed results for a grid experiment.
        
        Usage:
            experiment_results           - Show current experiment results
            experiment_results my_exp    - Show specific experiment results
        """
        exp_name = arg.strip() if arg else self._current_experiment
        
        if not exp_name:
            self._print_error("No experiment specified.")
            return
        
        if exp_name not in self._experiments:
            self._print_error(f"Experiment not found: {exp_name}")
            return
        
        exp = self._experiments[exp_name]
        
        if not exp.run_results:
            self._print_info("No run results available for this experiment.")
            return
        
        click.echo("\n" + "=" * 110)
        click.echo(f"Experiment Results: {exp_name}")
        click.echo("=" * 110)
        
        # Table header
        click.echo(
            f"\n{'#':<3} {'Model':<12} {'Dataset':<12} {'Config':<10} "
            f"{'Acc':<8} {'Prec':<8} {'Recall':<8} {'F1':<8} {'Loss':<8}"
        )
        click.echo("-" * 110)
        
        for i, run in enumerate(exp.run_results, 1):
            click.echo(
                f"{i:<3} "
                f"{run['model'][:10]:<12} "
                f"{run['dataset'][:10]:<12} "
                f"{run['config'][:8]:<10} "
                f"{run['best_accuracy']:<8.2f} "
                f"{run.get('final_precision', 0):<8.4f} "
                f"{run.get('final_recall', 0):<8.4f} "
                f"{run.get('best_f1_score', run.get('final_f1_score', 0)):<8.4f} "
                f"{run['final_loss']:<8.4f}"
            )
        
        click.echo("-" * 110)
        
        if exp.results.get("best_run"):
            best = exp.results["best_run"]
            click.echo(f"\nBest Configuration (by F1-Score):")
            click.echo(f"  Model: {best['model']}")
            click.echo(f"  Dataset: {best['dataset']}")
            click.echo(f"  Config: {best['config']}")
            click.echo(f"  Accuracy: {best['best_accuracy']:.2f}%")
            click.echo(f"  Precision: {best.get('final_precision', 0):.4f}")
            click.echo(f"  Recall: {best.get('final_recall', 0):.4f}")
            click.echo(f"  F1-Score: {best.get('best_f1_score', best.get('final_f1_score', 0)):.4f}")
        
        click.echo()
    
    def complete_add_model(self, text: str, line: str, begidx: int, endidx: int) -> List[str]:
        """Tab completion for add_model command."""
        from mlcli.core.registry import MODEL_REGISTRY
        models = MODEL_REGISTRY.list_registered()
        return [m for m in models if m.startswith(text)]
    
    def complete_remove_model(self, text: str, line: str, begidx: int, endidx: int) -> List[str]:
        """Tab completion for remove_model command."""
        return [name for name in self._models if name.startswith(text)]
    
    def complete_remove_dataset(self, text: str, line: str, begidx: int, endidx: int) -> List[str]:
        """Tab completion for remove_dataset command."""
        return [name for name in self._datasets if name.startswith(text)]
    
    def complete_remove_config(self, text: str, line: str, begidx: int, endidx: int) -> List[str]:
        """Tab completion for remove_config command."""
        return [name for name in self._configs if name.startswith(text)]
    
    def complete_experiment_add_models(self, text: str, line: str, begidx: int, endidx: int) -> List[str]:
        """Tab completion for experiment_add_models command."""
        return [name for name in self._models if name.startswith(text)]
    
    def complete_experiment_add_datasets(self, text: str, line: str, begidx: int, endidx: int) -> List[str]:
        """Tab completion for experiment_add_datasets command."""
        return [name for name in self._datasets if name.startswith(text)]
    
    def complete_experiment_add_configs(self, text: str, line: str, begidx: int, endidx: int) -> List[str]:
        """Tab completion for experiment_add_configs command."""
        return [name for name in self._configs if name.startswith(text)]

    # ==================== Utility Commands ====================
    
    def do_save(self, arg: str) -> None:
        """
        Save the current model.
        
        Usage:
            save ./model.pt
        """
        if not self._model:
            self._print_warning("No model loaded.")
            return
        
        if not arg:
            self._print_error("Please specify an output path.")
            click.echo("Usage: save <path>")
            return
        
        try:
            import torch
            path = Path(arg)
            path.parent.mkdir(parents=True, exist_ok=True)
            
            torch.save({
                "model_state_dict": self._model.state_dict(),
                "model_name": self._model_name,
            }, path)
            
            self._print_success(f"Model saved to: {path}")
            
        except Exception as e:
            self._print_error(f"Failed to save model: {e}")
    
    def do_export(self, arg: str) -> None:
        """
        Export model to ONNX or TorchScript.
        
        Usage:
            export ./model.onnx --format onnx
            export ./model.pt --format torchscript
        """
        if not self._model:
            self._print_warning("No model loaded.")
            return
        
        if not arg:
            self._print_error("Please specify an output path.")
            click.echo("Usage: export <path> [--format onnx|torchscript]")
            return
        
        parts = arg.split()
        output_path = Path(parts[0])
        
        # Parse format
        export_format = "onnx"
        for i, part in enumerate(parts[1:], 1):
            if part == "--format" and i < len(parts) - 1:
                export_format = parts[i + 1]
        
        try:
            from mlcli.utils.io import export_model
            
            self._print_info(f"Exporting model to {export_format}...")
            export_model(
                self._model,
                output_path,
                input_shape=(3, 224, 224),
                format=export_format,
            )
            
            self._print_success(f"Model exported to: {output_path}")
            
        except Exception as e:
            self._print_error(f"Export failed: {e}")
    
    def complete_load_model(self, text: str, line: str, begidx: int, endidx: int) -> List[str]:
        """Tab completion for load_model command."""
        from mlcli.core.registry import MODEL_REGISTRY
        models = MODEL_REGISTRY.list_registered()
        return [m for m in models if m.startswith(text)]
    
    def complete_device(self, text: str, line: str, begidx: int, endidx: int) -> List[str]:
        """Tab completion for device command."""
        options = ["cpu", "cuda", "cuda:0", "cuda:1", "mps"]
        return [o for o in options if o.startswith(text)]


@click.command()
@pass_context
def interactive(ctx: Context) -> None:
    """
    Start interactive mode.
    
    Launches a REPL-style interface for exploring models,
    loading datasets, and running experiments interactively.
    """
    try:
        shell = InteractiveShell(ctx)
        shell.cmdloop()
    except KeyboardInterrupt:
        click.echo("\nExiting...")

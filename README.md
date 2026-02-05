# MLCLI - Production-Grade Machine Learning CLI

A modular, production-grade Python CLI tool for machine learning that supports training, evaluation, and inference across multiple model families.

## Features

### Model Support

**Classification (CNN-based):**
- ResNet (18, 34, 50, 101, 152)
- EfficientNet (B0-B7)
- MobileNet (V2, V3 Small, V3 Large)

**Classification (Transformer-based):**
- Vision Transformer (ViT) - Tiny, Small, Base, Large
- Swin Transformer - Tiny, Small, Base, Large

**Object Detection:**
- YOLO (v8 variants)
- Faster R-CNN
- SSD
- DETR

### Architecture

- **Modular Design**: Plugin-based architecture for easy extension
- **Registry Pattern**: Dynamic component registration
- **Factory Pattern**: Flexible model/dataset/task creation
- **Task Abstraction**: Unified interface for different ML tasks

### Training Features

- Configurable training pipelines with callbacks
- Mixed precision training (FP16/BF16)
- Gradient clipping and accumulation
- Multiple optimizers (Adam, AdamW, SGD)
- Learning rate schedulers (Step, Cosine, Plateau, Warmup)
- Early stopping and model checkpointing

### Logging & Metrics

- Console, File, and JSON logging
- TensorBoard integration
- Metric tracking (Accuracy, F1, mAP, IoU)
- Experiment reproducibility

## Installation

```bash
# Clone the repository
git clone https://github.com/your-org/ml-utils.git
cd ml-utils

# Install in development mode
pip install -e ".[dev]"

# Or install with all optional dependencies
pip install -e ".[all]"
```

## Quick Start

### Training a Model

```bash
# Train a ResNet50 classifier
mlcli train --model resnet50 --dataset ./data/train --epochs 100

# Train with custom hyperparameters
mlcli train \
    --model vit_base \
    --dataset ./data/train \
    --val-dataset ./data/val \
    --epochs 100 \
    --batch-size 32 \
    --learning-rate 0.0001 \
    --optimizer adamw \
    --scheduler cosine \
    --mixed-precision

# Train an object detector
mlcli train \
    --model yolo \
    --dataset ./coco/train \
    --task detection \
    --epochs 50
```

### Evaluating a Model

```bash
# Evaluate on test set
mlcli evaluate --checkpoint ./model.pt --dataset ./data/test

# Evaluate with per-class metrics
mlcli evaluate -c ./model.pt -d ./data/test --per-class

# Generate confusion matrix
mlcli evaluate -c ./model.pt -d ./data/test --confusion-matrix
```

### Running Inference

```bash
# Single image inference
mlcli inference --model ./model.pt --input ./image.jpg

# Batch inference on directory
mlcli inference -m ./model.pt -i ./images/ -o ./predictions

# Detection with visualization
mlcli inference -m ./detector.pt -i ./image.jpg --save-visualization
```

### Interactive Mode

Launch an interactive REPL for exploring models and running experiments:

```bash
# Start interactive mode
mlcli interactive
```

Inside the interactive shell:

```
mlcli> models                    # List available models
mlcli> load_model resnet50 --classes 10
mlcli> model_info                # Show model details
mlcli> load_dataset ./data/train
mlcli> dataset_info              # Show dataset details
mlcli> predict ./image.jpg       # Run inference
mlcli> benchmark --iters 100     # Benchmark speed
mlcli> train --epochs 10 --lr 0.001
mlcli> save ./my_model.pt        # Save model
mlcli> export ./model.onnx --format onnx
mlcli> status                    # Show session status
mlcli> device cuda               # Switch device
mlcli> help                      # Show all commands
mlcli> quit                      # Exit
```

#### Experiment Runs

Manage and track experiment runs within interactive mode:

```
# Create a new experiment
mlcli> experiment my_exp --model resnet50 --epochs 50 --lr 0.001

# Create experiment from config file
mlcli> experiment my_exp --config config.yaml

# List all experiments
mlcli> experiments

# Switch to a specific experiment
mlcli> use_experiment my_exp

# View experiment details
mlcli> experiment_info my_exp

# Run the current experiment
mlcli> run_experiment

# Run a specific experiment
mlcli> run_experiment my_exp

# Compare multiple experiments
mlcli> compare_experiments exp1 exp2 exp3

# Export experiment results
mlcli> export_experiment my_exp results.json

# Delete an experiment
mlcli> delete_experiment old_exp
```

#### Multi-Model, Dataset, and Config Support

Register multiple models, datasets, and configurations for grid experiments:

```
# Register multiple models
mlcli> add_model resnet_small resnet18 --classes 10
mlcli> add_model resnet_large resnet50 --classes 10
mlcli> add_model vit_model vit_base --classes 10
mlcli> list_models

# Register multiple datasets
mlcli> add_dataset train_v1 ./data/train_v1
mlcli> add_dataset train_v2 ./data/train_v2 --type folder
mlcli> list_datasets

# Register multiple training configurations
mlcli> add_config fast --epochs 10 --lr 0.01 --batch-size 64
mlcli> add_config slow --epochs 100 --lr 0.0001 --batch-size 16
mlcli> add_config medium --epochs 50 --lr 0.001 --batch-size 32
mlcli> list_configs

# Create experiment and add models/datasets/configs
mlcli> experiment grid_search
mlcli> experiment_add_models resnet_small resnet_large vit_model
mlcli> experiment_add_datasets train_v1 train_v2
mlcli> experiment_add_configs fast slow medium

# Or add all registered items
mlcli> experiment_add_models --all
mlcli> experiment_add_datasets --all
mlcli> experiment_add_configs --all

# Run grid experiment (all combinations)
mlcli> run_grid_experiment

# View detailed results (includes Accuracy, Precision, Recall, F1-Score)
mlcli> experiment_results

# Remove items when no longer needed
mlcli> remove_model resnet_small
mlcli> remove_dataset train_v1
mlcli> remove_config fast
```

#### YAML-Based Grid Experiments

Create experiments with multiple models, datasets, and configs from a single YAML file:

```yaml
# configs/experiment_grid.yaml
name: grid_search_experiment
description: "Compare multiple models and configurations"
seed: 42

# Define multiple models
models:
  - name: resnet_small
    architecture: resnet18
    num_classes: 10
    pretrained: true
  
  - name: resnet_large
    architecture: resnet50
    num_classes: 10
    pretrained: true
  
  - name: vit_model
    architecture: vit_base_patch16
    num_classes: 10
    pretrained: true

# Define multiple datasets
datasets:
  - name: train_v1
    path: ./data/train_v1
    type: folder
  
  - name: train_v2
    path: ./data/train_v2
    type: folder

# Define multiple training configurations
configs:
  - name: fast
    epochs: 10
    learning_rate: 0.01
    batch_size: 64
    optimizer: adamw
  
  - name: standard
    epochs: 50
    learning_rate: 0.001
    batch_size: 32
    mixed_precision: true
  
  - name: slow_finetune
    epochs: 100
    learning_rate: 0.0001
    batch_size: 16
    weight_decay: 0.01
```

Load and run the grid experiment in interactive mode:

```
# Load experiment from YAML (automatically loads all models, datasets, configs)
mlcli> experiment my_grid --config configs/experiment_grid.yaml

# View what was loaded
mlcli> experiment_info

# Run all combinations (3 models × 2 datasets × 3 configs = 18 runs)
mlcli> run_grid_experiment

# View comprehensive results with Precision, Recall, F1-Score
mlcli> experiment_results

# Export results to file
mlcli> export_experiment my_grid results.json
```

### Other Commands

```bash
# Show system info and available models
mlcli info

# Generate configuration template
mlcli init --format yaml --task classification > config.yaml

# Describe a model architecture
mlcli describe resnet50
```

## Configuration

You can use YAML, JSON, or TOML configuration files:

```yaml
experiment:
  name: my_experiment
  seed: 42
  device: auto
  output_dir: ./outputs
  reproducible: true

model:
  task_type: classification
  model_type: cnn
  architecture: resnet50
  num_classes: 10
  pretrained: true

training:
  epochs: 100
  batch_size: 32
  learning_rate: 0.001
  optimizer: adamw
  scheduler: cosine
  mixed_precision: true

dataset:
  train_path: ./data/train
  val_path: ./data/val
  num_workers: 4

logging:
  console: true
  tensorboard: true
```

## Python API

```python
from mlcli.core.config import ModelConfig, TrainingConfig
from mlcli.core.factory import ModelFactory
from mlcli.training.trainer import Trainer
from mlcli.data.classification import ImageFolderDataset

# Create model
model_config = ModelConfig(
    task_type="classification",
    architecture="resnet50",
    num_classes=10,
    pretrained=True,
)
model = ModelFactory.create(model_config)

# Load dataset
dataset = ImageFolderDataset(root="./data/train")
train_loader = dataset.create_dataloader(batch_size=32, shuffle=True)

# Create trainer
trainer = Trainer(
    model=model,
    optimizer=torch.optim.AdamW(model.parameters(), lr=0.001),
    device="cuda",
    mixed_precision=True,
)

# Train
trainer.fit(train_loader, epochs=100)
```

## Plugin System

Create custom plugins to extend MLCLI:

```python
from mlcli.plugins.base import ModelPlugin
from mlcli.models.base import BaseModel

class MyCustomModel(BaseModel):
    def __init__(self, num_classes: int):
        super().__init__()
        # Define your model
    
    def forward(self, x):
        # Forward pass
        pass

class MyPlugin(ModelPlugin):
    name = "my_custom_plugin"
    version = "1.0.0"
    
    def get_models(self):
        return {"my_custom_model": MyCustomModel}

# Register plugin
from mlcli.plugins import load_plugin
load_plugin(MyPlugin)
```

## Project Structure

```
ml-utils/
├── src/
│   └── mlcli/
│       ├── __init__.py
│       ├── cli/              # CLI commands
│       │   ├── main.py
│       │   ├── train.py
│       │   ├── evaluate.py
│       │   └── inference.py
│       ├── core/             # Core infrastructure
│       │   ├── config.py
│       │   ├── registry.py
│       │   └── factory.py
│       ├── models/           # Model definitions
│       │   ├── base.py
│       │   ├── classification/
│       │   └── detection/
│       ├── data/             # Dataset handling
│       │   ├── base.py
│       │   ├── transforms.py
│       │   ├── classification.py
│       │   └── detection.py
│       ├── tasks/            # Task abstractions
│       │   ├── base.py
│       │   ├── classification.py
│       │   └── detection.py
│       ├── training/         # Training infrastructure
│       │   ├── trainer.py
│       │   ├── optimizer.py
│       │   ├── checkpoint.py
│       │   └── callbacks.py
│       ├── logging/          # Logging and metrics
│       │   ├── logger.py
│       │   ├── metrics.py
│       │   └── tensorboard.py
│       ├── plugins/          # Plugin system
│       │   ├── base.py
│       │   └── loader.py
│       └── utils/            # Utilities
│           ├── reproducibility.py
│           ├── device.py
│           └── io.py
├── tests/
├── pyproject.toml
└── README.md
```

## Requirements

- Python 3.9+
- PyTorch 2.0+
- torchvision
- timm
- click
- pydantic
- tqdm
- numpy
- Pillow

Optional:
- ultralytics (for YOLO)
- transformers (for DETR)
- tensorboard
- wandb

## License

MIT License

## Contributing

Contributions are welcome! Please read our contributing guidelines and submit a pull request.

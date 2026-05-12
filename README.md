<div align="center">

# MLCLI

**A production-grade Python CLI for training, evaluating, and running inference on computer vision models.**

[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

[Features](#features) · [Installation](#installation) · [Quick Start](#quick-start) · [Interactive Mode](#interactive-mode) · [Grid Experiments](#grid-experiments) · [API](#python-api) · [Contributing](#contributing)

</div>

---

## Overview

MLCLI is a modular toolkit for computer vision that covers the full ML lifecycle — from training and evaluation to grid search experiments and production inference. It wraps PyTorch with a clean CLI and Python API, supports 20+ model architectures out of the box, and includes a built-in interactive shell for exploratory work.

```bash
# Train a classifier in one line
mlcli train --model resnet50 --dataset ./data/train --epochs 100

# Or explore interactively
mlcli interactive
```

---

## Features

### Model Support

| Family | Task | Architectures |
|---|---|---|
| CNN | Classification | ResNet (18/34/50/101/152), EfficientNet (B0–B7), MobileNet (V2/V3) |
| Transformer | Classification | ViT (Base/Large), Swin (Tiny/Small/Base/Large) |
| — | Detection | YOLOv5/v8, Faster R-CNN, SSD (300/512), DETR |

### Training

- Mixed precision (FP16) with automatic gradient scaling
- Gradient accumulation and clipping
- Optimizers: Adam, AdamW, SGD
- Schedulers: Step, Cosine, Plateau, Warmup+Cosine
- Early stopping with best-weight restoration
- **Fail-safe checkpointing** — saves emergency checkpoint on crash; resume from any point with `--resume` or `--auto-resume`

### Experiments

- Interactive REPL for exploratory work (`mlcli interactive`)
- Grid search over models × datasets × configs — from CLI or YAML
- Per-combination state persistence — interrupted grid runs resume exactly where they left off
- TensorBoard and Weights & Biases logging

### Architecture

- Plugin system for custom models, datasets, and tasks
- Registry + Factory patterns for zero-boilerplate extension
- Pydantic-validated configs with YAML/JSON serialization
- Reproducibility helpers (seed, deterministic mode)

---

## Installation

**Requirements:** Python 3.9+, PyTorch 2.0+

```bash
git clone https://github.com/your-org/mlcli.git
cd mlcli

# Standard install
pip install -e .

# With transformer models (ViT, Swin, DETR)
pip install -e ".[transformers]"

# Full install including dev tools
pip install -e ".[all]"
```

---

## Quick Start

### Train

```bash
# Classification
mlcli train --model resnet50 --dataset ./data/train --epochs 100

# With validation, mixed precision, and cosine scheduler
mlcli train \
  --model vit_base \
  --dataset ./data/train \
  --val-dataset ./data/val \
  --epochs 100 --lr 0.0001 \
  --optimizer adamw --scheduler cosine \
  --mixed-precision

# Object detection
mlcli train --model yolo --dataset ./coco --task detection --epochs 50

# Resume after a crash (auto-finds latest checkpoint)
mlcli train --model resnet50 --dataset ./data/train --auto-resume

# Resume from a specific checkpoint
mlcli train --model resnet50 --dataset ./data/train --resume ./outputs/checkpoints/epoch_042.pt
```

### Evaluate

```bash
mlcli evaluate --checkpoint ./model.pt --dataset ./data/test

# Per-class metrics + confusion matrix
mlcli evaluate -c ./model.pt -d ./data/test --per-class --confusion-matrix
```

### Inference

```bash
# Single image
mlcli inference --model ./model.pt --input ./image.jpg

# Batch with saved visualizations
mlcli inference -m ./model.pt -i ./images/ -o ./predictions --save-visualization
```

---

## Fail-Safe Training

MLCLI saves an emergency checkpoint automatically if training crashes (OOM, code error, etc.):

```
outputs/my_experiment/
├── checkpoints/
│   ├── epoch_010.pt          # regular checkpoint
│   ├── emergency_epoch042.pt # saved on crash
│   └── best_model.pt
└── failure_info.json         # {epoch, error, timestamp, checkpoint_path}
```

To continue after fixing the issue:

```bash
# Auto-detect and resume from latest checkpoint
mlcli train --model resnet50 --dataset ./data/train --auto-resume

# Or point to the emergency checkpoint explicitly
mlcli train --model resnet50 --dataset ./data/train \
  --resume ./outputs/my_experiment/checkpoints/emergency_epoch042.pt
```

The checkpoint includes full training state: model weights, optimizer, scheduler, and gradient scaler — so training continues exactly where it left off.

---

## Interactive Mode

Launch a REPL for exploratory work:

```bash
mlcli interactive
```

```
mlcli> load_model resnet50 --classes 10
mlcli> load_dataset ./data/train
mlcli> train --epochs 10 --lr 0.001
mlcli> predict ./image.jpg
mlcli> benchmark --iters 100
mlcli> export ./model.onnx --format onnx
mlcli> help
```

### Experiment Management

```
mlcli> experiment my_exp --model resnet50 --epochs 50
mlcli> run_experiment
mlcli> experiments                    # list all
mlcli> compare_experiments exp1 exp2
mlcli> export_experiment my_exp results.json
```

Resume an interrupted experiment — already-completed combinations are skipped automatically:

```
mlcli> run_grid_experiment my_grid --resume
# Resumed: 6/18 combos already done, running remaining 12
```

---

## Grid Experiments

Run all combinations of models × datasets × configs from a single YAML:

```yaml
# experiment_configs/my_grid.yaml
name: grid_search
seed: 42

models:
  - name: resnet_small
    architecture: resnet18
    num_classes: 10
    pretrained: true
  - name: resnet_large
    architecture: resnet50
    num_classes: 10
    pretrained: true

datasets:
  - name: train_v1
    path: ./data/train_v1
    type: folder

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
```

```
mlcli> experiment my_grid --config experiment_configs/my_grid.yaml
mlcli> run_grid_experiment        # runs 2 models × 1 dataset × 2 configs = 4 runs
mlcli> experiment_results         # accuracy, precision, recall, F1 per run
```

State is saved after each combination to `outputs/my_grid/experiment_state.json`. If the session is interrupted, `run_grid_experiment my_grid --resume` skips already-completed runs.

---

## Python API

```python
from mlcli.core.config import ModelConfig, TrainingConfig, TaskType, ModelFamily
from mlcli.core.factory import ModelFactory
from mlcli.training.trainer import Trainer, train_from_config

# --- Quick path: train from a config object ---
from mlcli.core.config import ExperimentConfig, DatasetConfig
from pathlib import Path

config = ExperimentConfig(
    model=ModelConfig(
        task_type=TaskType.CLASSIFICATION,
        family=ModelFamily.CNN,
        architecture="resnet50",
        num_classes=10,
    ),
    dataset=DatasetConfig(name="my_data", root_path=Path("./data")),
)
results = train_from_config(config)

# --- Or build the Trainer directly ---
model_config = ModelConfig(
    task_type=TaskType.CLASSIFICATION,
    family=ModelFamily.CNN,
    architecture="resnet50",
    num_classes=10,
    pretrained=True,
)
model = ModelFactory.create(model_config)
trainer = Trainer(config)
results = trainer.train()
# results: {epochs_trained, best_metric, total_time, history}
```

---

## Plugin System

Register custom models without modifying the core:

```python
from mlcli.plugins.base import ModelPlugin
from mlcli.models.base import BaseModel
from mlcli.core.registry import ModelRegistry

class MyModel(BaseModel):
    def __init__(self, num_classes: int):
        super().__init__()
        self.head = torch.nn.Linear(512, num_classes)

    def forward(self, x):
        return self.head(x)

class MyPlugin(ModelPlugin):
    name = "my_plugin"
    version = "1.0.0"

    def get_models(self):
        return {"my_model": MyModel}

from mlcli.plugins.loader import load_plugin
load_plugin(MyPlugin)

# Now available everywhere:
# mlcli train --model my_model --dataset ./data
```

---

## Project Structure

```
mlcli/
├── src/mlcli/
│   ├── cli/          # click commands: train, evaluate, inference, interactive
│   ├── core/         # config (Pydantic), registry, factory
│   ├── models/       # classification/ and detection/ model definitions
│   ├── data/         # datasets and transforms
│   ├── tasks/        # task abstractions (classification, detection)
│   ├── training/     # Trainer, CheckpointManager, callbacks, optimizer
│   ├── logging/      # ExperimentLogger, metrics, TensorBoard writer
│   ├── plugins/      # plugin base class and loader
│   └── utils/        # device, reproducibility, I/O helpers
├── experiment_configs/   # example YAML grid configs
├── tests/
└── pyproject.toml
```

---

## Contributing

Contributions are welcome — bug fixes, new model support, documentation improvements, or new features. See [CONTRIBUTING.md](CONTRIBUTING.md) for the full guide.

**Setup:**
```bash
git clone https://github.com/nanmanat/ml-utils.git
cd ml-utils
pip install -e ".[dev]"
pre-commit install --hook-type commit-msg
```

**Workflow:**
1. Create a branch from `main`: `git checkout -b feat/my-feature`
2. Make changes and add tests where applicable
3. Commit using [Conventional Commits](https://www.conventionalcommits.org/) — enforced on PRs:
   ```
   feat(training): add cosine annealing with warm restarts
   fix(checkpoint): prevent emergency save from overwriting best_model.pt
   docs: update grid search example
   ```
4. Open a pull request — CI runs lint + tests automatically; direct pushes to `main` are blocked
5. On merge to `main`, `python-semantic-release` bumps the version and publishes a release automatically

For larger changes, open an issue first to discuss the approach.

---

## License

MIT — see [LICENSE](LICENSE) for details.

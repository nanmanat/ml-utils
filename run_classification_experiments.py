"""
Runner script for all-classification-models experiment.

Reads experiment_configs/all_classification_models_default.yaml and executes
mlcli train for each model with the configured hyperparameters.

Usage:
    python run_classification_experiments.py
    python run_classification_experiments.py --dry-run
    python run_classification_experiments.py --config path/to/config.yaml
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import yaml


def load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_command(model: dict, dataset: dict, config: dict) -> list[str]:
    cmd = [
        sys.executable, "-m", "mlcli",
        "train",
        "--model", model["architecture"],
        "--dataset", str(dataset["path"]),
        "--task", "classification",
        "--epochs", str(config["epochs"]),
        "--batch-size", str(config["batch_size"]),
        "--learning-rate", str(config["learning_rate"]),
        "--optimizer", config["optimizer"],
        "--scheduler", config["scheduler"],
        "--weight-decay", str(config["weight_decay"]),
        "--num-classes", str(model["num_classes"]),
        "--experiment-name", model["name"],
        "--num-workers", "4",
    ]
    if model.get("pretrained", True):
        cmd.append("--pretrained")
    else:
        cmd.append("--no-pretrained")
    if config.get("mixed_precision", True):
        cmd.append("--mixed-precision")
    else:
        cmd.append("--no-mixed-precision")
    if config.get("gradient_clip") is not None:
        cmd.extend(["--gradient-clip", str(config["gradient_clip"])])
    if config.get("early_stopping_patience") is not None:
        cmd.extend(["--early-stopping", str(config["early_stopping_patience"])])
    return cmd


def run_experiments(config_path: Path, dry_run: bool = False) -> None:
    cfg = load_config(config_path)
    models = cfg["models"]
    dataset = cfg["datasets"][0]
    train_cfg = cfg["configs"][0]

    print(f"\nExperiment: {cfg['name']}")
    print(f"Description: {cfg.get('description', '')}")
    print(f"Models: {len(models)}")
    print(f"Dataset: {dataset['path']}")
    print(f"Training config: {train_cfg['name']}")
    print("=" * 60)

    results: list[dict] = []

    for i, model in enumerate(models, 1):
        cmd = build_command(model, dataset, train_cfg)
        cmd_str = " ".join(cmd)

        print(f"\n[{i}/{len(models)}] Model: {model['name']}")
        print(f"  Command: {cmd_str}")

        if dry_run:
            results.append({"model": model["name"], "status": "dry-run", "exit_code": None, "duration": None})
            continue

        start = time.time()
        try:
            result = subprocess.run(cmd, capture_output=False, text=True)
            duration = time.time() - start
            status = "success" if result.returncode == 0 else "failed"
            results.append({
                "model": model["name"],
                "status": status,
                "exit_code": result.returncode,
                "duration": round(duration, 1),
            })
            print(f"  Status: {status} (exit {result.returncode}, {duration:.1f}s)")
        except Exception as e:
            duration = time.time() - start
            results.append({
                "model": model["name"],
                "status": "error",
                "exit_code": None,
                "duration": round(duration, 1),
            })
            print(f"  Error: {e}")

    # Summary table
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Model':<30} {'Status':<12} {'Exit':<6} {'Duration'}")
    print("-" * 60)
    for r in results:
        duration = f"{r['duration']}s" if r["duration"] is not None else "-"
        exit_code = str(r["exit_code"]) if r["exit_code"] is not None else "-"
        print(f"{r['model']:<30} {r['status']:<12} {exit_code:<6} {duration}")

    if not dry_run:
        success = sum(1 for r in results if r["status"] == "success")
        failed = sum(1 for r in results if r["status"] != "success")
        print(f"\nCompleted: {success}/{len(results)} successful, {failed} failed")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all classification model experiments")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("experiment_configs/all_classification_models_default.yaml"),
        help="Path to experiment config YAML",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them",
    )
    args = parser.parse_args()

    if not args.config.exists():
        print(f"Error: config not found: {args.config}", file=sys.stderr)
        sys.exit(1)

    run_experiments(args.config, dry_run=args.dry_run)


if __name__ == "__main__":
    main()

"""
Inference command for mlcli.

Provides inference/prediction functionality for trained models.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, List

import click

from mlcli.cli.main import Context, pass_context


@click.command()
@click.option(
    "--model", "-m",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to trained model checkpoint.",
)
@click.option(
    "--input", "-i",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Input image or directory of images.",
)
@click.option(
    "--output", "-o",
    type=click.Path(path_type=Path),
    default=None,
    help="Output directory for predictions.",
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
    default=1,
    help="Inference batch size.",
)
@click.option(
    "--threshold",
    type=float,
    default=0.5,
    help="Confidence threshold for predictions.",
)
@click.option(
    "--top-k",
    type=int,
    default=5,
    help="Number of top predictions to show (classification).",
)
@click.option(
    "--save-visualization/--no-save-visualization",
    default=False,
    help="Save visualization images (detection).",
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["json", "csv", "txt"]),
    default="json",
    help="Output format for predictions.",
)
@click.option(
    "--class-names",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to class names file (one per line).",
)
@pass_context
def inference(
    ctx: Context,
    model: Path,
    input: Path,
    output: Optional[Path],
    task: Optional[str],
    batch_size: int,
    threshold: float,
    top_k: int,
    save_visualization: bool,
    output_format: str,
    class_names: Optional[Path],
) -> None:
    """
    Run inference with a trained model.
    
    \b
    Examples:
      mlcli inference --model ./model.pt --input ./image.jpg
      mlcli inference -m ./model.pt -i ./images/ -o ./predictions
      mlcli inference -m ./detector.pt -i ./image.jpg --save-visualization
    """
    import json
    import torch
    from PIL import Image
    import torchvision.transforms as T
    
    from mlcli.core.factory import ModelFactory
    from mlcli.core.config import ModelConfig
    
    click.echo("\n" + "=" * 60)
    click.echo("MLCLI Inference")
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
    click.echo(f"Model: {model}")
    click.echo(f"Input: {input}")
    
    # Load checkpoint
    click.echo("\nLoading model...")
    ckpt = torch.load(model, map_location=device)
    
    # Get model config
    if "model_config" in ckpt:
        model_config = ModelConfig(**ckpt["model_config"])
        task = task or model_config.task_type
        num_classes = model_config.num_classes
    else:
        if task is None:
            click.echo("Error: Task type not found in checkpoint. Please specify --task.", err=True)
            sys.exit(1)
        model_config = None
        num_classes = None
    
    click.echo(f"Task: {task}")
    
    # Load class names
    class_labels = None
    if class_names:
        class_labels = class_names.read_text().strip().split("\n")
        click.echo(f"Loaded {len(class_labels)} class names")
    
    # Create model
    if model_config:
        model_instance = ModelFactory.create(model_config)
        model_instance.load_state_dict(ckpt["model_state_dict"])
    else:
        raise NotImplementedError("Please provide a checkpoint with model config.")
    
    model_instance = model_instance.to(device)
    model_instance.eval()
    
    # Collect input images
    image_paths: List[Path] = []
    
    if input.is_file():
        image_paths = [input]
    elif input.is_dir():
        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
        image_paths = [
            p for p in input.iterdir()
            if p.suffix.lower() in extensions
        ]
        image_paths.sort()
    
    click.echo(f"\nFound {len(image_paths)} image(s)")
    
    if not image_paths:
        click.echo("No images found.", err=True)
        sys.exit(1)
    
    # Set up transforms
    if task == "classification":
        transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
    else:
        transform = T.Compose([
            T.ToTensor(),
        ])
    
    # Run inference
    click.echo("\n" + "-" * 60)
    click.echo("Running inference...")
    click.echo("-" * 60)
    
    all_predictions = []
    
    with torch.no_grad():
        for i, image_path in enumerate(image_paths):
            # Load and preprocess image
            image = Image.open(image_path).convert("RGB")
            original_size = image.size
            
            input_tensor = transform(image).unsqueeze(0).to(device)
            
            # Run inference
            if task == "classification":
                outputs = model_instance(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                
                # Get top-k predictions
                top_probs, top_indices = torch.topk(probabilities, min(top_k, probabilities.size(1)))
                top_probs = top_probs[0].cpu().tolist()
                top_indices = top_indices[0].cpu().tolist()
                
                predictions = []
                for prob, idx in zip(top_probs, top_indices):
                    if prob >= threshold:
                        label = class_labels[idx] if class_labels else f"class_{idx}"
                        predictions.append({
                            "class_id": idx,
                            "class_name": label,
                            "confidence": prob,
                        })
                
                result = {
                    "file": str(image_path),
                    "predictions": predictions,
                }
                
                # Print result
                click.echo(f"\n{image_path.name}:")
                for pred in predictions:
                    click.echo(f"  {pred['class_name']}: {pred['confidence']:.4f}")
            
            else:  # detection
                outputs = model_instance([input_tensor[0]])
                
                if len(outputs) > 0:
                    output = outputs[0]
                    boxes = output.get("boxes", torch.tensor([]))
                    scores = output.get("scores", torch.tensor([]))
                    labels = output.get("labels", torch.tensor([]))
                    
                    # Filter by threshold
                    mask = scores >= threshold
                    boxes = boxes[mask].cpu().tolist()
                    scores = scores[mask].cpu().tolist()
                    labels = labels[mask].cpu().tolist()
                    
                    detections = []
                    for box, score, label in zip(boxes, scores, labels):
                        label_name = class_labels[label] if class_labels and label < len(class_labels) else f"class_{label}"
                        detections.append({
                            "class_id": label,
                            "class_name": label_name,
                            "confidence": score,
                            "bbox": box,  # [x1, y1, x2, y2]
                        })
                    
                    result = {
                        "file": str(image_path),
                        "image_size": list(original_size),
                        "detections": detections,
                    }
                    
                    # Print result
                    click.echo(f"\n{image_path.name}: {len(detections)} detection(s)")
                    for det in detections[:5]:  # Show first 5
                        click.echo(f"  {det['class_name']}: {det['confidence']:.4f} @ {det['bbox']}")
                    if len(detections) > 5:
                        click.echo(f"  ... and {len(detections) - 5} more")
                    
                    # Save visualization
                    if save_visualization and output:
                        _save_detection_visualization(
                            image,
                            detections,
                            output / f"{image_path.stem}_pred.jpg",
                        )
                else:
                    result = {
                        "file": str(image_path),
                        "image_size": list(original_size),
                        "detections": [],
                    }
                    click.echo(f"\n{image_path.name}: No detections")
            
            all_predictions.append(result)
            
            if ctx.verbose and (i + 1) % 10 == 0:
                click.echo(f"Processed {i + 1}/{len(image_paths)} images")
    
    # Save predictions
    if output:
        output.mkdir(parents=True, exist_ok=True)
        
        if output_format == "json":
            output_file = output / "predictions.json"
            with open(output_file, "w") as f:
                json.dump(all_predictions, f, indent=2)
        
        elif output_format == "csv":
            import csv
            output_file = output / "predictions.csv"
            
            with open(output_file, "w", newline="") as f:
                if task == "classification":
                    writer = csv.writer(f)
                    writer.writerow(["file", "class_id", "class_name", "confidence"])
                    for pred in all_predictions:
                        for p in pred["predictions"]:
                            writer.writerow([
                                pred["file"],
                                p["class_id"],
                                p["class_name"],
                                f"{p['confidence']:.6f}",
                            ])
                else:
                    writer = csv.writer(f)
                    writer.writerow(["file", "class_id", "class_name", "confidence", "x1", "y1", "x2", "y2"])
                    for pred in all_predictions:
                        for d in pred.get("detections", []):
                            bbox = d["bbox"]
                            writer.writerow([
                                pred["file"],
                                d["class_id"],
                                d["class_name"],
                                f"{d['confidence']:.6f}",
                                *bbox,
                            ])
        
        elif output_format == "txt":
            output_file = output / "predictions.txt"
            
            with open(output_file, "w") as f:
                for pred in all_predictions:
                    f.write(f"File: {pred['file']}\n")
                    
                    if task == "classification":
                        for p in pred["predictions"]:
                            f.write(f"  {p['class_name']}: {p['confidence']:.4f}\n")
                    else:
                        for d in pred.get("detections", []):
                            f.write(f"  {d['class_name']}: {d['confidence']:.4f} @ {d['bbox']}\n")
                    
                    f.write("\n")
        
        click.echo(f"\nPredictions saved to: {output_file}")
    
    click.echo("\n" + "=" * 60)
    click.echo(f"Inference completed: {len(image_paths)} image(s) processed")
    click.echo("=" * 60 + "\n")


def _save_detection_visualization(
    image: "Image.Image",
    detections: List[dict],
    output_path: Path,
) -> None:
    """Save detection visualization with bounding boxes."""
    from PIL import ImageDraw, ImageFont
    
    draw = ImageDraw.Draw(image)
    
    # Try to get a font, fall back to default
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 12)
    except Exception:
        font = ImageFont.load_default()
    
    # Color palette
    colors = [
        "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4",
        "#FFEAA7", "#DDA0DD", "#98D8C8", "#F7DC6F",
    ]
    
    for i, det in enumerate(detections):
        color = colors[det["class_id"] % len(colors)]
        bbox = det["bbox"]
        label = f"{det['class_name']}: {det['confidence']:.2f}"
        
        # Draw bounding box
        draw.rectangle(bbox, outline=color, width=2)
        
        # Draw label background
        text_bbox = draw.textbbox((bbox[0], bbox[1] - 15), label, font=font)
        draw.rectangle(text_bbox, fill=color)
        draw.text((bbox[0], bbox[1] - 15), label, fill="white", font=font)
    
    # Save image
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)

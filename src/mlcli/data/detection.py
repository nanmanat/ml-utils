"""
Detection datasets.

Provides dataset implementations for object detection tasks.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from PIL import Image
from torch.utils.data import Dataset

from mlcli.core.registry import register_dataset
from mlcli.data.base import BaseDataset, DatasetInfo


@register_dataset("detection", aliases=["det"])
class DetectionDataset(BaseDataset):
    """
    Base detection dataset.
    
    Expects annotations in a standard format with image paths
    and bounding box annotations.
    """
    
    EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
    
    def __init__(
        self,
        root: Union[str, Path],
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        annotation_file: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize detection dataset.
        
        Args:
            root: Root directory containing images.
            split: Dataset split.
            transform: Transform for images.
            target_transform: Transform for targets.
            annotation_file: Path to annotation file.
        """
        self.annotation_file = annotation_file
        self.images: List[Dict[str, Any]] = []
        self.annotations: Dict[int, List[Dict[str, Any]]] = {}
        self.classes: List[str] = []
        self.class_to_idx: Dict[str, int] = {}
        
        super().__init__(root, split, transform, target_transform, **kwargs)
    
    def _load_dataset(self) -> None:
        """Load dataset from annotation file."""
        if self.annotation_file is None:
            # Try to find annotation file
            possible_files = [
                self.root / f"{self.split}.json",
                self.root / "annotations" / f"{self.split}.json",
                self.root / "annotations.json",
            ]
            for f in possible_files:
                if f.exists():
                    self.annotation_file = str(f)
                    break
        
        if self.annotation_file and Path(self.annotation_file).exists():
            self._load_json_annotations()
        else:
            # Fallback to VOC-style annotations
            self._load_voc_annotations()
    
    def _load_json_annotations(self) -> None:
        """Load annotations from JSON file."""
        with open(self.annotation_file) as f:
            data = json.load(f)
        
        # Handle different annotation formats
        if "images" in data and "annotations" in data:
            # COCO-like format
            self._parse_coco_format(data)
        elif "annotations" in data:
            # Simple format
            self._parse_simple_format(data)
    
    def _parse_coco_format(self, data: Dict[str, Any]) -> None:
        """Parse COCO-format annotations."""
        self.images = data["images"]
        
        # Build category mapping
        if "categories" in data:
            categories = {cat["id"]: cat["name"] for cat in data["categories"]}
            self.classes = list(categories.values())
            self.class_to_idx = {name: idx for idx, name in enumerate(self.classes)}
        
        # Group annotations by image
        for ann in data["annotations"]:
            img_id = ann["image_id"]
            if img_id not in self.annotations:
                self.annotations[img_id] = []
            self.annotations[img_id].append(ann)
    
    def _parse_simple_format(self, data: Dict[str, Any]) -> None:
        """Parse simple annotation format."""
        for item in data["annotations"]:
            img_info = {
                "id": len(self.images),
                "file_name": item["image"],
            }
            self.images.append(img_info)
            
            self.annotations[img_info["id"]] = item.get("boxes", [])
    
    def _load_voc_annotations(self) -> None:
        """Load VOC-style XML annotations."""
        # Placeholder for VOC format support
        pass
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Get sample by index."""
        img_info = self.images[index]
        img_id = img_info["id"]
        
        # Load image
        img_path = self.root / "images" / img_info["file_name"]
        if not img_path.exists():
            img_path = self.root / img_info["file_name"]
        
        image = Image.open(img_path).convert("RGB")
        
        # Get annotations
        anns = self.annotations.get(img_id, [])
        
        boxes = []
        labels = []
        areas = []
        iscrowd = []
        
        for ann in anns:
            if "bbox" in ann:
                # COCO format: [x, y, width, height]
                x, y, w, h = ann["bbox"]
                boxes.append([x, y, x + w, y + h])
                labels.append(ann.get("category_id", 0))
                areas.append(ann.get("area", w * h))
                iscrowd.append(ann.get("iscrowd", 0))
            elif "box" in ann:
                # Simple format: [x1, y1, x2, y2]
                boxes.append(ann["box"])
                labels.append(ann.get("label", 0))
        
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4)),
            "labels": torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64),
            "image_id": torch.tensor([img_id]),
            "area": torch.tensor(areas, dtype=torch.float32) if areas else torch.zeros((0,)),
            "iscrowd": torch.tensor(iscrowd, dtype=torch.int64) if iscrowd else torch.zeros((0,), dtype=torch.int64),
        }
        
        if self.transform is not None:
            image = self.transform(image)
        
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return image, target
    
    @property
    def num_classes(self) -> int:
        return len(self.classes) if self.classes else 0
    
    @property
    def class_names(self) -> List[str]:
        return self.classes


@register_dataset("coco", aliases=["coco2017"])
class COCODataset(DetectionDataset):
    """
    COCO detection dataset.
    
    Wrapper for the COCO dataset with standard splits.
    """
    
    def __init__(
        self,
        root: Union[str, Path],
        split: str = "train",
        year: str = "2017",
        transform: Optional[Callable] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize COCO dataset.
        
        Args:
            root: Root directory containing COCO data.
            split: Dataset split (train, val).
            year: COCO version year.
            transform: Image transform.
        """
        self.year = year
        
        # Set up paths
        root = Path(root)
        annotation_file = root / "annotations" / f"instances_{split}{year}.json"
        
        super().__init__(
            root=root,
            split=split,
            transform=transform,
            annotation_file=str(annotation_file) if annotation_file.exists() else None,
            **kwargs,
        )
    
    def _load_dataset(self) -> None:
        """Load COCO dataset."""
        # Try to use pycocotools if available
        try:
            from pycocotools.coco import COCO
            
            ann_file = self.root / "annotations" / f"instances_{self.split}{self.year}.json"
            if ann_file.exists():
                self.coco = COCO(str(ann_file))
                self.images = [
                    {"id": img_id, **self.coco.imgs[img_id]}
                    for img_id in self.coco.getImgIds()
                ]
                
                # Get categories
                cats = self.coco.loadCats(self.coco.getCatIds())
                self.classes = [cat["name"] for cat in cats]
                self.class_to_idx = {cat["name"]: cat["id"] for cat in cats}
                
                # Load annotations
                for img_info in self.images:
                    img_id = img_info["id"]
                    ann_ids = self.coco.getAnnIds(imgIds=img_id)
                    self.annotations[img_id] = self.coco.loadAnns(ann_ids)
            else:
                super()._load_dataset()
        except ImportError:
            super()._load_dataset()
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Get sample by index."""
        img_info = self.images[index]
        img_id = img_info["id"]
        
        # Load image
        img_path = self.root / f"{self.split}{self.year}" / img_info["file_name"]
        if not img_path.exists():
            img_path = self.root / "images" / f"{self.split}{self.year}" / img_info["file_name"]
        
        image = Image.open(img_path).convert("RGB")
        orig_size = image.size
        
        # Get annotations
        anns = self.annotations.get(img_id, [])
        
        boxes = []
        labels = []
        areas = []
        iscrowd = []
        
        for ann in anns:
            if ann.get("iscrowd", 0):
                continue
            
            x, y, w, h = ann["bbox"]
            if w <= 0 or h <= 0:
                continue
            
            boxes.append([x, y, x + w, y + h])
            labels.append(ann["category_id"])
            areas.append(ann.get("area", w * h))
            iscrowd.append(ann.get("iscrowd", 0))
        
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4)),
            "labels": torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64),
            "image_id": torch.tensor([img_id]),
            "area": torch.tensor(areas, dtype=torch.float32) if areas else torch.zeros((0,)),
            "iscrowd": torch.tensor(iscrowd, dtype=torch.int64) if iscrowd else torch.zeros((0,), dtype=torch.int64),
            "orig_size": torch.tensor(orig_size),
        }
        
        if self.transform is not None:
            image = self.transform(image)
        
        return image, target
    
    @property
    def num_classes(self) -> int:
        return 80  # COCO has 80 object categories


@register_dataset("voc", aliases=["pascalvoc", "voc2012"])
class VOCDataset(DetectionDataset):
    """Pascal VOC detection dataset."""
    
    VOC_CLASSES = [
        "aeroplane", "bicycle", "bird", "boat", "bottle",
        "bus", "car", "cat", "chair", "cow",
        "diningtable", "dog", "horse", "motorbike", "person",
        "pottedplant", "sheep", "sofa", "train", "tvmonitor",
    ]
    
    def __init__(
        self,
        root: Union[str, Path],
        split: str = "train",
        year: str = "2012",
        transform: Optional[Callable] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize VOC dataset."""
        self.year = year
        self.classes = self.VOC_CLASSES
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        super().__init__(root, split, transform, **kwargs)
    
    def _load_dataset(self) -> None:
        """Load VOC dataset."""
        try:
            from torchvision.datasets import VOCDetection
            
            self._voc = VOCDetection(
                root=str(self.root),
                year=self.year,
                image_set=self.split,
                download=False,
            )
            
            # Build image list
            for i in range(len(self._voc)):
                self.images.append({"id": i, "index": i})
        except Exception:
            super()._load_dataset()
    
    def __len__(self) -> int:
        if hasattr(self, "_voc"):
            return len(self._voc)
        return len(self.images)
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Get sample by index."""
        if hasattr(self, "_voc"):
            image, target_dict = self._voc[index]
            
            boxes = []
            labels = []
            
            objects = target_dict["annotation"].get("object", [])
            if not isinstance(objects, list):
                objects = [objects]
            
            for obj in objects:
                bbox = obj["bndbox"]
                x1 = float(bbox["xmin"])
                y1 = float(bbox["ymin"])
                x2 = float(bbox["xmax"])
                y2 = float(bbox["ymax"])
                boxes.append([x1, y1, x2, y2])
                
                cls_name = obj["name"]
                labels.append(self.class_to_idx.get(cls_name, 0))
            
            target = {
                "boxes": torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4)),
                "labels": torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64),
                "image_id": torch.tensor([index]),
            }
            
            if self.transform is not None:
                image = self.transform(image)
            
            return image, target
        
        return super().__getitem__(index)
    
    @property
    def num_classes(self) -> int:
        return 20

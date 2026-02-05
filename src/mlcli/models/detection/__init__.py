"""Object detection model definitions."""

from mlcli.models.detection.yolo import YOLO
from mlcli.models.detection.rcnn import FasterRCNN
from mlcli.models.detection.ssd import SSD
from mlcli.models.detection.detr import DETR

__all__ = ["YOLO", "FasterRCNN", "SSD", "DETR"]

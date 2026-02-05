"""Task abstractions for MLCLI."""

from mlcli.tasks.base import BaseTask
from mlcli.tasks.classification import ClassificationTask
from mlcli.tasks.detection import DetectionTask

__all__ = ["BaseTask", "ClassificationTask", "DetectionTask"]

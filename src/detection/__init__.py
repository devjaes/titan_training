"""
Módulo de detección de objetos con YOLOv11.
"""

from .train import DetectionTrainer
from .validate import DetectionValidator
from .inference import DetectionInference

__all__ = ['DetectionTrainer', 'DetectionValidator', 'DetectionInference']

"""
Biofilm Image Analysis Package
Provides tools for segmenting biofilms using ADMM optimization and classifying them using deep learning models.
"""

from .admm_segmentation import (
    admm_biofilm_segmentation_2d,
    admm_biofilm_segmentation_3d,
)
from .classifier import build_efficientnet_model
from .dataset import create_structured_directory

__version__ = "0.1.0"

__all__ = [
    "admm_biofilm_segmentation_2d",
    "admm_biofilm_segmentation_3d",
    "create_structured_directory",
    "build_efficientnet_model",
]

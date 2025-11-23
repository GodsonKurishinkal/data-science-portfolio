"""
Demand Forecasting System

A production-ready machine learning system for forecasting product demand.
"""

__version__ = "0.1.0"
__author__ = "Godson Kurishinkal"
__email__ = "godson.kurishinkal+github@gmail.com"

from src.data import preprocessing
from src.features import build_features
from src.models import train, predict

__all__ = [
    "preprocessing",
    "build_features",
    "train",
    "predict",
]

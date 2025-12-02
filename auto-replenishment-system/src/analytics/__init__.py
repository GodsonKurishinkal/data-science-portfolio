"""Demand analytics module."""

from .demand import DemandAnalyzer, WeightedDemandCalculator
from .trends import TrendDetector
from .seasonality import SeasonalityAnalyzer

__all__ = [
    "DemandAnalyzer",
    "WeightedDemandCalculator",
    "TrendDetector",
    "SeasonalityAnalyzer",
]

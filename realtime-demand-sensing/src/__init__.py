"""Real-Time Demand Sensing & Intelligent Replenishment

An intelligent system for real-time demand sensing, anomaly detection,
and automated replenishment with interactive dashboard monitoring.

Modules:
- utils: Stream simulation and helpers
- sensing: Real-time demand estimation
- detection: Anomaly detection and alerts
- forecasting: Short-term forecasting
- replenishment: Automated inventory replenishment
"""

__version__ = "1.0.0"
__author__ = "Godson Kurishinkal"

from .utils import StreamSimulator
from .sensing import DemandSensor, DemandSensorBatch
from .detection import (
    AnomalyDetector,
    Anomaly,
    AnomalySeverity,
    AnomalyType,
    AlertManager,
    Alert
)
from .forecasting import (
    ShortTermForecaster,
    EnsembleForecaster,
    Forecast
)
from .replenishment import (
    ReplenishmentEngine,
    InventoryPosition,
    ReplenishmentOrder
)

__all__ = [
    # Version
    '__version__',
    '__author__',
    # Utils
    'StreamSimulator',
    # Sensing
    'DemandSensor',
    'DemandSensorBatch',
    # Detection
    'AnomalyDetector',
    'Anomaly',
    'AnomalySeverity',
    'AnomalyType',
    'AlertManager',
    'Alert',
    # Forecasting
    'ShortTermForecaster',
    'EnsembleForecaster',
    'Forecast',
    # Replenishment
    'ReplenishmentEngine',
    'InventoryPosition',
    'ReplenishmentOrder'
]

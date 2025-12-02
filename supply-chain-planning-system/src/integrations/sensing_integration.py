"""
Real-Time Sensing Integration.
"""

from typing import Optional, Any, List, Dict
import pandas as pd
import logging

from src.data.models import SensingResult

logger = logging.getLogger(__name__)


class SensingIntegration:
    """Integration with the Real-Time Demand Sensing module."""
    
    def __init__(self, config: Any):
        self.config = config
        self._active_alerts: List[Dict[str, Any]] = []
        logger.info("SensingIntegration initialized")
    
    def run(
        self,
        data: Optional[pd.DataFrame] = None,
        **_kwargs
    ) -> SensingResult:
        """Run demand sensing."""
        logger.info("Running demand sensing")
        
        # Detect anomalies
        anomalies = self._detect_anomalies(data)
        
        # Generate alerts
        alerts = self._generate_alerts(anomalies)
        self._active_alerts = alerts
        
        return SensingResult(
            current_demand=pd.DataFrame({
                'item_id': ['SKU001', 'SKU002'],
                'current_demand': [120, 85],
                'forecast_demand': [100, 90],
                'variance': [0.20, -0.05]
            }),
            anomalies=anomalies,
            alerts=alerts,
            short_term_forecast=pd.DataFrame({
                'item_id': ['SKU001', 'SKU002'],
                'forecast_24h': [115, 88],
                'forecast_48h': [110, 90]
            }),
            trend_indicators={
                'overall': 'increasing',
                'confidence': 0.85
            }
        )
    
    def _detect_anomalies(self, _data: Optional[pd.DataFrame]) -> List[Dict[str, Any]]:
        """Detect demand anomalies."""
        return [
            {
                'item_id': 'SKU001',
                'type': 'spike',
                'severity': 'HIGH',
                'z_score': 2.8,
                'detected_at': '2025-12-02T08:30:00'
            }
        ]
    
    def _generate_alerts(self, anomalies: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate alerts from anomalies."""
        alerts = []
        for anomaly in anomalies:
            alerts.append({
                'id': f"ALT_{anomaly['item_id']}",
                'type': 'demand_anomaly',
                'severity': anomaly['severity'],
                'item_id': anomaly['item_id'],
                'message': f"Demand spike detected for {anomaly['item_id']}"
            })
        return alerts
    
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get currently active alerts."""
        return self._active_alerts

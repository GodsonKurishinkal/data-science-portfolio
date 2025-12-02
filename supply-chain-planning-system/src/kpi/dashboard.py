"""
KPI Dashboard for Supply Chain Planning System.
"""

from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)


class KPIDashboard:
    """
    Dashboard for displaying KPIs and planning metrics.
    """
    
    def __init__(self):
        self.metrics: Dict[str, Any] = {}
        logger.info("KPIDashboard initialized")
    
    def update(self, kpis: Dict[str, float]) -> None:
        """Update dashboard with new KPI values."""
        self.metrics.update(kpis)
    
    def get_display_data(self) -> Dict[str, Any]:
        """Get data formatted for display."""
        return {
            'strategic': self._get_strategic_kpis(),
            'operational': self._get_operational_kpis(),
            'alerts': self._get_alerts()
        }
    
    def _get_strategic_kpis(self) -> List[Dict[str, Any]]:
        """Get strategic KPIs for display."""
        return [
            {'name': 'Forecast Accuracy', 'value': self.metrics.get('forecast_accuracy', 0), 'target': 0.85},
            {'name': 'Service Level', 'value': self.metrics.get('service_level', 0), 'target': 0.98},
            {'name': 'Revenue Lift', 'value': self.metrics.get('revenue_lift', 0), 'target': 0.08}
        ]
    
    def _get_operational_kpis(self) -> List[Dict[str, Any]]:
        """Get operational KPIs for display."""
        return [
            {'name': 'Automation Rate', 'value': self.metrics.get('automation_rate', 0), 'target': 0.80},
            {'name': 'Cost Reduction', 'value': self.metrics.get('logistics_cost_reduction', 0), 'target': 0.15}
        ]
    
    def _get_alerts(self) -> List[Dict[str, Any]]:
        """Get active alerts."""
        alerts = []
        if self.metrics.get('service_level', 1) < 0.95:
            alerts.append({'severity': 'HIGH', 'message': 'Service level below threshold'})
        return alerts

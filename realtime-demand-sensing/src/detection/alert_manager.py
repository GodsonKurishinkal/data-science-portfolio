"""
Alert Manager - Anomaly Alert Generation and Management

Manages alerts from anomaly detection:
- Alert creation and formatting
- Alert deduplication
- Severity-based filtering
- Alert history tracking

Author: Godson Kurishinkal
Date: December 2025
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
import logging

from .anomaly_detector import Anomaly, AnomalySeverity, AnomalyType

logger = logging.getLogger(__name__)


class AlertStatus(Enum):
    """Alert status states."""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    DISMISSED = "dismissed"


@dataclass
class Alert:
    """Alert from anomaly detection."""
    id: str
    anomaly: Anomaly
    status: AlertStatus = AlertStatus.ACTIVE
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    notes: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'product_id': self.anomaly.product_id,
            'timestamp': self.anomaly.timestamp,
            'type': self.anomaly.anomaly_type.value,
            'severity': self.anomaly.severity.value,
            'message': self.anomaly.message,
            'value': self.anomaly.value,
            'expected_value': self.anomaly.expected_value,
            'status': self.status.value,
            'created_at': self.created_at,
            'acknowledged_by': self.acknowledged_by
        }


class AlertManager:
    """
    Manages alerts from anomaly detection system.
    
    Features:
    - Alert creation from anomalies
    - Deduplication within time window
    - Status tracking (active, acknowledged, resolved)
    - Filtering by severity and type
    - Alert history
    
    Example:
        >>> manager = AlertManager()
        >>> alerts = manager.create_alerts(anomalies)
        >>> active = manager.get_active_alerts(severity='critical')
        >>> manager.acknowledge(alert_id, user='analyst')
    """
    
    def __init__(
        self,
        dedup_window_hours: float = 4.0,
        max_history: int = 1000
    ):
        """
        Initialize alert manager.
        
        Args:
            dedup_window_hours: Time window for deduplication
            max_history: Maximum alerts to keep in history
        """
        self.dedup_window_hours = dedup_window_hours
        self.max_history = max_history
        
        self.alerts: Dict[str, Alert] = {}
        self.alert_history: deque = deque(maxlen=max_history)
        self._alert_counter = 0
        
        logger.info(
            "Initialized AlertManager: dedup_window=%.1f hrs, max_history=%d",
            dedup_window_hours, max_history
        )
    
    def _generate_alert_id(self) -> str:
        """Generate unique alert ID."""
        self._alert_counter += 1
        return f"ALERT_{datetime.now().strftime('%Y%m%d%H%M%S')}_{self._alert_counter:04d}"
    
    def _is_duplicate(self, anomaly: Anomaly) -> bool:
        """Check if anomaly is duplicate of existing alert."""
        cutoff_time = datetime.now() - timedelta(hours=self.dedup_window_hours)
        
        for alert in self.alerts.values():
            if alert.status != AlertStatus.ACTIVE:
                continue
            
            if alert.created_at < cutoff_time:
                continue
            
            # Check if same product, type, and within time window
            if (alert.anomaly.product_id == anomaly.product_id and
                alert.anomaly.anomaly_type == anomaly.anomaly_type):
                return True
        
        return False
    
    def create_alerts(
        self,
        anomalies: List[Anomaly],
        deduplicate: bool = True
    ) -> List[Alert]:
        """
        Create alerts from anomalies.
        
        Args:
            anomalies: List of detected anomalies
            deduplicate: Whether to skip duplicate alerts
        
        Returns:
            List of created alerts
        """
        created = []
        
        for anomaly in anomalies:
            if deduplicate and self._is_duplicate(anomaly):
                logger.debug("Skipping duplicate alert for %s", anomaly.product_id)
                continue
            
            alert_id = self._generate_alert_id()
            alert = Alert(
                id=alert_id,
                anomaly=anomaly
            )
            
            self.alerts[alert_id] = alert
            created.append(alert)
            
            logger.info(
                "Created alert %s: [%s] %s - %s",
                alert_id,
                anomaly.severity.value.upper(),
                anomaly.product_id,
                anomaly.message
            )
        
        return created
    
    def get_alert(self, alert_id: str) -> Optional[Alert]:
        """Get alert by ID."""
        return self.alerts.get(alert_id)
    
    def get_active_alerts(
        self,
        severity: Optional[str] = None,
        anomaly_type: Optional[str] = None,
        product_id: Optional[str] = None
    ) -> List[Alert]:
        """
        Get active alerts with optional filtering.
        
        Args:
            severity: Filter by severity ('critical', 'warning', 'info')
            anomaly_type: Filter by type
            product_id: Filter by product
        
        Returns:
            List of matching alerts
        """
        alerts = [a for a in self.alerts.values() if a.status == AlertStatus.ACTIVE]
        
        if severity:
            alerts = [a for a in alerts if a.anomaly.severity.value == severity]
        
        if anomaly_type:
            alerts = [a for a in alerts if a.anomaly.anomaly_type.value == anomaly_type]
        
        if product_id:
            alerts = [a for a in alerts if a.anomaly.product_id == product_id]
        
        # Sort by severity (critical first) then by time
        severity_order = {'critical': 0, 'warning': 1, 'info': 2}
        alerts.sort(key=lambda a: (
            severity_order.get(a.anomaly.severity.value, 3),
            -a.created_at.timestamp()
        ))
        
        return alerts
    
    def acknowledge(
        self,
        alert_id: str,
        user: str = "system",
        notes: Optional[str] = None
    ) -> bool:
        """
        Acknowledge an alert.
        
        Args:
            alert_id: Alert to acknowledge
            user: User acknowledging
            notes: Optional notes
        
        Returns:
            True if successful
        """
        alert = self.alerts.get(alert_id)
        if not alert:
            return False
        
        alert.status = AlertStatus.ACKNOWLEDGED
        alert.acknowledged_by = user
        alert.acknowledged_at = datetime.now()
        alert.updated_at = datetime.now()
        
        if notes:
            alert.notes.append(f"{datetime.now()}: {notes}")
        
        logger.info("Alert %s acknowledged by %s", alert_id, user)
        return True
    
    def resolve(self, alert_id: str, notes: Optional[str] = None) -> bool:
        """
        Resolve an alert.
        
        Args:
            alert_id: Alert to resolve
            notes: Resolution notes
        
        Returns:
            True if successful
        """
        alert = self.alerts.get(alert_id)
        if not alert:
            return False
        
        alert.status = AlertStatus.RESOLVED
        alert.updated_at = datetime.now()
        
        if notes:
            alert.notes.append(f"{datetime.now()}: Resolved - {notes}")
        
        # Move to history
        self.alert_history.append(alert)
        del self.alerts[alert_id]
        
        logger.info("Alert %s resolved", alert_id)
        return True
    
    def dismiss(self, alert_id: str) -> bool:
        """
        Dismiss an alert (false positive).
        
        Args:
            alert_id: Alert to dismiss
        
        Returns:
            True if successful
        """
        alert = self.alerts.get(alert_id)
        if not alert:
            return False
        
        alert.status = AlertStatus.DISMISSED
        alert.updated_at = datetime.now()
        
        # Move to history
        self.alert_history.append(alert)
        del self.alerts[alert_id]
        
        logger.info("Alert %s dismissed", alert_id)
        return True
    
    def get_summary(self) -> Dict:
        """Get alert summary statistics."""
        active = [a for a in self.alerts.values() if a.status == AlertStatus.ACTIVE]
        
        severity_counts = {}
        type_counts = {}
        
        for alert in active:
            sev = alert.anomaly.severity.value
            typ = alert.anomaly.anomaly_type.value
            
            severity_counts[sev] = severity_counts.get(sev, 0) + 1
            type_counts[typ] = type_counts.get(typ, 0) + 1
        
        return {
            'total_active': len(active),
            'total_acknowledged': len([a for a in self.alerts.values() 
                                       if a.status == AlertStatus.ACKNOWLEDGED]),
            'by_severity': severity_counts,
            'by_type': type_counts,
            'critical_count': severity_counts.get('critical', 0),
            'history_size': len(self.alert_history)
        }
    
    def to_dataframe(self, include_history: bool = False) -> pd.DataFrame:
        """
        Convert alerts to DataFrame.
        
        Args:
            include_history: Include resolved/dismissed alerts
        
        Returns:
            DataFrame of alerts
        """
        alerts = list(self.alerts.values())
        
        if include_history:
            alerts.extend(list(self.alert_history))
        
        if not alerts:
            return pd.DataFrame()
        
        return pd.DataFrame([a.to_dict() for a in alerts])
    
    def cleanup_old_alerts(self, max_age_hours: float = 24.0) -> int:
        """
        Cleanup old acknowledged alerts.
        
        Args:
            max_age_hours: Maximum age for acknowledged alerts
        
        Returns:
            Number of alerts cleaned up
        """
        cutoff = datetime.now() - timedelta(hours=max_age_hours)
        to_remove = []
        
        for alert_id, alert in self.alerts.items():
            if alert.status == AlertStatus.ACKNOWLEDGED and alert.updated_at < cutoff:
                to_remove.append(alert_id)
        
        for alert_id in to_remove:
            alert = self.alerts[alert_id]
            self.alert_history.append(alert)
            del self.alerts[alert_id]
        
        if to_remove:
            logger.info("Cleaned up %d old alerts", len(to_remove))
        
        return len(to_remove)


if __name__ == "__main__":
    # Quick test
    logging.basicConfig(level=logging.INFO)
    
    from anomaly_detector import Anomaly, AnomalySeverity, AnomalyType
    
    # Create test anomalies
    anomalies = [
        Anomaly(
            product_id="PROD_001",
            timestamp=datetime.now(),
            anomaly_type=AnomalyType.STOCKOUT_RISK,
            severity=AnomalySeverity.CRITICAL,
            value=1.2,
            expected_value=3.0,
            deviation=-1.8,
            z_score=-2.5,
            message="Critical: Only 1.2 days of stock!",
            detection_method="business_rule"
        ),
        Anomaly(
            product_id="PROD_002",
            timestamp=datetime.now(),
            anomaly_type=AnomalyType.DEMAND_SPIKE,
            severity=AnomalySeverity.WARNING,
            value=200,
            expected_value=100,
            deviation=100,
            z_score=2.5,
            message="Demand spike: 100% above baseline",
            detection_method="z_score"
        )
    ]
    
    # Test alert manager
    manager = AlertManager()
    alerts = manager.create_alerts(anomalies)
    
    print(f"\nCreated {len(alerts)} alerts")
    print(f"Summary: {manager.get_summary()}")
    
    # Get active alerts
    print("\nActive alerts:")
    for alert in manager.get_active_alerts():
        print(f"  [{alert.anomaly.severity.value.upper()}] {alert.anomaly.message}")
    
    # Acknowledge first alert
    manager.acknowledge(alerts[0].id, user="analyst", notes="Investigating")
    print(f"\nAfter acknowledge: {manager.get_summary()}")

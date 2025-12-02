"""
Alert Manager for Supply Chain Planning System.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, List, Optional
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class Alert:
    """Planning alert."""
    id: str
    severity: AlertSeverity
    category: str
    message: str
    source_module: str
    created_at: datetime = field(default_factory=datetime.now)
    acknowledged: bool = False
    resolved: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class AlertManager:
    """
    Manages alerts across all planning modules.
    """
    
    def __init__(self):
        self.alerts: List[Alert] = []
        self._alert_counter = 0
        logger.info("AlertManager initialized")
    
    def create_alert(
        self,
        severity: AlertSeverity,
        category: str,
        message: str,
        source_module: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Alert:
        """Create a new alert."""
        self._alert_counter += 1
        
        alert = Alert(
            id=f"ALT{self._alert_counter:06d}",
            severity=severity,
            category=category,
            message=message,
            source_module=source_module,
            metadata=metadata or {}
        )
        
        self.alerts.append(alert)
        logger.info("Created alert %s: %s", alert.id, message)
        
        return alert
    
    def get_active_alerts(self, severity: Optional[AlertSeverity] = None) -> List[Alert]:
        """Get active (unresolved) alerts."""
        active = [a for a in self.alerts if not a.resolved]
        
        if severity:
            active = [a for a in active if a.severity == severity]
        
        return sorted(active, key=lambda a: list(AlertSeverity).index(a.severity))
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert."""
        for alert in self.alerts:
            if alert.id == alert_id:
                alert.acknowledged = True
                return True
        return False
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert."""
        for alert in self.alerts:
            if alert.id == alert_id:
                alert.resolved = True
                return True
        return False
    
    def get_summary(self) -> Dict[str, Any]:
        """Get alert summary."""
        active = self.get_active_alerts()
        
        by_severity = {}
        for severity in AlertSeverity:
            count = sum(1 for a in active if a.severity == severity)
            by_severity[severity.value] = count
        
        return {
            'total_alerts': len(self.alerts),
            'active_alerts': len(active),
            'by_severity': by_severity,
            'critical_count': by_severity.get('critical', 0),
            'high_count': by_severity.get('high', 0)
        }

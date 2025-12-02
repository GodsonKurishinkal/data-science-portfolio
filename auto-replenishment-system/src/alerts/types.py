"""Alert types and data structures."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional


class AlertSeverity(Enum):
    """Alert severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class AlertType(Enum):
    """Types of replenishment alerts."""
    STOCKOUT_IMMINENT = "stockout_imminent"
    STOCKOUT = "stockout"
    OVERSTOCK = "overstock"
    LOW_COVERAGE = "low_coverage"
    CAPACITY_EXCEEDED = "capacity_exceeded"
    SOURCE_SHORTAGE = "source_shortage"
    LEAD_TIME_DELAY = "lead_time_delay"
    DEMAND_SPIKE = "demand_spike"
    DEMAND_DROP = "demand_drop"
    SLOW_MOVER = "slow_mover"
    EXPIRY_RISK = "expiry_risk"
    VELOCITY_CHANGE = "velocity_change"


@dataclass
class Alert:
    """Replenishment alert data structure.
    
    Attributes:
        alert_type: Type of alert
        severity: Alert severity level
        item_id: Affected item ID
        location_id: Affected location
        message: Human-readable alert message
        details: Additional alert details
        recommended_action: Suggested action
        created_at: Alert creation timestamp
        expires_at: Alert expiration timestamp
        acknowledged: Whether alert has been acknowledged
    """
    alert_type: AlertType
    severity: AlertSeverity
    item_id: str
    location_id: Optional[str] = None
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    recommended_action: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    acknowledged: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary."""
        return {
            "alert_type": self.alert_type.value,
            "severity": self.severity.value,
            "item_id": self.item_id,
            "location_id": self.location_id,
            "message": self.message,
            "details": self.details,
            "recommended_action": self.recommended_action,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "acknowledged": self.acknowledged,
        }
    
    def __str__(self) -> str:
        """String representation of alert."""
        location = f" @ {self.location_id}" if self.location_id else ""
        return (
            f"[{self.severity.value.upper()}] {self.alert_type.value}: "
            f"Item {self.item_id}{location} - {self.message}"
        )

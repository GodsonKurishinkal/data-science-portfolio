"""Alert generation for replenishment events.

Generates actionable alerts for:
- Stockout imminent/actual
- Overstock conditions
- Low coverage
- Capacity constraints
- Source shortages
- Demand anomalies
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd

from ..interfaces.base import IAlertGenerator
from .types import Alert, AlertSeverity, AlertType

logger = logging.getLogger(__name__)


@dataclass
class AlertThresholds:
    """Configurable alert thresholds.

    Attributes:
        critical_days_supply: Days of supply for critical stockout alert
        low_days_supply: Days of supply for low stock alert
        overstock_days_supply: Days of supply for overstock alert
        capacity_utilization_high: Threshold for high capacity alert
        capacity_utilization_critical: Threshold for critical capacity alert
        demand_spike_threshold: % increase for demand spike alert
        demand_drop_threshold: % decrease for demand drop alert
        coverage_critical: Coverage % for critical alert
        coverage_warning: Coverage % for warning alert
    """
    critical_days_supply: float = 1.0
    low_days_supply: float = 3.0
    overstock_days_supply: float = 60.0
    capacity_utilization_high: float = 0.85
    capacity_utilization_critical: float = 0.95
    demand_spike_threshold: float = 0.50  # 50% increase
    demand_drop_threshold: float = 0.30   # 30% decrease
    coverage_critical: float = 0.80
    coverage_warning: float = 0.90


class AlertGenerator(IAlertGenerator):
    """Generate replenishment alerts based on inventory conditions.

    Features:
    - Multi-level severity classification
    - Configurable thresholds
    - Actionable recommendations
    - Alert prioritization

    Examples:
        >>> generator = AlertGenerator()
        >>> alerts = generator.generate(policy_results_df)
        >>> for alert in alerts:
        ...     print(alert)
    """

    def __init__(
        self,
        thresholds: Optional[AlertThresholds] = None,
        item_column: str = "item_id",
        location_column: Optional[str] = None,
    ):
        """Initialize alert generator.

        Args:
            thresholds: Alert thresholds configuration
            item_column: Item identifier column name
            location_column: Location identifier column name
        """
        self.thresholds = thresholds or AlertThresholds()
        self.item_column = item_column
        self.location_column = location_column

    def generate(self, df: pd.DataFrame) -> List[Alert]:
        """Generate alerts from policy calculation results.

        Args:
            df: DataFrame with policy calculations

        Returns:
            List of alerts sorted by severity
        """
        alerts: List[Alert] = []

        for _, row in df.iterrows():
            item_id = str(row.get(self.item_column, "unknown"))
            location_id = (
                str(row.get(self.location_column))
                if self.location_column and self.location_column in row
                else None
            )

            # Check for various alert conditions
            item_alerts = self._check_all_conditions(row, item_id, location_id)
            alerts.extend(item_alerts)

        # Sort by severity (critical first)
        severity_order = {
            AlertSeverity.CRITICAL: 0,
            AlertSeverity.HIGH: 1,
            AlertSeverity.MEDIUM: 2,
            AlertSeverity.LOW: 3,
            AlertSeverity.INFO: 4,
        }
        alerts.sort(key=lambda x: severity_order[x.severity])

        logger.info("Generated %d alerts", len(alerts))
        return alerts

    def _check_all_conditions(
        self,
        row: pd.Series,
        item_id: str,
        location_id: Optional[str],
    ) -> List[Alert]:
        """Check all alert conditions for an item."""
        alerts = []

        # Get key metrics
        current_stock = row.get("inventory_position", row.get("current_stock", 0))
        daily_demand = row.get("daily_demand_rate", row.get("demand_mean", 0))
        max_capacity = row.get("max_capacity")
        source_available = row.get("source_available")
        reorder_point = row.get("reorder_point", 0)

        # Calculate days of supply
        days_of_supply = (
            current_stock / daily_demand if daily_demand > 0 else float("inf")
        )

        # Stockout alerts
        stockout_alert = self._check_stockout(
            item_id, location_id, current_stock,
            days_of_supply, reorder_point
        )
        if stockout_alert:
            alerts.append(stockout_alert)

        # Overstock alerts
        overstock_alert = self._check_overstock(
            item_id, location_id, days_of_supply
        )
        if overstock_alert:
            alerts.append(overstock_alert)

        # Capacity alerts
        if max_capacity:
            capacity_alert = self._check_capacity(
                item_id, location_id, current_stock, max_capacity
            )
            if capacity_alert:
                alerts.append(capacity_alert)

        # Source shortage alerts
        if source_available is not None:
            source_alert = self._check_source_shortage(
                item_id, location_id, row, source_available
            )
            if source_alert:
                alerts.append(source_alert)

        # Coverage alerts
        coverage = row.get("coverage_probability")
        if coverage is not None:
            coverage_alert = self._check_coverage(
                item_id, location_id, coverage
            )
            if coverage_alert:
                alerts.append(coverage_alert)

        # Demand anomaly alerts
        demand_change = row.get("demand_change_pct")
        if demand_change is not None:
            demand_alert = self._check_demand_anomaly(
                item_id, location_id, demand_change
            )
            if demand_alert:
                alerts.append(demand_alert)

        return alerts

    def _check_stockout(
        self,
        item_id: str,
        location_id: Optional[str],
        current_stock: float,
        days_of_supply: float,
        reorder_point: float,
    ) -> Optional[Alert]:
        """Check for stockout conditions."""
        if current_stock <= 0:
            return Alert(
                alert_type=AlertType.STOCKOUT,
                severity=AlertSeverity.CRITICAL,
                item_id=item_id,
                location_id=location_id,
                message="Item is out of stock",
                details={
                    "current_stock": current_stock,
                    "days_of_supply": 0,
                },
                recommended_action="Expedite replenishment order immediately",
            )

        if days_of_supply <= self.thresholds.critical_days_supply:
            return Alert(
                alert_type=AlertType.STOCKOUT_IMMINENT,
                severity=AlertSeverity.CRITICAL,
                item_id=item_id,
                location_id=location_id,
                message=f"Stockout imminent - only {days_of_supply:.1f} days of supply",
                details={
                    "current_stock": current_stock,
                    "days_of_supply": days_of_supply,
                    "reorder_point": reorder_point,
                },
                recommended_action="Place emergency order or expedite existing order",
            )

        if days_of_supply <= self.thresholds.low_days_supply:
            return Alert(
                alert_type=AlertType.LOW_COVERAGE,
                severity=AlertSeverity.HIGH,
                item_id=item_id,
                location_id=location_id,
                message=f"Low stock - {days_of_supply:.1f} days of supply remaining",
                details={
                    "current_stock": current_stock,
                    "days_of_supply": days_of_supply,
                },
                recommended_action="Review replenishment order timing",
            )

        return None

    def _check_overstock(
        self,
        item_id: str,
        location_id: Optional[str],
        days_of_supply: float,
    ) -> Optional[Alert]:
        """Check for overstock conditions."""
        if days_of_supply >= self.thresholds.overstock_days_supply:
            return Alert(
                alert_type=AlertType.OVERSTOCK,
                severity=AlertSeverity.MEDIUM,
                item_id=item_id,
                location_id=location_id,
                message=f"Overstock condition - {days_of_supply:.0f} days of supply",
                details={"days_of_supply": days_of_supply},
                recommended_action="Consider promotional activity or transfer to other locations",
            )
        return None

    def _check_capacity(
        self,
        item_id: str,
        location_id: Optional[str],
        current_stock: float,
        max_capacity: float,
    ) -> Optional[Alert]:
        """Check for capacity constraints."""
        utilization = current_stock / max_capacity if max_capacity > 0 else 0

        if utilization >= self.thresholds.capacity_utilization_critical:
            return Alert(
                alert_type=AlertType.CAPACITY_EXCEEDED,
                severity=AlertSeverity.HIGH,
                item_id=item_id,
                location_id=location_id,
                message=f"Storage capacity critical at {utilization:.0%}",
                details={
                    "current_stock": current_stock,
                    "max_capacity": max_capacity,
                    "utilization": utilization,
                },
                recommended_action="Reduce incoming orders or expand storage",
            )

        if utilization >= self.thresholds.capacity_utilization_high:
            return Alert(
                alert_type=AlertType.CAPACITY_EXCEEDED,
                severity=AlertSeverity.MEDIUM,
                item_id=item_id,
                location_id=location_id,
                message=f"Storage capacity high at {utilization:.0%}",
                details={"utilization": utilization},
                recommended_action="Monitor capacity and adjust order quantities",
            )

        return None

    def _check_source_shortage(
        self,
        item_id: str,
        location_id: Optional[str],
        row: pd.Series,
        source_available: float,
    ) -> Optional[Alert]:
        """Check for source inventory shortage."""
        recommended_qty = row.get("recommended_quantity", 0)

        if recommended_qty > 0 and source_available < recommended_qty:
            shortage_pct = 1 - (source_available / recommended_qty)

            severity = (
                AlertSeverity.CRITICAL if shortage_pct > 0.5
                else AlertSeverity.HIGH if shortage_pct > 0.2
                else AlertSeverity.MEDIUM
            )

            return Alert(
                alert_type=AlertType.SOURCE_SHORTAGE,
                severity=severity,
                item_id=item_id,
                location_id=location_id,
                message=f"Source shortage - only {source_available:.0f} available of {recommended_qty:.0f} needed",
                details={
                    "recommended_quantity": recommended_qty,
                    "source_available": source_available,
                    "shortage_percent": shortage_pct,
                },
                recommended_action="Seek alternative source or adjust order quantity",
            )

        return None

    def _check_coverage(
        self,
        item_id: str,
        location_id: Optional[str],
        coverage: float,
    ) -> Optional[Alert]:
        """Check for low coverage probability."""
        if coverage < self.thresholds.coverage_critical:
            return Alert(
                alert_type=AlertType.LOW_COVERAGE,
                severity=AlertSeverity.HIGH,
                item_id=item_id,
                location_id=location_id,
                message=f"Low service coverage at {coverage:.0%}",
                details={"coverage_probability": coverage},
                recommended_action="Increase order quantity or safety stock",
            )

        if coverage < self.thresholds.coverage_warning:
            return Alert(
                alert_type=AlertType.LOW_COVERAGE,
                severity=AlertSeverity.MEDIUM,
                item_id=item_id,
                location_id=location_id,
                message=f"Service coverage below target at {coverage:.0%}",
                details={"coverage_probability": coverage},
                recommended_action="Review safety stock parameters",
            )

        return None

    def _check_demand_anomaly(
        self,
        item_id: str,
        location_id: Optional[str],
        demand_change: float,
    ) -> Optional[Alert]:
        """Check for demand spikes or drops."""
        if demand_change >= self.thresholds.demand_spike_threshold:
            return Alert(
                alert_type=AlertType.DEMAND_SPIKE,
                severity=AlertSeverity.HIGH,
                item_id=item_id,
                location_id=location_id,
                message=f"Demand spike detected - {demand_change:+.0%} change",
                details={"demand_change_percent": demand_change},
                recommended_action="Review inventory levels and increase orders",
            )

        if demand_change <= -self.thresholds.demand_drop_threshold:
            return Alert(
                alert_type=AlertType.DEMAND_DROP,
                severity=AlertSeverity.MEDIUM,
                item_id=item_id,
                location_id=location_id,
                message=f"Demand drop detected - {demand_change:+.0%} change",
                details={"demand_change_percent": demand_change},
                recommended_action="Review inventory levels and reduce orders",
            )

        return None

    def summarize_alerts(
        self,
        alerts: List[Alert],
    ) -> Dict[str, Any]:
        """Generate summary statistics for alerts.

        Args:
            alerts: List of generated alerts

        Returns:
            Summary dictionary
        """
        if not alerts:
            return {
                "total_alerts": 0,
                "by_severity": {},
                "by_type": {},
                "items_affected": 0,
            }

        # Count by severity
        by_severity = {}
        for severity in AlertSeverity:
            count = sum(1 for a in alerts if a.severity == severity)
            if count > 0:
                by_severity[severity.value] = count

        # Count by type
        by_type = {}
        for alert_type in AlertType:
            count = sum(1 for a in alerts if a.alert_type == alert_type)
            if count > 0:
                by_type[alert_type.value] = count

        # Unique items
        items_affected = len(set(a.item_id for a in alerts))

        return {
            "total_alerts": len(alerts),
            "by_severity": by_severity,
            "by_type": by_type,
            "items_affected": items_affected,
        }

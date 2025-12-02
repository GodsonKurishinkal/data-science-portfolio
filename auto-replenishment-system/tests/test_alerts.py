"""Tests for alert generation."""

import pytest
import pandas as pd
import numpy as np

from src.alerts.generator import AlertGenerator, AlertThresholds
from src.alerts.types import Alert, AlertSeverity, AlertType


class TestAlertGenerator:
    """Tests for alert generation."""

    def test_generate_stockout_alert(self):
        """Test stockout alert generation."""
        generator = AlertGenerator(
            thresholds=AlertThresholds(critical_days_supply=1.0)
        )

        test_data = pd.DataFrame({
            "item_id": ["SKU001"],
            "current_stock": [0],  # Out of stock
            "daily_demand_rate": [10],
        })

        alerts = generator.generate(test_data)

        assert len(alerts) >= 1
        stockout_alerts = [a for a in alerts if a.alert_type == AlertType.STOCKOUT]
        assert len(stockout_alerts) == 1
        assert stockout_alerts[0].severity == AlertSeverity.CRITICAL

    def test_generate_stockout_imminent_alert(self):
        """Test stockout imminent alert."""
        generator = AlertGenerator(
            thresholds=AlertThresholds(
                critical_days_supply=1.0,
                low_days_supply=3.0,
            )
        )

        test_data = pd.DataFrame({
            "item_id": ["SKU001"],
            "current_stock": [5],  # 0.5 days of supply
            "daily_demand_rate": [10],
            "inventory_position": [5],
        })

        alerts = generator.generate(test_data)

        stockout_imminent = [
            a for a in alerts
            if a.alert_type == AlertType.STOCKOUT_IMMINENT
        ]
        assert len(stockout_imminent) == 1
        assert stockout_imminent[0].severity == AlertSeverity.CRITICAL

    def test_generate_overstock_alert(self):
        """Test overstock alert generation."""
        generator = AlertGenerator(
            thresholds=AlertThresholds(overstock_days_supply=60.0)
        )

        test_data = pd.DataFrame({
            "item_id": ["SKU001"],
            "current_stock": [1000],  # 100 days of supply
            "daily_demand_rate": [10],
            "inventory_position": [1000],
        })

        alerts = generator.generate(test_data)

        overstock_alerts = [
            a for a in alerts
            if a.alert_type == AlertType.OVERSTOCK
        ]
        assert len(overstock_alerts) == 1
        assert overstock_alerts[0].severity == AlertSeverity.MEDIUM

    def test_generate_capacity_alert(self):
        """Test capacity exceeded alert."""
        generator = AlertGenerator(
            thresholds=AlertThresholds(
                capacity_utilization_critical=0.95,
            )
        )

        test_data = pd.DataFrame({
            "item_id": ["SKU001"],
            "current_stock": [980],  # 98% utilization
            "max_capacity": [1000],
            "daily_demand_rate": [10],
            "inventory_position": [980],
        })

        alerts = generator.generate(test_data)

        capacity_alerts = [
            a for a in alerts
            if a.alert_type == AlertType.CAPACITY_EXCEEDED
        ]
        assert len(capacity_alerts) == 1

    def test_generate_source_shortage_alert(self):
        """Test source shortage alert."""
        generator = AlertGenerator()

        test_data = pd.DataFrame({
            "item_id": ["SKU001"],
            "current_stock": [10],
            "daily_demand_rate": [10],
            "inventory_position": [10],
            "recommended_quantity": [100],
            "source_available": [20],  # Only 20% available
        })

        alerts = generator.generate(test_data)

        source_alerts = [
            a for a in alerts
            if a.alert_type == AlertType.SOURCE_SHORTAGE
        ]
        assert len(source_alerts) == 1

    def test_no_alerts_healthy_inventory(self):
        """Test no alerts for healthy inventory."""
        generator = AlertGenerator()

        test_data = pd.DataFrame({
            "item_id": ["SKU001"],
            "current_stock": [200],  # 20 days of supply
            "daily_demand_rate": [10],
            "inventory_position": [200],
        })

        alerts = generator.generate(test_data)

        # Should have no critical alerts
        critical = [a for a in alerts if a.severity == AlertSeverity.CRITICAL]
        assert len(critical) == 0

    def test_alert_sorting_by_severity(self):
        """Test alerts are sorted by severity."""
        generator = AlertGenerator()

        test_data = pd.DataFrame({
            "item_id": ["SKU001", "SKU002", "SKU003"],
            "current_stock": [0, 5, 1000],  # stockout, low, overstock
            "daily_demand_rate": [10, 10, 10],
            "inventory_position": [0, 5, 1000],
        })

        alerts = generator.generate(test_data)

        # Critical should come first
        if len(alerts) >= 2:
            assert alerts[0].severity in [
                AlertSeverity.CRITICAL, AlertSeverity.HIGH
            ]

    def test_summarize_alerts(self):
        """Test alert summary generation."""
        generator = AlertGenerator()

        test_data = pd.DataFrame({
            "item_id": ["SKU001", "SKU002"],
            "current_stock": [0, 5],
            "daily_demand_rate": [10, 10],
            "inventory_position": [0, 5],
        })

        alerts = generator.generate(test_data)
        summary = generator.summarize_alerts(alerts)

        assert "total_alerts" in summary
        assert "by_severity" in summary
        assert "by_type" in summary
        assert "items_affected" in summary

        assert summary["total_alerts"] == len(alerts)


class TestAlertTypes:
    """Tests for alert types and structures."""

    def test_alert_to_dict(self):
        """Test alert serialization to dictionary."""
        alert = Alert(
            alert_type=AlertType.STOCKOUT,
            severity=AlertSeverity.CRITICAL,
            item_id="SKU001",
            message="Test alert",
        )

        alert_dict = alert.to_dict()

        assert alert_dict["alert_type"] == "stockout"
        assert alert_dict["severity"] == "critical"
        assert alert_dict["item_id"] == "SKU001"
        assert alert_dict["message"] == "Test alert"

    def test_alert_str_representation(self):
        """Test alert string representation."""
        alert = Alert(
            alert_type=AlertType.STOCKOUT,
            severity=AlertSeverity.CRITICAL,
            item_id="SKU001",
            location_id="LOC01",
            message="Out of stock",
        )

        alert_str = str(alert)

        assert "CRITICAL" in alert_str
        assert "SKU001" in alert_str
        assert "Out of stock" in alert_str

"""
Tests for Supply Chain Planning System integration modules.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from src.config.loader import PlanningConfig
from src.integrations.demand_integration import DemandIntegration
from src.integrations.inventory_integration import InventoryIntegration
from src.integrations.pricing_integration import PricingIntegration
from src.integrations.network_integration import NetworkIntegration
from src.integrations.sensing_integration import SensingIntegration
from src.integrations.replenishment_integration import ReplenishmentIntegration


class TestDemandIntegration:
    """Tests for DemandIntegration class."""

    def test_initialization(self):
        """Test demand integration initialization."""
        config = PlanningConfig()
        integration = DemandIntegration(config.demand)

        assert integration is not None

    def test_run(self):
        """Test running demand forecast."""
        config = PlanningConfig()
        integration = DemandIntegration(config.demand)

        result = integration.run(horizon_months=3)

        assert result is not None
        assert result.mape is not None
        assert 0 <= result.mape <= 1
        assert result.forecast is not None
        assert isinstance(result.forecast, pd.DataFrame)


class TestInventoryIntegration:
    """Tests for InventoryIntegration class."""

    def test_initialization(self):
        """Test inventory integration initialization."""
        config = PlanningConfig()
        integration = InventoryIntegration(config.inventory)

        assert integration is not None

    def test_run(self):
        """Test running inventory optimization."""
        config = PlanningConfig()
        integration = InventoryIntegration(config.inventory)

        result = integration.run()

        assert result is not None
        assert result.service_level >= 0
        assert result.positions is not None
        assert result.classifications is not None

    def test_get_exceptions(self):
        """Test getting inventory exceptions."""
        config = PlanningConfig()
        integration = InventoryIntegration(config.inventory)

        exceptions = integration.get_exceptions()

        assert exceptions is not None
        assert isinstance(exceptions, list)


class TestPricingIntegration:
    """Tests for PricingIntegration class."""

    def test_initialization(self):
        """Test pricing integration initialization."""
        config = PlanningConfig()
        integration = PricingIntegration(config.pricing)

        assert integration is not None

    def test_run(self):
        """Test running pricing optimization."""
        config = PlanningConfig()
        integration = PricingIntegration(config.pricing)

        result = integration.run()

        assert result is not None
        assert result.revenue_lift is not None
        assert result.optimal_prices is not None
        assert isinstance(result.optimal_prices, pd.DataFrame)


class TestNetworkIntegration:
    """Tests for NetworkIntegration class."""

    def test_initialization(self):
        """Test network integration initialization."""
        config = PlanningConfig()
        integration = NetworkIntegration(config.network)

        assert integration is not None

    def test_run(self):
        """Test running network optimization."""
        config = PlanningConfig()
        integration = NetworkIntegration(config.network)

        result = integration.run()

        assert result is not None
        assert result.cost_reduction is not None
        assert result.facility_decisions is not None
        assert result.routes is not None


class TestSensingIntegration:
    """Tests for SensingIntegration class."""

    def test_initialization(self):
        """Test sensing integration initialization."""
        config = PlanningConfig()
        integration = SensingIntegration(config.sensing)

        assert integration is not None

    def test_run(self):
        """Test running real-time sensing."""
        config = PlanningConfig()
        integration = SensingIntegration(config.sensing)

        result = integration.run()

        assert result is not None
        assert result.anomalies is not None
        assert result.alerts is not None

    def test_get_active_alerts(self):
        """Test getting active alerts."""
        config = PlanningConfig()
        integration = SensingIntegration(config.sensing)

        # Run sensing first to populate alerts
        integration.run()

        alerts = integration.get_active_alerts()

        assert alerts is not None
        assert isinstance(alerts, list)


class TestReplenishmentIntegration:
    """Tests for ReplenishmentIntegration class."""

    def test_initialization(self):
        """Test replenishment integration initialization."""
        config = PlanningConfig()
        integration = ReplenishmentIntegration(config.replenishment)

        assert integration is not None

    def test_run(self):
        """Test running replenishment."""
        config = PlanningConfig()
        integration = ReplenishmentIntegration(config.replenishment)

        result = integration.run()

        assert result is not None
        assert result.automation_rate >= 0
        assert result.orders is not None
        assert isinstance(result.orders, pd.DataFrame)

    def test_run_scenario(self):
        """Test running replenishment for a scenario."""
        config = PlanningConfig()
        integration = ReplenishmentIntegration(config.replenishment)

        result = integration.run_scenario('dc_to_store', '2024-01-01')

        assert result is not None
        assert result.automation_rate >= 0

    def test_get_urgent_items(self):
        """Test getting urgent items."""
        config = PlanningConfig()
        integration = ReplenishmentIntegration(config.replenishment)

        # Run replenishment first
        integration.run()

        urgent = integration.get_urgent_items()

        assert urgent is not None
        assert isinstance(urgent, list)

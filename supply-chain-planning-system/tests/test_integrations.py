"""
Tests for Supply Chain Planning System integration modules.
"""

import pytest
import pandas as pd
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

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
        integration = DemandIntegration(config)
        
        assert integration is not None
    
    def test_run_forecast(self, sample_sales_data):
        """Test running demand forecast."""
        config = PlanningConfig()
        integration = DemandIntegration(config)
        
        result = integration.run_forecast(
            data=sample_sales_data,
            horizon_days=30
        )
        
        assert result is not None
        assert result.mape is not None
        assert 0 <= result.mape <= 1
    
    def test_get_forecast(self):
        """Test getting forecast results."""
        config = PlanningConfig()
        integration = DemandIntegration(config)
        
        # Run forecast first
        integration.run_forecast(horizon_days=30)
        
        forecast = integration.get_forecast()
        
        assert forecast is not None
        assert isinstance(forecast, pd.DataFrame)


class TestInventoryIntegration:
    """Tests for InventoryIntegration class."""
    
    def test_initialization(self):
        """Test inventory integration initialization."""
        config = PlanningConfig()
        integration = InventoryIntegration(config)
        
        assert integration is not None
    
    def test_run_optimization(self, sample_inventory_data):
        """Test running inventory optimization."""
        config = PlanningConfig()
        integration = InventoryIntegration(config)
        
        result = integration.run_optimization(
            data=sample_inventory_data,
            service_level=0.95
        )
        
        assert result is not None
        assert result.service_level >= 0
    
    def test_get_reorder_recommendations(self):
        """Test getting reorder recommendations."""
        config = PlanningConfig()
        integration = InventoryIntegration(config)
        
        # Run optimization first
        integration.run_optimization()
        
        recommendations = integration.get_reorder_recommendations()
        
        assert recommendations is not None
        assert isinstance(recommendations, pd.DataFrame)


class TestPricingIntegration:
    """Tests for PricingIntegration class."""
    
    def test_initialization(self):
        """Test pricing integration initialization."""
        config = PlanningConfig()
        integration = PricingIntegration(config)
        
        assert integration is not None
    
    def test_run_optimization(self, sample_pricing_data):
        """Test running pricing optimization."""
        config = PlanningConfig()
        integration = PricingIntegration(config)
        
        result = integration.run_optimization(
            data=sample_pricing_data
        )
        
        assert result is not None
        assert result.elasticity is not None
    
    def test_get_optimal_prices(self):
        """Test getting optimal prices."""
        config = PlanningConfig()
        integration = PricingIntegration(config)
        
        # Run optimization first
        integration.run_optimization()
        
        prices = integration.get_optimal_prices()
        
        assert prices is not None
        assert isinstance(prices, pd.DataFrame)


class TestNetworkIntegration:
    """Tests for NetworkIntegration class."""
    
    def test_initialization(self):
        """Test network integration initialization."""
        config = PlanningConfig()
        integration = NetworkIntegration(config)
        
        assert integration is not None
    
    def test_run_optimization(self, sample_network_data):
        """Test running network optimization."""
        config = PlanningConfig()
        integration = NetworkIntegration(config)
        
        result = integration.run_optimization(
            facilities=sample_network_data['facilities'],
            customers=sample_network_data['customers']
        )
        
        assert result is not None
        assert result.cost_reduction is not None
    
    def test_get_optimal_routes(self):
        """Test getting optimal routes."""
        config = PlanningConfig()
        integration = NetworkIntegration(config)
        
        # Run optimization first
        integration.run_optimization()
        
        routes = integration.get_optimal_routes()
        
        assert routes is not None


class TestSensingIntegration:
    """Tests for SensingIntegration class."""
    
    def test_initialization(self):
        """Test sensing integration initialization."""
        config = PlanningConfig()
        integration = SensingIntegration(config)
        
        assert integration is not None
    
    def test_run_sensing(self, sample_sales_data):
        """Test running real-time sensing."""
        config = PlanningConfig()
        integration = SensingIntegration(config)
        
        result = integration.run_sensing(
            data=sample_sales_data
        )
        
        assert result is not None
        assert result.anomaly_count >= 0
    
    def test_get_anomalies(self):
        """Test getting detected anomalies."""
        config = PlanningConfig()
        integration = SensingIntegration(config)
        
        # Run sensing first
        integration.run_sensing()
        
        anomalies = integration.get_anomalies()
        
        assert anomalies is not None
        assert isinstance(anomalies, list)


class TestReplenishmentIntegration:
    """Tests for ReplenishmentIntegration class."""
    
    def test_initialization(self):
        """Test replenishment integration initialization."""
        config = PlanningConfig()
        integration = ReplenishmentIntegration(config)
        
        assert integration is not None
    
    def test_run_replenishment(self, sample_inventory_data):
        """Test running replenishment."""
        config = PlanningConfig()
        integration = ReplenishmentIntegration(config)
        
        result = integration.run_replenishment(
            inventory_data=sample_inventory_data
        )
        
        assert result is not None
        assert result.automation_rate >= 0
    
    def test_get_orders(self):
        """Test getting generated orders."""
        config = PlanningConfig()
        integration = ReplenishmentIntegration(config)
        
        # Run replenishment first
        integration.run_replenishment()
        
        orders = integration.get_orders()
        
        assert orders is not None
        assert isinstance(orders, pd.DataFrame)
    
    def test_approve_order(self):
        """Test order approval."""
        config = PlanningConfig()
        integration = ReplenishmentIntegration(config)
        
        # Generate orders first
        integration.run_replenishment()
        
        result = integration.approve_order('ORD001')
        
        assert result is not None
        assert 'status' in result

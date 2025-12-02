"""Tests for the main replenishment engine."""

import pytest
import pandas as pd
import numpy as np

from src.engine.replenishment import ReplenishmentEngine, ReplenishmentResult


class TestReplenishmentEngine:
    """Tests for the main replenishment engine."""
    
    def test_initialize_with_config(self, sample_config):
        """Test engine initialization with config dict."""
        engine = ReplenishmentEngine(config_dict=sample_config)
        
        assert engine.config is not None
        assert "scenarios" in engine.config
    
    def test_run_basic(self, sample_config, sample_inventory_data):
        """Test basic engine run."""
        engine = ReplenishmentEngine(config_dict=sample_config)
        
        result = engine.run(
            scenario="test_scenario",
            inventory_data=sample_inventory_data,
        )
        
        assert isinstance(result, ReplenishmentResult)
        assert result.scenario == "test_scenario"
        assert not result.recommendations.empty
        assert result.summary is not None
    
    def test_run_with_demand_data(
        self, sample_config, sample_inventory_data, sample_demand_data
    ):
        """Test engine run with demand analytics."""
        engine = ReplenishmentEngine(config_dict=sample_config)
        
        result = engine.run(
            scenario="test_scenario",
            inventory_data=sample_inventory_data,
            demand_data=sample_demand_data,
        )
        
        assert isinstance(result, ReplenishmentResult)
        # Should have analytics results
        assert result.analytics is not None
    
    def test_run_with_source_inventory(
        self, sample_config, sample_inventory_data, sample_source_inventory
    ):
        """Test engine run with source availability."""
        engine = ReplenishmentEngine(config_dict=sample_config)
        
        result = engine.run(
            scenario="test_scenario",
            inventory_data=sample_inventory_data,
            source_inventory=sample_source_inventory,
        )
        
        assert isinstance(result, ReplenishmentResult)
        # Should have source_available in recommendations
        assert "source_available" in result.recommendations.columns
    
    def test_run_unknown_scenario(self, sample_config, sample_inventory_data):
        """Test engine run with unknown scenario uses defaults."""
        engine = ReplenishmentEngine(config_dict=sample_config)
        
        result = engine.run(
            scenario="unknown_scenario",
            inventory_data=sample_inventory_data,
        )
        
        # Should still work with defaults
        assert isinstance(result, ReplenishmentResult)
    
    def test_result_to_dict(self, sample_config, sample_inventory_data):
        """Test result serialization."""
        engine = ReplenishmentEngine(config_dict=sample_config)
        
        result = engine.run(
            scenario="test_scenario",
            inventory_data=sample_inventory_data,
        )
        
        result_dict = result.to_dict()
        
        assert "scenario" in result_dict
        assert "timestamp" in result_dict
        assert "summary" in result_dict
    
    def test_get_priority_orders(self, sample_config, sample_inventory_data):
        """Test getting priority orders."""
        engine = ReplenishmentEngine(config_dict=sample_config)
        
        result = engine.run(
            scenario="test_scenario",
            inventory_data=sample_inventory_data,
        )
        
        priority = result.get_priority_orders(top_n=5)
        
        assert len(priority) <= 5
    
    def test_validate_scenario(self, sample_config):
        """Test scenario validation."""
        engine = ReplenishmentEngine(config_dict=sample_config)
        
        validation = engine.validate_scenario("test_scenario")
        
        assert "valid" in validation
        assert "issues" in validation
        assert "warnings" in validation
    
    def test_get_available_scenarios(self, sample_config):
        """Test listing available scenarios."""
        engine = ReplenishmentEngine(config_dict=sample_config)
        
        scenarios = engine.get_available_scenarios()
        
        assert "test_scenario" in scenarios


class TestReplenishmentResult:
    """Tests for ReplenishmentResult dataclass."""
    
    def test_create_result(self):
        """Test creating a result object."""
        result = ReplenishmentResult(
            scenario="test",
            recommendations=pd.DataFrame({"item_id": ["SKU001"]}),
            alerts=[],
            summary={"total_items": 1},
        )
        
        assert result.scenario == "test"
        assert len(result.recommendations) == 1
        assert result.summary["total_items"] == 1
    
    def test_get_priority_orders_empty(self):
        """Test priority orders with empty recommendations."""
        result = ReplenishmentResult(
            scenario="test",
            recommendations=pd.DataFrame(),
            alerts=[],
            summary={},
        )
        
        priority = result.get_priority_orders()
        
        assert priority.empty

"""Tests for replenishment policies."""

import pytest
import pandas as pd
import numpy as np

from src.policies.periodic_review import PeriodicReviewPolicy
from src.policies.continuous_review import ContinuousReviewPolicy
from src.policies.min_max import MinMaxPolicy
from src.safety_stock.calculator import SafetyStockCalculator


class TestPeriodicReviewPolicy:
    """Tests for Periodic Review (s,S) policy."""
    
    def test_calculate_basic(self, sample_inventory_data):
        """Test basic policy calculation."""
        policy = PeriodicReviewPolicy(
            review_period=7,
            lead_time=14,
            service_level=0.95,
        )
        
        result = policy.calculate(sample_inventory_data)
        
        assert "reorder_point" in result.columns
        assert "order_up_to_level" in result.columns
        assert "safety_stock" in result.columns
        assert "recommended_quantity" in result.columns
        assert "needs_order" in result.columns
    
    def test_reorder_point_formula(self):
        """Test that reorder point follows correct formula."""
        policy = PeriodicReviewPolicy(
            review_period=7,
            lead_time=14,
            service_level=0.95,
        )
        
        # Create simple test data
        test_data = pd.DataFrame({
            "item_id": ["SKU001"],
            "current_stock": [100],
            "daily_demand_rate": [10],
            "demand_std": [2],
        })
        
        result = policy.calculate(test_data)
        
        # s = DDR × LT + SS
        expected_s = 10 * 14 + result["safety_stock"].iloc[0]
        assert abs(result["reorder_point"].iloc[0] - expected_s) < 0.01
    
    def test_order_up_to_formula(self):
        """Test that order-up-to level follows correct formula."""
        policy = PeriodicReviewPolicy(
            review_period=7,
            lead_time=14,
            service_level=0.95,
        )
        
        test_data = pd.DataFrame({
            "item_id": ["SKU001"],
            "current_stock": [50],
            "daily_demand_rate": [10],
            "demand_std": [2],
        })
        
        result = policy.calculate(test_data)
        
        # S = DDR × (LT + RP) + SS
        expected_S = 10 * (14 + 7) + result["safety_stock"].iloc[0]
        assert abs(result["order_up_to_level"].iloc[0] - expected_S) < 0.01
    
    def test_needs_order_detection(self):
        """Test that items below reorder point are flagged."""
        policy = PeriodicReviewPolicy(
            review_period=7,
            lead_time=7,
            service_level=0.95,
        )
        
        test_data = pd.DataFrame({
            "item_id": ["SKU001", "SKU002"],
            "current_stock": [10, 1000],  # Low and high stock
            "daily_demand_rate": [10, 10],
            "demand_std": [2, 2],
        })
        
        result = policy.calculate(test_data)
        
        # First item should need order (low stock)
        assert result.loc[0, "needs_order"] == True
        # Second item should not need order (high stock)
        assert result.loc[1, "needs_order"] == False
    
    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        policy = PeriodicReviewPolicy()
        result = policy.calculate(pd.DataFrame())
        
        assert result.empty


class TestContinuousReviewPolicy:
    """Tests for Continuous Review (s,Q) policy."""
    
    def test_calculate_basic(self, sample_inventory_data):
        """Test basic policy calculation."""
        policy = ContinuousReviewPolicy(
            lead_time=7,
            service_level=0.95,
            ordering_cost=50,
            holding_cost_rate=0.25,
        )
        
        result = policy.calculate(sample_inventory_data)
        
        assert "reorder_point" in result.columns
        assert "order_quantity_Q" in result.columns
        assert "safety_stock" in result.columns
    
    def test_eoq_calculation(self):
        """Test EOQ calculation."""
        policy = ContinuousReviewPolicy(
            lead_time=7,
            ordering_cost=50,
            holding_cost_rate=0.25,
            use_eoq=True,
        )
        
        test_data = pd.DataFrame({
            "item_id": ["SKU001"],
            "current_stock": [100],
            "daily_demand_rate": [10],
            "demand_std": [2],
            "unit_cost": [20],
        })
        
        result = policy.calculate(test_data)
        
        # EOQ = sqrt(2DS/H)
        annual_demand = 10 * 365
        holding_cost = 20 * 0.25
        expected_eoq = np.sqrt(2 * annual_demand * 50 / holding_cost)
        
        # Allow some tolerance
        assert abs(result["order_quantity_Q"].iloc[0] - expected_eoq) < 1


class TestMinMaxPolicy:
    """Tests for Min-Max policy."""
    
    def test_calculate_basic(self, sample_inventory_data):
        """Test basic policy calculation."""
        policy = MinMaxPolicy(
            min_days_supply=7,
            max_days_supply=21,
        )
        
        result = policy.calculate(sample_inventory_data)
        
        assert "min_level" in result.columns
        assert "max_level" in result.columns
        assert "needs_order" in result.columns
    
    def test_min_max_levels(self):
        """Test min/max level calculation."""
        policy = MinMaxPolicy(
            min_days_supply=7,
            max_days_supply=21,
        )
        
        test_data = pd.DataFrame({
            "item_id": ["SKU001"],
            "current_stock": [50],
            "daily_demand_rate": [10],
        })
        
        result = policy.calculate(test_data)
        
        # Min = DDR × min_days = 10 × 7 = 70
        assert result["min_level"].iloc[0] == 70
        # Max = DDR × max_days = 10 × 21 = 210
        assert result["max_level"].iloc[0] == 210


class TestSafetyStockCalculator:
    """Tests for safety stock calculations."""
    
    def test_standard_method(self):
        """Test standard safety stock calculation."""
        calculator = SafetyStockCalculator(method="standard")
        
        ss = calculator.calculate(
            demand_mean=100,
            demand_std=20,
            lead_time=7,
            service_level=0.95,
        )
        
        # SS = Z × σ × √LT
        # Z(0.95) ≈ 1.65
        expected_ss = 1.65 * 20 * np.sqrt(7)
        
        assert abs(ss - expected_ss) < 1
    
    def test_dynamic_method(self):
        """Test dynamic safety stock with lead time variability."""
        calculator = SafetyStockCalculator(method="dynamic")
        
        ss = calculator.calculate(
            demand_mean=100,
            demand_std=20,
            lead_time=7,
            service_level=0.95,
            lead_time_std=1,
        )
        
        # Dynamic SS should be higher than standard due to LT variability
        standard_calc = SafetyStockCalculator(method="standard")
        standard_ss = standard_calc.calculate(
            demand_mean=100,
            demand_std=20,
            lead_time=7,
            service_level=0.95,
        )
        
        assert ss > standard_ss
    
    def test_z_score_lookup(self):
        """Test Z-score lookup for common service levels."""
        calculator = SafetyStockCalculator()
        
        # Test known Z-scores
        assert abs(calculator._get_z_score(0.95) - 1.65) < 0.01
        assert abs(calculator._get_z_score(0.99) - 2.33) < 0.01
    
    def test_min_max_constraints(self):
        """Test min/max safety stock constraints."""
        calculator = SafetyStockCalculator(
            method="standard",
            min_safety_stock=50,
            max_safety_stock=500,
        )
        
        # Test minimum constraint
        ss_low = calculator.calculate(
            demand_mean=10,
            demand_std=1,
            lead_time=1,
            service_level=0.90,
        )
        assert ss_low >= 50
        
        # Test maximum constraint
        ss_high = calculator.calculate(
            demand_mean=1000,
            demand_std=500,
            lead_time=30,
            service_level=0.99,
        )
        assert ss_high <= 500

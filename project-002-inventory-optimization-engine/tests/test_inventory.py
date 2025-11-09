"""Tests for inventory calculations."""

import pytest
import pandas as pd
import numpy as np
from src.inventory.safety_stock import SafetyStockCalculator
from src.inventory.reorder_point import ReorderPointCalculator
from src.inventory.eoq import EOQCalculator


class TestSafetyStock:
    """Tests for SafetyStockCalculator."""
    
    def test_basic_safety_stock(self):
        """Test basic safety stock calculation."""
        calc = SafetyStockCalculator(service_level=0.95, lead_time=7)
        ss = calc.calculate_basic_safety_stock(demand_std=5.0)
        
        assert ss > 0
        assert isinstance(ss, float)
    
    def test_safety_stock_by_service_level(self):
        """Test safety stock for multiple service levels."""
        calc = SafetyStockCalculator()
        results = calc.calculate_by_service_level(demand_std=5.0)
        
        assert '95%' in results
        assert results['99%'] > results['95%']
        assert results['95%'] > results['90%']


class TestReorderPoint:
    """Tests for ReorderPointCalculator."""
    
    def test_basic_reorder_point(self):
        """Test basic reorder point calculation."""
        calc = ReorderPointCalculator(lead_time=7)
        rop = calc.calculate_reorder_point(
            demand_mean=10.0,
            safety_stock=15.0
        )
        
        assert rop == 10.0 * 7 + 15.0
    
    def test_time_to_reorder(self):
        """Test days until reorder calculation."""
        calc = ReorderPointCalculator()
        days = calc.calculate_time_to_reorder(
            current_inventory=100,
            reorder_point=50,
            demand_mean=10
        )
        
        assert days == 5.0  # (100-50)/10


class TestEOQ:
    """Tests for EOQCalculator."""
    
    def test_basic_eoq(self):
        """Test basic EOQ calculation."""
        calc = EOQCalculator(ordering_cost=100, holding_cost_rate=0.25)
        eoq = calc.calculate_eoq(
            annual_demand=1000,
            unit_cost=10
        )
        
        assert eoq > 0
        expected = np.sqrt((2 * 1000 * 100) / (10 * 0.25))
        assert abs(eoq - expected) < 0.01
    
    def test_total_cost(self):
        """Test total cost calculation."""
        calc = EOQCalculator(ordering_cost=100, holding_cost_rate=0.25)
        costs = calc.calculate_total_cost(
            order_quantity=200,
            annual_demand=1000,
            unit_cost=10
        )
        
        assert 'ordering_cost' in costs
        assert 'holding_cost' in costs
        assert 'total_cost' in costs
        assert costs['total_cost'] > 0

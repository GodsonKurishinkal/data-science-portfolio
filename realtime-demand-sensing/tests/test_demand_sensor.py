"""
Tests for Demand Sensor module.

Tests the DemandSensor class for processing streaming demand data
and maintaining state for real-time analysis.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.sensing.demand_sensor import DemandSensor, DemandSensorBatch


class TestDemandSensor:
    """Tests for DemandSensor class."""
    
    @pytest.fixture
    def sample_historical(self):
        """Create sample historical data."""
        np.random.seed(42)
        n = 168  # 7 days hourly
        base_time = datetime.now() - timedelta(hours=n)
        
        return pd.DataFrame({
            'timestamp': [base_time + timedelta(hours=i) for i in range(n)],
            'product_id': ['PROD_001'] * n,
            'sales': np.random.normal(100, 15, n).clip(0)
        })
    
    def test_initialization(self):
        """Test DemandSensor initializes correctly."""
        sensor = DemandSensor(
            alpha=0.3,
            drift_threshold=2.0
        )
        
        assert sensor.alpha == 0.3
        assert sensor.drift_threshold == 2.0
    
    def test_initialization_with_custom_params(self):
        """Test initialization with custom parameters."""
        sensor = DemandSensor(
            alpha=0.5,
            drift_threshold=3.0,
            lookback_hours=336,
            min_observations=48
        )
        
        assert sensor.alpha == 0.5
        assert sensor.drift_threshold == 3.0
        assert sensor.lookback_hours == 336
        assert sensor.min_observations == 48
    
    def test_invalid_alpha(self):
        """Test that invalid alpha raises error."""
        with pytest.raises(ValueError):
            DemandSensor(alpha=0)
        
        with pytest.raises(ValueError):
            DemandSensor(alpha=1.5)
    
    def test_initialize_baseline(self, sample_historical):
        """Test initializing baseline from historical data."""
        sensor = DemandSensor(alpha=0.3)
        sensor.initialize_baseline(sample_historical)
        
        # Check baseline was set
        baseline = sensor.get_baseline('PROD_001')
        assert baseline is not None
        assert 'mean' in baseline
        assert 'std' in baseline
    
    def test_update_with_single_value(self, sample_historical):
        """Test updating sensor with a single observation."""
        sensor = DemandSensor(alpha=0.3)
        sensor.initialize_baseline(sample_historical)
        
        result = sensor.update(
            product_id='PROD_001',
            sales=100.0
        )
        
        assert 'smoothed_demand' in result
        assert 'baseline_demand' in result
        assert 'z_score' in result
        assert 'is_drift' in result
    
    def test_update_with_multiple_values(self, sample_historical):
        """Test updating sensor with multiple observations."""
        sensor = DemandSensor(alpha=0.3)
        sensor.initialize_baseline(sample_historical)
        
        results = []
        for i in range(10):
            result = sensor.update(
                product_id='PROD_001',
                sales=100 + i * 5  # Increasing values
            )
            results.append(result)
        
        # Smoothed demand should reflect trend
        assert len(results) == 10
        # Each result should be valid
        for result in results:
            assert 'smoothed_demand' in result
    
    def test_drift_detection(self, sample_historical):
        """Test drift detection with extreme value."""
        sensor = DemandSensor(alpha=0.3, drift_threshold=2.0)
        sensor.initialize_baseline(sample_historical)
        
        # Large value should trigger drift
        result = sensor.update(
            product_id='PROD_001',
            sales=300  # Very high compared to ~100 baseline
        )
        
        # After sufficient observations, drift should be detected
        # Need to accumulate observations first
        for _ in range(30):
            sensor.update('PROD_001', 300)
        
        result = sensor.update('PROD_001', 300)
        
        # Check drift detection after observations threshold
        assert result['z_score'] > 2.0 or result['is_drift']
    
    def test_get_current_demand(self, sample_historical):
        """Test getting current demand."""
        sensor = DemandSensor(alpha=0.3)
        sensor.initialize_baseline(sample_historical)
        
        sensor.update('PROD_001', 120)
        
        demand = sensor.get_current_demand('PROD_001')
        assert demand is not None
        assert isinstance(demand, float)
    
    def test_get_all_demands(self, sample_historical):
        """Test getting all current demands."""
        sensor = DemandSensor(alpha=0.3)
        sensor.initialize_baseline(sample_historical)
        
        demands = sensor.get_all_demands()
        
        assert isinstance(demands, dict)
        assert 'PROD_001' in demands
    
    def test_status_summary(self, sample_historical):
        """Test getting status summary."""
        sensor = DemandSensor(alpha=0.3)
        sensor.initialize_baseline(sample_historical)
        
        summary = sensor.get_status_summary()
        
        assert isinstance(summary, pd.DataFrame)
        assert 'product_id' in summary.columns
        assert 'current_demand' in summary.columns
        assert 'status' in summary.columns
    
    def test_new_product_initialization(self):
        """Test handling of new product without history."""
        sensor = DemandSensor(alpha=0.3)
        
        # Update with new product (no baseline)
        result = sensor.update(
            product_id='NEW_PROD',
            sales=100.0
        )
        
        assert result['product_id'] == 'NEW_PROD'
        assert result['smoothed_demand'] == 100.0
    
    def test_update_baseline(self, sample_historical):
        """Test updating baseline."""
        sensor = DemandSensor(alpha=0.3)
        sensor.initialize_baseline(sample_historical)
        
        # Update several times
        for _ in range(50):
            sensor.update('PROD_001', 150)
        
        # Recalculate baseline
        new_baseline = sensor.update_baseline('PROD_001', recalculate=True)
        
        assert 'mean' in new_baseline
        # Mean should have shifted toward 150
        assert new_baseline['mean'] > 100


class TestDemandSensorBatch:
    """Tests for DemandSensorBatch class."""
    
    @pytest.fixture
    def sample_historical(self):
        """Create sample historical data for multiple products."""
        np.random.seed(42)
        n = 168  # 7 days hourly
        base_time = datetime.now() - timedelta(hours=n)
        
        records = []
        for product in ['PROD_001', 'PROD_002', 'PROD_003']:
            for i in range(n):
                records.append({
                    'timestamp': base_time + timedelta(hours=i),
                    'product_id': product,
                    'sales': np.random.normal(100, 15, 1)[0].clip(0)
                })
        
        return pd.DataFrame(records)
    
    def test_initialization(self):
        """Test DemandSensorBatch initialization."""
        batch_sensor = DemandSensorBatch(alpha=0.3)
        assert batch_sensor is not None
    
    def test_initialize_from_historical(self, sample_historical):
        """Test initializing from historical data."""
        batch_sensor = DemandSensorBatch(alpha=0.3)
        batch_sensor.initialize(sample_historical)
        
        # Check all products have baselines
        for product in ['PROD_001', 'PROD_002', 'PROD_003']:
            baseline = batch_sensor.sensor.get_baseline(product)
            assert baseline is not None
    
    def test_update_batch(self, sample_historical):
        """Test batch update."""
        batch_sensor = DemandSensorBatch(alpha=0.3)
        batch_sensor.initialize(sample_historical)
        
        # Create batch data
        batch_data = pd.DataFrame({
            'timestamp': [datetime.now()] * 3,
            'product_id': ['PROD_001', 'PROD_002', 'PROD_003'],
            'sales': [100, 110, 90]
        })
        
        results = batch_sensor.update_batch(batch_data)
        
        assert isinstance(results, pd.DataFrame)
        assert len(results) == 3


class TestDemandSensorEdgeCases:
    """Test edge cases for DemandSensor."""
    
    def test_zero_sales(self):
        """Test handling of zero sales."""
        sensor = DemandSensor(alpha=0.3)
        
        result = sensor.update('TEST', sales=0.0)
        
        assert result['smoothed_demand'] == 0.0
    
    def test_very_large_values(self):
        """Test handling of very large values."""
        sensor = DemandSensor(alpha=0.3)
        
        result = sensor.update('TEST', sales=1e6)
        
        assert result['smoothed_demand'] == 1e6
    
    def test_consecutive_updates(self):
        """Test many consecutive updates."""
        sensor = DemandSensor(alpha=0.3, lookback_hours=100)
        
        # Initialize with baseline
        sensor.update('TEST', 100)
        
        # Many updates
        for i in range(200):
            sensor.update('TEST', 100 + np.random.randn() * 10)
        
        # Should still work
        state = sensor.get_current_demand('TEST')
        assert state is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

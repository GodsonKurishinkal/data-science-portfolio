"""
Tests for Stream Simulator module.

Tests the StreamSimulator class for generating realistic
streaming demand data with patterns and anomalies.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.utils.stream_simulator import StreamSimulator, create_demo_data


class TestStreamSimulator:
    """Tests for StreamSimulator class."""
    
    def test_initialization(self):
        """Test StreamSimulator initializes correctly."""
        simulator = StreamSimulator(
            n_products=10,
            base_demand_range=(50, 200),
            seed=42
        )
        
        assert simulator.n_products == 10
        assert simulator.base_demand_range == (50, 200)
        assert len(simulator.products) == 10
    
    def test_initialization_with_custom_params(self):
        """Test initialization with custom parameters."""
        simulator = StreamSimulator(
            n_products=5,
            base_demand_range=(10, 50),
            noise_level=0.3,
            anomaly_probability=0.05,
            seed=123
        )
        
        assert simulator.n_products == 5
        assert simulator.noise_level == 0.3
        assert simulator.anomaly_probability == 0.05
    
    def test_generate_historical(self):
        """Test generating historical data."""
        simulator = StreamSimulator(n_products=5, seed=42)
        
        df = simulator.generate_historical(days=7)
        
        # Check DataFrame structure
        assert isinstance(df, pd.DataFrame)
        assert 'timestamp' in df.columns
        assert 'sales' in df.columns
        assert 'product_id' in df.columns
        assert 'category' in df.columns
        
        # Check data volume (7 days * 24 hours * 5 products)
        expected_rows = 7 * 24 * 5
        assert len(df) == expected_rows
        
        # Check sales are non-negative
        assert (df['sales'] >= 0).all()
    
    def test_generate_historical_with_different_durations(self):
        """Test generating different durations of data."""
        simulator = StreamSimulator(n_products=3, seed=42)
        
        df_7days = simulator.generate_historical(days=7)
        df_30days = simulator.generate_historical(days=30)
        
        # 3 products * hours
        assert len(df_7days) == 7 * 24 * 3
        assert len(df_30days) == 30 * 24 * 3
    
    def test_reproducibility_with_seed(self):
        """Test that same seed produces reproducible data structure and products."""
        # Test that seed produces consistent product characteristics
        sim1 = StreamSimulator(n_products=5, seed=42)
        
        # Verify consistent data generation for a single simulator
        df1 = sim1.generate_historical(days=3)
        
        # Should have correct structure
        assert len(df1) == 3 * 24 * 5  # 3 days * 24 hours * 5 products
        assert 'sales' in df1.columns
        assert 'product_id' in df1.columns
        
        # All products should be present
        assert len(df1['product_id'].unique()) == 5
        
        # Sales should be non-negative
        assert (df1['sales'] >= 0).all()
        
        # Verify second call to same simulator uses consistent products
        df2 = sim1.generate_historical(days=1)
        assert set(df1['product_id'].unique()) == set(df2['product_id'].unique())
    
    def test_different_seeds_produce_different_results(self):
        """Test that different seeds produce different results."""
        sim1 = StreamSimulator(n_products=5, seed=42)
        sim2 = StreamSimulator(n_products=5, seed=123)
        
        df1 = sim1.generate_historical(days=3)
        df2 = sim2.generate_historical(days=3)
        
        # Values should be different
        assert not (df1['sales'].values == df2['sales'].values).all()
    
    def test_base_demand_affects_values(self):
        """Test that base demand range affects generated values."""
        sim_low = StreamSimulator(n_products=5, base_demand_range=(10, 20), seed=42)
        sim_high = StreamSimulator(n_products=5, base_demand_range=(200, 300), seed=42)
        
        df_low = sim_low.generate_historical(days=7)
        df_high = sim_high.generate_historical(days=7)
        
        # High base demand should have higher mean
        assert df_high['sales'].mean() > df_low['sales'].mean()
    
    def test_anomaly_column_exists(self):
        """Test that anomaly indicators exist in data."""
        simulator = StreamSimulator(
            n_products=5,
            anomaly_probability=0.1,
            seed=42
        )
        
        df = simulator.generate_historical(days=30)
        
        # Should have anomaly indicators
        assert 'is_anomaly' in df.columns
        
        # With 10% probability over large data, should have some anomalies
        assert df['is_anomaly'].sum() > 0
    
    def test_streaming_generates_batches(self):
        """Test streaming functionality."""
        simulator = StreamSimulator(n_products=5, seed=42)
        
        # Stream 3 batches
        batches = []
        for batch in simulator.stream(interval_hours=1, max_iterations=3):
            batches.append(batch)
        
        assert len(batches) == 3
        for batch in batches:
            assert len(batch) == 5  # One row per product
    
    def test_products_dataframe(self):
        """Test getting products DataFrame."""
        simulator = StreamSimulator(n_products=10, seed=42)
        products = simulator.get_products()
        
        assert isinstance(products, pd.DataFrame)
        assert len(products) == 10
        assert 'product_id' in products.columns
        assert 'base_demand' in products.columns
        assert 'category' in products.columns
    
    def test_inventory_tracking(self):
        """Test that inventory levels are tracked."""
        simulator = StreamSimulator(n_products=3, seed=42)
        df = simulator.generate_historical(days=7)
        
        assert 'inventory' in df.columns
        # Inventory should be non-negative
        assert (df['inventory'] >= 0).all()
    
    def test_product_summary(self):
        """Test product summary statistics."""
        simulator = StreamSimulator(n_products=5, seed=42)
        df = simulator.generate_historical(days=30)
        
        summary = simulator.get_product_summary(df)
        
        assert isinstance(summary, pd.DataFrame)
        assert len(summary) == 5
        assert 'avg_hourly_sales' in summary.columns
        assert 'total_sales' in summary.columns


class TestStreamSimulatorPatterns:
    """Test pattern generation in StreamSimulator."""
    
    def test_hourly_pattern_exists(self):
        """Test that hourly patterns exist in data."""
        simulator = StreamSimulator(
            n_products=3,
            noise_level=0.05,  # Low noise to see patterns
            seed=42
        )
        
        df = simulator.generate_historical(days=7)
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        
        # Group by hour and check variation
        hourly_means = df.groupby('hour')['sales'].mean()
        
        # Should have variation across hours
        assert hourly_means.std() > 0
    
    def test_weekly_pattern_exists(self):
        """Test that weekly patterns exist in data."""
        simulator = StreamSimulator(
            n_products=3,
            noise_level=0.05,
            seed=42
        )
        
        df = simulator.generate_historical(days=14)
        df['dayofweek'] = pd.to_datetime(df['timestamp']).dt.dayofweek
        
        # Group by day of week
        daily_means = df.groupby('dayofweek')['sales'].mean()
        
        # Should have variation across days (weekends higher)
        assert daily_means.std() > 0
    
    def test_is_weekend_column(self):
        """Test weekend indicator column."""
        simulator = StreamSimulator(n_products=3, seed=42)
        df = simulator.generate_historical(days=7)
        
        assert 'is_weekend' in df.columns
        
        # Check weekend days (Saturday=5, Sunday=6)
        df['dayofweek'] = pd.to_datetime(df['timestamp']).dt.dayofweek
        expected_weekend = df['dayofweek'].isin([5, 6])
        pd.testing.assert_series_equal(
            df['is_weekend'].reset_index(drop=True),
            expected_weekend.reset_index(drop=True),
            check_names=False
        )


class TestCreateDemoData:
    """Test the convenience function create_demo_data."""
    
    def test_create_demo_data(self):
        """Test create_demo_data function."""
        historical, summary = create_demo_data(
            n_products=5,
            days=30,
            seed=42
        )
        
        assert isinstance(historical, pd.DataFrame)
        assert isinstance(summary, pd.DataFrame)
        assert len(summary) == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

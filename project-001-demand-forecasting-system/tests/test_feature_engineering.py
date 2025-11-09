"""
Unit tests for feature engineering module.
"""

import pytest
import pandas as pd
import numpy as np
from src.features.build_features import (
    create_time_features,
    create_lag_features,
    create_rolling_features,
    create_all_features
)


@pytest.fixture
def time_series_data():
    """Create sample time series data for testing."""
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    df = pd.DataFrame({
        'demand': np.random.randint(50, 200, 100)
    }, index=dates)
    return df


@pytest.fixture
def time_series_data_with_column():
    """Create sample time series data with date as column."""
    return pd.DataFrame({
        'date': pd.date_range('2023-01-01', periods=100, freq='D'),
        'demand': np.random.randint(50, 200, 100)
    })


class TestCreateTimeFeatures:
    """Test cases for create_time_features function."""
    
    def test_creates_time_features(self, time_series_data):
        """Test that time features are created."""
        df_features = create_time_features(time_series_data)
        
        expected_features = ['year', 'month', 'day', 'dayofweek', 'quarter', 
                           'weekofyear', 'is_weekend', 'is_month_start', 'is_month_end']
        for feature in expected_features:
            assert feature in df_features.columns
    
    def test_time_features_with_date_column(self, time_series_data_with_column):
        """Test time features creation with date column."""
        df_features = create_time_features(time_series_data_with_column, date_column='date')
        assert 'year' in df_features.columns
        assert 'month' in df_features.columns
    
    def test_weekend_feature_correct(self, time_series_data):
        """Test that weekend feature is correctly calculated."""
        df_features = create_time_features(time_series_data)
        # Saturday is 5, Sunday is 6
        saturdays_sundays = df_features.index.dayofweek.isin([5, 6])
        assert (df_features['is_weekend'] == saturdays_sundays.astype(int)).all()
    
    def test_preserves_original_data(self, time_series_data):
        """Test that original columns are preserved."""
        original_cols = time_series_data.columns.tolist()
        df_features = create_time_features(time_series_data)
        assert all(col in df_features.columns for col in original_cols)


class TestCreateLagFeatures:
    """Test cases for create_lag_features function."""
    
    def test_creates_lag_features(self, time_series_data):
        """Test that lag features are created."""
        lags = [1, 7, 14]
        df_lags = create_lag_features(time_series_data, 'demand', lags=lags)
        
        for lag in lags:
            assert f'demand_lag_{lag}' in df_lags.columns
    
    def test_lag_values_correct(self, time_series_data):
        """Test that lag values are calculated correctly."""
        df_lags = create_lag_features(time_series_data, 'demand', lags=[1])
        
        # Check that lag_1 matches previous day (where not NaN)
        for i in range(1, len(df_lags)):
            if not pd.isna(df_lags['demand_lag_1'].iloc[i]):
                assert df_lags['demand_lag_1'].iloc[i] == df_lags['demand'].iloc[i-1]
    
    def test_raises_on_invalid_column(self, time_series_data):
        """Test that error is raised for invalid target column."""
        with pytest.raises(ValueError, match="Target column.*not found"):
            create_lag_features(time_series_data, 'nonexistent_column', lags=[1])


class TestCreateRollingFeatures:
    """Test cases for create_rolling_features function."""
    
    def test_creates_rolling_features(self, time_series_data):
        """Test that rolling features are created."""
        windows = [7, 14]
        df_rolling = create_rolling_features(time_series_data, 'demand', windows=windows)
        
        for window in windows:
            assert f'demand_rolling_mean_{window}' in df_rolling.columns
            assert f'demand_rolling_std_{window}' in df_rolling.columns
            assert f'demand_rolling_min_{window}' in df_rolling.columns
            assert f'demand_rolling_max_{window}' in df_rolling.columns
    
    def test_rolling_mean_calculation(self, time_series_data):
        """Test that rolling mean is calculated correctly."""
        df_rolling = create_rolling_features(time_series_data, 'demand', windows=[3])
        
        # Manually calculate rolling mean for position 3 and compare
        idx = 3
        expected_mean = time_series_data['demand'].iloc[1:4].mean()
        actual_mean = df_rolling['demand_rolling_mean_3'].iloc[idx]
        
        if not pd.isna(actual_mean):
            assert abs(expected_mean - actual_mean) < 0.01


class TestCreateAllFeatures:
    """Test cases for create_all_features function."""
    
    def test_creates_all_feature_types(self, time_series_data):
        """Test that all feature types are created."""
        df_all = create_all_features(time_series_data, 'demand', lags=[1, 7], windows=[7])
        
        # Check for time features
        assert 'year' in df_all.columns
        assert 'month' in df_all.columns
        
        # Check for lag features
        assert 'demand_lag_1' in df_all.columns
        assert 'demand_lag_7' in df_all.columns
        
        # Check for rolling features
        assert 'demand_rolling_mean_7' in df_all.columns
    
    def test_feature_engineering_integration(self, time_series_data):
        """Test complete feature engineering pipeline."""
        df_all = create_all_features(
            time_series_data,
            'demand',
            lags=[1, 7],
            windows=[7]
        )
        
        # Should have original + time + lag + rolling features
        assert len(df_all.columns) > len(time_series_data.columns)
        
        # Original data should be preserved
        assert 'demand' in df_all.columns

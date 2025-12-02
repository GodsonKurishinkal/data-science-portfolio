"""
Tests for Short-Term Forecasting module.

Tests the forecasting classes for generating short-term
demand forecasts using multiple methods.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.forecasting.short_term import (
    ShortTermForecaster,
    EnsembleForecaster,
    EWMForecaster,
    MovingAverageForecaster,
    NaiveForecaster,
    Forecast
)


class TestForecast:
    """Tests for Forecast dataclass."""
    
    def test_forecast_creation(self):
        """Test creating a Forecast object."""
        forecast = Forecast(
            timestamp=datetime.now(),
            value=100.0,
            lower_bound=90.0,
            upper_bound=110.0,
            horizon=1,
            model='test'
        )
        
        assert forecast.value == 100.0
        assert forecast.lower_bound == 90.0
        assert forecast.upper_bound == 110.0
        assert forecast.horizon == 1
        assert forecast.model == 'test'
    
    def test_forecast_with_confidence(self):
        """Test forecast with confidence level."""
        forecast = Forecast(
            timestamp=datetime.now(),
            value=100.0,
            lower_bound=90.0,
            upper_bound=110.0,
            horizon=1,
            model='test',
            confidence=0.95
        )
        
        assert forecast.value == 100.0
        assert forecast.confidence == 0.95


class TestNaiveForecaster:
    """Tests for NaiveForecaster class."""
    
    @pytest.fixture
    def sample_history(self):
        """Create sample historical data."""
        n = 48  # 2 days of hourly data
        base_time = datetime.now() - timedelta(hours=n)
        
        return pd.DataFrame({
            'timestamp': [base_time + timedelta(hours=i) for i in range(n)],
            'value': [100 + np.sin(i/4) * 10 for i in range(n)]  # Some pattern
        })
    
    def test_initialization(self):
        """Test NaiveForecaster initializes correctly."""
        forecaster = NaiveForecaster()
        assert forecaster is not None
    
    def test_fit(self, sample_history):
        """Test fitting forecaster."""
        forecaster = NaiveForecaster()
        result = forecaster.fit(sample_history)
        assert result is forecaster  # Returns self for chaining
    
    def test_predict(self, sample_history):
        """Test generating predictions."""
        forecaster = NaiveForecaster()
        forecaster.fit(sample_history)
        
        forecasts = forecaster.predict(horizon=6)
        
        assert len(forecasts) == 6
        for f in forecasts:
            assert isinstance(f, Forecast)
            assert f.value > 0
            assert f.model == 'naive'
    
    def test_predict_uses_last_value(self, sample_history):
        """Test naive forecaster uses last value."""
        forecaster = NaiveForecaster()
        forecaster.fit(sample_history)
        
        last_value = sample_history['value'].iloc[-1]
        forecasts = forecaster.predict(horizon=3)
        
        # All forecasts should be equal to last value
        for f in forecasts:
            assert f.value == last_value


class TestMovingAverageForecaster:
    """Tests for MovingAverageForecaster class."""
    
    @pytest.fixture
    def sample_history(self):
        """Create sample historical data."""
        n = 100
        base_time = datetime.now() - timedelta(hours=n)
        np.random.seed(42)
        
        return pd.DataFrame({
            'timestamp': [base_time + timedelta(hours=i) for i in range(n)],
            'value': 100 + np.cumsum(np.random.randn(n) * 2)
        })
    
    def test_initialization(self):
        """Test MovingAverageForecaster initializes correctly."""
        forecaster = MovingAverageForecaster(window=24)
        assert forecaster.window == 24
    
    def test_fit(self, sample_history):
        """Test fitting forecaster."""
        forecaster = MovingAverageForecaster(window=24)
        result = forecaster.fit(sample_history)
        assert result is forecaster  # Returns self for chaining
    
    def test_predict(self, sample_history):
        """Test generating predictions."""
        forecaster = MovingAverageForecaster(window=24)
        forecaster.fit(sample_history)
        
        forecasts = forecaster.predict(horizon=12)
        
        assert len(forecasts) == 12
        for f in forecasts:
            assert isinstance(f, Forecast)
            assert f.model == 'moving_average'
    
    def test_different_windows(self, sample_history):
        """Test with different window sizes."""
        forecaster_short = MovingAverageForecaster(window=12)
        forecaster_long = MovingAverageForecaster(window=48)
        
        forecaster_short.fit(sample_history)
        forecaster_long.fit(sample_history)
        
        fc_short = forecaster_short.predict(horizon=6)
        fc_long = forecaster_long.predict(horizon=6)
        
        # Both should produce valid forecasts
        assert len(fc_short) == 6
        assert len(fc_long) == 6


class TestEWMForecaster:
    """Tests for EWMForecaster (Exponential Weighted Moving Average)."""
    
    @pytest.fixture
    def sample_history(self):
        """Create sample historical data with trend."""
        n = 100
        base_time = datetime.now() - timedelta(hours=n)
        
        # Data with trend
        values = [100 + i * 0.5 + np.sin(i/12) * 5 for i in range(n)]
        
        return pd.DataFrame({
            'timestamp': [base_time + timedelta(hours=i) for i in range(n)],
            'value': values
        })
    
    def test_initialization(self):
        """Test EWMForecaster initializes correctly."""
        forecaster = EWMForecaster(span=24)
        assert forecaster.span == 24
    
    def test_default_span(self):
        """Test default span value."""
        forecaster = EWMForecaster()
        assert forecaster.span == 24  # Default
    
    def test_fit(self, sample_history):
        """Test fitting forecaster."""
        forecaster = EWMForecaster()
        result = forecaster.fit(sample_history)
        assert result is forecaster  # Returns self for chaining
    
    def test_predict(self, sample_history):
        """Test generating predictions."""
        forecaster = EWMForecaster()
        forecaster.fit(sample_history)
        
        forecasts = forecaster.predict(horizon=12)
        
        assert len(forecasts) == 12
        for f in forecasts:
            assert isinstance(f, Forecast)
            assert f.model == 'ewm'
    
    def test_forecast_with_bounds(self, sample_history):
        """Test forecasts have prediction intervals."""
        forecaster = EWMForecaster()
        forecaster.fit(sample_history)
        
        forecasts = forecaster.predict(horizon=6)
        
        for f in forecasts:
            # Bounds should be present
            assert f.lower_bound <= f.value
            assert f.upper_bound >= f.value


class TestEnsembleForecaster:
    """Tests for EnsembleForecaster."""
    
    @pytest.fixture
    def sample_history(self):
        """Create sample historical data."""
        n = 100
        base_time = datetime.now() - timedelta(hours=n)
        np.random.seed(42)
        
        return pd.DataFrame({
            'timestamp': [base_time + timedelta(hours=i) for i in range(n)],
            'value': 100 + np.random.randn(n) * 10
        })
    
    def test_initialization_default(self):
        """Test EnsembleForecaster initializes with default models."""
        forecaster = EnsembleForecaster()
        assert len(forecaster.forecasters) > 0
    
    def test_initialization_with_models(self):
        """Test EnsembleForecaster with custom models."""
        forecaster = EnsembleForecaster(models=['ewm', 'moving_average'])
        assert len(forecaster.forecasters) == 2
    
    def test_fit(self, sample_history):
        """Test fitting ensemble."""
        forecaster = EnsembleForecaster()
        result = forecaster.fit(sample_history)
        assert result is forecaster  # Returns self for chaining
    
    def test_predict(self, sample_history):
        """Test ensemble prediction."""
        forecaster = EnsembleForecaster()
        forecaster.fit(sample_history)
        
        forecasts = forecaster.predict(horizon=12)
        
        assert len(forecasts) == 12
        for f in forecasts:
            assert isinstance(f, Forecast)
            assert f.model == 'ensemble'
    
    def test_ensemble_averages_methods(self, sample_history):
        """Test ensemble combines multiple methods."""
        forecaster = EnsembleForecaster()
        forecaster.fit(sample_history)
        
        forecasts = forecaster.predict(horizon=6)
        
        # Ensemble should produce reasonable values
        for f in forecasts:
            assert f.value > 0


class TestShortTermForecaster:
    """Tests for ShortTermForecaster high-level interface."""
    
    @pytest.fixture
    def sample_history(self):
        """Create sample historical data."""
        n = 168  # 1 week of hourly data
        base_time = datetime.now() - timedelta(hours=n)
        np.random.seed(42)
        
        # Realistic pattern: base + trend + daily seasonality + noise
        values = []
        for i in range(n):
            hour = i % 24
            # Higher demand during day hours
            daily_pattern = 1.2 if 8 <= hour <= 18 else 0.8
            values.append(100 * daily_pattern + np.random.randn() * 10)
        
        return pd.DataFrame({
            'timestamp': [base_time + timedelta(hours=i) for i in range(n)],
            'value': values
        })
    
    def test_initialization_default(self):
        """Test ShortTermForecaster default initialization."""
        forecaster = ShortTermForecaster()
        assert forecaster.model_type == 'ensemble'
    
    def test_initialization_with_model(self):
        """Test ShortTermForecaster with specific model."""
        forecaster = ShortTermForecaster(model='ewm')
        assert forecaster.model_type == 'ewm'
    
    def test_fit(self, sample_history):
        """Test fitting forecaster."""
        forecaster = ShortTermForecaster()
        forecaster.fit(sample_history)
    
    def test_forecast_returns_dataframe(self, sample_history):
        """Test forecast returns DataFrame."""
        forecaster = ShortTermForecaster()
        forecaster.fit(sample_history)
        
        result = forecaster.forecast(horizon=24)
        
        assert isinstance(result, pd.DataFrame)
        assert 'timestamp' in result.columns
        assert 'forecast' in result.columns  # Note: column name is 'forecast', not 'value'
        assert len(result) == 24
    
    def test_forecast_custom_horizon(self, sample_history):
        """Test forecasting with custom horizon."""
        forecaster = ShortTermForecaster()
        forecaster.fit(sample_history)
        
        result = forecaster.forecast(horizon=48)
        
        assert len(result) == 48
    
    def test_different_models(self, sample_history):
        """Test different model selection."""
        models = ['ewm', 'moving_average', 'ensemble']
        
        for model in models:
            forecaster = ShortTermForecaster(model=model)
            forecaster.fit(sample_history)
            result = forecaster.forecast(horizon=6)
            assert len(result) == 6
    
    def test_update(self, sample_history):
        """Test updating forecaster with new data."""
        forecaster = ShortTermForecaster()
        forecaster.fit(sample_history)
        
        # Update with new observation
        forecaster.update(125.0, datetime.now())
        
        # Should be able to forecast after update
        result = forecaster.forecast(horizon=6)
        assert len(result) == 6


class TestForecasterEdgeCases:
    """Test edge cases for forecasters."""
    
    def test_small_history(self):
        """Test with very small history."""
        base_time = datetime.now()
        data = pd.DataFrame({
            'timestamp': [base_time + timedelta(hours=i) for i in range(5)],
            'value': [100, 102, 98, 105, 101]
        })
        
        forecaster = ShortTermForecaster()
        forecaster.fit(data)
        
        result = forecaster.forecast(horizon=3)
        assert len(result) == 3
    
    def test_constant_values(self):
        """Test with constant values."""
        base_time = datetime.now()
        data = pd.DataFrame({
            'timestamp': [base_time + timedelta(hours=i) for i in range(50)],
            'value': [100.0] * 50
        })
        
        forecaster = ShortTermForecaster()
        forecaster.fit(data)
        
        result = forecaster.forecast(horizon=6)
        
        # Forecast should be close to the constant value
        for val in result['forecast']:
            assert abs(val - 100.0) < 10
    
    def test_high_variability(self):
        """Test with high variability data."""
        np.random.seed(42)
        base_time = datetime.now()
        data = pd.DataFrame({
            'timestamp': [base_time + timedelta(hours=i) for i in range(100)],
            'value': np.random.randn(100) * 100 + 100
        })
        
        forecaster = ShortTermForecaster()
        forecaster.fit(data)
        
        result = forecaster.forecast(horizon=6)
        
        # Should still produce forecasts
        assert len(result) == 6
        for val in result['forecast']:
            assert isinstance(val, (int, float, np.floating))
    
    def test_update_incremental(self):
        """Test incremental updates."""
        base_time = datetime.now()
        data = pd.DataFrame({
            'timestamp': [base_time + timedelta(hours=i) for i in range(50)],
            'value': [100 + i * 0.5 for i in range(50)]
        })
        
        forecaster = EWMForecaster()
        forecaster.fit(data)
        
        # Update with new observation
        new_ts = base_time + timedelta(hours=50)
        forecaster.update(125.0, new_ts)
        
        # Should incorporate new value
        forecasts = forecaster.predict(horizon=3)
        assert len(forecasts) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

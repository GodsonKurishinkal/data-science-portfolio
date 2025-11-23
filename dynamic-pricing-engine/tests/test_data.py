"""
Test suite for data modules

Tests for data loading and preprocessing functionality.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from src.data.loader import PricingDataLoader
from src.data.preprocessing import PricingDataPreprocessor


class TestPricingDataLoader:
    """Test data loader functionality."""
    
    def test_loader_initialization(self):
        """Test loader initializes correctly."""
        loader = PricingDataLoader(data_path='data/raw')
        assert loader.data_path == Path('data/raw')
    
    def test_load_sales_data(self):
        """Test sales data loading."""
        loader = PricingDataLoader(data_path='data/raw')
        
        # This test requires actual M5 data
        if not (Path('data/raw') / 'sales_train_validation.csv').exists():
            pytest.skip("M5 data not available")
        
        sales = loader.load_sales_data(validation=True)
        
        assert isinstance(sales, pd.DataFrame)
        assert 'sales' in sales.columns
        assert 'item_id' in sales.columns
        assert 'store_id' in sales.columns
        assert len(sales) > 0
    
    def test_load_price_data(self):
        """Test price data loading."""
        loader = PricingDataLoader(data_path='data/raw')
        
        if not (Path('data/raw') / 'sell_prices.csv').exists():
            pytest.skip("M5 data not available")
        
        prices = loader.load_price_data()
        
        assert isinstance(prices, pd.DataFrame)
        assert 'sell_price' in prices.columns
        assert 'store_id' in prices.columns
        assert 'item_id' in prices.columns
        assert len(prices) > 0
    
    def test_load_calendar_data(self):
        """Test calendar data loading."""
        loader = PricingDataLoader(data_path='data/raw')
        
        if not (Path('data/raw') / 'calendar.csv').exists():
            pytest.skip("M5 data not available")
        
        calendar = loader.load_calendar_data()
        
        assert isinstance(calendar, pd.DataFrame)
        assert 'date' in calendar.columns
        assert 'd' in calendar.columns
        assert pd.api.types.is_datetime64_any_dtype(calendar['date'])


class TestPricingDataPreprocessor:
    """Test data preprocessing functionality."""
    
    @pytest.fixture
    def sample_pricing_data(self):
        """Create sample pricing data for testing."""
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=30, freq='D')
        
        data = []
        for item in ['ITEM_A', 'ITEM_B']:
            for store in ['STORE_1']:
                for date in dates:
                    data.append({
                        'date': date,
                        'item_id': item,
                        'store_id': store,
                        'sell_price': np.random.uniform(3.0, 7.0),
                        'sales': np.random.randint(10, 50),
                        'cat_id': 'FOODS'
                    })
        
        return pd.DataFrame(data)
    
    def test_preprocessor_initialization(self):
        """Test preprocessor initializes correctly."""
        preprocessor = PricingDataPreprocessor()
        assert preprocessor is not None
    
    def test_extract_price_history(self, sample_pricing_data):
        """Test price history extraction."""
        preprocessor = PricingDataPreprocessor()
        result = preprocessor.extract_price_history(sample_pricing_data)
        
        assert 'price_lag_1' in result.columns
        assert 'price_change' in result.columns
        assert 'price_change_pct' in result.columns
        assert 'price_changed' in result.columns
        assert 'days_since_price_change' in result.columns
    
    def test_calculate_price_statistics(self, sample_pricing_data):
        """Test price statistics calculation."""
        preprocessor = PricingDataPreprocessor()
        result = preprocessor.calculate_price_statistics(sample_pricing_data)
        
        assert 'price_mean' in result.columns
        assert 'price_std' in result.columns
        assert 'price_min' in result.columns
        assert 'price_max' in result.columns
        assert 'price_vs_mean' in result.columns
        assert 'price_volatility' in result.columns
    
    def test_identify_promotions(self, sample_pricing_data):
        """Test promotion identification."""
        preprocessor = PricingDataPreprocessor()
        
        # Add price statistics first
        data_with_stats = preprocessor.calculate_price_statistics(sample_pricing_data)
        result = preprocessor.identify_promotions(data_with_stats)
        
        assert 'is_promotion' in result.columns
        assert 'discount_depth' in result.columns
        assert result['is_promotion'].dtype == bool
    
    def test_engineer_pricing_features(self, sample_pricing_data):
        """Test pricing feature engineering."""
        preprocessor = PricingDataPreprocessor()
        result = preprocessor.engineer_pricing_features(
            sample_pricing_data,
            include_lags=True,
            include_rolling=True
        )
        
        # Check lag features
        assert 'price_lag_7' in result.columns
        assert 'price_change_7d' in result.columns
        
        # Check rolling features
        assert 'price_rolling_mean_7' in result.columns
        assert 'price_rolling_std_7' in result.columns
        
        # Check trend indicators
        assert 'price_trend_7d' in result.columns
    
    def test_create_pricing_dataset(self, sample_pricing_data):
        """Test complete pricing dataset creation."""
        preprocessor = PricingDataPreprocessor()
        result = preprocessor.create_pricing_dataset(
            sample_pricing_data,
            include_all_features=True
        )
        
        # Should have all preprocessing steps applied
        assert 'price_change' in result.columns
        assert 'price_mean' in result.columns
        assert 'is_promotion' in result.columns
        assert 'price_lag_7' in result.columns
        assert len(result) > 0
    
    def test_get_product_price_summary(self, sample_pricing_data):
        """Test product price summary generation."""
        preprocessor = PricingDataPreprocessor()
        
        # Create dataset first
        processed = preprocessor.create_pricing_dataset(sample_pricing_data)
        summary = preprocessor.get_product_price_summary(processed)
        
        assert len(summary) == 2  # 2 items in sample data
        assert 'sell_price_mean' in summary.columns
        assert 'sales_sum' in summary.columns

"""
Unit tests for data preprocessing module.
"""

import pytest
import pandas as pd
import numpy as np
from src.data.preprocessing import (
    load_data,
    clean_data,
    preprocess_data,
    load_and_preprocess_data
)


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame({
        'date': pd.date_range('2023-01-01', periods=100, freq='D'),
        'demand': np.random.randint(50, 200, 100),
        'price': np.random.uniform(10, 50, 100),
        'promotion': np.random.choice([0, 1], 100)
    })


@pytest.fixture
def sample_dataframe_with_missing():
    """Create a sample DataFrame with missing values."""
    df = pd.DataFrame({
        'date': pd.date_range('2023-01-01', periods=100, freq='D'),
        'demand': np.random.randint(50, 200, 100),
        'price': np.random.uniform(10, 50, 100)
    })
    # Add some missing values
    df.loc[5:10, 'demand'] = np.nan
    df.loc[20:25, 'price'] = np.nan
    return df


@pytest.fixture
def sample_dataframe_with_duplicates():
    """Create a sample DataFrame with duplicate rows."""
    df = pd.DataFrame({
        'date': pd.date_range('2023-01-01', periods=50, freq='D'),
        'demand': np.random.randint(50, 200, 50),
    })
    # Add duplicate rows
    df = pd.concat([df, df.iloc[:10]], ignore_index=True)
    return df


class TestCleanData:
    """Test cases for clean_data function."""

    def test_clean_data_drops_duplicates(self, sample_dataframe_with_duplicates):
        """Test that duplicates are removed."""
        df_clean = clean_data(sample_dataframe_with_duplicates, drop_duplicates=True)
        assert len(df_clean) < len(sample_dataframe_with_duplicates)
        assert df_clean.duplicated().sum() == 0

    def test_clean_data_handles_missing_drop(self, sample_dataframe_with_missing):
        """Test that missing values are dropped."""
        df_clean = clean_data(sample_dataframe_with_missing, handle_missing='drop')
        assert df_clean.isnull().sum().sum() == 0

    def test_clean_data_handles_missing_fill_mean(self, sample_dataframe_with_missing):
        """Test that missing values are filled with mean."""
        df_clean = clean_data(sample_dataframe_with_missing, handle_missing='fill_mean')
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        assert df_clean[numeric_cols].isnull().sum().sum() == 0

    def test_clean_data_preserves_original(self, sample_dataframe):
        """Test that original DataFrame is not modified."""
        original_shape = sample_dataframe.shape
        _ = clean_data(sample_dataframe)
        assert sample_dataframe.shape == original_shape


class TestPreprocessData:
    """Test cases for preprocess_data function."""

    def test_preprocess_data_returns_tuple(self, sample_dataframe):
        """Test that function returns X and y."""
        X, y = preprocess_data(sample_dataframe, target_column='demand')
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)

    def test_preprocess_data_separates_target(self, sample_dataframe):
        """Test that target is properly separated."""
        X, y = preprocess_data(sample_dataframe, target_column='demand')
        assert 'demand' not in X.columns
        assert y.name == 'demand'

    def test_preprocess_data_parses_dates(self, sample_dataframe):
        """Test that date column is parsed and set as index."""
        X, y = preprocess_data(sample_dataframe, date_column='date', target_column='demand')
        assert isinstance(X.index, pd.DatetimeIndex)

    def test_preprocess_data_raises_on_missing_target(self, sample_dataframe):
        """Test that error is raised when target column doesn't exist."""
        with pytest.raises(ValueError, match="Target column.*not found"):
            preprocess_data(sample_dataframe, target_column='nonexistent_column')


class TestDataWorkflow:
    """Integration tests for complete data workflow."""

    def test_clean_and_preprocess_workflow(self, sample_dataframe_with_missing):
        """Test complete workflow from dirty data to clean features."""
        # Clean data
        df_clean = clean_data(sample_dataframe_with_missing, handle_missing='drop')

        # Preprocess
        X, y = preprocess_data(df_clean, date_column='date', target_column='demand')

        # Verify results
        assert X.isnull().sum().sum() == 0
        assert y.isnull().sum() == 0
        assert len(X) == len(y)
        assert len(X) > 0


def test_data_shape_consistency(sample_dataframe):
    """Test that data shapes are consistent after preprocessing."""
    X, y = preprocess_data(sample_dataframe, target_column='demand')
    assert len(X) == len(y)
    assert X.shape[0] > 0


def test_feature_names(sample_dataframe):
    """Test that feature columns are correct."""
    X, y = preprocess_data(sample_dataframe, target_column='demand')
    expected_features = ['price', 'promotion']
    assert all(feat in X.columns for feat in expected_features)


# M5-Specific Tests

@pytest.fixture
def sample_m5_sales():
    """Create a sample M5 sales DataFrame."""
    return pd.DataFrame({
        'id': ['FOODS_1_001_CA_1', 'FOODS_1_002_CA_1'],
        'item_id': ['FOODS_1_001', 'FOODS_1_002'],
        'dept_id': ['FOODS_1', 'FOODS_1'],
        'cat_id': ['FOODS', 'FOODS'],
        'store_id': ['CA_1', 'CA_1'],
        'state_id': ['CA', 'CA'],
        'd_1': [3, 0],
        'd_2': [5, 2],
        'd_3': [4, 1]
    })


@pytest.fixture
def sample_m5_calendar():
    """Create a sample M5 calendar DataFrame."""
    return pd.DataFrame({
        'd': ['d_1', 'd_2', 'd_3'],
        'date': ['2021-01-01', '2021-01-02', '2021-01-03'],
        'wm_yr_wk': [11101, 11101, 11101],
        'weekday': [1, 2, 3],
        'event_name_1': [None, 'Event1', None],
        'event_type_1': [None, 'Cultural', None],
        'snap_CA': [0, 1, 0]
    })


@pytest.fixture
def sample_m5_prices():
    """Create a sample M5 prices DataFrame."""
    return pd.DataFrame({
        'store_id': ['CA_1', 'CA_1'],
        'item_id': ['FOODS_1_001', 'FOODS_1_002'],
        'wm_yr_wk': [11101, 11101],
        'sell_price': [1.5, 2.0]
    })


class TestM5DataProcessing:
    """Test cases for M5-specific data processing functions."""

    def test_melt_sales_data(self, sample_m5_sales):
        """Test melting M5 sales data from wide to long format."""
        from src.data.preprocessing import melt_sales_data

        sales_long = melt_sales_data(sample_m5_sales)

        assert 'd' in sales_long.columns
        assert 'sales' in sales_long.columns
        assert len(sales_long) == 6  # 2 items * 3 days
        assert sales_long['sales'].dtype in [np.int64, np.int32]

    def test_merge_m5_data(self, sample_m5_sales, sample_m5_calendar, sample_m5_prices):
        """Test merging M5 datasets."""
        from src.data.preprocessing import melt_sales_data, merge_m5_data

        sales_long = melt_sales_data(sample_m5_sales)
        df_merged = merge_m5_data(sales_long, sample_m5_calendar, sample_m5_prices)

        assert 'date' in df_merged.columns
        assert 'sell_price' in df_merged.columns
        assert 'event_name_1' in df_merged.columns
        assert len(df_merged) > 0

    def test_create_datetime_features(self):
        """Test datetime feature creation."""
        from src.data.preprocessing import create_datetime_features

        df = pd.DataFrame({
            'date': pd.date_range('2021-01-01', periods=10, freq='D'),
            'value': range(10)
        })

        df_features = create_datetime_features(df, date_col='date')

        expected_features = ['year', 'month', 'day', 'dayofweek', 'is_weekend']
        assert all(feat in df_features.columns for feat in expected_features)
        assert df_features['is_weekend'].dtype in [np.int64, np.int32]
        assert df_features['year'].iloc[0] == 2021

    def test_create_datetime_features_missing_column(self):
        """Test that error is raised for missing date column."""
        from src.data.preprocessing import create_datetime_features

        df = pd.DataFrame({'value': range(10)})

        with pytest.raises(ValueError, match="Date column.*not found"):
            create_datetime_features(df, date_col='date')

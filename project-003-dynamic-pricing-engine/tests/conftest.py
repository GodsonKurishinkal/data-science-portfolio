"""
pytest configuration for Dynamic Pricing Engine tests
"""

import pytest
import pandas as pd
import numpy as np


@pytest.fixture
def sample_price_data():
    """Sample price data for testing."""
    return pd.DataFrame({
        'product_id': ['A', 'A', 'A', 'B', 'B', 'B'],
        'date': pd.date_range('2020-01-01', periods=6, freq='D')[:6],
        'price': [5.0, 4.5, 4.0, 10.0, 9.5, 9.0],
        'sales': [100, 120, 145, 50, 55, 60]
    })


@pytest.fixture
def sample_sales_data():
    """Sample sales data for testing."""
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    return pd.DataFrame({
        'date': dates,
        'product_id': ['A'] * 100,
        'sales': np.random.randint(80, 120, 100),
        'price': np.random.uniform(4.0, 6.0, 100)
    })


@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return {
        'elasticity': {
            'default_method': 'log-log',
            'min_observations': 10
        },
        'optimization': {
            'objective': 'maximize_revenue',
            'constraints': {
                'min_price_factor': 0.8,
                'max_price_factor': 1.2
            }
        }
    }

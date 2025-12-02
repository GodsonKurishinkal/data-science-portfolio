"""Test configuration and fixtures."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path


@pytest.fixture
def sample_sales_data():
    """Generate sample sales data for testing."""
    np.random.seed(42)

    dates = pd.date_range('2015-01-01', '2015-12-31', freq='D')
    stores = ['CA_1', 'TX_1']
    items = ['ITEM_001', 'ITEM_002', 'ITEM_003']

    data = []
    for store in stores:
        for item in items:
            for date in dates:
                data.append({
                    'date': date,
                    'store_id': store,
                    'item_id': item,
                    'sales': max(0, np.random.poisson(10)),
                    'sell_price': np.random.uniform(10, 50)
                })

    df = pd.DataFrame(data)
    df['revenue'] = df['sales'] * df['sell_price']
    return df


@pytest.fixture
def sample_demand_stats():
    """Generate sample demand statistics for testing."""
    np.random.seed(42)

    data = {
        'store_id': ['CA_1'] * 10,
        'item_id': [f'ITEM_{i:03d}' for i in range(10)],
        'sales_sum': np.random.randint(1000, 10000, 10),
        'sales_mean': np.random.uniform(5, 50, 10),
        'sales_std': np.random.uniform(1, 10, 10),
        'revenue_sum': np.random.uniform(10000, 100000, 10),
        'sell_price_mean': np.random.uniform(10, 50, 10),
    }

    df = pd.DataFrame(data)
    df['demand_cv'] = df['sales_std'] / df['sales_mean']
    return df


@pytest.fixture
def config():
    """Sample configuration dictionary."""
    return {
        'inventory': {
            'abc_thresholds': {'A': 0.80, 'B': 0.95, 'C': 1.00},
            'xyz_thresholds': {'X': 0.5, 'Y': 1.0, 'Z': 999},
            'service_levels': {
                'high_value': 0.99,
                'medium_value': 0.95,
                'low_value': 0.90
            },
            'costs': {
                'holding_cost_rate': 0.25,
                'ordering_cost': 100,
                'stockout_cost_rate': 2.0
            },
            'lead_time': {
                'default': 7
            }
        }
    }

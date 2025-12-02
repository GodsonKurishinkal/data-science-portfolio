"""
Pytest configuration and shared fixtures for Supply Chain Planning System.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def sample_config():
    """Sample configuration dictionary."""
    return {
        'global': {
            'environment': 'test',
            'log_level': 'DEBUG',
        },
        'demand': {
            'model_type': 'lightgbm',
            'forecast_horizon_days': 30,
        },
        'inventory': {
            'service_level_target': 0.95,
        },
        'pricing': {
            'elasticity_method': 'log-log',
        },
    }


@pytest.fixture
def sample_sales_data():
    """Generate sample sales data for testing."""
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=365, freq='D')
    
    data = {
        'date': dates,
        'item_id': ['SKU001'] * 365,
        'store_id': ['STORE001'] * 365,
        'sales': np.random.poisson(100, 365),
        'price': np.random.uniform(9, 11, 365),
        'inventory': np.random.randint(50, 200, 365),
    }
    
    return pd.DataFrame(data)


@pytest.fixture
def sample_inventory_data():
    """Generate sample inventory data for testing."""
    np.random.seed(42)
    
    items = [f'SKU{str(i).zfill(3)}' for i in range(1, 101)]
    
    data = {
        'item_id': items,
        'current_stock': np.random.randint(10, 500, 100),
        'reorder_point': np.random.randint(20, 100, 100),
        'safety_stock': np.random.randint(10, 50, 100),
        'lead_time_days': np.random.randint(3, 14, 100),
        'unit_cost': np.random.uniform(5, 50, 100),
        'annual_demand': np.random.randint(500, 5000, 100),
    }
    
    return pd.DataFrame(data)


@pytest.fixture
def sample_pricing_data():
    """Generate sample pricing data for testing."""
    np.random.seed(42)
    n_records = 1000
    
    # Generate price-quantity pairs with inverse relationship
    prices = np.random.uniform(8, 15, n_records)
    base_demand = 100
    elasticity = -1.5
    quantities = base_demand * (prices / 10) ** elasticity * np.random.uniform(0.9, 1.1, n_records)
    
    data = {
        'date': pd.date_range('2024-01-01', periods=n_records, freq='H'),
        'item_id': np.random.choice(['SKU001', 'SKU002', 'SKU003'], n_records),
        'price': prices,
        'quantity': quantities.astype(int),
        'revenue': prices * quantities,
    }
    
    return pd.DataFrame(data)


@pytest.fixture
def sample_network_data():
    """Generate sample network data for testing."""
    np.random.seed(42)
    
    # Facilities
    facilities = pd.DataFrame({
        'facility_id': ['DC001', 'DC002', 'DC003'],
        'lat': [40.7128, 34.0522, 41.8781],
        'lon': [-74.0060, -118.2437, -87.6298],
        'capacity': [10000, 8000, 12000],
        'fixed_cost': [50000, 40000, 60000],
    })
    
    # Customers
    customers = pd.DataFrame({
        'customer_id': [f'C{i:03d}' for i in range(1, 21)],
        'lat': np.random.uniform(30, 45, 20),
        'lon': np.random.uniform(-120, -70, 20),
        'demand': np.random.randint(100, 1000, 20),
    })
    
    return {'facilities': facilities, 'customers': customers}


@pytest.fixture
def sample_alerts():
    """Generate sample alerts for testing."""
    return [
        {
            'alert_id': 'ALT001',
            'severity': 'CRITICAL',
            'type': 'stockout_imminent',
            'item_id': 'SKU001',
            'message': 'Stockout expected in 2 days',
            'timestamp': datetime.now(),
        },
        {
            'alert_id': 'ALT002',
            'severity': 'HIGH',
            'type': 'demand_spike',
            'item_id': 'SKU002',
            'message': 'Demand 150% above forecast',
            'timestamp': datetime.now(),
        },
        {
            'alert_id': 'ALT003',
            'severity': 'MEDIUM',
            'type': 'excess_inventory',
            'item_id': 'SKU003',
            'message': '90 days of supply on hand',
            'timestamp': datetime.now(),
        },
    ]


@pytest.fixture
def sample_kpis():
    """Generate sample KPI values for testing."""
    return {
        'forecast_accuracy': 0.87,
        'service_level': 0.96,
        'inventory_turns': 11.5,
        'fill_rate': 0.98,
        'stockout_rate': 0.015,
        'order_accuracy': 0.997,
        'on_time_delivery': 0.94,
        'automation_rate': 0.75,
        'gross_margin': 0.32,
        'total_cost_to_serve': 0.075,
    }


@pytest.fixture
def temp_config_file(tmp_path, sample_config):
    """Create a temporary config file."""
    import yaml
    
    config_file = tmp_path / "config.yaml"
    with open(config_file, 'w') as f:
        yaml.dump(sample_config, f)
    
    return config_file

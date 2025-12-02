"""Test fixtures and configuration for pytest."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


@pytest.fixture
def sample_inventory_data():
    """Create sample inventory data for testing."""
    np.random.seed(42)

    n_items = 50

    return pd.DataFrame({
        "item_id": [f"SKU{i:04d}" for i in range(n_items)],
        "current_stock": np.random.randint(10, 500, n_items),
        "max_capacity": np.random.randint(500, 1000, n_items),
        "daily_demand_rate": np.random.uniform(5, 50, n_items),
        "demand_std": np.random.uniform(2, 20, n_items),
        "unit_cost": np.random.uniform(5, 100, n_items),
        "revenue": np.random.uniform(1000, 100000, n_items),
        "lead_time": np.random.choice([7, 14, 21], n_items),
    })


@pytest.fixture
def sample_demand_data():
    """Create sample demand history for testing."""
    np.random.seed(42)

    n_items = 20
    n_days = 90

    items = [f"SKU{i:04d}" for i in range(n_items)]
    dates = pd.date_range(
        end=datetime.now(),
        periods=n_days,
        freq="D",
    )

    data = []
    for item in items:
        base_demand = np.random.uniform(10, 100)
        for date in dates:
            # Add seasonality and noise
            seasonal = 1 + 0.2 * np.sin(2 * np.pi * date.dayofyear / 365)
            noise = np.random.normal(0, base_demand * 0.2)
            demand = max(0, base_demand * seasonal + noise)

            data.append({
                "item_id": item,
                "date": date,
                "quantity": demand,
                "price": np.random.uniform(10, 50),
            })

    return pd.DataFrame(data)


@pytest.fixture
def sample_source_inventory():
    """Create sample source inventory for testing."""
    np.random.seed(42)

    n_items = 50

    return pd.DataFrame({
        "item_id": [f"SKU{i:04d}" for i in range(n_items)],
        "available_quantity": np.random.randint(100, 5000, n_items),
        "location_id": "DC001",
    })


@pytest.fixture
def sample_config():
    """Create sample configuration for testing."""
    return {
        "scenarios": {
            "test_scenario": {
                "policy_type": "periodic_review",
                "review_period": 7,
                "lead_time": 7,
                "service_level": 0.95,
                "order_strategy": "policy_target",
            }
        },
        "classification": {
            "abc_enabled": True,
            "xyz_enabled": True,
        },
        "alerts": {
            "enabled": True,
            "critical_days_supply": 1.0,
            "low_days_supply": 3.0,
        },
    }


@pytest.fixture
def abc_test_data():
    """Create data specifically for ABC classification testing."""
    return pd.DataFrame({
        "item_id": ["A1", "A2", "B1", "B2", "C1", "C2", "C3", "C4"],
        "revenue": [50000, 30000, 10000, 5000, 2000, 1500, 1000, 500],
    })


@pytest.fixture
def xyz_test_data():
    """Create data for XYZ classification testing."""
    np.random.seed(42)

    data = []
    items = {
        "X1": {"mean": 100, "std": 10},   # CV = 0.1 (X)
        "Y1": {"mean": 100, "std": 70},   # CV = 0.7 (Y)
        "Z1": {"mean": 100, "std": 150},  # CV = 1.5 (Z)
    }

    dates = pd.date_range(end=datetime.now(), periods=30, freq="D")

    for item, params in items.items():
        for date in dates:
            qty = max(0, np.random.normal(params["mean"], params["std"]))
            data.append({
                "item_id": item,
                "date": date,
                "quantity": qty,
            })

    return pd.DataFrame(data)

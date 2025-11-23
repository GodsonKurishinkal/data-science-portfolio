"""Tests for demand sensing and anomaly detection."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def test_data_generation():
    """Test sample data generation."""
    dates = pd.date_range(end=datetime.now(), periods=100, freq='H')
    data = pd.DataFrame({
        'timestamp': dates,
        'sales': np.random.uniform(50, 150, 100)
    })
    
    assert len(data) == 100
    assert 'timestamp' in data.columns
    assert 'sales' in data.columns
    assert data['sales'].min() >= 0


def test_anomaly_detection():
    """Test anomaly detection with z-score."""
    data = pd.Series(np.random.normal(100, 10, 1000))
    
    # Insert obvious anomalies
    data[50] = 200
    data[100] = 10
    
    zscore = (data - data.mean()) / data.std()
    anomalies = np.abs(zscore) > 3
    
    # Should detect at least our 2 inserted anomalies
    assert anomalies.sum() >= 2


def test_replenishment_trigger():
    """Test replenishment trigger logic."""
    inventory = 200
    daily_sales = 100
    safety_stock = 150
    reorder_point = 300
    
    days_of_stock = inventory / daily_sales
    
    assert days_of_stock == 2.0
    assert inventory < reorder_point  # Should trigger

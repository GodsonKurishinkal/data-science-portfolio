"""Tests for facility location optimization."""

import pytest
import pandas as pd
import numpy as np
from src.network.facility_location import FacilityLocationOptimizer


def test_facility_location_basic():
    """Test basic facility location optimization."""
    # Simple test case
    facilities = ['DC_1', 'DC_2', 'DC_3']
    fixed_costs = {'DC_1': 100000, 'DC_2': 120000, 'DC_3': 110000}
    capacities = {'DC_1': 1000, 'DC_2': 1200, 'DC_3': 1100}
    
    stores = pd.DataFrame({
        'id': ['S1', 'S2', 'S3'],
        'latitude': [34.05, 36.17, 38.0],
        'longitude': [-118.24, -115.14, -120.0]
    })
    
    demand = pd.Series([200, 150, 180], index=['S1', 'S2', 'S3'])
    
    # Distance matrix
    distance_matrix = pd.DataFrame(
        [[0, 100, 200], [100, 0, 150], [200, 150, 0]],
        index=facilities,
        columns=stores['id']
    )
    
    optimizer = FacilityLocationOptimizer(fixed_costs, capacities)
    
    # This test verifies the optimizer can be instantiated
    assert optimizer is not None
    assert optimizer.fixed_costs == fixed_costs
    assert optimizer.capacities == capacities


def test_distance_calculator():
    """Test distance calculations."""
    from src.utils.distance import haversine_distance
    
    # LA to SF (approx 380 miles)
    dist = haversine_distance(34.05, -118.24, 37.77, -122.42)
    
    assert dist > 300  # Approximately 380 miles
    assert dist < 450

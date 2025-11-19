"""
Simple test for Project 002 optimizations.
"""
import pandas as pd
import numpy as np
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "project-002-inventory-optimization-engine"))

print("Testing Project 002 Optimizations...")
print("=" * 70)

from src.optimization.optimizer import InventoryOptimizer
from src.optimization.cost_calculator import CostCalculator

# Create sample inventory data
np.random.seed(42)
n_items = 100

inventory_df = pd.DataFrame({
    'item_id': [f'ITEM_{i:03d}' for i in range(n_items)],
    'store_id': np.random.choice(['STORE_1', 'STORE_2'], n_items),
    'abc_class': np.random.choice(['A', 'B', 'C'], n_items),
    'xyz_class': np.random.choice(['X', 'Y', 'Z'], n_items),
    'abc_xyz_class': [f"{a}{x}" for a, x in zip(
        np.random.choice(['A', 'B', 'C'], n_items),
        np.random.choice(['X', 'Y', 'Z'], n_items)
    )],
    'priority_score': np.random.randint(1, 10, n_items),
    'revenue_sum': np.random.uniform(1000, 100000, n_items),
    'sales_sum': np.random.uniform(100, 1000, n_items),
    'demand_cv': np.random.uniform(0.1, 2.0, n_items),
    'sell_price_mean': np.random.uniform(10, 100, n_items),
    'target_service_level': np.random.choice([0.90, 0.95, 0.99], n_items),
    'eoq': np.random.uniform(10, 200, n_items),
    'reorder_point': np.random.uniform(5, 100, n_items),
    'safety_stock': np.random.uniform(5, 50, n_items),
    'total_annual_cost': np.random.uniform(500, 10000, n_items),
    'average_inventory': np.random.uniform(20, 500, n_items),
    'orders_per_year': np.random.randint(4, 52, n_items)
})

print("\n1. Testing optimizer generate_recommendations (vectorized)...")
config = {
    'inventory': {
        'abc_thresholds': {'A': 0.80, 'B': 0.95, 'C': 1.00},
        'xyz_thresholds': {'X': 0.5, 'Y': 1.0, 'Z': 999},
        'service_levels': {'high_value': 0.99, 'medium_value': 0.95, 'low_value': 0.90},
        'costs': {
            'holding_cost_rate': 0.25,
            'ordering_cost': 100,
            'stockout_cost_rate': 2.0
        },
        'lead_time': {'default': 7}
    }
}

optimizer = InventoryOptimizer(config)
start = time.time()
recommendations = optimizer.generate_recommendations(inventory_df, top_n=20)
time_vectorized = time.time() - start

assert isinstance(recommendations, pd.DataFrame)
assert len(recommendations) > 0
assert 'recommended_policy' in recommendations.columns
print(f"   ✓ Vectorized recommendations work: {time_vectorized:.4f}s")
print(f"   Generated {len(recommendations)} recommendations")

print("\n2. Testing cost calculator service level optimization...")
cost_calc = CostCalculator(
    holding_cost_rate=0.25,
    ordering_cost=100,
    stockout_cost_rate=2.0
)

start = time.time()
sl_analysis = cost_calc.calculate_service_level_cost_tradeoff(
    demand_mean=100,
    demand_std=20,
    unit_cost=25,
    service_levels=[0.85, 0.90, 0.95, 0.99]
)
time_vectorized_sl = time.time() - start

assert isinstance(sl_analysis, pd.DataFrame)
assert len(sl_analysis) == 4
assert 'total_cost' in sl_analysis.columns
print(f"   ✓ Vectorized service level optimization: {time_vectorized_sl:.4f}s")
print(f"   Results:\n{sl_analysis[['service_level', 'safety_stock', 'total_cost']].to_string(index=False)}")

print("\n3. Testing ABC class cost calculation (vectorized groupby)...")
start = time.time()
abc_costs = cost_calc.calculate_abc_class_costs(
    inventory_df,
    group_col='abc_class',
    inventory_col='average_inventory',
    sales_col='sales_sum',
    price_col='sell_price_mean',
    orders_col='orders_per_year'
)
time_groupby = time.time() - start

assert isinstance(abc_costs, pd.DataFrame)
assert len(abc_costs) == 3  # A, B, C classes
assert 'total_operational_cost' in abc_costs.columns
print(f"   ✓ Vectorized groupby costs work: {time_groupby:.4f}s")
print(f"   Calculated costs for {len(abc_costs)} ABC classes")

print("\n✅ All Project 002 optimizations verified!")
print("=" * 70)

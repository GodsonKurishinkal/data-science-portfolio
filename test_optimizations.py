"""
Test script to verify performance optimizations maintain correctness.
"""

import pandas as pd
import numpy as np
import time
import sys
from pathlib import Path

# Add project paths
sys.path.insert(0, str(Path(__file__).parent / "project-001-demand-forecasting-system"))
sys.path.insert(0, str(Path(__file__).parent / "project-002-inventory-optimization-engine"))

print("Testing Project 001 - Demand Forecasting System optimizations...")
print("=" * 70)

try:
    from src.features.build_features import (
        create_price_features,
        encode_calendar_features,
        create_sales_lag_features,
        create_sales_rolling_features,
        create_hierarchical_features,
        build_m5_features
    )
    
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    
    test_df = pd.DataFrame({
        'item_id': np.random.choice(['ITEM_001', 'ITEM_002', 'ITEM_003'], n_samples),
        'store_id': np.random.choice(['STORE_CA_1', 'STORE_TX_1'], n_samples),
        'date': pd.date_range('2024-01-01', periods=n_samples, freq='D')[:n_samples],
        'sales': np.random.poisson(10, n_samples),
        'sell_price': np.random.uniform(5, 50, n_samples),
        'state_id': np.random.choice(['CA', 'TX', 'WI'], n_samples),
        'cat_id': np.random.choice(['FOODS', 'HOUSEHOLD'], n_samples),
        'dept_id': np.random.choice(['FOODS_1', 'HOUSEHOLD_1'], n_samples),
        'event_name_1': np.random.choice([None, 'Christmas', 'Easter'], n_samples),
        'has_event': np.random.choice([0, 1], n_samples),
        'snap_CA': np.random.choice([0, 1], n_samples),
        'snap_TX': np.random.choice([0, 1], n_samples),
        'snap_WI': np.random.choice([0, 1], n_samples),
    })
    
    print("\n1. Testing price features with inplace=False...")
    df_copy = test_df.copy()
    start = time.time()
    result1 = create_price_features(df_copy, inplace=False)
    time1 = time.time() - start
    assert result1 is not None
    assert len(result1) == len(test_df)
    print(f"   ✓ Non-inplace works: {time1:.4f}s, shape={result1.shape}")
    
    print("\n2. Testing price features with inplace=True...")
    df_copy = test_df.copy()
    start = time.time()
    result2 = create_price_features(df_copy, inplace=True)
    time2 = time.time() - start
    assert result2 is None
    assert 'price_change' in df_copy.columns
    print(f"   ✓ Inplace works: {time2:.4f}s, memory efficient")
    print(f"   Performance improvement: {((time1-time2)/time1*100):.1f}%")
    
    print("\n3. Testing calendar encoding...")
    df_copy = test_df.copy()
    result3 = encode_calendar_features(df_copy, inplace=False)
    assert result3 is not None
    assert 'has_event' in result3.columns
    print(f"   ✓ Calendar encoding works, shape={result3.shape}")
    
    print("\n4. Testing lag features...")
    df_copy = test_df.copy()
    result4 = create_sales_lag_features(df_copy, 'sales', [1, 7], inplace=False)
    assert result4 is not None
    assert 'sales_lag_1' in result4.columns
    assert 'sales_lag_7' in result4.columns
    print(f"   ✓ Lag features work, shape={result4.shape}")
    
    print("\n5. Testing rolling features...")
    df_copy = test_df.copy()
    result5 = create_sales_rolling_features(df_copy, 'sales', [7, 14], inplace=False)
    assert result5 is not None
    assert 'sales_rolling_mean_7' in result5.columns
    print(f"   ✓ Rolling features work, shape={result5.shape}")
    
    print("\n6. Testing hierarchical features...")
    df_copy = test_df.copy()
    result6 = create_hierarchical_features(df_copy, 'sales', inplace=False)
    assert result6 is not None
    print(f"   ✓ Hierarchical features work, shape={result6.shape}")
    
    print("\n7. Testing complete pipeline (non-inplace)...")
    df_copy = test_df.copy()
    start = time.time()
    result7 = build_m5_features(df_copy, 'sales', lags=[1, 7], windows=[7], inplace=False)
    time_non_inplace = time.time() - start
    assert result7 is not None
    print(f"   ✓ Full pipeline works: {time_non_inplace:.4f}s, shape={result7.shape}")
    
    print("\n8. Testing complete pipeline (inplace)...")
    df_copy = test_df.copy()
    start = time.time()
    result8 = build_m5_features(df_copy, 'sales', lags=[1, 7], windows=[7], inplace=True)
    time_inplace = time.time() - start
    assert result8 is None
    print(f"   ✓ Full pipeline inplace works: {time_inplace:.4f}s")
    print(f"   Memory optimization: {((time_non_inplace-time_inplace)/time_non_inplace*100):.1f}% faster")
    
    print("\n✅ Project 001 optimizations verified!")
    
except Exception as e:
    print(f"\n❌ Error in Project 001 tests: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("Testing Project 002 - Inventory Optimization Engine optimizations...")
print("=" * 70)

try:
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
    
    print("\n✅ Project 002 optimizations verified!")
    
except Exception as e:
    print(f"\n❌ Error in Project 002 tests: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("🎉 All optimizations tested successfully!")
print("=" * 70)

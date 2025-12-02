"""Demo script for Inventory Optimization Engine."""

import sys
from pathlib import Path
import pandas as pd

# Add project root to path for proper imports
sys.path.insert(0, str(Path(__file__).parent))

from src.data import DataLoader, DemandCalculator
from src.inventory import ABCAnalyzer, SafetyStockCalculator, ReorderPointCalculator, EOQCalculator
from src.optimization import InventoryOptimizer, CostCalculator
from src.utils import load_config, setup_logging


def main():
    """Run demo of inventory optimization system."""

    print("=" * 70)
    print("📦 INVENTORY OPTIMIZATION ENGINE - DEMO")
    print("=" * 70)
    print()

    # Setup logging
    setup_logging(level='INFO')

    # Load configuration
    print("📋 Loading configuration...")
    config = load_config('config/config.yaml')
    print("✅ Configuration loaded")
    print()

    # Load data
    print("📊 Loading M5 Walmart data...")
    loader = DataLoader(config['data']['raw_data_path'])
    raw_data = loader.load_raw_data()
    print(f"✅ Loaded {len(loader.sales_train)} products")
    print()

    # Process data for sample stores
    print("🔄 Processing data (using sample stores for demo)...")
    full_data = loader.process_data()

    # Filter to sample stores for quick demo
    sample_stores = ['CA_1', 'TX_1']
    data = loader.filter_data(
        full_data,
        stores=sample_stores,
        start_date='2015-01-01',
        end_date='2016-03-27'
    )
    print(f"✅ Processed {len(data):,} records")
    print(f"   Stores: {', '.join(sample_stores)}")
    print(f"   Date range: {data['date'].min()} to {data['date'].max()}")
    print()

    # Calculate demand statistics
    print("📈 Calculating demand statistics...")
    calc = DemandCalculator()
    demand_stats = calc.calculate_demand_statistics(
        data,
        group_cols=['store_id', 'item_id']
    )
    print(f"✅ Calculated stats for {len(demand_stats)} store-item combinations")
    print()

    # ABC/XYZ Analysis
    print("🎯 Performing ABC/XYZ classification...")
    abc_analyzer = ABCAnalyzer(
        abc_thresholds=config['inventory']['abc_thresholds'],
        xyz_thresholds=config['inventory']['xyz_thresholds']
    )
    classified = abc_analyzer.perform_combined_analysis(
        demand_stats,
        value_col='revenue_sum',
        cv_col='demand_cv'
    )

    # Show ABC-XYZ distribution
    print("\n📊 ABC-XYZ Classification Matrix:")
    matrix = classified.groupby(['abc_class', 'xyz_class']).size().unstack(fill_value=0)
    print(matrix)
    print()

    # Calculate EOQ
    print("💰 Calculating Economic Order Quantity...")
    eoq_calc = EOQCalculator(
        ordering_cost=config['inventory']['costs']['ordering_cost'],
        holding_cost_rate=config['inventory']['costs']['holding_cost_rate']
    )
    inventory_data = eoq_calc.calculate_for_dataframe(classified)
    print(f"✅ Average EOQ: {inventory_data['eoq'].mean():.0f} units")
    print(f"   Total annual cost: ${inventory_data['total_annual_cost'].sum():,.0f}")
    print()

    # Calculate safety stock
    print("🛡️ Calculating safety stock levels...")
    ss_calc = SafetyStockCalculator(
        service_level=0.95,
        lead_time=config['inventory']['lead_time']['default']
    )
    inventory_data = ss_calc.calculate_for_dataframe(inventory_data, method='basic')
    print(f"✅ Average safety stock: {inventory_data['safety_stock'].mean():.0f} units")
    print()

    # Calculate reorder points
    print("📍 Calculating reorder points...")
    rop_calc = ReorderPointCalculator(
        lead_time=config['inventory']['lead_time']['default']
    )
    inventory_data = rop_calc.calculate_for_dataframe(inventory_data)
    print(f"✅ Average reorder point: {inventory_data['reorder_point'].mean():.0f} units")
    print()

    # Show top items by revenue
    print("🏆 Top 10 Items by Revenue:")
    print("-" * 70)
    top_items = inventory_data.nlargest(10, 'revenue_sum')[
        ['item_id', 'store_id', 'abc_xyz_class', 'revenue_sum', 'eoq',
         'safety_stock', 'reorder_point', 'total_annual_cost']
    ]
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    print(top_items.to_string(index=False))
    print()

    # Cost analysis by ABC class
    print("💵 Cost Analysis by ABC Class:")
    print("-" * 70)
    cost_calc = CostCalculator(
        holding_cost_rate=config['inventory']['costs']['holding_cost_rate'],
        ordering_cost=config['inventory']['costs']['ordering_cost']
    )

    abc_costs = inventory_data.groupby('abc_class').agg({
        'revenue_sum': 'sum',
        'eoq': 'mean',
        'safety_stock': 'mean',
        'total_annual_cost': 'sum',
        'item_id': 'count'
    }).round(2)
    abc_costs.columns = ['Total Revenue', 'Avg EOQ', 'Avg Safety Stock',
                         'Total Annual Cost', 'Item Count']
    print(abc_costs)
    print()

    # Recommendations
    print("💡 Sample Inventory Policy Recommendations:")
    print("-" * 70)

    for abc_xyz_class in ['AX', 'BX', 'CX']:
        policy = abc_analyzer.recommend_inventory_policy(abc_xyz_class)
        items_in_class = len(inventory_data[inventory_data['abc_xyz_class'] == abc_xyz_class])

        print(f"\n{abc_xyz_class} Items ({items_in_class} items):")
        print(f"  Policy: {policy['policy']}")
        print(f"  Service Level: {policy['service_level']}")
        print(f"  Safety Stock: {policy['safety_stock']}")
        print(f"  Review Frequency: {policy['review_frequency']}")
        print(f"  Attention Level: {policy['attention']}")

    print()
    print("=" * 70)
    print("✅ Demo Complete!")
    print("=" * 70)
    print()
    print("Next Steps:")
    print("1. Explore Jupyter notebooks in notebooks/")
    print("2. Run full optimization: python scripts/run_optimization.py")
    print("3. Customize config/config.yaml for your needs")
    print("4. Review documentation in docs/")
    print()


if __name__ == "__main__":
    main()

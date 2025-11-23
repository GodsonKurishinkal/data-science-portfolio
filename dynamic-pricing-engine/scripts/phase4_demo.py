"""
Phase 4 Demonstration: Demand Response Modeling

This script demonstrates the demand response modeling capabilities:
1. Load elasticity estimates from Phase 3
2. Predict demand at various price points
3. Generate demand curves
4. Simulate pricing scenarios
5. Incorporate seasonality and promotions
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.pricing.demand_response import DemandResponseModel, create_standard_scenarios
from src.pricing.elasticity import ElasticityAnalyzer

print("=" * 80)
print("PHASE 4: DEMAND RESPONSE MODELING DEMONSTRATION")
print("=" * 80)
print()

# Step 1: Load data and elasticity estimates
print("[1] Loading data and elasticity estimates...")
print("-" * 80)

# Load processed data
try:
    df = pd.read_parquet('data/processed/processed_pricing_data.parquet')
    print(f"✓ Loaded {len(df):,} rows of pricing data")
    print(f"✓ Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"✓ Unique products: {df['item_id'].nunique()}")
except Exception as e:
    print(f"⚠ Warning: Could not load processed data: {e}")
    print("  Using sample data for demonstration...")
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=180, freq='D')
    products = [f'FOODS_{i}_001' for i in range(1, 6)]
    
    data = []
    for product in products:
        base_price = np.random.uniform(3, 10)
        base_demand = np.random.uniform(50, 200)
        elasticity = np.random.uniform(-2.0, -0.5)
        
        for date in dates:
            price = base_price * (1 + np.random.normal(0, 0.1))
            demand = base_demand * (price / base_price) ** elasticity
            demand = max(0, demand + np.random.normal(0, 10))
            
            data.append({
                'item_id': product,
                'product_id': product,
                'date': date,
                'sell_price': price,
                'sales': demand
            })
    
    df = pd.DataFrame(data)

print()

# Step 2: Initialize demand response model
print("[2] Initializing Demand Response Model...")
print("-" * 80)

model = DemandResponseModel(
    elasticity_analyzer=ElasticityAnalyzer(method='log-log'),
    use_confidence_intervals=True,
    confidence_level=0.95
)

print("✓ Model initialized with 95% confidence intervals")
print()

# Step 3: Load or calculate elasticities
print("[3] Calculating price elasticities...")
print("-" * 80)

# Prepare data for elasticity calculation
price_data = df[['product_id', 'sell_price']].copy()
sales_data = df[['product_id', 'sales']].copy()

model.load_elasticity_from_analyzer(price_data, sales_data)

elasticity_summary = model.get_elasticity_summary()
print(f"✓ Calculated elasticity for {len(elasticity_summary)} products")
print()
print("Sample elasticity estimates:")
print(elasticity_summary.head(10).to_string())
print()

# Step 4: Learn seasonality patterns
print("[4] Learning seasonality patterns...")
print("-" * 80)

model.learn_seasonality(df, product_id_col='product_id', date_col='date', sales_col='sales')
print(f"✓ Learned seasonality for {len(model.seasonality_patterns)} products")
print()

# Step 5: Single product prediction
print("[5] Single Product Demand Prediction")
print("-" * 80)

# Select a product for detailed analysis
sample_product = elasticity_summary.iloc[0]
product_id = sample_product['product_id']
elasticity = sample_product['elasticity']

# Get baseline metrics
product_data = df[df['product_id'] == product_id]
baseline_demand = product_data['sales'].mean()
current_price = product_data['sell_price'].mean()

print(f"Product: {product_id}")
print(f"Baseline demand: {baseline_demand:.1f} units")
print(f"Current price: ${current_price:.2f}")
print(f"Elasticity: {elasticity:.3f}")
print()

# Predict demand at different price points
print("Demand predictions at various prices:")
print()

test_prices = [current_price * 0.85, current_price * 0.95, current_price, current_price * 1.05, current_price * 1.15]

results = []
for price in test_prices:
    prediction = model.predict_demand_at_price(
        product_id=product_id,
        new_price=price,
        baseline_demand=baseline_demand,
        current_price=current_price,
        elasticity=elasticity
    )
    results.append(prediction)

results_df = pd.DataFrame(results)
print(results_df[['new_price', 'price_change_pct', 'predicted_demand', 'demand_change_pct', 'revenue_new', 'revenue_change_pct']].to_string())
print()

# Step 6: Generate demand curve
print("[6] Generating Demand Curve")
print("-" * 80)

demand_curve = model.predict_demand_curve(
    product_id=product_id,
    baseline_demand=baseline_demand,
    current_price=current_price,
    price_range=(current_price * 0.7, current_price * 1.3),
    num_points=30,
    elasticity=elasticity
)

print(f"Generated demand curve with {len(demand_curve)} price points")
print()
print("Key points on demand curve:")
print(demand_curve[['price', 'demand', 'revenue', 'is_optimal']].head(10).to_string())
print()

optimal_price = demand_curve[demand_curve['is_optimal']]['price'].iloc[0]
optimal_revenue = demand_curve[demand_curve['is_optimal']]['revenue'].iloc[0]
print(f"💰 Optimal price: ${optimal_price:.2f} (revenue: ${optimal_revenue:.2f})")
print(f"   Price change from current: {((optimal_price / current_price) - 1) * 100:+.1f}%")
print()

# Step 7: Scenario simulation
print("[7] Pricing Scenario Analysis")
print("-" * 80)

scenarios = create_standard_scenarios()
scenario_results = model.simulate_price_scenarios(
    product_id=product_id,
    baseline_demand=baseline_demand,
    current_price=current_price,
    scenarios=scenarios,
    elasticity=elasticity
)

print("Scenario comparison:")
print()
print(scenario_results[['scenario_name', 'new_price', 'predicted_demand', 'revenue_new', 'revenue_change_pct']].to_string())
print()

# Find best scenario
best_scenario = scenario_results.loc[scenario_results['revenue_new'].idxmax()]
print(f"🏆 Best scenario: {best_scenario['scenario_name']}")
print(f"   Revenue increase: {best_scenario['revenue_change_pct']:.1f}%")
print()

# Step 8: Seasonality demonstration
print("[8] Seasonality Impact Analysis")
print("-" * 80)

if product_id in model.seasonality_patterns:
    print(f"Day-of-week seasonality for {product_id}:")
    dow_pattern = model.seasonality_patterns[product_id]['day_of_week']
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    
    for day_idx, day_name in enumerate(days):
        if day_idx in dow_pattern:
            factor = dow_pattern[day_idx]
            print(f"  {day_name}: {factor:.2f}x ({(factor - 1) * 100:+.1f}%)")
    print()
    
    # Compare predictions for weekday vs weekend
    weekday = datetime(2024, 11, 18)  # Monday
    weekend = datetime(2024, 11, 16)  # Saturday
    
    pred_weekday = model.predict_demand_at_price(
        product_id=product_id,
        new_price=current_price,
        baseline_demand=baseline_demand,
        current_price=current_price,
        elasticity=elasticity,
        date=weekday
    )
    
    pred_weekend = model.predict_demand_at_price(
        product_id=product_id,
        new_price=current_price,
        baseline_demand=baseline_demand,
        current_price=current_price,
        elasticity=elasticity,
        date=weekend
    )
    
    print(f"Weekday demand: {pred_weekday['predicted_demand']:.1f} units")
    print(f"Weekend demand: {pred_weekend['predicted_demand']:.1f} units")
    print(f"Weekend lift: {((pred_weekend['predicted_demand'] / pred_weekday['predicted_demand']) - 1) * 100:+.1f}%")
    print()

# Step 9: Bulk predictions
print("[9] Bulk Predictions for Top Products")
print("-" * 80)

# Get top 5 products by total sales
top_products = df.groupby('product_id')['sales'].sum().nlargest(5).index

bulk_data = []
for pid in top_products:
    if pid in elasticity_summary['product_id'].values:
        product_data = df[df['product_id'] == pid]
        elasticity_val = elasticity_summary[elasticity_summary['product_id'] == pid]['elasticity'].iloc[0]
        
        bulk_data.append({
            'product_id': pid,
            'baseline_demand': product_data['sales'].mean(),
            'current_price': product_data['sell_price'].mean(),
            'new_price': product_data['sell_price'].mean() * 0.9,  # 10% discount
            'elasticity': elasticity_val
        })

if bulk_data:
    bulk_df = pd.DataFrame(bulk_data)
    bulk_results = model.predict_bulk(bulk_df, elasticity_col='elasticity')
    
    print(f"10% price reduction impact on top {len(bulk_results)} products:")
    print()
    print(bulk_results[['product_id', 'current_price', 'new_price', 'predicted_demand', 'demand_change_pct', 'revenue_change_pct']].to_string())
    print()
    
    total_revenue_change = bulk_results['revenue_change'].sum()
    print(f"💵 Total revenue impact: ${total_revenue_change:.2f} ({(total_revenue_change / bulk_results['revenue_base'].sum() * 100):+.1f}%)")
    print()

# Summary
print("=" * 80)
print("PHASE 4 SUMMARY")
print("=" * 80)
print()
print(f"✓ Implemented DemandResponseModel class")
print(f"✓ Calculated elasticity for {len(elasticity_summary)} products")
print(f"✓ Learned seasonality patterns for {len(model.seasonality_patterns)} products")
print(f"✓ Generated demand curves showing optimal pricing")
print(f"✓ Simulated {len(scenarios)} pricing scenarios")
print(f"✓ Incorporated seasonality and promotional effects")
print()
print("Key capabilities demonstrated:")
print("  • Predict demand at any price point")
print("  • Generate complete demand curves")
print("  • Identify revenue-maximizing prices")
print("  • Analyze pricing scenarios")
print("  • Account for seasonality (day of week, monthly patterns)")
print("  • Model promotional lifts")
print("  • Compute confidence intervals")
print("  • Bulk predictions for multiple products")
print()
print("Next steps:")
print("  → Phase 5: Price Optimization Engine")
print("  → Phase 6: Markdown Strategy Module")
print("=" * 80)

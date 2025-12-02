# 🚀 Quick Start Guide

Get up and running with the Inventory Optimization Engine in 5 minutes!

## ⚡ 5-Minute Setup

### Step 1: Environment Setup

```bash
# Navigate to project directory
cd inventory-optimization-engine

# Activate portfolio virtual environment
source ../venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

### Step 2: Verify Data Access

The project uses data from `demand-forecasting-system`:

```bash
# Check data availability
ls ../demand-forecasting-system/data/raw/
```

You should see:
- `calendar.csv`
- `sales_train_evaluation.csv`
- `sell_prices.csv`

### Step 3: Run Quick Demo

```bash
python demo.py
```

Expected output:
```
✅ Loading M5 data...
✅ Calculating demand statistics...
✅ Performing ABC/XYZ analysis...
✅ Calculating EOQ and safety stock...
✅ Generating recommendations...

Top 10 Inventory Recommendations:
[Table showing optimized inventory parameters]
```

## 📊 Interactive Analysis

### Launch Jupyter Notebooks

```bash
jupyter notebook notebooks/
```

Recommended order:
1. **`exploratory/01_inventory_analysis.ipynb`** - Data exploration
2. **`analysis/02_abc_xyz_classification.ipynb`** - Classification analysis
3. **`analysis/03_optimization_results.ipynb`** - Optimization results

## 💻 Basic Usage Examples

### Example 1: ABC Analysis

```python
from src.data import DataLoader, DemandCalculator
from src.inventory import ABCAnalyzer
from src.utils import load_config

# Load configuration
config = load_config('config/config.yaml')

# Load data
loader = DataLoader(config['data']['raw_data_path'])
data = loader.process_data()

# Filter to specific store
store_data = loader.filter_data(data, stores=['CA_1'])

# Calculate demand statistics
calc = DemandCalculator()
stats = calc.calculate_demand_statistics(store_data)

# Perform ABC/XYZ analysis
analyzer = ABCAnalyzer()
classified = analyzer.perform_combined_analysis(stats)

# Get recommendations
for _, item in classified.head(10).iterrows():
    policy = analyzer.recommend_inventory_policy(item['abc_xyz_class'])
    print(f"{item['item_id']}: {item['abc_xyz_class']} - {policy['policy']}")
```

### Example 2: Calculate Safety Stock

```python
from src.inventory import SafetyStockCalculator

# Initialize calculator with 95% service level
ss_calc = SafetyStockCalculator(service_level=0.95, lead_time=7)

# Calculate for single item
safety_stock = ss_calc.calculate_basic_safety_stock(
    demand_std=5.2,  # Standard deviation of daily demand
    lead_time=7      # 7-day lead time
)

print(f"Safety Stock: {safety_stock:.0f} units")

# Calculate for multiple service levels
ss_by_level = ss_calc.calculate_by_service_level(
    demand_std=5.2,
    service_levels={'85%': 0.85, '95%': 0.95, '99%': 0.99}
)

for level, quantity in ss_by_level.items():
    print(f"Service Level {level}: {quantity:.0f} units")
```

### Example 3: Calculate EOQ

```python
from src.inventory import EOQCalculator

# Initialize with cost parameters
eoq_calc = EOQCalculator(
    ordering_cost=100,        # $100 per order
    holding_cost_rate=0.25    # 25% of unit cost per year
)

# Calculate EOQ
annual_demand = 5000  # units per year
unit_cost = 25       # $25 per unit

eoq = eoq_calc.calculate_eoq(annual_demand, unit_cost)
print(f"Economic Order Quantity: {eoq:.0f} units")

# Get total cost breakdown
costs = eoq_calc.calculate_total_cost(eoq, annual_demand, unit_cost)
print(f"Annual Ordering Cost: ${costs['ordering_cost']:,.2f}")
print(f"Annual Holding Cost: ${costs['holding_cost']:,.2f}")
print(f"Total Cost: ${costs['total_cost']:,.2f}")
```

### Example 4: Full Optimization Pipeline

```python
from src.optimization import InventoryOptimizer
from src.utils import load_config

# Load configuration
config = load_config('config/config.yaml')

# Initialize optimizer
optimizer = InventoryOptimizer(config)

# Run optimization (assuming you have demand_stats)
optimized = optimizer.optimize_inventory_policy(demand_stats)

# Get top recommendations
recommendations = optimizer.generate_recommendations(optimized, top_n=20)

# Save results
recommendations.to_csv('data/processed/recommendations.csv', index=False)
optimized.to_csv('data/processed/optimized_inventory.csv', index=False)

print("✅ Optimization complete!")
print(f"Total items optimized: {len(optimized)}")
print(f"Total annual cost: ${optimized['total_annual_cost'].sum():,.2f}")
```

## 🎯 Common Use Cases

### Use Case 1: Store-Level Optimization

```python
# Optimize inventory for a specific store
store_id = 'CA_1'
store_data = data[data['store_id'] == store_id]
store_stats = calc.calculate_demand_statistics(store_data)
store_optimized = optimizer.optimize_inventory_policy(store_stats)
```

### Use Case 2: Category-Level Analysis

```python
# Analyze by product category
category_analysis = data.groupby('cat_id').agg({
    'sales': ['sum', 'mean', 'std'],
    'revenue': 'sum'
}).reset_index()
```

### Use Case 3: Service Level Impact Analysis

```python
from src.optimization import CostCalculator

cost_calc = CostCalculator()

# Compare costs across service levels
tradeoff = cost_calc.calculate_service_level_cost_tradeoff(
    demand_mean=10,
    demand_std=3,
    unit_cost=25,
    service_levels=[0.85, 0.90, 0.95, 0.99]
)

print(tradeoff)
```

## 📈 Interpreting Results

### ABC-XYZ Classification

- **AX**: High value + Predictable → Monitor closely, high service level
- **AY/AZ**: High value + Variable → Extra safety stock
- **CZ**: Low value + Erratic → Consider discontinuation

### Key Metrics

- **EOQ**: Optimal order quantity that minimizes costs
- **Safety Stock**: Buffer against demand variability
- **Reorder Point**: Trigger level for placing orders
- **Service Level**: % of demand met from stock

### Cost Breakdown

- **Holding Cost**: Cost to store inventory (typically 20-30% of value)
- **Ordering Cost**: Fixed cost per order placement
- **Stockout Cost**: Lost sales/customer dissatisfaction

## 🔧 Configuration

Edit `config/config.yaml` to adjust:

- Service level targets by ABC class
- Cost parameters (holding, ordering, stockout)
- Lead times
- Review periods
- ABC/XYZ thresholds

## 🐛 Troubleshooting

### Issue: "Module not found" errors

```bash
# Reinstall in development mode
pip install -e .
```

### Issue: Data not found

```bash
# Check data path in config.yaml
# Verify project-001 data exists
ls ../demand-forecasting-system/data/raw/
```

### Issue: Memory errors with large datasets

```python
# Use sampling in config.yaml
config['analysis']['sample_size'] = 10000
```

## 📚 Next Steps

1. ✅ Complete Quick Start (You are here!)
2. 📓 Explore Jupyter notebooks
3. 📖 Read [Project Roadmap](PROJECT_ROADMAP.md)
4. 🔬 Review [Model Card](docs/MODEL_CARD.md)
5. 🧪 Run tests: `pytest tests/ -v`
6. 🚀 Customize for your use case

## 💡 Tips

- Start with a single store to understand the workflow
- Use ABC classification to prioritize optimization efforts
- Adjust service levels based on business priorities
- Monitor actual vs. predicted stockout rates
- Iterate on cost parameters for your industry

## 🆘 Getting Help

- Check [README.md](README.md) for detailed documentation
- Review example notebooks in `notebooks/`
- Open an issue on GitHub
- Review test files in `tests/` for more examples

---

Ready to optimize! 🚀📦

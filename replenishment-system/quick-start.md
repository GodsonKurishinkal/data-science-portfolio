# Quick Start Guide - Replenishment System

Get started with the Replenishment System in under 5 minutes.

## Prerequisites

- Python 3.11+
- pandas, numpy, scipy, PyYAML

## Installation

```bash
# Clone repository
cd data-science-portfolio/replenishment-system

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

## Quick Demo

Run the interactive demo to explore all features:

```bash
python demo.py
```

Select options to see:
1. ABC/XYZ Classification
2. Safety Stock Calculations  
3. Replenishment Policies
4. Alert Generation
5. Full Engine Demo
6. Multi-Scenario Comparison

## Basic Usage

### 1. Simple Replenishment Calculation

```python
import pandas as pd
from src.engine.replenishment import ReplenishmentEngine

# Create inventory data
inventory = pd.DataFrame({
    "item_id": ["SKU001", "SKU002", "SKU003"],
    "current_stock": [100, 50, 200],
    "daily_demand_rate": [20, 15, 10],
    "demand_std": [5, 4, 3],
    "max_capacity": [500, 500, 500],
    "revenue": [10000, 5000, 2000],
})

# Configure engine
config = {
    "scenarios": {
        "dc_to_store": {
            "policy_type": "periodic_review",
            "review_period": 1,
            "lead_time": 2,
            "service_level": 0.97,
        }
    }
}

# Run engine
engine = ReplenishmentEngine(config_dict=config)
result = engine.run(scenario="dc_to_store", inventory_data=inventory)

# View results
print(f"Items needing order: {result.summary['items_needing_order']}")
print(result.recommendations[['item_id', 'recommended_quantity', 'needs_order']])
```

### 2. Using Classification

```python
from src.classification.abc_classifier import ABCClassifier
from src.classification.xyz_classifier import XYZClassifier

# ABC Classification (by revenue)
abc = ABCClassifier(value_column="revenue", thresholds=(0.80, 0.95))
abc_result = abc.classify(inventory)

# XYZ Classification (by demand variability) - requires demand history
xyz = XYZClassifier(cv_thresholds=(0.5, 1.0))
xyz_result = xyz.classify(demand_history)

print(abc_result[['item_id', 'abc_class']])
```

### 3. Safety Stock Calculation

```python
from src.safety_stock.calculator import SafetyStockCalculator

calculator = SafetyStockCalculator(method="standard")

ss = calculator.calculate(
    demand_mean=100,      # Daily demand
    demand_std=20,        # Demand std dev
    lead_time=7,          # Days
    service_level=0.95    # Target service level
)

print(f"Safety Stock: {ss:.0f} units")
```

### 4. Generate Alerts

```python
from src.alerts.generator import AlertGenerator

generator = AlertGenerator()
alerts = generator.generate(result.recommendations)

for alert in alerts[:5]:
    print(f"[{alert.severity.value}] {alert.item_id}: {alert.message}")
```

## Supported Scenarios

| Scenario | Description | Typical Lead Time |
|----------|-------------|-------------------|
| `supplier_to_dc` | Supplier to Distribution Center | 7-21 days |
| `dc_to_store` | DC to Retail Store | 1-3 days |
| `store_to_dc` | Returns from Store to DC | 1-7 days |
| `storage_to_picking` | Reserve to Forward Pick | 0.5-1 day |
| `backroom_to_floor` | Store Backroom to Sales Floor | 0.1-0.5 days |
| `cross_dock` | Cross-dock Flow-through | 0.25 days |
| `inter_store_transfer` | Between Stores | 1-3 days |
| `ecommerce_fulfillment` | E-commerce FC | 3-7 days |

## Configuration

Edit `config/config.yaml` to customize:
- Classification thresholds (ABC/XYZ)
- Service level matrix
- Alert thresholds
- Scenario-specific parameters

## Running Tests

```bash
# All tests
pytest tests/ -v

# Specific test file
pytest tests/test_policies.py -v

# With coverage
pytest tests/ --cov=src
```

## Need Help?

- Full documentation: [README.md](README.md)
- Configuration guide: [config/config.yaml](config/config.yaml)
- API examples: [demo.py](demo.py)

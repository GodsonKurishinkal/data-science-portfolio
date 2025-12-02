# Data Directory

This directory contains data files for the Universal Replenishment Engine.

## Structure

```
data/
├── raw/           # Original source data
├── processed/     # Transformed data ready for analysis
└── external/      # Reference data (calendars, holidays, etc.)
```

## Data Requirements

### Inventory Data

Required columns:
- `item_id`: Unique item identifier
- `current_stock`: Current inventory level
- `daily_demand_rate`: Average daily demand

Optional columns:
- `max_capacity`: Maximum storage capacity
- `demand_std`: Demand standard deviation
- `unit_cost`: Unit cost for EOQ calculations
- `revenue`: Revenue for ABC classification
- `lead_time`: Item-specific lead time
- `category`: Product category

### Demand History Data

Required columns:
- `item_id`: Item identifier
- `date`: Transaction date
- `quantity`: Demand quantity

Optional columns:
- `price`: Selling price
- `location_id`: Location identifier

### Source Inventory Data

Required columns:
- `item_id`: Item identifier
- `available_quantity`: Available quantity at source

## Sample Data Generation

Use the demo script to generate sample data:

```python
from demo import generate_sample_inventory, generate_sample_demand

inventory = generate_sample_inventory(n_items=100)
demand = generate_sample_demand(n_items=100, n_days=90)
```

## Notes

- Data files are gitignored to prevent large files in version control
- Use `scripts/` for data download and preprocessing scripts

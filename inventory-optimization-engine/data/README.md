# Data README

## Directory Structure

```
data/
├── processed/          # Processed inventory data
└── external/           # Additional data sources
```

## Data Source

This project uses the **M5 Walmart dataset** from the sibling project:
```
../demand-forecasting-system/data/raw/
```

The raw data includes:
- `calendar.csv` - Calendar information and events
- `sales_train_evaluation.csv` - Historical sales data
- `sell_prices.csv` - Product pricing information

## Processed Data

Processed data files will be generated in `processed/` directory:

- `demand_statistics.csv` - Aggregated demand statistics by store-item
- `abc_xyz_classified.csv` - Items with ABC-XYZ classification
- `optimized_inventory.csv` - Complete optimization results
- `recommendations.csv` - Top inventory policy recommendations

## Data Processing

To process the raw data:

```python
from src.data import DataLoader, DemandCalculator

loader = DataLoader('../demand-forecasting-system/data/raw')
data = loader.process_data()

calc = DemandCalculator()
stats = calc.calculate_demand_statistics(data)
```

## External Data (Optional)

Additional data sources can be placed in `external/`:
- Supplier lead times
- Warehouse capacities
- Transportation costs
- Seasonal factors

## Notes

- All data files are gitignored except this README
- Use `.gitkeep` files to maintain directory structure
- Data is shared with project-001 to avoid duplication

# Data Directory

This directory contains all data files used in the demand forecasting system.

## Structure

### `raw/`
**Purpose**: Original, immutable data files as received from the source.

**Guidelines**:
- Never modify files in this directory
- Store original data exactly as received
- Document data source and collection date
- Files in this directory are excluded from Git (see .gitignore)

**Example files**:
- `sales_data_2023.csv`
- `product_catalog.xlsx`
- `promotional_calendar.json`

### `processed/`
**Purpose**: Cleaned and transformed data ready for analysis and modeling.

**Guidelines**:
- Store data after preprocessing and feature engineering
- Use consistent naming conventions
- Include version numbers or dates in filenames
- Files in this directory are excluded from Git (see .gitignore)

**Example files**:
- `train_data_v1.csv`
- `test_data_v1.csv`
- `features_20231101.parquet`

### `external/`
**Purpose**: External reference data from third-party sources.

**Guidelines**:
- Store supplementary data (holidays, weather, economic indicators)
- Document the source and update frequency
- Keep external data separate from primary datasets
- Files in this directory are excluded from Git (see .gitignore)

**Example files**:
- `us_holidays_2023.csv`
- `weather_data.csv`
- `economic_indicators.xlsx`

## Data Management Best Practices

1. **Version Control**: Track data versions and changes in a data catalog
2. **Documentation**: Maintain a data dictionary describing all fields
3. **Size Management**: Use efficient formats (Parquet, HDF5) for large files
4. **Security**: Never commit sensitive or proprietary data to Git
5. **Backup**: Keep backups of raw data in a secure location

## Sample Data

For demonstration purposes, you can generate sample data using:

```python
from src.data.preprocessing import load_data
# Sample data generation code here
```

## Notes

⚠️ **Important**: All data files are excluded from version control. Only code and documentation should be committed to Git.

For data access and permissions, contact the project maintainer.

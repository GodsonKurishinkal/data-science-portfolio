# M5 Competition Dataset - Walmart Demand Forecasting

## Dataset Overview

The M5 Competition dataset is a hierarchical sales dataset from Walmart, involving the unit sales of 3,049 products, classified in 3 product categories (Hobbies, Foods, and Household) and 7 product departments. The products are sold across 10 stores in three states (CA, TX, WI).

## Dataset Files

The dataset consists of the following files:

1. **calendar.csv** - Contains information about the dates on which the products are sold
2. **sales_train_validation.csv** - Historical daily unit sales data per product and store
3. **sell_prices.csv** - Contains information about the price of the products sold per store and date
4. **sales_train_evaluation.csv** - Extended version of sales data (optional)
5. **sample_submission.csv** - Sample submission file format

## Download Instructions

### Option 1: Manual Download from Kaggle

1. Visit the M5 Competition page: https://www.kaggle.com/competitions/m5-forecasting-accuracy/data
2. Download all CSV files
3. Place them in: `data/raw/`

### Option 2: Using Kaggle API (Recommended)

```bash
# Install Kaggle API
pip install kaggle

# Set up Kaggle credentials (one-time setup)
# Download kaggle.json from: https://www.kaggle.com/settings
# Place it in ~/.kaggle/kaggle.json
# Set permissions: chmod 600 ~/.kaggle/kaggle.json

# Navigate to project directory
cd /Users/godsonkurishinkal/Projects/data-science-portfolio/demand-forecasting-system

# Download M5 dataset
kaggle competitions download -c m5-forecasting-accuracy -p data/raw/

# Unzip files
cd data/raw/
unzip m5-forecasting-accuracy.zip
rm m5-forecasting-accuracy.zip
```

## Dataset Structure

### calendar.csv
- **date**: Calendar date (1,969 days from 2011-01-29 to 2016-06-19)
- **wm_yr_wk**: Walmart year-week identifier
- **weekday**: Day of week
- **wday**: Day number (1-7)
- **month**: Month
- **year**: Year
- **event_name_1, event_name_2**: Special event names
- **event_type_1, event_type_2**: Event types (Cultural, National, Religious, Sporting)
- **snap_CA, snap_TX, snap_WI**: SNAP (food stamps) indicators by state

### sales_train_validation.csv
- **id**: Product and store identifier (e.g., HOBBIES_1_001_CA_1)
- **item_id**: Product identifier
- **dept_id**: Department identifier
- **cat_id**: Category identifier (HOBBIES, HOUSEHOLD, FOODS)
- **store_id**: Store identifier (CA_1, CA_2, ..., WI_3)
- **state_id**: State identifier (CA, TX, WI)
- **d_1, d_2, ..., d_1913**: Daily sales for each day

### sell_prices.csv
- **store_id**: Store identifier
- **item_id**: Product identifier
- **wm_yr_wk**: Week identifier
- **sell_price**: Selling price of the product for the week

## Data Size Information

- **Total products**: 3,049
- **Total stores**: 10
- **Time series length**: 1,913 days (~5.25 years)
- **States**: 3 (CA, TX, WI)
- **Product categories**: 3 (HOBBIES, FOODS, HOUSEHOLD)
- **Product departments**: 7

## Dataset Characteristics

### Hierarchical Structure
```
State Level (3)
├── Store Level (10)
│   ├── Category Level (3)
│   │   ├── Department Level (7)
│   │   │   └── Item Level (3,049)
```

### Key Challenges
1. **Hierarchical aggregation**: Sales at different levels need to be consistent
2. **Zero inflation**: Many products have zero sales on many days
3. **Intermittent demand**: Products may not sell every day
4. **Calendar effects**: Events, holidays, SNAP days affect sales
5. **Price dynamics**: Prices change over time and affect demand
6. **Store-specific patterns**: Different stores have different characteristics

## Expected File Sizes
- calendar.csv: ~63 KB
- sales_train_validation.csv: ~113 MB
- sell_prices.csv: ~143 MB

## Verification

After downloading, verify the data:

```python
import pandas as pd

# Load datasets
calendar = pd.read_csv('data/raw/calendar.csv')
sales = pd.read_csv('data/raw/sales_train_validation.csv')
prices = pd.read_csv('data/raw/sell_prices.csv')

print(f"Calendar shape: {calendar.shape}")
print(f"Sales shape: {sales.shape}")
print(f"Prices shape: {prices.shape}")

# Expected output:
# Calendar shape: (1969, 14)
# Sales shape: (30490, 1919)
# Prices shape: (6841121, 4)
```

## Next Steps

After downloading the data:
1. Run `notebooks/exploratory/01_data_exploration.ipynb` for initial EDA
2. Explore data quality and patterns
3. Begin feature engineering
4. Start modeling

## References

- Competition page: https://www.kaggle.com/competitions/m5-forecasting-accuracy
- Original paper: https://www.sciencedirect.com/science/article/pii/S0169207021001874
- Documentation: https://mofc.unic.ac.cy/m5-competition/

---

**Note**: This is a large dataset. Ensure you have at least 1GB of free disk space.

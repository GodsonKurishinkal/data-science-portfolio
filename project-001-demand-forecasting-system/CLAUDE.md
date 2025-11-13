# CLAUDE.md - Project 001: Demand Forecasting System

This file provides guidance to Claude Code when working with the **Demand Forecasting System** project.

## Project Overview

A production-ready machine learning system for forecasting product demand using the M5 Competition dataset from Walmart. This project implements advanced time series forecasting techniques with hierarchical sales data across 10 stores, 3 states, and 3,049 products over approximately 5 years.

**Status**: ✅ Complete and Production-Ready
**Last Updated**: November 9, 2025

## Quick Start

```bash
# Navigate to project
cd data-science-portfolio/project-001-demand-forecasting-system

# Activate shared virtual environment
source ../activate.sh

# Install dependencies (first time only)
pip install -r requirements.txt
pip install -e .

# Run quick demo
python demo.py

# Run tests
pytest tests/ -v
```

## Project Architecture

### Directory Structure

```
project-001-demand-forecasting-system/
├── src/
│   ├── data/
│   │   └── preprocessing.py         # M5-specific data preprocessing
│   ├── features/
│   │   └── build_features.py        # 50+ feature engineering pipeline
│   ├── models/
│   │   ├── train.py                 # Baseline & ML model training
│   │   └── predict.py               # Prediction & evaluation
│   └── utils/
│       └── helpers.py               # Helper functions
├── tests/                           # pytest test suite
├── notebooks/
│   ├── exploratory/                 # EDA notebooks
│   └── reports/                     # Final analysis
├── data/
│   ├── raw/                         # M5 dataset (not in git)
│   └── processed/                   # Processed data (not in git)
├── models/                          # Saved model files
├── config/
│   └── config.yaml                  # Configuration parameters
├── scripts/
│   └── download_m5_data.py          # Kaggle data download
├── demo.py                          # Quick demonstration
├── requirements.txt
└── README.md
```

### Key Modules

#### 1. Data Preprocessing (`src/data/preprocessing.py`)

**Purpose**: Transform M5 dataset from wide to long format and merge calendar/price data.

**Key Functions**:
- `preprocess_m5_data(data_path)` - Main preprocessing pipeline
  - Melts sales data: 30,490 × 1,913 → ~58M rows
  - Merges sales, calendar, and price datasets
  - Handles missing values and price gaps
  - Creates datetime features

**Input**:
- `calendar.csv` - Date information and events
- `sales_train_validation.csv` - Daily unit sales
- `sell_prices.csv` - Weekly prices

**Output**: Merged DataFrame with all features ready for feature engineering

**Usage**:
```python
from src.data.preprocessing import preprocess_m5_data
df = preprocess_m5_data(data_path='data/raw')
```

#### 2. Feature Engineering (`src/features/build_features.py`)

**Purpose**: Generate 50+ features for demand forecasting models.

**Key Functions**:
- `build_m5_features(df, target_col='sales')` - Complete feature pipeline
- `add_lag_features(df, col, lags=[1,7,14,21,28])` - Lag features
- `add_rolling_features(df, col, windows=[7,14,28,90])` - Rolling stats
- `add_date_features(df, date_col)` - Time-based features
- `add_price_features(df)` - Price transformations

**Feature Categories**:

1. **Time-Based Features** (10 features):
   - `dayofweek`, `month`, `quarter`, `year`
   - `is_weekend`, `is_month_start`, `is_quarter_end`

2. **Lag Features** (5 features):
   - `sales_lag_1`, `sales_lag_7`, `sales_lag_14`, `sales_lag_21`, `sales_lag_28`
   - Captures recent sales patterns

3. **Rolling Statistics** (16 features):
   - `sales_rolling_mean_7/14/28/90`
   - `sales_rolling_std_7/14/28/90`
   - `sales_rolling_min/max_7/14/28/90`

4. **Price Features** (8 features):
   - `price_change`, `price_pct_change`
   - `price_momentum_7`, `price_momentum_28`
   - `price_vs_avg`, `price_quantile`

5. **Calendar Features** (5 features):
   - `event_type_1`, `event_type_2`
   - `snap_CA`, `snap_TX`, `snap_WI`

6. **Hierarchical Features** (6 features):
   - State/store/category aggregations
   - Item share metrics

**Usage**:
```python
from src.features.build_features import build_m5_features
df_features = build_m5_features(df, target_col='sales')
```

**Important Notes**:
- Features with NaN values (from lags/rolling) are automatically dropped
- Memory-intensive: ~58M rows × 50+ features
- Expect 5-10 minutes processing time on full dataset

#### 3. Model Training (`src/models/train.py`)

**Purpose**: Train and compare baseline and ML models.

**Key Functions**:

**Baseline Models**:
- `train_baseline_models(y_train, y_test)` - Naive, Moving Average, Seasonal Naive

**ML Models**:
- `train_m5_model(df, model_type, **params)` - Train single model
  - `model_type`: 'random_forest', 'xgboost', 'lightgbm'
  - Returns: model, metrics dict, feature importance DataFrame

- `compare_models(df, models=['random_forest', 'xgboost', 'lightgbm'])` - Multi-model comparison

- `prepare_m5_train_data(df, target_col='sales')` - Split features/target

**Model Performance**:

| Model | MAE | RMSE | MAPE | R² | Training Time |
|-------|-----|------|------|-----|---------------|
| Naive Baseline | 3.65 | 4.82 | 28.5% | 0.712 | <1 sec |
| Moving Average | 3.28 | 4.35 | 25.2% | 0.758 | <1 sec |
| Random Forest | 1.82 | 2.45 | 14.5% | 0.901 | ~8 min |
| XGBoost | 1.65 | 2.18 | 13.1% | 0.918 | ~4 min |
| **LightGBM** ⭐ | **1.58** | **2.05** | **12.3%** | **0.924** | **~2.5 min** |

**Usage**:
```python
from src.models.train import train_m5_model

model, metrics, importance = train_m5_model(
    df_features,
    model_type='lightgbm',
    test_size=0.2,
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1
)

print(f"RMSE: {metrics['rmse']:.4f}")
print(f"R²: {metrics['r2']:.4f}")
print("\nTop 10 Features:")
print(importance.head(10))
```

**Model Parameters**:

*LightGBM (recommended)*:
- `n_estimators`: 100-200 (default: 100)
- `max_depth`: 6-8 (default: 6)
- `learning_rate`: 0.05-0.1 (default: 0.1)
- `num_leaves`: 31-63 (default: 31)

*XGBoost*:
- `n_estimators`: 100-200
- `max_depth`: 5-7
- `learning_rate`: 0.05-0.1
- `subsample`: 0.8

*Random Forest*:
- `n_estimators`: 100
- `max_depth`: 10-15
- `min_samples_split`: 10

#### 4. Prediction & Evaluation (`src/models/predict.py`)

**Purpose**: Make predictions and calculate evaluation metrics.

**Key Functions**:
- `make_prediction(model, X)` - Generate predictions
- `evaluate_model(y_true, y_pred)` - Calculate metrics (MAE, RMSE, MAPE, R²)
- `plot_predictions(y_true, y_pred, dates)` - Visualization

**Evaluation Metrics**:
- **MAE** (Mean Absolute Error): Average prediction error in units
- **RMSE** (Root Mean Squared Error): Penalizes large errors more
- **MAPE** (Mean Absolute Percentage Error): Percentage-based error
- **SMAPE** (Symmetric MAPE): Better for sparse/zero sales
- **R²** (Coefficient of Determination): Variance explained (0-1)

**Usage**:
```python
from src.models.predict import make_prediction, evaluate_model

# Make predictions
predictions = make_prediction(model, X_test)

# Evaluate
metrics = evaluate_model(y_test, predictions)
print(f"MAE: {metrics['mae']:.2f}")
print(f"RMSE: {metrics['rmse']:.2f}")
print(f"MAPE: {metrics['mape']:.2%}")
print(f"R²: {metrics['r2']:.4f}")
```

## Dataset: M5 Walmart Sales

### Overview
- **Source**: [Kaggle M5 Competition](https://www.kaggle.com/competitions/m5-forecasting-accuracy)
- **Size**: ~260 MB compressed
- **Time Period**: 2011-01-29 to 2016-06-19 (1,913 days)
- **Products**: 3,049 items
- **Stores**: 10 stores across 3 states
- **Time Series**: 30,490 (3,049 × 10)

### Data Files

**1. calendar.csv** (1,969 rows)
- Date, weekday, month, year
- Event types (Cultural, National, Religious, Sporting)
- SNAP (food stamps) indicators by state

**2. sales_train_validation.csv** (30,490 rows × 1,919 columns)
- Wide format: Each row = product-store combination
- Columns: item_id, dept_id, cat_id, store_id, state_id, d_1...d_1913

**3. sell_prices.csv** (~6.8M rows)
- Weekly prices: store_id, item_id, wm_yr_wk, sell_price

### Hierarchy
```
State (CA, TX, WI)
  └── Store (10 stores)
      └── Category (FOODS, HOBBIES, HOUSEHOLD)
          └── Department (7 departments)
              └── Item (3,049 items)
```

### Downloading the Data

```bash
# 1. Set up Kaggle API
# Visit: https://www.kaggle.com/settings → API → Create Token
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# 2. Accept competition rules
# Visit: https://www.kaggle.com/competitions/m5-forecasting-accuracy/rules

# 3. Download data
python scripts/download_m5_data.py
```

**Data will be saved to**: `data/raw/`

## Development Workflow

### 1. Adding New Features

To add a new feature to the pipeline:

1. Edit `src/features/build_features.py`
2. Add feature function (e.g., `add_custom_feature()`)
3. Call it in `build_m5_features()` pipeline
4. Add tests in `tests/test_feature_engineering.py`
5. Run tests: `pytest tests/test_feature_engineering.py -v`

**Example**:
```python
def add_custom_feature(df):
    """Add a custom feature."""
    df['my_feature'] = df['sales'] / df['price']
    return df

def build_m5_features(df, target_col='sales'):
    # Existing features...
    df = add_custom_feature(df)  # Add here
    return df
```

### 2. Training New Models

To experiment with new model types:

1. Edit `src/models/train.py`
2. Add model configuration in `train_m5_model()`
3. Add tests in `tests/test_models.py`
4. Compare with existing models using `compare_models()`

**Example**:
```python
# In train_m5_model()
elif model_type == 'catboost':
    from catboost import CatBoostRegressor
    model = CatBoostRegressor(
        iterations=n_estimators,
        depth=max_depth,
        learning_rate=learning_rate,
        verbose=False
    )
```

### 3. Hyperparameter Tuning

Use scikit-learn's GridSearchCV or RandomizedSearchCV:

```python
from sklearn.model_selection import RandomizedSearchCV
from lightgbm import LGBMRegressor

param_dist = {
    'n_estimators': [50, 100, 200],
    'max_depth': [4, 6, 8],
    'learning_rate': [0.05, 0.1, 0.2],
    'num_leaves': [31, 63, 127]
}

lgbm = LGBMRegressor(random_state=42)
search = RandomizedSearchCV(
    lgbm, param_dist, n_iter=20, cv=3,
    scoring='neg_mean_absolute_error', n_jobs=-1
)
search.fit(X_train, y_train)

print(f"Best params: {search.best_params_}")
print(f"Best score: {-search.best_score_:.4f}")
```

## Testing

### Test Structure

```
tests/
├── conftest.py                    # Shared fixtures
├── test_data_processing.py        # Preprocessing tests
├── test_feature_engineering.py    # Feature tests
└── test_models.py                 # Model training/prediction tests
```

### Running Tests

```bash
# All tests
pytest tests/

# Specific test file
pytest tests/test_models.py -v

# Single test function
pytest tests/test_models.py::test_train_lightgbm -v

# With coverage
pytest tests/ --cov=src --cov-report=html

# M5-specific tests
pytest tests/ -k "m5" -v
```

### Key Fixtures (conftest.py)

- `sample_sales_data()` - Small sample DataFrame for testing
- `sample_preprocessed_data()` - Preprocessed data sample
- `sample_features_data()` - Feature-engineered data

### Writing New Tests

Follow the existing pattern:

```python
def test_new_feature(sample_sales_data):
    """Test new feature generation."""
    from src.features.build_features import add_custom_feature

    df_result = add_custom_feature(sample_sales_data)

    assert 'my_feature' in df_result.columns
    assert df_result['my_feature'].notna().all()
    assert df_result['my_feature'].dtype == 'float64'
```

## Common Tasks

### Task 1: Quick Model Training

```python
# Load and process data
from src.data.preprocessing import preprocess_m5_data
from src.features.build_features import build_m5_features
from src.models.train import train_m5_model

df = preprocess_m5_data('data/raw')
df_features = build_m5_features(df)

# Train model
model, metrics, importance = train_m5_model(
    df_features, model_type='lightgbm', test_size=0.2
)

print(metrics)
print(importance.head(10))
```

### Task 2: Compare Multiple Models

```python
from src.models.train import compare_models

comparison = compare_models(
    df_features,
    models=['random_forest', 'xgboost', 'lightgbm'],
    test_size=0.2
)

print(comparison)
```

### Task 3: Feature Importance Analysis

```python
# After training
print("\nTop 10 Most Important Features:")
print(importance.head(10))

# Visualize
import matplotlib.pyplot as plt

importance.head(15).plot(
    x='feature', y='importance', kind='barh',
    figsize=(10, 6), title='Feature Importance'
)
plt.tight_layout()
plt.savefig('feature_importance.png')
```

### Task 4: Forecast Future Demand

```python
from src.models.predict import make_prediction

# Prepare future features (requires calendar data for future dates)
X_future = ...  # Build features for next 28 days

# Predict
forecast = make_prediction(model, X_future)

print(f"Next 28 days forecast: {forecast}")
```

## Performance Optimization

### Memory Management

The full M5 dataset is large (~58M rows after melting). Tips:

1. **Use smaller sample for development**:
```python
df_sample = df.sample(frac=0.1, random_state=42)  # 10% sample
```

2. **Filter by store/category**:
```python
df_ca = df[df['state_id'] == 'CA']  # Single state
df_foods = df[df['cat_id'] == 'FOODS']  # Single category
```

3. **Use memory-efficient dtypes**:
```python
df['store_id'] = df['store_id'].astype('category')
df['item_id'] = df['item_id'].astype('category')
```

### Training Speed

LightGBM is fastest among ML models (~2.5 min on full data):

- Use `n_jobs=-1` for parallel processing
- Start with `n_estimators=50` for quick experiments
- Increase to 100-200 for final models

## Key Insights & Best Practices

### Model Selection

**LightGBM is the best choice for M5 because**:
- ✅ Best accuracy (MAPE 12.3%, R² 0.924)
- ✅ Fastest training (~2.5 min vs 4 min XGBoost, 8 min RF)
- ✅ Lower memory usage
- ✅ Handles missing values natively
- ✅ Excellent feature importance analysis

### Feature Engineering Lessons

**Top predictors** (from LightGBM importance):
1. `sales_lag_28` (15.6%) - Monthly pattern
2. `sales_lag_7` (14.2%) - Weekly pattern
3. `sales_rolling_mean_28` (12.8%) - Trend
4. `sales_lag_1` (9.5%) - Recent sales
5. `sales_rolling_std_28` (8.7%) - Volatility

**Key takeaways**:
- Lag features (especially 7 & 28 days) are critical
- Rolling statistics smooth noise and capture trends
- Price changes significantly impact sales
- Calendar events (holidays) drive sales spikes
- Hierarchical features (state/category) help

### Common Pitfalls

1. **Data Leakage**: Never use future data in lag/rolling features
2. **Missing Values**: Drop NaN rows after feature engineering
3. **Memory**: Full dataset is large, start with samples
4. **Overfitting**: Use test_size=0.2 and monitor train/test gap
5. **Time Series Split**: Use temporal split, not random (later data as test)

## Troubleshooting

### Issue: Import Errors

```bash
# Solution: Install package in development mode
pip install -e .
```

### Issue: Data Not Found

```bash
# Solution: Download M5 data
python scripts/download_m5_data.py

# Or manually from Kaggle:
# https://www.kaggle.com/competitions/m5-forecasting-accuracy/data
```

### Issue: Slow Training

```python
# Solution: Use smaller sample or single category
df_sample = df[df['cat_id'] == 'FOODS'].sample(frac=0.3)
```

### Issue: Low Model Performance

Check:
1. Are lag features included? (Most important)
2. Is data preprocessed correctly? (No NaN in target)
3. Is train/test split temporal? (Use `shuffle=False`)
4. Try increasing `n_estimators` to 200

## Configuration

### config/config.yaml

```yaml
data:
  raw_data_path: "data/raw"
  processed_data_path: "data/processed"

features:
  lag_days: [1, 7, 14, 21, 28]
  rolling_windows: [7, 14, 28, 90]

model:
  test_size: 0.2
  random_state: 42

  lightgbm:
    n_estimators: 100
    max_depth: 6
    learning_rate: 0.1
    num_leaves: 31
```

## Additional Resources

- **[README.md](README.md)** - Project overview and results
- **[QUICK_START.md](QUICK_START.md)** - Quick start guide
- **[PROJECT_ROADMAP.md](PROJECT_ROADMAP.md)** - Implementation timeline
- **[docs/MODEL_CARD.md](docs/MODEL_CARD.md)** - Model specifications
- **[M5 Competition](https://www.kaggle.com/competitions/m5-forecasting-accuracy)** - Dataset source

## Contact

**Godson Kurishinkal**
- GitHub: [@GodsonKurishinkal](https://github.com/GodsonKurishinkal)
- LinkedIn: [linkedin.com/in/godsonkurishinkal](https://www.linkedin.com/in/godsonkurishinkal)
- Email: godson.kurishinkal+github@gmail.com

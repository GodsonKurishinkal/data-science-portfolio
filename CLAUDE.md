# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a **Supply Chain Analytics Portfolio** containing 5 interconnected projects demonstrating end-to-end supply chain intelligence from demand forecasting to real-time operations. The portfolio includes:

- **Project 001**: Demand Forecasting System (✅ Complete)
- **Project 002**: Inventory Optimization Engine (✅ Complete)
- **Project 003**: Dynamic Pricing Engine (Template)
- **Project 004**: Supply Chain Network Optimization (Template)
- **Project 005**: Real-Time Demand Sensing (Template)

## Environment Setup

### Python Version
This portfolio requires **Python 3.11.14** specifically. The virtual environment is configured for this version.

### Virtual Environment
A **single shared virtual environment** (`.venv`) at the root manages dependencies for all projects.

**Activate environment:**
```bash
# Quick activation (auto-creates .venv if missing)
source activate.sh

# Manual activation
source .venv/bin/activate
```

**Verify activation:**
```bash
which python  # Should show: .../data-science-portfolio/.venv/bin/python
python --version  # Should show: Python 3.11.14
```

**Deactivate:**
```bash
deactivate
```

### Installing Dependencies

Each project has its own `requirements.txt`. Install dependencies after activating the virtual environment:

```bash
# For a specific project
cd project-001-demand-forecasting-system
pip install -r requirements.txt

# Install in development mode (allows editing source code)
pip install -e .
```

The root also has `requirements-full.txt` with all installed packages from a complete setup.

## Common Development Commands

### Running Projects

#### Quick Demo
Each complete project has a `demo.py` for quick demonstrations:
```bash
cd project-001-demand-forecasting-system
python demo.py  # Runs complete pipeline in <3 minutes
```

#### Jupyter Notebooks
```bash
# Start Jupyter from project directory
cd project-001-demand-forecasting-system
jupyter notebook

# Notebooks are organized in:
# - notebooks/exploratory/  : EDA and experiments
# - notebooks/reports/      : Final analysis
```

### Testing

Projects use **pytest** with coverage reporting:

```bash
cd project-001-demand-forecasting-system

# Run all tests
pytest tests/

# Run with coverage report
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_data_processing.py -v

# Run tests matching a pattern
pytest tests/test_models.py -k "m5" -v
```

### Code Quality

```bash
# Linting with flake8 (config in .flake8)
flake8 src/ tests/

# Formatting with black
black src/ tests/

# Type checking with mypy
mypy src/
```

### Data Management

#### M5 Dataset (Projects 001, 002, 003)
The M5 Walmart dataset is used across multiple projects:

```bash
# Download M5 data (requires Kaggle API setup)
cd project-001-demand-forecasting-system
python scripts/download_m5_data.py
```

**Kaggle API Setup:**
1. Visit https://www.kaggle.com/settings → API → Create New API Token
2. Move kaggle.json to `~/.kaggle/`
3. Set permissions: `chmod 600 ~/.kaggle/kaggle.json`
4. Accept competition rules at: https://www.kaggle.com/competitions/m5-forecasting-accuracy/rules

#### Data Location
```
project-XXX/
├── data/
│   ├── raw/           # Original datasets (not in git)
│   ├── processed/     # Preprocessed data (not in git)
│   └── external/      # Reference data
```

## Project Architecture

### Common Structure Pattern

Each project follows a consistent structure:

```
project-XXX/
├── src/                    # Source code modules
│   ├── data/              # Data processing
│   ├── features/          # Feature engineering
│   ├── models/            # Model training/prediction
│   └── utils/             # Helper functions
├── tests/                 # Unit tests
├── notebooks/             # Jupyter notebooks
│   ├── exploratory/      # EDA and experiments
│   └── reports/          # Final analysis
├── data/                  # Datasets (in .gitignore)
├── models/                # Saved models (in .gitignore)
├── config/                # Configuration files
├── scripts/               # Utility scripts
├── docs/                  # Documentation and images
├── demo.py               # Quick demonstration script
├── requirements.txt      # Project dependencies
├── setup.py              # Package configuration
└── README.md             # Project documentation
```

### Import Conventions

Projects use `setup.py` for package installation, enabling clean imports:

```python
# After pip install -e .
from data.preprocessing import preprocess_m5_data
from features.build_features import build_m5_features
from models.train import train_m5_model
from models.predict import make_prediction
```

### Key Modules by Project

#### Project 001: Demand Forecasting System
- `src/data/preprocessing.py` - M5 data preprocessing pipeline
- `src/features/build_features.py` - 50+ feature engineering (lag, rolling, price)
- `src/models/train.py` - Baseline & ML models (LightGBM, XGBoost, RF)
- `src/models/predict.py` - Evaluation metrics & predictions
- **Dataset**: M5 Walmart (30,490 time series, 1,913 days)
- **Best Model**: LightGBM (MAPE 12.3%, R² 0.924)

#### Project 002: Inventory Optimization Engine
- `src/inventory/abc_classification.py` - ABC/XYZ analysis
- `src/inventory/eoq.py` - Economic Order Quantity
- `src/inventory/safety_stock.py` - Safety stock calculations
- `src/inventory/reorder_point.py` - ROP automation
- **Key Output**: Cost optimization recommendations
- **Results**: 20% cost reduction, 98% service level

## Development Workflow

### Working on a Project

1. **Activate environment:**
   ```bash
   cd data-science-portfolio
   source activate.sh
   ```

2. **Navigate to project:**
   ```bash
   cd project-001-demand-forecasting-system
   ```

3. **Install dependencies (first time):**
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

4. **Run tests to verify setup:**
   ```bash
   pytest tests/ -v
   ```

5. **Start developing:**
   - Edit code in `src/`
   - Run tests frequently
   - Use notebooks for experimentation

### Adding New Features

1. Create feature branch: `git checkout -b feature/your-feature`
2. Implement with proper docstrings and type hints
3. Add tests in `tests/`
4. Run quality checks: `pytest`, `flake8`, `black`
5. Update relevant notebook if applicable
6. Commit with clear message

### Creating a New Project

Follow the established pattern:

1. Copy structure from completed projects (001 or 002)
2. Update `setup.py` with new project name
3. Create `requirements.txt` with specific dependencies
4. Implement core modules in `src/`
5. Add tests in `tests/`
6. Create demo notebook
7. Write comprehensive README.md

## Architecture Patterns

### Data Processing Pipeline

Standard flow across projects:

```
Raw Data → Preprocessing → Feature Engineering → Model Training → Prediction → Evaluation
```

Each stage is modular and testable:

```python
# 1. Load and preprocess
df = preprocess_m5_data(data_path='data/raw')

# 2. Feature engineering
df_features = build_m5_features(df, target_col='sales')

# 3. Train model
model, metrics, importance = train_m5_model(
    df_features,
    model_type='lightgbm',
    test_size=0.2
)

# 4. Make predictions
predictions = make_prediction(model, X_test)
```

### Feature Engineering Patterns

Projects implement comprehensive feature engineering:

- **Lag features**: `sales_lag_1`, `sales_lag_7`, `sales_lag_28`
- **Rolling statistics**: `sales_rolling_mean_7`, `sales_rolling_std_28`
- **Date features**: `dayofweek`, `month`, `quarter`, `is_weekend`
- **Domain features**: Price changes, event indicators, hierarchical aggregations

### Model Training Patterns

Multi-model comparison approach:

1. **Baseline models**: Naive, Moving Average, Seasonal Naive
2. **ML models**: Random Forest, XGBoost, LightGBM
3. **Evaluation**: Multiple metrics (MAE, RMSE, MAPE, R²)
4. **Selection**: Best model based on accuracy-speed tradeoff

### Testing Patterns

Projects follow pytest conventions:

- `tests/conftest.py` - Shared fixtures and configuration
- `tests/test_*.py` - Test modules mirroring src structure
- Use `@pytest.fixture` for reusable test data
- Use `@pytest.mark.parametrize` for multiple test cases
- Include both unit tests and integration tests

## Key Conventions

### Code Style
- **Formatter**: Black (line length: 100)
- **Linter**: Flake8 (config in `.flake8`)
- **Type hints**: Use for function signatures
- **Docstrings**: Google style format

### Naming Conventions
- **Functions**: `snake_case` (e.g., `build_m5_features`)
- **Classes**: `PascalCase` (e.g., `InventoryOptimizer`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `DEFAULT_LAG_DAYS`)
- **Private**: `_leading_underscore` (e.g., `_validate_data`)

### Git Workflow
- Meaningful commit messages
- Feature branches for development
- Projects 001 and 002 are complete and stable
- Projects 003-005 are templates ready for implementation

### Documentation
- Comprehensive README.md for each project
- Docstrings for all public functions
- Jupyter notebooks with markdown explanations
- Code comments for complex logic only

## Project-Specific Notes

### Project 001: Demand Forecasting
- **Data size**: ~58M rows after melting M5 data
- **Training time**: LightGBM ~2.5 minutes on full dataset
- **Feature count**: 50+ engineered features
- **Key insight**: Lag features (7, 28 days) are most important

### Project 002: Inventory Optimization
- **No ML training**: Pure optimization algorithms
- **Four modules**: ABC/XYZ, EOQ, Safety Stock, ROP
- **Visualization focus**: 6 professional charts in `scripts/generate_visualizations.py`
- **Key insight**: ABC/XYZ classification drives service level differentiation

### Projects 003-005: Templates
These are **fully-scoped templates** with:
- Comprehensive README (400+ lines)
- Complete architecture outlined
- All dependencies specified in requirements.txt
- Ready for implementation following the pattern from 001-002

## Troubleshooting

### Virtual Environment Issues

**Wrong Python version:**
```bash
rm -rf .venv
python3.11 -m venv .venv
source .venv/bin/activate
```

**Permission errors:**
```bash
chmod -R u+w .venv
# Never use sudo with virtual environments
```

### Import Errors

**Module not found:**
```bash
# Ensure project is installed in development mode
pip install -e .

# Verify installation
pip list | grep demand-forecasting
```

### Data Issues

**M5 data missing:**
- Check `data/raw/` directory exists
- Re-run `python scripts/download_m5_data.py`
- Verify Kaggle API is configured correctly

### Test Failures

**Tests fail after code changes:**
- Check if test fixtures need updating
- Verify data preprocessing still produces expected output
- Run specific test with `-v` for detailed output

## Additional Resources

- **[README.md](README.md)** - Portfolio overview and project status
- **[ENVIRONMENT_SETUP.md](ENVIRONMENT_SETUP.md)** - Detailed environment setup
- **[PROJECT_ROADMAP.md](PROJECT_ROADMAP.md)** - Implementation plan for templates
- **[GETTING_STARTED.md](GETTING_STARTED.md)** - Quick start guide
- Individual project READMEs for detailed documentation

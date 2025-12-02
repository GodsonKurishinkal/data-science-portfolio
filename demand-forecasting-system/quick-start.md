# Quick Start Guide - M5 Walmart Demand Forecasting Project

## Prerequisites

1. **Python 3.9+** installed
2. **Kaggle account** (free at kaggle.com)

## Step 1: Setup Environment

```bash
# Navigate to project directory
cd /Users/godsonkurishinkal/Projects/data-science-portfolio/demand-forecasting-system

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # On macOS/Linux

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
pip install kaggle  # For data download
```

## Step 2: Configure Kaggle API

```bash
# 1. Go to https://www.kaggle.com/settings
# 2. Scroll to "API" section
# 3. Click "Create New API Token"
# 4. This downloads kaggle.json

# 5. Move kaggle.json to the right location
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

## Step 3: Download M5 Dataset

```bash
# Accept competition rules first!
# Visit: https://www.kaggle.com/competitions/m5-forecasting-accuracy/rules
# Click "I Understand and Accept"

# Download the dataset (~ 260 MB)
python scripts/download_m5_data.py
```

**Alternative manual download:**
1. Visit https://www.kaggle.com/competitions/m5-forecasting-accuracy/data
2. Download these files:
   - calendar.csv
   - sales_train_validation.csv
   - sell_prices.csv
3. Place them in `data/raw/` directory

## Step 4: Run the Project

### Option A: Jupyter Notebook (Recommended)

```bash
# Start Jupyter
jupyter notebook

# Open: notebooks/exploratory/01_m5_data_exploration.ipynb
# Run all cells: Cell > Run All
```

### Option B: Python Scripts

```bash
# Run data preprocessing
python src/data/preprocessing.py

# Run feature engineering
python src/features/build_features.py

# Train models
python src/models/train.py
```

## Step 5: Run Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src

# Run specific test file
pytest tests/test_data_processing.py -v
```

## Project Structure Quick Reference

```
demand-forecasting-system/
├── data/
│   ├── raw/               # Download M5 data here
│   └── processed/         # Processed data goes here
├── notebooks/
│   └── exploratory/       # Start here: 01_m5_data_exploration.ipynb
├── src/                   # Reusable Python modules
├── tests/                 # Unit tests
├── requirements.txt       # Dependencies
└── QUICK_START.md        # This file
```

## Common Issues & Solutions

### Issue: "kaggle: command not found"
**Solution**: Install Kaggle API: `pip install kaggle`

### Issue: "403 Forbidden" when downloading
**Solution**: Accept competition rules at https://www.kaggle.com/competitions/m5-forecasting-accuracy/rules

### Issue: "Could not find kaggle.json"
**Solution**: Place kaggle.json in ~/.kaggle/ and run `chmod 600 ~/.kaggle/kaggle.json`

### Issue: Import errors in notebooks
**Solution**: Make sure virtual environment is activated and install package: `pip install -e .`

## Next Steps

1. ✅ Complete setup above
2. 📊 Run `01_m5_data_exploration.ipynb` - Understand the data
3. 🔧 Run `02_feature_engineering.ipynb` - Create features
4. 🤖 Run `03_model_training.ipynb` - Train models
5. 📈 Run `04_results_analysis.ipynb` - Analyze results
6. 📝 Update README with your findings

## Expected Timeline

- **Setup**: 15-30 minutes
- **Data Exploration**: 2-3 hours
- **Feature Engineering**: 3-4 hours
- **Model Training**: 4-6 hours
- **Analysis & Documentation**: 2-3 hours

**Total**: 12-16 hours for complete project

## Resources

- M5 Competition: https://www.kaggle.com/competitions/m5-forecasting-accuracy
- Dataset Info: `data/M5_DATASET_INFO.md`
- Project README: `README.md`

## Getting Help

If you encounter issues:
1. Check `data/M5_DATASET_INFO.md` for dataset details
2. Review test files in `tests/` for usage examples
3. Check Kaggle discussion: https://www.kaggle.com/competitions/m5-forecasting-accuracy/discussion

---

**Ready to start? Run the download script:**
```bash
python scripts/download_m5_data.py
```

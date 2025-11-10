# 🚀 Quick Start Guide

## Prerequisites

- Python 3.9+
- M5 Walmart dataset (from project-001)
- Virtual environment activated

## Installation (5 minutes)

### 1. Navigate to Project
```bash
cd project-003-dynamic-pricing-engine
```

### 2. Activate Virtual Environment
```bash
# Use portfolio-level virtual environment
cd ..
source .venv/bin/activate  # macOS/Linux
# or .venv\Scripts\activate on Windows
cd project-003-dynamic-pricing-engine
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
pip install -e .
```

### 4. Verify Installation
```bash
python demo.py
```

You should see:
```
🎯 DYNAMIC PRICING ENGINE - DEMO
✅ Configuration loaded successfully
```

## Project Structure

```
project-003-dynamic-pricing-engine/
├── src/
│   ├── pricing/          # Elasticity, optimization, markdown
│   ├── models/           # Demand response models
│   ├── competitive/      # Competitive analysis
│   ├── data/            # Data loading and preprocessing
│   └── utils/           # Helpers and validators
├── notebooks/           # Jupyter analysis notebooks
├── tests/              # Unit tests
├── config/             # Configuration files
├── data/               # Data directory (linked to project-001)
├── models/             # Saved models
├── docs/               # Documentation and visualizations
└── demo.py            # Quick demonstration script
```

## Next Steps

### Phase 2: Data Preparation
Create symlink to M5 data:
```bash
cd data
ln -s ../../project-001-demand-forecasting-system/data/raw raw
cd ..
```

### Run Tests
```bash
pytest tests/ -v
```

### Start Development
See `IMPLEMENTATION_PLAN.md` for the complete development roadmap.

## Quick Commands

```bash
# Run demo
python demo.py

# Run tests
pytest tests/ -v --cov=src

# Run specific test
pytest tests/test_utils.py -v

# Start Jupyter
jupyter notebook notebooks/

# Format code
black src/ tests/

# Lint code
flake8 src/ tests/
```

## Common Issues

### Import Errors
If you see module import errors:
```bash
pip install -e .
```

### Configuration Not Found
Make sure you're in the project directory:
```bash
cd project-003-dynamic-pricing-engine
python demo.py
```

### Data Not Found
Link M5 data from project-001:
```bash
cd data
ln -s ../../project-001-demand-forecasting-system/data/raw raw
```

## Getting Help

- See `IMPLEMENTATION_PLAN.md` for detailed implementation guide
- See `README.md` for project overview
- Check `config/config.yaml` for configuration options

---

**Ready to build? Follow the phases in `IMPLEMENTATION_PLAN.md`!** 🚀

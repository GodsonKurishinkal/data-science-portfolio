# Project Setup Request: Production-Ready Data Science Portfolio Project

Establish a professional, industry-standard data science project structure in this directory. This is a portfolio project connected to GitHub that will be evaluated by hiring managers for a data scientist. Implement production-ready best practices that demonstrate technical excellence and professional maturity.

## Required Directory Structure

```
project_root/
├── data/
│   ├── raw/           # Original, immutable data
│   ├── processed/     # Cleaned, transformed data
│   └── external/      # External reference data
├── notebooks/
│   ├── exploratory/   # EDA and experiments
│   └── reports/       # Final analysis notebooks
├── src/
│   ├── __init__.py
│   ├── data/          # Data processing modules
│   ├── features/      # Feature engineering
│   ├── models/        # Model training and prediction
│   └── utils/         # Helper functions
├── tests/
│   ├── __init__.py
│   └── test_*.py      # Unit tests
├── config/
│   └── config.yaml    # Configuration parameters
├── docs/              # Additional documentation
├── .github/
│   └── workflows/     # CI/CD pipelines (optional)
├── .gitignore
├── .gitattributes
├── requirements.txt
├── setup.py
├── README.md
└── LICENSE
```

## Specific Implementation Requirements

### 1. Version Control (.gitignore)
Create comprehensive .gitignore including:
- `data/raw/`, `data/processed/`, `data/external/`
- `*.pkl`, `*.h5`, `*.joblib`, `*.csv`, `*.parquet`
- `.env`, `*.log`, `.DS_Store`
- `__pycache__/`, `*.pyc`, `.pytest_cache/`
- `.ipynb_checkpoints/`, `*.egg-info/`
- `models/*.pkl`, `models/*.h5`
- Virtual environment folders: `venv/`, `env/`, `.venv/`

### 2. Environment Management (requirements.txt)
Pin exact versions of core libraries:
```
numpy==1.24.3
pandas==2.0.3
scikit-learn==1.3.0
matplotlib==3.7.2
seaborn==0.12.2
jupyter==1.0.0
pytest==7.4.0
```

Include Python version requirement in README: `Python 3.9+`

### 3. Package Structure (setup.py)
Create proper Python package with:
```python
from setuptools import setup, find_packages

setup(
    name='project_name',
    version='0.1.0',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    python_requires='>=3.9',
)
```

### 4. Code Quality Configuration
Create `.flake8` file:
```
[flake8]
max-line-length = 88
extend-ignore = E203, W503
exclude = .git,__pycache__,venv,env
```

Include type hints in all function definitions:
```python
def process_data(df: pd.DataFrame) -> pd.DataFrame:
    """Process input DataFrame."""
    pass
```

### 5. Professional README.md Structure
```markdown
# Project Title

## Overview
Brief description (2-3 sentences)

## Business Problem
Clear problem statement

## Data
- Source
- Size
- Key features

## Methodology
- Data preprocessing steps
- Feature engineering approach
- Model selection rationale
- Evaluation metrics

## Results
- Key findings
- Model performance metrics
- Business impact

## Installation
```bash
pip install -r requirements.txt
```

## Usage
```python
# Example code
```

## Project Structure
Tree view of directories

## Requirements
- Python 3.11+
- See requirements.txt

## License
MIT
```

### 6. Testing Framework (pytest)
Create `tests/test_data_processing.py`:
```python
import pytest
from src.data.preprocessing import clean_data

def test_clean_data():
    """Test data cleaning function."""
    # Test implementation
    pass
```

### 7. Notebook Standards
- Use meaningful names: `01_data_exploration.ipynb`, `02_feature_engineering.ipynb`
- Clear markdown headers for each section
- Remove all debugging outputs and test cells
- Restart kernel and run all cells before committing
- Add execution time stamps for reproducibility

### 8. Code Documentation
All functions must have numpy-style docstrings:
```python
def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Calculate evaluation metrics for predictions.
    
    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
        
    Returns
    -------
    dict
        Dictionary containing accuracy, precision, recall, f1
    """
    pass
```

### 9. Git Best Practices
- Never commit data files, models, or credentials
- Use clear commit messages: `feat: Add feature engineering pipeline`
- Create .gitattributes for notebook diffs
- Keep repository under 100MB total size

## Quality Checklist
- [ ] All code follows PEP 8 standards
- [ ] Every function has docstrings and type hints
- [ ] No hardcoded paths or credentials
- [ ] README is complete and professional
- [ ] All notebooks run without errors from top to bottom
- [ ] requirements.txt has pinned versions
- [ ] .gitignore prevents committing sensitive files
- [ ] Code is modular and reusable
- [ ] Project structure is clean and organized

---

**Objective:** Create a portfolio project that demonstrates production-ready code, not just working code. Every file should showcase professional software engineering practices expected in industry data science teams.
# Notebooks Directory

This directory contains Jupyter notebooks for exploratory data analysis, experimentation, and reporting.

## Structure

### `exploratory/`
**Purpose**: Exploratory data analysis (EDA) and experimental work.

Notebooks in this directory are used for:
- Initial data exploration and understanding
- Testing hypotheses and ideas
- Prototyping features and models
- Ad-hoc analysis

**Naming Convention**: `##_descriptive_name.ipynb`

Example notebooks:
- `01_data_exploration.ipynb` - Initial data exploration and statistics
- `02_feature_analysis.ipynb` - Feature distribution and correlation analysis
- `03_model_experiments.ipynb` - Testing different modeling approaches
- `04_hyperparameter_tuning.ipynb` - Model optimization experiments

### `reports/`
**Purpose**: Final, polished analysis notebooks for presentation.

Notebooks in this directory are:
- Clean, well-documented, and production-ready
- Run from top to bottom without errors
- Suitable for sharing with stakeholders
- Include clear conclusions and recommendations

**Naming Convention**: `##_report_name.ipynb`

Example notebooks:
- `01_final_model_evaluation.ipynb` - Complete model evaluation and metrics
- `02_business_insights.ipynb` - Key findings and business recommendations
- `03_forecast_results.ipynb` - Final forecasting results and visualizations

## Notebook Best Practices

### Code Quality
- ✅ Clear markdown headers for each section
- ✅ Explanatory text before and after code cells
- ✅ Well-commented code
- ✅ Proper error handling
- ✅ Consistent code style (PEP 8)

### Reproducibility
- ✅ Set random seeds for reproducibility
- ✅ Document package versions used
- ✅ Include installation instructions if needed
- ✅ Clear cell execution order
- ✅ Restart kernel and run all cells before committing

### Output Management
- ✅ Remove debugging outputs before committing
- ✅ Keep essential visualizations and results
- ✅ Clear sensitive information from outputs
- ✅ Use `%matplotlib inline` for static plots

### Organization
- ✅ One notebook per major analysis or task
- ✅ Keep notebooks focused and concise
- ✅ Move reusable code to `src/` modules
- ✅ Reference shared functions from `src/`

## Running Notebooks

### Setup
```bash
# Install Jupyter
pip install jupyter

# Launch Jupyter Notebook
jupyter notebook

# Or use Jupyter Lab
jupyter lab
```

### Kernel Setup
```bash
# Create kernel with project environment
python -m ipykernel install --user --name=demand-forecasting --display-name "Demand Forecasting"
```

## Template Notebook Structure

```markdown
# Notebook Title

**Author**: Your Name  
**Date**: YYYY-MM-DD  
**Purpose**: Brief description of the notebook's objective

## 1. Setup
- Import libraries
- Set configurations
- Load data

## 2. Data Exploration
- Basic statistics
- Visualizations
- Data quality checks

## 3. Analysis
- Main analysis or modeling work
- Experiments and iterations

## 4. Results
- Key findings
- Visualizations
- Metrics and evaluation

## 5. Conclusions
- Summary of insights
- Next steps
- Recommendations
```

## Version Control

⚠️ **Important**: 
- Always clear outputs before committing exploratory notebooks
- Keep final report notebooks with outputs for documentation
- Use `.gitattributes` for better notebook diff visualization

## Converting Notebooks

Convert notebooks to Python scripts when code is production-ready:

```bash
# Convert notebook to Python script
jupyter nbconvert --to script notebook_name.ipynb --output ../src/module_name.py
```

---

For questions or issues with notebooks, refer to the main [README.md](../README.md)

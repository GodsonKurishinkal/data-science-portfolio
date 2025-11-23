# Notebooks

## Directory Structure

```
notebooks/
├── exploratory/        # Data exploration and EDA
└── analysis/          # Inventory optimization analysis
```

## Notebook Guide

### Exploratory Notebooks

1. **`01_inventory_data_exploration.ipynb`** *(To be created)*
   - Load and explore M5 data
   - Understand demand patterns
   - Analyze sales and price distributions

### Analysis Notebooks

2. **`02_abc_xyz_classification.ipynb`** *(To be created)*
   - Perform ABC/XYZ analysis
   - Visualize classification matrix
   - Analyze class characteristics

3. **`03_optimization_results.ipynb`** *(To be created)*
   - Run complete optimization
   - Analyze results
   - Generate recommendations

## Getting Started

```bash
# Launch Jupyter
jupyter notebook

# Or use JupyterLab
jupyter lab
```

## Best Practices

- Keep notebooks focused on one topic
- Clear all outputs before committing
- Use markdown for documentation
- Save key figures to `docs/images/`

## Kernel

Use the portfolio virtual environment:
```bash
python -m ipykernel install --user --name=venv --display-name="Data Science Portfolio"
```

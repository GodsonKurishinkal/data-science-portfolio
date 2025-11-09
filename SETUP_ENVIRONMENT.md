# Virtual Environment Setup Guide

This guide explains how to use the Python virtual environment for this portfolio.

## Virtual Environment Location

The virtual environment is located at: `.venv/`

This keeps your project dependencies isolated from system Python packages.

---

## Activating the Virtual Environment

### On macOS/Linux:
```bash
source .venv/bin/activate
```

### On Windows:
```bash
.venv\Scripts\activate
```

### Verification:
After activation, your terminal prompt should show `(.venv)` at the beginning:
```
(.venv) user@computer:~/data-science-portfolio$
```

---

## Installing Dependencies

Once activated, install project dependencies:

### For Project 001 (Demand Forecasting):
```bash
cd project-001-demand-forecasting-system
pip install -r requirements.txt
```

### For All Projects (when available):
```bash
# Install from root
pip install -e project-001-demand-forecasting-system/
```

---

## Deactivating the Virtual Environment

When you're done working:
```bash
deactivate
```

---

## Quick Start Workflow

```bash
# 1. Navigate to portfolio
cd ~/Projects/data-science-portfolio

# 2. Activate virtual environment
source .venv/bin/activate

# 3. Install dependencies (first time only)
cd project-001-demand-forecasting-system
pip install -r requirements.txt

# 4. Run demo or work on project
python demo.py

# 5. When done, deactivate
deactivate
```

---

## Checking Installed Packages

```bash
# List all installed packages
pip list

# Check specific package
pip show pandas

# Save current environment
pip freeze > installed_packages.txt
```

---

## Recreating the Environment

If you need to recreate the virtual environment:

```bash
# Remove old environment
rm -rf .venv

# Create new environment
python3 -m venv .venv

# Activate
source .venv/bin/activate

# Install dependencies
pip install -r project-001-demand-forecasting-system/requirements.txt
```

---

## IDE Integration

### VS Code:
1. Open Command Palette (Cmd/Ctrl + Shift + P)
2. Type "Python: Select Interpreter"
3. Choose `.venv/bin/python`

VS Code should automatically detect and use this environment.

### PyCharm:
1. Go to Settings/Preferences
2. Project → Python Interpreter
3. Click gear icon → Add
4. Select "Existing environment"
5. Browse to `.venv/bin/python`

---

## Troubleshooting

### Issue: Virtual environment not activating

**Solution:**
```bash
# Ensure you're in the correct directory
pwd  # Should show: .../data-science-portfolio

# Try activating with full path
source /Users/godsonkurishinkal/Projects/data-science-portfolio/.venv/bin/activate
```

### Issue: pip command not found

**Solution:**
```bash
# Ensure virtual environment is activated
which python  # Should show: .../data-science-portfolio/.venv/bin/python

# Upgrade pip
python -m pip install --upgrade pip
```

### Issue: Permission denied

**Solution:**
```bash
# Fix permissions
chmod +x .venv/bin/activate
```

---

## Environment Variables

For project-specific environment variables, create a `.env` file:

```bash
# Create .env file
cat > .env << EOF
# Kaggle API credentials
KAGGLE_USERNAME=your_username
KAGGLE_KEY=your_api_key

# Other configurations
DATA_PATH=./data
MODEL_PATH=./models
EOF
```

**Note:** `.env` is in `.gitignore` and won't be committed.

---

## Best Practices

1. **Always activate** the virtual environment before working
2. **Keep requirements.txt updated** when adding new packages:
   ```bash
   pip freeze > requirements.txt
   ```
3. **Don't commit** the `.venv/` directory (already in `.gitignore`)
4. **Use the same Python version** across team members
5. **Document** any system-level dependencies in README

---

## Package Management Tips

### Installing a new package:
```bash
# Install
pip install package-name

# Add to requirements
pip freeze | grep package-name >> requirements.txt
```

### Updating packages:
```bash
# Update specific package
pip install --upgrade package-name

# Update all packages
pip list --outdated
pip install --upgrade package-name1 package-name2
```

### Removing a package:
```bash
# Uninstall
pip uninstall package-name

# Update requirements.txt
pip freeze > requirements.txt
```

---

**Last Updated:** November 9, 2025  
**Python Version:** 3.9+

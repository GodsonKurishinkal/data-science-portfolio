# Virtual Environment Setup Guide

This portfolio uses a **single virtual environment** (`.venv`) at the root level to manage dependencies for all projects.

## Quick Start

### 1. Activate Virtual Environment

**On macOS/Linux:**
```bash
source .venv/bin/activate
```

**On Windows:**
```bash
.venv\Scripts\activate
```

**Verify activation:**
```bash
which python  # Should show: .../data-science-portfolio/.venv/bin/python
```

### 2. Install Dependencies

**For Project 001 (Demand Forecasting):**
```bash
cd demand-forecasting-system
pip install -r requirements.txt
```

**For all projects (if there's a root requirements.txt):**
```bash
pip install -r requirements.txt
```

### 3. Deactivate When Done

```bash
deactivate
```

---

## Complete Setup Instructions

### Initial Setup (One-time)

1. **Create virtual environment** (already done):
   ```bash
   python3 -m venv .venv
   ```

2. **Activate it**:
   ```bash
   source .venv/bin/activate  # macOS/Linux
   # or
   .venv\Scripts\activate     # Windows
   ```

3. **Upgrade pip**:
   ```bash
   pip install --upgrade pip
   ```

4. **Install project dependencies**:
   ```bash
   # Project 001
   pip install -r demand-forecasting-system/requirements.txt
   
   # Add more projects as you create them
   # pip install -r inventory-optimization-engine/requirements.txt
   ```

5. **Verify installation**:
   ```bash
   pip list
   python --version
   ```

---

## Daily Workflow

### Start Working
```bash
# 1. Navigate to portfolio
cd /Users/godsonkurishinkal/Projects/data-science-portfolio

# 2. Activate virtual environment
source .venv/bin/activate

# 3. Navigate to specific project
cd demand-forecasting-system

# 4. Start working!
python demo.py
# or
jupyter notebook
```

### Stop Working
```bash
# Deactivate virtual environment
deactivate
```

---

## VS Code Integration

VS Code should automatically detect `.venv`. To manually select it:

1. **Open Command Palette**: `Cmd+Shift+P` (Mac) or `Ctrl+Shift+P` (Windows)
2. **Type**: `Python: Select Interpreter`
3. **Choose**: `.venv` from the list

**Recommended VS Code settings** (`.vscode/settings.json`):
```json
{
    "python.defaultInterpreterPath": "${workspaceFolder}/.venv/bin/python",
    "python.terminal.activateEnvironment": true,
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": false,
    "python.linting.flake8Enabled": true,
    "python.formatting.provider": "black",
    "python.testing.pytestEnabled": true,
    "python.testing.unittestEnabled": false,
    "jupyter.notebookFileRoot": "${workspaceFolder}"
}
```

---

## Managing Dependencies

### Install New Package
```bash
# Activate environment first
source .venv/bin/activate

# Install package
pip install package-name

# Update requirements.txt (for specific project)
cd demand-forecasting-system
pip freeze > requirements.txt
```

### Update All Packages
```bash
pip list --outdated
pip install --upgrade package-name
```

### Create Requirements File
```bash
pip freeze > requirements.txt
```

---

## Troubleshooting

### Issue: "python3: command not found"
**Solution:**
```bash
# Check Python installation
which python3

# On Mac, install via Homebrew
brew install python3
```

### Issue: Virtual environment not activating
**Solution:**
```bash
# Recreate virtual environment
rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate
```

### Issue: "pip: command not found"
**Solution:**
```bash
# Use python -m pip instead
python -m pip install --upgrade pip
```

### Issue: Permission denied
**Solution:**
```bash
# Don't use sudo with virtual environments
# If needed, fix permissions:
chmod -R u+w .venv
```

### Issue: Wrong Python version in .venv
**Solution:**
```bash
# Specify Python version when creating
python3.9 -m venv .venv
# or
python3.11 -m venv .venv
```

---

## Best Practices

### ✅ Do's
- ✅ Always activate `.venv` before working
- ✅ Keep requirements.txt updated for each project
- ✅ Use meaningful package versions in requirements
- ✅ Deactivate when switching projects/environments
- ✅ Commit requirements.txt to git
- ✅ Test your code after installing new packages

### ❌ Don'ts
- ❌ Don't commit `.venv/` to git (already in .gitignore)
- ❌ Don't use system Python for development
- ❌ Don't install packages without activating .venv
- ❌ Don't use `sudo pip install` inside virtual environment
- ❌ Don't mix pip and conda in same environment

---

## Project Structure with Virtual Environment

```
data-science-portfolio/
├── .venv/                          # Virtual environment (NOT in git)
├── .gitignore                      # Excludes .venv/
├── ENVIRONMENT_SETUP.md            # This file
├── README.md                       # Portfolio documentation
│
├── demand-forecasting-system/
│   ├── requirements.txt            # Project 001 dependencies
│   ├── src/                        # Source code
│   └── ...
│
├── inventory-optimization-engine/
│   ├── requirements.txt            # Project 002 dependencies
│   ├── src/                        # Source code
│   └── ...
│
└── ...
```

---

## Environment Information

**Created:** November 9, 2025  
**Python Version:** 3.9+ (check with `python --version`)  
**Location:** `/Users/godsonkurishinkal/Projects/data-science-portfolio/.venv`

**To check your environment:**
```bash
source .venv/bin/activate
python --version
pip list
```

---

## Quick Reference

| Command | Description |
|---------|-------------|
| `source .venv/bin/activate` | Activate (Mac/Linux) |
| `.venv\Scripts\activate` | Activate (Windows) |
| `deactivate` | Deactivate |
| `which python` | Check Python location |
| `pip list` | List installed packages |
| `pip install package` | Install package |
| `pip freeze > requirements.txt` | Save dependencies |
| `pip install -r requirements.txt` | Install from requirements |

---

## Additional Resources

- [Python Virtual Environments Guide](https://docs.python.org/3/tutorial/venv.html)
- [pip Documentation](https://pip.pypa.io/)
- [VS Code Python Environments](https://code.visualstudio.com/docs/python/environments)

---

**Need Help?** Review this guide or check the troubleshooting section above.

## 🎯 Virtual Environment Quick Reference

### Created `.venv` at portfolio root level
**Location:** `/Users/godsonkurishinkal/Projects/data-science-portfolio/.venv`

---

## 🚀 Daily Usage

### Start Working
```bash
cd ~/Projects/data-science-portfolio
source activate.sh  # One command to activate!
```

### Navigate to Project
```bash
cd project-001-demand-forecasting-system
python demo.py
```

### Stop Working
```bash
deactivate
```

---

## 📦 Installation Commands

### Activate First
```bash
source .venv/bin/activate
```

### Install Project 001 Dependencies
```bash
cd project-001-demand-forecasting-system
pip install -r requirements.txt
```

### Install a New Package
```bash
pip install package-name
```

---

## ✅ Verification

### Check If Activated
```bash
which python
# Should show: .../data-science-portfolio/.venv/bin/python
```

### Check Installed Packages
```bash
pip list
```

### Check Python Version
```bash
python --version
```

---

## 🔧 Troubleshooting

### Re-create Environment
```bash
cd ~/Projects/data-science-portfolio
rm -rf .venv
python3 -m venv .venv
source activate.sh
```

### Fix Permissions
```bash
chmod -R u+w .venv
```

---

## 📁 Structure

```
data-science-portfolio/
├── .venv/              ← Your virtual environment (not in git)
├── activate.sh         ← Quick activation script
├── ENVIRONMENT_SETUP.md  ← Full documentation
│
├── project-001-demand-forecasting-system/
│   ├── requirements.txt  ← Project dependencies
│   ├── demo.py
│   └── ...
│
└── project-002-inventory-optimization-engine/
    └── ...
```

---

## 💡 Pro Tips

1. **Always activate before working:**
   ```bash
   source activate.sh
   ```

2. **VS Code will auto-detect `.venv`** when you open the portfolio folder

3. **One environment for all projects** keeps things clean

4. **Update requirements.txt after installing new packages:**
   ```bash
   pip freeze > requirements.txt
   ```

5. **Use the activation script** instead of manually sourcing:
   - It checks if .venv exists
   - Creates it if missing
   - Shows useful info

---

**📖 Full Documentation:** See [ENVIRONMENT_SETUP.md](ENVIRONMENT_SETUP.md)

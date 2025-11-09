# Git & GitHub Workflow Guide for Data Science Portfolio Projects

A professional guide for using Git and GitHub to manage your data science portfolio projects. This workflow demonstrates industry-standard version control practices expected by hiring managers.

## Initial Setup

### 1. Configure Git (One-time setup)
```bash
# Set your identity
git config --global user.name "Godson Kurishinkal"
git config --global user.email "godson.kurishinkal+github@gmail.com"

# Set default branch name
git config --global init.defaultBranch main

# Enable color output
git config --global color.ui auto
```

### 2. Create New Repository on GitHub
1. Go to github.com and click "New repository"
2. Name it descriptively: `demand-forecasting-retail` or `customer-churn-prediction`
3. Add description: "Machine learning model for predicting customer churn in retail"
4. Choose "Public" for portfolio visibility
5. Do NOT initialize with README (you'll create locally)
6. Click "Create repository"

### 3. Initialize Local Project
```bash
# Navigate to your project folder
cd /path/to/your/project

# Initialize Git repository
git init

# Create .gitignore first (critical!)
# Copy the .gitignore content from your setup prompt

# Add all files
git add .

# Create first commit
git commit -m "Initial commit: Project structure setup"

# Connect to GitHub
git remote add origin https://github.com/yourusername/project-name.git

# Push to GitHub
git branch -M main
git push -u origin main
```

## Daily Workflow

### Standard Git Workflow Cycle
```bash
# 1. Check current status
git status

# 2. See what changed
git diff

# 3. Add specific files
git add src/data/preprocessing.py
git add notebooks/01_data_exploration.ipynb

# Or add all changes
git add .

# 4. Commit with clear message
git commit -m "feat: Add data preprocessing pipeline"

# 5. Push to GitHub
git push origin main
```

### Viewing History
```bash
# See commit history
git log --oneline --graph --all

# See what changed in last commit
git show

# See changes in specific file
git log -p src/models/train.py
```

## Commit Message Standards

### Format
```
<type>: <description>

[optional body]
```

### Types
- **feat**: New feature or functionality
- **fix**: Bug fix
- **docs**: Documentation changes
- **refactor**: Code restructuring without functionality change
- **test**: Adding or updating tests
- **style**: Formatting, missing semicolons, etc.
- **chore**: Maintenance tasks, dependency updates

### Examples
```bash
git commit -m "feat: Add feature engineering pipeline for customer data"
git commit -m "fix: Resolve missing value handling in preprocessing"
git commit -m "docs: Update README with model performance metrics"
git commit -m "refactor: Optimize data loading for large datasets"
git commit -m "test: Add unit tests for feature engineering functions"
```

## Working with Branches

### Why Use Branches
- Keep main branch stable and production-ready
- Experiment safely without breaking working code
- Show employers you understand collaborative workflows

### Basic Branch Workflow
```bash
# Create and switch to new branch
git checkout -b feature/add-random-forest

# Work on your feature, commit changes
git add .
git commit -m "feat: Implement random forest model"

# Switch back to main
git checkout main

# Merge feature into main
git merge feature/add-random-forest

# Delete feature branch (optional)
git branch -d feature/add-random-forest

# Push to GitHub
git push origin main
```

### Branch Naming Conventions
```
feature/add-lstm-model
fix/memory-leak-in-training
docs/update-readme
refactor/optimize-preprocessing
experiment/test-new-algorithm
```

## Common Scenarios

### Scenario 1: Made Changes, Haven't Committed Yet
```bash
# Discard changes to specific file
git checkout -- src/models/train.py

# Discard all changes
git reset --hard HEAD
```

### Scenario 2: Committed But Haven't Pushed
```bash
# Undo last commit, keep changes
git reset --soft HEAD~1

# Undo last commit, discard changes
git reset --hard HEAD~1

# Amend last commit (fix message or add files)
git commit --amend -m "New commit message"
```

### Scenario 3: Already Pushed to GitHub
```bash
# Create new commit that reverses changes
git revert HEAD

# Force push (use with caution, avoid on shared branches)
git push --force origin main
```

### Scenario 4: Need to Update from GitHub
```bash
# Fetch and merge changes from GitHub
git pull origin main
```

### Scenario 5: Accidentally Committed Large File
```bash
# Remove file from Git but keep locally
git rm --cached data/large_file.csv

# Update .gitignore
echo "data/*.csv" >> .gitignore

# Commit the removal
git commit -m "chore: Remove large data files from tracking"
git push origin main
```

## GitHub-Specific Features

### Creating Good README
- Write clear project description
- Include installation instructions
- Add usage examples with code
- Show results and visualizations
- Link to live demos or notebooks

### Using GitHub Issues (Optional)
- Track bugs and feature ideas
- Shows planning and organization skills
```
Title: "Add cross-validation to model training"
Body: "Implement k-fold cross-validation to improve model evaluation reliability"
Labels: enhancement, model-training
```

### GitHub Releases (For Completed Projects)
```bash
# Tag a release version
git tag -a v1.0.0 -m "First stable release"
git push origin v1.0.0
```
Then create release on GitHub with release notes.

## Best Practices for Portfolio Projects

### DO:
- ✅ Commit frequently with clear messages
- ✅ Keep commits small and focused (one logical change per commit)
- ✅ Write descriptive commit messages
- ✅ Use .gitignore to exclude data files and credentials
- ✅ Keep repository size under 100MB
- ✅ Update README regularly
- ✅ Clean up notebooks before committing (clear outputs, remove debugging)

### DON'T:
- ❌ Commit large datasets or model files
- ❌ Commit passwords, API keys, or credentials
- ❌ Use vague messages like "fixed stuff" or "updated files"
- ❌ Commit broken or untested code to main branch
- ❌ Force push to shared branches
- ❌ Commit generated files (__pycache__, .pyc, .DS_Store)

## Workflow for Portfolio Project

```bash
# Day 1: Setup
git init
# Create .gitignore, README, project structure
git add .
git commit -m "Initial commit: Project structure and documentation"
git push origin main

# Day 2: Data exploration
# Work in notebooks/01_data_exploration.ipynb
git add notebooks/01_data_exploration.ipynb
git commit -m "feat: Complete initial data exploration and EDA"
git push origin main

# Day 3: Data preprocessing
# Create src/data/preprocessing.py
git add src/data/preprocessing.py
git commit -m "feat: Add data cleaning and preprocessing pipeline"
git push origin main

# Day 4: Feature engineering
# Create src/features/build_features.py
git add src/features/build_features.py
git commit -m "feat: Implement feature engineering for predictive modeling"
git push origin main

# Day 5: Model training
# Create src/models/train.py
git add src/models/train.py
git commit -m "feat: Add model training pipeline with hyperparameter tuning"
git push origin main

# Day 6: Documentation
# Update README with results
git add README.md
git commit -m "docs: Add model results and performance metrics"
git push origin main
```

## Quick Reference Commands

```bash
# Status and Changes
git status                    # Check what's changed
git diff                      # See line-by-line changes
git log --oneline            # View commit history

# Basic Operations
git add <file>               # Stage specific file
git add .                    # Stage all changes
git commit -m "message"      # Commit with message
git push origin main         # Push to GitHub

# Branching
git branch                   # List branches
git checkout -b <name>       # Create and switch to branch
git merge <branch>           # Merge branch into current

# Undoing
git checkout -- <file>       # Discard changes to file
git reset --soft HEAD~1      # Undo last commit, keep changes
git revert HEAD              # Create new commit that undoes last

# Remote Operations
git clone <url>              # Clone repository
git pull origin main         # Get latest changes
git remote -v                # View remote connections
```

## Checking Your Work

Before pushing, always:
1. Run `git status` - ensure you're committing the right files
2. Run `git diff` - review your changes
3. Test your code - make sure it runs without errors
4. Clear notebook outputs - restart kernel and run all cells
5. Update README if needed

## Resources

- GitHub Documentation: https://docs.github.com
- Git Cheat Sheet: https://education.github.com/git-cheat-sheet-education.pdf
- Conventional Commits: https://www.conventionalcommits.org

---

**Remember:** Your Git history is part of your portfolio. Clean, professional commits demonstrate your attention to detail and understanding of software engineering practices.
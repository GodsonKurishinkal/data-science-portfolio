# M5 Walmart Demand Forecasting - Project Roadmap

## 🎯 Project Overview

Build a complete, production-ready demand forecasting system using the M5 Competition dataset from Walmart. This portfolio project demonstrates end-to-end data science capabilities including EDA, feature engineering, model training, evaluation, and deployment-ready code.

**Dataset**: M5 Forecasting Accuracy Competition  
**Source**: Kaggle  
**Size**: 30,490 time series, 1,913 days, ~260MB  
**Business Goal**: Forecast daily sales for Walmart products across multiple stores

---

## 📋 Phase 1: Project Setup & Data Acquisition ✅

### Completed
- [x] Created production-ready project structure
- [x] Set up Python package with proper modules
- [x] Created comprehensive .gitignore and configuration files
- [x] Built automated data download script
- [x] Created M5 dataset documentation
- [x] Set up testing framework with pytest
- [x] Created QUICK_START.md guide

### Files Created
- `scripts/download_m5_data.py` - Automated M5 data download
- `data/M5_DATASET_INFO.md` - Comprehensive dataset documentation
- `QUICK_START.md` - Step-by-step setup guide
- `notebooks/exploratory/01_m5_data_exploration.ipynb` - EDA notebook (started)

### Next Action
**Download the M5 dataset:**
```bash
# Option 1: Automated (requires Kaggle API setup)
python scripts/download_m5_data.py

# Option 2: Manual download from Kaggle
# Visit: https://www.kaggle.com/competitions/m5-forecasting-accuracy/data
```

---

## 📊 Phase 2: Exploratory Data Analysis (EDA)

### Objectives
1. Understand M5 dataset structure and characteristics
2. Analyze sales patterns and distributions
3. Identify trends, seasonality, and anomalies
4. Examine calendar effects (events, SNAP days, holidays)
5. Analyze price dynamics and their impact on sales
6. Explore hierarchical structure (state → store → category → department → item)

### Notebooks to Create
1. **01_m5_data_exploration.ipynb** (In Progress)
   - Load and examine all three datasets
   - Basic statistics and data quality checks
   - Visualize overall sales trends
   - Analyze calendar features and events
   
2. **02_sales_pattern_analysis.ipynb**
   - Time series decomposition (trend, seasonality, residuals)
   - Weekly, monthly, yearly patterns
   - Store-level and category-level analysis
   - Product performance comparison

3. **03_feature_correlation_analysis.ipynb**
   - Price vs. sales relationships
   - Event impact on sales
   - SNAP day effects
   - Cross-feature correlations

### Key Questions to Answer
- What are the overall sales trends?
- Which products/stores have highest sales?
- How do calendar events affect sales?
- What role does pricing play?
- Are there strong seasonal patterns?
- How much zero-inflation exists?

### Expected Insights
- Identification of high-performing products/stores
- Calendar effects quantification
- Seasonality patterns
- Data quality issues to address

**Estimated Time**: 6-8 hours

---

## 🔧 Phase 3: Data Preprocessing & Feature Engineering

### Preprocessing Tasks

#### 3.1 Data Transformation
- [ ] Melt sales data from wide to long format
- [ ] Merge calendar, sales, and price datasets
- [ ] Handle missing values in price data
- [ ] Create datetime features from calendar
- [ ] Aggregate data at different hierarchy levels

#### 3.2 Data Cleaning
- [ ] Identify and handle outliers
- [ ] Deal with zero-sales days
- [ ] Validate data consistency across datasets
- [ ] Remove or impute missing prices

### Feature Engineering Tasks

#### 3.3 Time-Based Features
- [ ] Day of week, month, quarter, year
- [ ] Week of year, day of month
- [ ] Is weekend, is month start/end
- [ ] Days since/until major holidays

#### 3.4 Lag Features
- [ ] Sales lags: 1, 7, 14, 28 days
- [ ] Price lags: 1, 4, 8 weeks
- [ ] Moving averages: 7, 14, 28, 90 days
- [ ] Rolling standard deviation

#### 3.5 Calendar Event Features
- [ ] Event type encoding (Cultural, National, Religious, Sporting)
- [ ] Event proximity features (days before/after)
- [ ] SNAP indicators by state
- [ ] Holiday indicators

#### 3.6 Price Features
- [ ] Price per item
- [ ] Price changes and trends
- [ ] Discount indicators
- [ ] Price relative to category average

#### 3.7 Hierarchical Features
- [ ] Store-level aggregations
- [ ] Category-level aggregations
- [ ] Department-level statistics
- [ ] State-level patterns

### Code Modules to Update
- `src/data/m5_preprocessing.py` - M5-specific preprocessing
- `src/features/m5_features.py` - M5-specific features
- `src/features/time_features.py` - Time-based features
- `src/features/price_features.py` - Price-related features

**Estimated Time**: 8-10 hours

---

## 🤖 Phase 4: Model Development

### 4.1 Baseline Models
- [ ] Naive forecast (last value)
- [ ] Moving average (7, 14, 28 days)
- [ ] Seasonal naive (same day last week)

### 4.2 Statistical Models
- [ ] ARIMA for select time series
- [ ] SARIMA with seasonal component
- [ ] Prophet (Facebook's forecasting tool)

### 4.3 Machine Learning Models
- [ ] Random Forest Regressor
- [ ] XGBoost Regressor
- [ ] LightGBM Regressor (recommended for M5)
- [ ] CatBoost Regressor

### 4.4 Advanced Techniques (Optional)
- [ ] LSTM/GRU neural networks
- [ ] Transformer-based models
- [ ] Ensemble methods

### Training Strategy
1. **Time-based cross-validation**: 5-fold time series split
2. **Hierarchy levels**: Start with aggregated, then drill down
3. **Sample subset**: Begin with single store or category
4. **Evaluation metrics**: RMSE, MAE, WRMSSE (M5 competition metric)

### Model Selection Criteria
- Predictive accuracy (RMSE, MAE)
- Training time and scalability
- Interpretability for business stakeholders
- Robustness to zero-inflation

### Code Modules
- `src/models/m5_train.py` - M5-specific training pipeline
- `src/models/baseline_models.py` - Baseline implementations
- `src/models/ml_models.py` - ML model wrappers
- `src/models/evaluate.py` - Evaluation functions

**Estimated Time**: 10-12 hours

---

## 📈 Phase 5: Model Evaluation & Analysis

### 5.1 Performance Metrics
- [ ] Calculate RMSE, MAE, MAPE per model
- [ ] Compute WRMSSE (weighted metric from M5)
- [ ] Evaluate at different hierarchy levels
- [ ] Analyze errors by product category and store

### 5.2 Model Comparison
- [ ] Create comparison tables
- [ ] Statistical significance testing
- [ ] Visualize predictions vs actuals
- [ ] Residual analysis

### 5.3 Feature Importance
- [ ] Analyze top predictive features
- [ ] SHAP values for interpretability
- [ ] Feature ablation studies

### 5.4 Business Insights
- [ ] Which products are easiest/hardest to forecast?
- [ ] Impact of promotions and events
- [ ] Store-specific patterns
- [ ] Actionable recommendations for inventory management

### Notebooks
- `notebooks/reports/01_model_evaluation.ipynb`
- `notebooks/reports/02_feature_importance.ipynb`
- `notebooks/reports/03_business_insights.ipynb`

**Estimated Time**: 6-8 hours

---

## 📝 Phase 6: Documentation & Presentation

### 6.1 README Update
- [ ] Add M5 dataset description
- [ ] Document model results and metrics
- [ ] Include visualizations
- [ ] Add business insights
- [ ] Update installation instructions

### 6.2 Final Report Notebook
- [ ] Executive summary
- [ ] Problem statement
- [ ] Methodology overview
- [ ] Key findings and results
- [ ] Visualizations and charts
- [ ] Conclusions and recommendations
- [ ] Future improvements

### 6.3 Code Documentation
- [ ] Ensure all functions have docstrings
- [ ] Add usage examples
- [ ] Create API documentation
- [ ] Update config.yaml with M5 settings

### 6.4 Results Visualization
- [ ] Sales forecast plots
- [ ] Performance comparison charts
- [ ] Feature importance plots
- [ ] Error distribution analysis
- [ ] Interactive dashboards (optional)

**Estimated Time**: 4-6 hours

---

## 🚀 Phase 7: GitHub Deployment

### 7.1 Pre-commit Checklist
- [ ] Run all unit tests: `pytest tests/`
- [ ] Run linting: `flake8 src/ tests/`
- [ ] Format code: `black src/ tests/`
- [ ] Clear notebook outputs (exploratory)
- [ ] Ensure no sensitive data in commits
- [ ] Update requirements.txt with exact versions

### 7.2 Git Workflow
```bash
# Initialize repository (if not done)
git init
git remote add origin [your-repo-url]

# Stage all files
git add .

# Commit with descriptive message
git commit -m "feat: Complete M5 demand forecasting project

- Comprehensive EDA of M5 dataset
- Feature engineering pipeline with 50+ features
- Trained and evaluated multiple models (RF, XGBoost, LightGBM)
- Achieved RMSE of [X] on validation set
- Production-ready code with tests and documentation"

# Push to GitHub
git push -u origin main
```

### 7.3 Repository Enhancements
- [ ] Add project banner/logo
- [ ] Create GitHub Actions for CI/CD (optional)
- [ ] Add badges (tests, coverage, license)
- [ ] Create CONTRIBUTING.md
- [ ] Add example outputs in results/

**Estimated Time**: 2-3 hours

---

## 📊 Success Metrics

### Technical Metrics
- **Model Performance**: RMSE < 2.5 on normalized sales
- **Code Quality**: 80%+ test coverage, passes flake8
- **Documentation**: All functions documented, comprehensive README

### Portfolio Metrics
- **Professional Structure**: Production-ready code organization
- **Reproducibility**: Anyone can run the project start-to-finish
- **Insights**: Clear business value and actionable recommendations
- **Presentation**: Polished notebooks and visualizations

---

## 🎓 Learning Outcomes

By completing this project, you will demonstrate:

1. **Data Engineering**: Handling large hierarchical datasets
2. **Feature Engineering**: Creating domain-specific features
3. **Time Series Forecasting**: Multiple modeling approaches
4. **Model Evaluation**: Proper validation and metrics
5. **Software Engineering**: Production-ready code structure
6. **Communication**: Clear documentation and insights
7. **Business Acumen**: Translating technical results to business value

---

## 📚 Key Resources

### M5 Competition
- Competition page: https://www.kaggle.com/competitions/m5-forecasting-accuracy
- Winning solutions: https://www.kaggle.com/competitions/m5-forecasting-accuracy/discussion
- Official paper: https://www.sciencedirect.com/science/article/pii/S0169207021001874

### Technical References
- LightGBM docs: https://lightgbm.readthedocs.io/
- Prophet docs: https://facebook.github.io/prophet/
- Time series in Python: https://www.statsmodels.org/stable/tsa.html

### Portfolio Examples
- Kaggle notebooks: https://www.kaggle.com/competitions/m5-forecasting-accuracy/code
- GitHub searches: "M5 forecasting"

---

## ⏱️ Total Estimated Time

| Phase | Time Estimate |
|-------|---------------|
| Setup & Data | 1-2 hours |
| EDA | 6-8 hours |
| Preprocessing & Features | 8-10 hours |
| Model Development | 10-12 hours |
| Evaluation & Analysis | 6-8 hours |
| Documentation | 4-6 hours |
| GitHub Deployment | 2-3 hours |
| **Total** | **37-49 hours** |

Recommended timeline: **1-2 weeks** working 4-6 hours per day

---

## ✅ Current Status

**Phase 1**: ✅ Complete  
**Phase 2**: 🚧 In Progress (EDA notebook started)  
**Next Step**: Complete M5 data download and finish EDA notebook

**Command to continue:**
```bash
# Ensure you're in the project directory
cd /Users/godsonkurishinkal/Projects/data-science-portfolio/demand-forecasting-system

# Download M5 data
python scripts/download_m5_data.py

# Start Jupyter and open EDA notebook
jupyter notebook notebooks/exploratory/01_m5_data_exploration.ipynb
```

---

**Last Updated**: November 9, 2025

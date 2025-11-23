# Model Card: Inventory Optimization Engine

## Model Details

**Model Name**: Inventory Optimization Engine  
**Version**: 0.1.0  
**Date**: November 2025  
**Model Type**: Operations Research / Optimization System  
**Framework**: Classical inventory theory with data-driven parameters

## Intended Use

### Primary Use Cases
- Retail inventory management and optimization
- Multi-location inventory allocation
- Safety stock and reorder point calculation
- Cost-service level tradeoff analysis
- Inventory policy recommendations

### Target Users
- Supply chain managers
- Inventory planners
- Operations analysts
- Data scientists working on supply chain problems

### Out-of-Scope Uses
- Real-time trading or financial decisions
- Medical or safety-critical inventory
- Perishable goods without modification
- Military or defense applications

## Model Components

### 1. ABC/XYZ Classification
**Purpose**: Segment items by value and variability

**Method**:
- **ABC**: Pareto analysis on revenue (80-15-5 rule)
- **XYZ**: Coefficient of variation on demand

**Outputs**: 9 classes (AX, AY, AZ, BX, BY, BZ, CX, CY, CZ)

### 2. Economic Order Quantity (EOQ)
**Formula**: $EOQ = \sqrt{\frac{2DS}{H}}$

**Parameters**:
- D: Annual demand
- S: Ordering cost per order ($100 default)
- H: Holding cost per unit per year (25% of unit cost)

**Assumptions**:
- Constant demand rate
- Instantaneous replenishment
- No quantity discounts
- Known costs

### 3. Safety Stock
**Formula**: $SS = Z \times \sigma_{demand} \times \sqrt{LT}$

**Parameters**:
- Z: Service level z-score (1.65 for 95%)
- σ: Standard deviation of daily demand
- LT: Lead time in days (7 default)

**Methods**:
- Basic (demand variability only)
- With lead time variability
- Periodic review
- Forecast error-based

### 4. Reorder Point
**Formula**: $ROP = (Demand_{avg} \times LT) + SS$

**Logic**:
- Trigger for placing orders
- Covers demand during lead time + buffer

### 5. Cost Calculator
**Components**:
- **Holding Cost**: Average inventory × Unit cost × Holding rate
- **Ordering Cost**: Number of orders × Fixed cost per order
- **Stockout Cost**: Unfilled demand × Unit cost × Penalty multiplier
- **Purchase Cost**: Total units × Unit cost

## Training Data

**Source**: M5 Walmart Forecasting Competition Dataset

**Characteristics**:
- **Temporal Coverage**: 2011-2016 (5+ years)
- **Geographic Scope**: 10 stores across CA, TX, WI
- **Product Range**: 3,049 products across 3 categories
- **Granularity**: Daily sales data
- **Size**: ~30,000 SKU-store combinations

**Data Quality**:
- Zero sales days included (demand = 0)
- Price changes captured
- Promotional events marked
- Holiday effects included

## Performance Metrics

### Optimization Metrics
- **Total Cost Reduction**: Target 10-15% vs. baseline
- **Service Level Achievement**: 95%+ fill rate
- **Inventory Turnover**: 15-20% improvement
- **Stockout Reduction**: 30%+ fewer incidents

### Model Validation
- Backtesting on historical data (2015-2016)
- Comparison with industry benchmarks
- Sensitivity analysis on key parameters

## Limitations

### Assumptions
1. **Stationary Demand**: Assumes demand patterns are relatively stable
2. **Lead Time Certainty**: Default fixed lead time (7 days)
3. **Cost Estimates**: Generic cost parameters, may need calibration
4. **No Capacity Constraints**: Unlimited warehouse space assumed
5. **Single-Echelon**: Store-level only, not multi-tier supply chain

### Known Issues
- **Seasonal Products**: Standard EOQ may not suit highly seasonal items
- **New Products**: Requires sufficient history (minimum 28 days)
- **Promotions**: Heavy promotional periods may distort parameters
- **Substitution**: Doesn't account for product substitution

### Biases
- **High-Volume Bias**: Works best for items with regular sales
- **Data Quality**: Sensitive to outliers and data errors
- **Industry-Specific**: Tuned for retail, may need adjustment for other industries

## Ethical Considerations

### Fairness
- Equal treatment across product categories and stores
- No discrimination based on geographic location

### Transparency
- All formulas and parameters documented
- Clear traceability of recommendations

### Environmental Impact
- Inventory optimization can reduce waste
- Lower holding costs encourage less overstocking

## Recommendations

### Best Practices
1. **Calibrate Costs**: Adjust holding, ordering, stockout costs for your business
2. **Monitor Service Levels**: Track actual vs. target regularly
3. **Review Periodically**: Re-optimize quarterly or when patterns change
4. **Start with A Items**: Focus on high-value items first
5. **Validate Assumptions**: Check if classical assumptions hold

### Customization
- Adjust service levels by item class
- Modify ABC/XYZ thresholds for your business
- Incorporate lead time variability if known
- Add constraints (budget, space) as needed

## Technical Specifications

**Language**: Python 3.9+  
**Key Libraries**:
- NumPy, Pandas (data processing)
- SciPy (statistical calculations)
- CVXPY, PuLP (optimization)

**Computational Requirements**:
- Memory: < 2GB for 30K SKUs
- Time: < 5 minutes for full optimization
- Hardware: Standard laptop/desktop sufficient

## Maintenance

**Update Frequency**: 
- Quarterly re-optimization recommended
- Monthly for fast-changing items
- Ad-hoc for major events (promotions, supply disruptions)

**Monitoring**:
- Track actual service levels
- Monitor stockout incidents
- Review cost performance
- Validate demand forecasts

## Version History

**v0.1.0** (November 2025)
- Initial release
- Core functionality: ABC/XYZ, EOQ, Safety Stock, ROP
- Basic cost optimization

## References

1. Harris, F. W. (1913). "How Many Parts to Make at Once"
2. Silver, E. A., Pyke, D. F., & Thomas, D. J. (2016). "Inventory and Production Management"
3. Axsäter, S. (2015). "Inventory Control" (3rd ed.)
4. M5 Forecasting Dataset: https://www.kaggle.com/competitions/m5-forecasting-accuracy

## Citation

```bibtex
@software{inventory_optimization_engine,
  title = {Inventory Optimization Engine},
  author = {Kurishinkal, Godson},
  year = {2025},
  version = {0.1.0},
  url = {https://github.com/GodsonKurishinkal/data-science-portfolio}
}
```

## Contact

For questions, issues, or contributions:
- GitHub Issues: [data-science-portfolio](https://github.com/GodsonKurishinkal/data-science-portfolio/issues)
- Author: Godson Kurishinkal

---

**Last Updated**: November 9, 2025  
**Next Review**: December 2025

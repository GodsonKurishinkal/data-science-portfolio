"""Safety stock calculation methods.

Provides multiple methods for calculating safety stock:
- Standard: Z × σ × √LT
- Dynamic: Accounts for lead time variability
- Capacity-aware: Adjusts for storage constraints
"""

import logging
from typing import Any, Dict, Optional

import pandas as pd
import numpy as np
from scipy import stats

from ..interfaces.base import ISafetyStockCalculator

logger = logging.getLogger(__name__)


# Z-scores for common service levels
Z_SCORES = {
    0.50: 0.00,
    0.75: 0.67,
    0.80: 0.84,
    0.85: 1.04,
    0.90: 1.28,
    0.92: 1.41,
    0.95: 1.65,
    0.97: 1.88,
    0.98: 2.05,
    0.99: 2.33,
    0.995: 2.58,
    0.999: 3.09,
}


class SafetyStockCalculator(ISafetyStockCalculator):
    """Multi-method safety stock calculator.
    
    Supports:
    - Standard method: Z × σ_demand × √LT
    - Dynamic method with lead time variability
    - Capacity-aware adjustments
    
    Examples:
        >>> calculator = SafetyStockCalculator(method='standard')
        >>> ss = calculator.calculate(
        ...     demand_mean=100,
        ...     demand_std=20,
        ...     lead_time=7,
        ...     service_level=0.95
        ... )
        >>> print(f"Safety Stock: {ss:.0f} units")
        Safety Stock: 87 units
    """
    
    def __init__(
        self,
        method: str = "standard",
        min_safety_stock: Optional[int] = None,
        max_safety_stock: Optional[int] = None,
        capacity_threshold: float = 0.85,
        capacity_multiplier: float = 1.2,
    ):
        """Initialize safety stock calculator.
        
        Args:
            method: Calculation method ('standard', 'dynamic', 'capacity_aware')
            min_safety_stock: Minimum safety stock (floor)
            max_safety_stock: Maximum safety stock (ceiling)
            capacity_threshold: Utilization threshold for capacity-aware method
            capacity_multiplier: Multiplier when above capacity threshold
        """
        valid_methods = ["standard", "dynamic", "capacity_aware"]
        if method not in valid_methods:
            raise ValueError(f"Method must be one of {valid_methods}")
        
        self.method = method
        self.min_safety_stock = min_safety_stock
        self.max_safety_stock = max_safety_stock
        self.capacity_threshold = capacity_threshold
        self.capacity_multiplier = capacity_multiplier
    
    @property
    def method_name(self) -> str:
        """Name of the calculation method."""
        return self.method
    
    def calculate(
        self,
        demand_mean: float,
        demand_std: float,
        lead_time: float,
        service_level: float,
        lead_time_std: Optional[float] = None,
        capacity_utilization: Optional[float] = None,
        **kwargs,
    ) -> float:
        """Calculate safety stock.
        
        Args:
            demand_mean: Average daily demand
            demand_std: Standard deviation of daily demand
            lead_time: Lead time in days
            service_level: Target service level (0-1)
            lead_time_std: Standard deviation of lead time (for dynamic method)
            capacity_utilization: Current capacity utilization (for capacity_aware)
            
        Returns:
            Safety stock quantity
        """
        # Get Z-score for service level
        z_score = self._get_z_score(service_level)
        
        # Calculate base safety stock based on method
        if self.method == "standard":
            safety_stock = self._standard_method(z_score, demand_std, lead_time)
        
        elif self.method == "dynamic":
            lt_std = lead_time_std or 0
            safety_stock = self._dynamic_method(
                z_score, demand_mean, demand_std, lead_time, lt_std
            )
        
        elif self.method == "capacity_aware":
            base_ss = self._standard_method(z_score, demand_std, lead_time)
            util = capacity_utilization or 0
            safety_stock = self._capacity_aware_adjustment(base_ss, util)
        
        else:
            safety_stock = self._standard_method(z_score, demand_std, lead_time)
        
        # Apply min/max constraints
        safety_stock = self._apply_constraints(safety_stock)
        
        return safety_stock
    
    def _get_z_score(self, service_level: float) -> float:
        """Get Z-score for a service level."""
        # Check if exact match in lookup table
        if service_level in Z_SCORES:
            return Z_SCORES[service_level]
        
        # Otherwise use scipy to calculate
        return stats.norm.ppf(service_level)
    
    def _standard_method(
        self,
        z_score: float,
        demand_std: float,
        lead_time: float,
    ) -> float:
        """Standard safety stock calculation.
        
        SS = Z × σ_demand × √LT
        """
        return z_score * demand_std * np.sqrt(lead_time)
    
    def _dynamic_method(
        self,
        z_score: float,
        demand_mean: float,
        demand_std: float,
        lead_time: float,
        lead_time_std: float,
    ) -> float:
        """Dynamic safety stock with lead time variability.
        
        SS = Z × √(LT × σ²_demand + DDR² × σ²_LT)
        
        Where:
            DDR = Daily Demand Rate (demand_mean)
            σ_demand = demand standard deviation
            σ_LT = lead time standard deviation
        """
        variance_component = (
            lead_time * (demand_std ** 2) +
            (demand_mean ** 2) * (lead_time_std ** 2)
        )
        return z_score * np.sqrt(variance_component)
    
    def _capacity_aware_adjustment(
        self,
        base_safety_stock: float,
        capacity_utilization: float,
    ) -> float:
        """Adjust safety stock based on capacity utilization.
        
        When capacity utilization is high, we may need more safety stock
        to prevent stockouts due to storage constraints.
        """
        if capacity_utilization >= self.capacity_threshold:
            return base_safety_stock * self.capacity_multiplier
        return base_safety_stock
    
    def _apply_constraints(self, safety_stock: float) -> float:
        """Apply min/max constraints to safety stock."""
        if self.min_safety_stock is not None:
            safety_stock = max(safety_stock, self.min_safety_stock)
        if self.max_safety_stock is not None:
            safety_stock = min(safety_stock, self.max_safety_stock)
        return safety_stock
    
    def calculate_for_items(
        self,
        demand_stats: pd.DataFrame,
        lead_time: float,
        service_level_column: Optional[str] = None,
        default_service_level: float = 0.95,
    ) -> pd.DataFrame:
        """Calculate safety stock for multiple items.
        
        Args:
            demand_stats: DataFrame with demand statistics per item
            lead_time: Lead time in days (or column name for variable LT)
            service_level_column: Column with item-specific service levels
            default_service_level: Default service level if column not provided
            
        Returns:
            DataFrame with safety stock calculations
        """
        result = demand_stats.copy()
        
        safety_stocks = []
        for _, row in demand_stats.iterrows():
            # Get service level (item-specific or default)
            if service_level_column and service_level_column in demand_stats.columns:
                sl = row[service_level_column]
            else:
                sl = default_service_level
            
            # Get lead time (could be item-specific)
            lt = row.get("lead_time", lead_time)
            lt_std = row.get("lead_time_std", 0)
            
            # Calculate safety stock
            ss = self.calculate(
                demand_mean=row.get("demand_mean", row.get("daily_demand_rate", 0)),
                demand_std=row.get("demand_std", 0),
                lead_time=lt,
                service_level=sl,
                lead_time_std=lt_std,
            )
            safety_stocks.append(ss)
        
        result["safety_stock"] = safety_stocks
        return result


class DynamicSafetyStock:
    """Context-aware dynamic safety stock adjustments.
    
    Adjusts safety stock based on:
    - Promotional periods
    - Seasonal peaks
    - New product launches
    - Supply uncertainty
    """
    
    def __init__(
        self,
        base_calculator: Optional[SafetyStockCalculator] = None,
    ):
        """Initialize dynamic safety stock.
        
        Args:
            base_calculator: Base safety stock calculator
        """
        self.base_calculator = base_calculator or SafetyStockCalculator()
        
        # Adjustment factors
        self.promotional_multiplier = 1.5
        self.seasonal_peak_multiplier = 1.3
        self.new_product_multiplier = 2.0
        self.supply_risk_multiplier = 1.4
    
    def calculate_with_adjustments(
        self,
        demand_mean: float,
        demand_std: float,
        lead_time: float,
        service_level: float,
        is_promotional: bool = False,
        is_seasonal_peak: bool = False,
        is_new_product: bool = False,
        supply_risk: str = "normal",
    ) -> Dict[str, Any]:
        """Calculate safety stock with contextual adjustments.
        
        Args:
            demand_mean: Average daily demand
            demand_std: Demand standard deviation
            lead_time: Lead time in days
            service_level: Target service level
            is_promotional: Whether item is in promotional period
            is_seasonal_peak: Whether in seasonal peak period
            is_new_product: Whether item is a new product
            supply_risk: Supply risk level ('low', 'normal', 'high')
            
        Returns:
            Dictionary with safety stock and breakdown
        """
        # Calculate base safety stock
        base_ss = self.base_calculator.calculate(
            demand_mean=demand_mean,
            demand_std=demand_std,
            lead_time=lead_time,
            service_level=service_level,
        )
        
        # Apply adjustments
        adjusted_ss = base_ss
        adjustments = []
        
        if is_promotional:
            adjusted_ss *= self.promotional_multiplier
            adjustments.append(f"promotional: ×{self.promotional_multiplier}")
        
        if is_seasonal_peak:
            adjusted_ss *= self.seasonal_peak_multiplier
            adjustments.append(f"seasonal_peak: ×{self.seasonal_peak_multiplier}")
        
        if is_new_product:
            adjusted_ss *= self.new_product_multiplier
            adjustments.append(f"new_product: ×{self.new_product_multiplier}")
        
        if supply_risk == "high":
            adjusted_ss *= self.supply_risk_multiplier
            adjustments.append(f"supply_risk: ×{self.supply_risk_multiplier}")
        
        return {
            "base_safety_stock": base_ss,
            "adjusted_safety_stock": adjusted_ss,
            "adjustments_applied": adjustments,
            "total_multiplier": adjusted_ss / base_ss if base_ss > 0 else 1.0,
        }
    
    def recommend_safety_stock(
        self,
        item_profile: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Recommend safety stock based on item profile.
        
        Args:
            item_profile: Dictionary with item characteristics
            
        Returns:
            Safety stock recommendation with reasoning
        """
        # Extract item characteristics
        abc_class = item_profile.get("abc_class", "B")
        xyz_class = item_profile.get("xyz_class", "Y")
        
        # Determine base service level from classification
        service_level_matrix = {
            "AX": 0.99, "AY": 0.97, "AZ": 0.95,
            "BX": 0.97, "BY": 0.95, "BZ": 0.92,
            "CX": 0.95, "CY": 0.92, "CZ": 0.90,
        }
        
        matrix_key = f"{abc_class}{xyz_class}"
        service_level = service_level_matrix.get(matrix_key, 0.95)
        
        # Calculate with adjustments
        result = self.calculate_with_adjustments(
            demand_mean=item_profile.get("demand_mean", 0),
            demand_std=item_profile.get("demand_std", 0),
            lead_time=item_profile.get("lead_time", 7),
            service_level=service_level,
            is_promotional=item_profile.get("is_promotional", False),
            is_seasonal_peak=item_profile.get("is_seasonal_peak", False),
            is_new_product=item_profile.get("is_new_product", False),
            supply_risk=item_profile.get("supply_risk", "normal"),
        )
        
        result["recommended_service_level"] = service_level
        result["classification"] = matrix_key
        
        return result

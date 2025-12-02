"""Periodic Review (s,S) Inventory Policy.

The (s,S) policy is a periodic review system where:
- Inventory is reviewed at fixed intervals (review period)
- If inventory position ≤ s (reorder point), order up to S (order-up-to level)

Key formulas:
- s (Reorder Point) = DDR × LT + Safety Stock
- S (Order-Up-To) = DDR × (LT + RP) + Safety Stock
- Order Quantity = min(S - IP, Available Source Inventory)
"""

import logging
from typing import Any, Dict, List, Optional

import pandas as pd
import numpy as np

from ..interfaces.base import IPolicy
from ..safety_stock.calculator import SafetyStockCalculator

logger = logging.getLogger(__name__)


class PeriodicReviewPolicy(IPolicy):
    """Periodic Review (s,S) inventory policy implementation.
    
    This policy is ideal for:
    - Items replenished from the same source
    - Economies of scale in ordering
    - Moderate to high demand items
    
    Examples:
        >>> policy = PeriodicReviewPolicy(
        ...     review_period=7,
        ...     lead_time=14,
        ...     service_level=0.95
        ... )
        >>> result = policy.calculate(demand_df)
    """
    
    def __init__(
        self,
        review_period: int = 7,
        lead_time: float = 7.0,
        lead_time_std: float = 0.0,
        service_level: float = 0.95,
        order_strategy: str = "policy_target",
        safety_stock_method: str = "standard",
        min_order_qty: Optional[int] = None,
        max_order_qty: Optional[int] = None,
        order_multiple: Optional[int] = None,
        item_column: str = "item_id",
        location_column: Optional[str] = None,
    ):
        """Initialize periodic review policy.
        
        Args:
            review_period: Days between inventory reviews
            lead_time: Average lead time in days
            lead_time_std: Standard deviation of lead time
            service_level: Target service level (0-1)
            order_strategy: 'policy_target' (order to S) or 'fill_to_capacity'
            safety_stock_method: Method for safety stock calculation
            min_order_qty: Minimum order quantity (MOQ)
            max_order_qty: Maximum order quantity
            order_multiple: Order must be multiple of this value
            item_column: Item identifier column
            location_column: Optional location column
        """
        self.review_period = review_period
        self.lead_time = lead_time
        self.lead_time_std = lead_time_std
        self.service_level = service_level
        self.order_strategy = order_strategy
        self.min_order_qty = min_order_qty
        self.max_order_qty = max_order_qty
        self.order_multiple = order_multiple
        self.item_column = item_column
        self.location_column = location_column
        
        # Initialize safety stock calculator
        self.ss_calculator = SafetyStockCalculator(
            method=safety_stock_method
        )
        
        self._parameters: Dict[str, Any] = {}
    
    @property
    def policy_type(self) -> str:
        """Policy type identifier."""
        return "periodic_review_sS"
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get current policy parameters."""
        return {
            "review_period": self.review_period,
            "lead_time": self.lead_time,
            "lead_time_std": self.lead_time_std,
            "service_level": self.service_level,
            "order_strategy": self.order_strategy,
            "min_order_qty": self.min_order_qty,
            "max_order_qty": self.max_order_qty,
            "order_multiple": self.order_multiple,
        }
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate replenishment parameters for all items.
        
        Required columns in df:
        - item_id (or configured item_column)
        - demand_mean or daily_demand_rate
        - demand_std
        - current_stock or inventory_position
        
        Optional columns:
        - service_level (item-specific)
        - lead_time (item-specific)
        - max_capacity
        - source_available
        
        Args:
            df: DataFrame with inventory and demand data
            
        Returns:
            DataFrame with policy parameters and recommendations
        """
        if df.empty:
            logger.warning("Empty DataFrame provided for policy calculation")
            return df
        
        result = df.copy()
        
        # Calculate for each item
        reorder_points = []
        order_up_to_levels = []
        safety_stocks = []
        recommended_quantities = []
        needs_order = []
        
        for _, row in result.iterrows():
            # Get demand rate
            ddr = row.get("daily_demand_rate", row.get("demand_mean", 0))
            demand_std = row.get("demand_std", 0)
            
            # Get item-specific parameters or use defaults
            sl = row.get("target_service_level", self.service_level)
            lt = row.get("lead_time", self.lead_time)
            lt_std = row.get("lead_time_std", self.lead_time_std)
            
            # Calculate safety stock
            ss = self.ss_calculator.calculate(
                demand_mean=ddr,
                demand_std=demand_std,
                lead_time=lt,
                service_level=sl,
                lead_time_std=lt_std,
            )
            
            # Calculate reorder point (s)
            s = ddr * lt + ss
            
            # Calculate order-up-to level (S)
            S = ddr * (lt + self.review_period) + ss
            
            # Get current inventory position
            ip = row.get("inventory_position", row.get("current_stock", 0))
            
            # Determine if order is needed
            order_needed = ip <= s
            
            # Calculate recommended quantity
            if order_needed:
                if self.order_strategy == "policy_target":
                    rec_qty = S - ip
                elif self.order_strategy == "fill_to_capacity":
                    max_cap = row.get("max_capacity", float("inf"))
                    rec_qty = min(S - ip, max_cap - ip)
                else:
                    rec_qty = S - ip
                
                # Apply constraints
                rec_qty = self._apply_order_constraints(rec_qty)
                
                # Check source availability
                source_avail = row.get("source_available", float("inf"))
                rec_qty = min(rec_qty, source_avail)
            else:
                rec_qty = 0
            
            # Collect results
            safety_stocks.append(ss)
            reorder_points.append(s)
            order_up_to_levels.append(S)
            recommended_quantities.append(max(0, rec_qty))
            needs_order.append(order_needed)
        
        # Add columns to result
        result["safety_stock"] = safety_stocks
        result["reorder_point"] = reorder_points
        result["order_up_to_level"] = order_up_to_levels
        result["recommended_quantity"] = recommended_quantities
        result["needs_order"] = needs_order
        
        # Calculate additional metrics
        result["days_of_supply"] = np.where(
            result.get("daily_demand_rate", result.get("demand_mean", 1)) > 0,
            result.get("inventory_position", result.get("current_stock", 0)) / 
            result.get("daily_demand_rate", result.get("demand_mean", 1)),
            float("inf")
        )
        
        logger.info(
            "Policy calculation complete: %d items need ordering",
            result["needs_order"].sum()
        )
        
        return result
    
    def _apply_order_constraints(self, quantity: float) -> float:
        """Apply order quantity constraints."""
        # Minimum order quantity
        if self.min_order_qty and quantity > 0 and quantity < self.min_order_qty:
            quantity = self.min_order_qty
        
        # Maximum order quantity
        if self.max_order_qty:
            quantity = min(quantity, self.max_order_qty)
        
        # Order multiple
        if self.order_multiple and quantity > 0:
            quantity = (
                np.ceil(quantity / self.order_multiple) * self.order_multiple
            )
        
        return quantity
    
    def simulate_coverage(
        self,
        demand_mean: float,
        demand_std: float,
        current_stock: float,
        order_quantity: float,
        n_simulations: int = 1000,
    ) -> Dict[str, float]:
        """Simulate coverage probability for an order decision.
        
        Args:
            demand_mean: Average daily demand
            demand_std: Demand standard deviation
            current_stock: Current inventory position
            order_quantity: Proposed order quantity
            n_simulations: Number of Monte Carlo simulations
            
        Returns:
            Dictionary with coverage statistics
        """
        # Simulate demand during lead time + review period
        period_length = self.lead_time + self.review_period
        
        # Total demand follows approximately normal distribution
        period_demand_mean = demand_mean * period_length
        period_demand_std = demand_std * np.sqrt(period_length)
        
        # Simulate
        simulated_demands = np.random.normal(
            period_demand_mean, period_demand_std, n_simulations
        )
        simulated_demands = np.maximum(simulated_demands, 0)  # No negative demand
        
        # Calculate stock after order arrives
        stock_after_order = current_stock + order_quantity
        
        # Count how many times we can meet demand
        covered = simulated_demands <= stock_after_order
        
        return {
            "coverage_probability": covered.mean(),
            "expected_demand": period_demand_mean,
            "demand_std": period_demand_std,
            "stock_after_order": stock_after_order,
            "expected_remaining": stock_after_order - period_demand_mean,
        }


class PeriodicReviewPolicyFactory:
    """Factory for creating periodic review policies based on scenario."""
    
    SCENARIO_DEFAULTS = {
        "supplier_to_dc": {
            "review_period": 7,
            "lead_time": 14,
            "order_strategy": "policy_target",
        },
        "dc_to_store": {
            "review_period": 1,
            "lead_time": 2,
            "order_strategy": "fill_to_capacity",
        },
        "storage_to_picking": {
            "review_period": 1,
            "lead_time": 0.5,
            "order_strategy": "fill_to_capacity",
        },
        "backroom_to_shelf": {
            "review_period": 1,
            "lead_time": 0.1,
            "order_strategy": "fill_to_capacity",
        },
    }
    
    @classmethod
    def create_for_scenario(
        cls,
        scenario_type: str,
        **override_params,
    ) -> PeriodicReviewPolicy:
        """Create policy configured for a specific scenario.
        
        Args:
            scenario_type: Type of replenishment scenario
            **override_params: Parameters to override defaults
            
        Returns:
            Configured PeriodicReviewPolicy
        """
        defaults = cls.SCENARIO_DEFAULTS.get(
            scenario_type, 
            cls.SCENARIO_DEFAULTS["supplier_to_dc"]
        )
        
        params = {**defaults, **override_params}
        return PeriodicReviewPolicy(**params)

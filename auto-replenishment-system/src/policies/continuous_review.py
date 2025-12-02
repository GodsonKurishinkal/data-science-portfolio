"""Continuous Review (s,Q) Inventory Policy.

The (s,Q) policy continuously monitors inventory:
- When inventory falls to or below s (reorder point), place order
- Order a fixed quantity Q (economic order quantity)

Ideal for:
- High-value items requiring close monitoring
- Fast-moving items
- Items with consistent demand
"""

import logging
from typing import Any, Dict, Optional

import pandas as pd
import numpy as np

from ..interfaces.base import IPolicy
from ..safety_stock.calculator import SafetyStockCalculator

logger = logging.getLogger(__name__)


class ContinuousReviewPolicy(IPolicy):
    """Continuous Review (s,Q) inventory policy.
    
    Features:
    - Reorder point (s) triggers ordering
    - Fixed order quantity (Q) - typically EOQ
    - Continuous monitoring of inventory
    
    Examples:
        >>> policy = ContinuousReviewPolicy(
        ...     lead_time=7,
        ...     service_level=0.95,
        ...     ordering_cost=50,
        ...     holding_cost_rate=0.25
        ... )
        >>> result = policy.calculate(demand_df)
    """
    
    def __init__(
        self,
        lead_time: float = 7.0,
        lead_time_std: float = 0.0,
        service_level: float = 0.95,
        ordering_cost: float = 50.0,
        holding_cost_rate: float = 0.25,
        use_eoq: bool = True,
        fixed_order_qty: Optional[int] = None,
        min_order_qty: Optional[int] = None,
        max_order_qty: Optional[int] = None,
        order_multiple: Optional[int] = None,
        item_column: str = "item_id",
    ):
        """Initialize continuous review policy.
        
        Args:
            lead_time: Lead time in days
            lead_time_std: Lead time standard deviation
            service_level: Target service level
            ordering_cost: Fixed cost per order
            holding_cost_rate: Annual holding cost as % of unit cost
            use_eoq: Whether to use EOQ for order quantity
            fixed_order_qty: Fixed order quantity (if not using EOQ)
            min_order_qty: Minimum order quantity
            max_order_qty: Maximum order quantity
            order_multiple: Order multiple constraint
            item_column: Item identifier column
        """
        self.lead_time = lead_time
        self.lead_time_std = lead_time_std
        self.service_level = service_level
        self.ordering_cost = ordering_cost
        self.holding_cost_rate = holding_cost_rate
        self.use_eoq = use_eoq
        self.fixed_order_qty = fixed_order_qty
        self.min_order_qty = min_order_qty
        self.max_order_qty = max_order_qty
        self.order_multiple = order_multiple
        self.item_column = item_column
        
        self.ss_calculator = SafetyStockCalculator(method="standard")
    
    @property
    def policy_type(self) -> str:
        return "continuous_review_sQ"
    
    def get_parameters(self) -> Dict[str, Any]:
        return {
            "lead_time": self.lead_time,
            "service_level": self.service_level,
            "ordering_cost": self.ordering_cost,
            "holding_cost_rate": self.holding_cost_rate,
            "use_eoq": self.use_eoq,
        }
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate reorder point and order quantity for all items.
        
        Args:
            df: DataFrame with demand and cost data
            
        Returns:
            DataFrame with s, Q, and recommendations
        """
        if df.empty:
            return df
        
        result = df.copy()
        
        reorder_points = []
        order_quantities = []
        safety_stocks = []
        recommended_quantities = []
        needs_order = []
        
        for _, row in result.iterrows():
            ddr = row.get("daily_demand_rate", row.get("demand_mean", 0))
            demand_std = row.get("demand_std", 0)
            unit_cost = row.get("unit_cost", 1.0)
            
            # Calculate safety stock
            ss = self.ss_calculator.calculate(
                demand_mean=ddr,
                demand_std=demand_std,
                lead_time=self.lead_time,
                service_level=self.service_level,
            )
            
            # Calculate reorder point: s = DDR × LT + SS
            s = ddr * self.lead_time + ss
            
            # Calculate order quantity
            if self.use_eoq:
                q = self._calculate_eoq(ddr, unit_cost)
            else:
                q = self.fixed_order_qty or ddr * 7  # Default to 1 week
            
            q = self._apply_constraints(q)
            
            # Check if order needed
            ip = row.get("inventory_position", row.get("current_stock", 0))
            order_needed = ip <= s
            
            rec_qty = q if order_needed else 0
            
            safety_stocks.append(ss)
            reorder_points.append(s)
            order_quantities.append(q)
            recommended_quantities.append(rec_qty)
            needs_order.append(order_needed)
        
        result["safety_stock"] = safety_stocks
        result["reorder_point"] = reorder_points
        result["order_quantity_Q"] = order_quantities
        result["recommended_quantity"] = recommended_quantities
        result["needs_order"] = needs_order
        
        return result
    
    def _calculate_eoq(
        self,
        daily_demand: float,
        unit_cost: float,
    ) -> float:
        """Calculate Economic Order Quantity.
        
        EOQ = √(2 × D × S / H)
        
        Where:
            D = Annual demand
            S = Ordering cost per order
            H = Annual holding cost per unit
        """
        annual_demand = daily_demand * 365
        holding_cost_per_unit = unit_cost * self.holding_cost_rate
        
        if holding_cost_per_unit <= 0 or annual_demand <= 0:
            return daily_demand * 7  # Default to 1 week supply
        
        eoq = np.sqrt(
            (2 * annual_demand * self.ordering_cost) / holding_cost_per_unit
        )
        
        return max(1, eoq)
    
    def _apply_constraints(self, quantity: float) -> float:
        """Apply order quantity constraints."""
        if self.min_order_qty:
            quantity = max(quantity, self.min_order_qty)
        if self.max_order_qty:
            quantity = min(quantity, self.max_order_qty)
        if self.order_multiple:
            quantity = np.ceil(quantity / self.order_multiple) * self.order_multiple
        return quantity

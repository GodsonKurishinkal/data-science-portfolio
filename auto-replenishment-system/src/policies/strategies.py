"""Order quantity strategies for replenishment policies."""

import logging
from typing import Any, Dict, Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class OrderQuantityStrategy:
    """Strategies for determining final order quantities.
    
    Strategies:
    - policy_target: Order up to policy-defined target (S or Max)
    - fill_to_capacity: Fill to storage capacity
    - demand_based: Based on expected demand
    - economic: Minimize total cost (EOQ-based)
    """
    
    def __init__(
        self,
        strategy: str = "policy_target",
        min_order_qty: Optional[int] = None,
        max_order_qty: Optional[int] = None,
        order_multiple: Optional[int] = None,
        round_up: bool = True,
    ):
        """Initialize order quantity strategy.
        
        Args:
            strategy: Strategy name
            min_order_qty: Minimum order quantity
            max_order_qty: Maximum order quantity
            order_multiple: Order must be multiple of this
            round_up: Whether to round up (True) or down (False)
        """
        valid_strategies = [
            "policy_target", "fill_to_capacity", 
            "demand_based", "economic"
        ]
        if strategy not in valid_strategies:
            raise ValueError(f"Strategy must be one of {valid_strategies}")
        
        self.strategy = strategy
        self.min_order_qty = min_order_qty
        self.max_order_qty = max_order_qty
        self.order_multiple = order_multiple
        self.round_up = round_up
    
    def calculate_quantity(
        self,
        target_level: float,
        current_position: float,
        max_capacity: Optional[float] = None,
        expected_demand: Optional[float] = None,
        source_available: Optional[float] = None,
        unit_cost: float = 1.0,
        ordering_cost: float = 50.0,
        holding_cost_rate: float = 0.25,
    ) -> Dict[str, Any]:
        """Calculate order quantity using configured strategy.
        
        Args:
            target_level: Policy target level (S or Max)
            current_position: Current inventory position
            max_capacity: Maximum storage capacity
            expected_demand: Expected demand during lead time + review
            source_available: Available inventory at source
            unit_cost: Unit cost for economic calculations
            ordering_cost: Fixed ordering cost
            holding_cost_rate: Annual holding cost rate
            
        Returns:
            Dictionary with quantity and calculation details
        """
        # Calculate base quantity
        if self.strategy == "policy_target":
            base_qty = target_level - current_position
        
        elif self.strategy == "fill_to_capacity":
            if max_capacity:
                base_qty = max_capacity - current_position
            else:
                base_qty = target_level - current_position
        
        elif self.strategy == "demand_based":
            if expected_demand:
                base_qty = expected_demand - current_position
            else:
                base_qty = target_level - current_position
        
        elif self.strategy == "economic":
            # EOQ-based quantity
            annual_demand = (expected_demand or target_level) * (365 / 14)
            holding_cost = unit_cost * holding_cost_rate
            if holding_cost > 0 and annual_demand > 0:
                eoq = np.sqrt(2 * annual_demand * ordering_cost / holding_cost)
                base_qty = max(eoq, target_level - current_position)
            else:
                base_qty = target_level - current_position
        
        else:
            base_qty = target_level - current_position
        
        # Apply constraints
        final_qty = self._apply_constraints(base_qty)
        
        # Check source availability
        if source_available is not None:
            final_qty = min(final_qty, source_available)
        
        # Ensure non-negative
        final_qty = max(0, final_qty)
        
        return {
            "base_quantity": base_qty,
            "final_quantity": final_qty,
            "strategy_used": self.strategy,
            "constrained_by_source": (
                source_available is not None and 
                final_qty < base_qty
            ),
            "constrained_by_capacity": (
                max_capacity is not None and 
                final_qty < base_qty and 
                self.strategy == "fill_to_capacity"
            ),
        }
    
    def _apply_constraints(self, quantity: float) -> float:
        """Apply order quantity constraints."""
        # Apply order multiple first
        if self.order_multiple and quantity > 0:
            if self.round_up:
                quantity = np.ceil(quantity / self.order_multiple) * self.order_multiple
            else:
                quantity = np.floor(quantity / self.order_multiple) * self.order_multiple
        
        # Apply min/max
        if self.min_order_qty and quantity > 0:
            quantity = max(quantity, self.min_order_qty)
        
        if self.max_order_qty:
            quantity = min(quantity, self.max_order_qty)
        
        return quantity
    
    def apply_to_dataframe(
        self,
        df: pd.DataFrame,
        target_column: str = "order_up_to_level",
        position_column: str = "inventory_position",
    ) -> pd.DataFrame:
        """Apply strategy to entire DataFrame.
        
        Args:
            df: DataFrame with policy calculations
            target_column: Column with target levels
            position_column: Column with current positions
            
        Returns:
            DataFrame with final quantities
        """
        result = df.copy()
        
        final_quantities = []
        for _, row in result.iterrows():
            qty_result = self.calculate_quantity(
                target_level=row.get(target_column, 0),
                current_position=row.get(position_column, 
                                        row.get("current_stock", 0)),
                max_capacity=row.get("max_capacity"),
                expected_demand=row.get("expected_demand"),
                source_available=row.get("source_available"),
                unit_cost=row.get("unit_cost", 1.0),
            )
            final_quantities.append(qty_result["final_quantity"])
        
        result["final_order_quantity"] = final_quantities
        return result

"""Min-Max Inventory Policy.

Simple threshold-based policy:
- When inventory falls below Min, order up to Max
- Easy to understand and implement
- Good for low-value items or simple operations
"""

import logging
from typing import Any, Dict, Optional

import pandas as pd
import numpy as np

from ..interfaces.base import IPolicy

logger = logging.getLogger(__name__)


class MinMaxPolicy(IPolicy):
    """Min-Max inventory policy.

    When inventory ≤ Min:
        Order Quantity = Max - Current Inventory

    Simple to implement, good for C items.

    Examples:
        >>> policy = MinMaxPolicy(min_days_supply=7, max_days_supply=21)
        >>> result = policy.calculate(demand_df)
    """

    def __init__(
        self,
        min_days_supply: int = 7,
        max_days_supply: int = 21,
        min_quantity: Optional[int] = None,
        max_quantity: Optional[int] = None,
        item_column: str = "item_id",
    ):
        """Initialize Min-Max policy.

        Args:
            min_days_supply: Days of supply for minimum level
            max_days_supply: Days of supply for maximum level
            min_quantity: Absolute minimum quantity (override days)
            max_quantity: Absolute maximum quantity (override days)
            item_column: Item identifier column
        """
        self.min_days_supply = min_days_supply
        self.max_days_supply = max_days_supply
        self.min_quantity = min_quantity
        self.max_quantity = max_quantity
        self.item_column = item_column

    @property
    def policy_type(self) -> str:
        return "min_max"

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "min_days_supply": self.min_days_supply,
            "max_days_supply": self.max_days_supply,
            "min_quantity": self.min_quantity,
            "max_quantity": self.max_quantity,
        }

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate min/max levels and recommendations.

        Args:
            df: DataFrame with demand data

        Returns:
            DataFrame with min, max, and recommendations
        """
        if df.empty:
            return df

        result = df.copy()

        # Calculate min/max based on days of supply
        ddr = result.get("daily_demand_rate", result.get("demand_mean", 0))

        result["min_level"] = ddr * self.min_days_supply
        result["max_level"] = ddr * self.max_days_supply

        # Apply absolute constraints
        if self.min_quantity:
            result["min_level"] = result["min_level"].clip(lower=self.min_quantity)
        if self.max_quantity:
            result["max_level"] = result["max_level"].clip(upper=self.max_quantity)

        # Check current position
        ip = result.get("inventory_position", result.get("current_stock", 0))

        result["needs_order"] = ip <= result["min_level"]
        result["recommended_quantity"] = np.where(
            result["needs_order"],
            result["max_level"] - ip,
            0
        )
        result["recommended_quantity"] = result["recommended_quantity"].clip(lower=0)

        # Use min_level as reorder_point for consistency
        result["reorder_point"] = result["min_level"]
        result["order_up_to_level"] = result["max_level"]

        return result

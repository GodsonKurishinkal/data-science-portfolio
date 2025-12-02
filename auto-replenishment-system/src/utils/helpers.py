"""Helper functions for replenishment calculations."""

import logging
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def ensure_columns(
    df: pd.DataFrame,
    required_columns: List[str],
    fill_value: Any = 0,
) -> pd.DataFrame:
    """Ensure DataFrame has required columns.

    Args:
        df: Input DataFrame
        required_columns: List of required columns
        fill_value: Value to fill for missing columns

    Returns:
        DataFrame with all required columns
    """
    result = df.copy()

    for col in required_columns:
        if col not in result.columns:
            logger.warning(f"Adding missing column '{col}' with default value")
            result[col] = fill_value

    return result


def calculate_days_of_supply(
    current_stock: Union[float, pd.Series],
    daily_demand: Union[float, pd.Series],
    default_value: float = float("inf"),
) -> Union[float, pd.Series]:
    """Calculate days of supply.

    Args:
        current_stock: Current inventory level
        daily_demand: Daily demand rate
        default_value: Value when demand is zero

    Returns:
        Days of supply
    """
    if isinstance(current_stock, pd.Series):
        return np.where(
            daily_demand > 0,
            current_stock / daily_demand,
            default_value,
        )
    else:
        if daily_demand > 0:
            return current_stock / daily_demand
        return default_value


def calculate_inventory_position(
    on_hand: float,
    on_order: float = 0,
    backorder: float = 0,
) -> float:
    """Calculate inventory position.

    Inventory Position = On Hand + On Order - Backorders

    Args:
        on_hand: Current on-hand inventory
        on_order: Quantity on order
        backorder: Backorder quantity

    Returns:
        Inventory position
    """
    return on_hand + on_order - backorder


def format_quantity(
    quantity: float,
    decimals: int = 0,
    unit: str = "",
) -> str:
    """Format quantity for display.

    Args:
        quantity: Quantity value
        decimals: Decimal places
        unit: Unit suffix

    Returns:
        Formatted string
    """
    formatted = f"{quantity:,.{decimals}f}"
    if unit:
        formatted = f"{formatted} {unit}"
    return formatted


def classify_urgency(
    days_of_supply: float,
    critical_threshold: float = 1.0,
    high_threshold: float = 3.0,
    medium_threshold: float = 7.0,
) -> str:
    """Classify replenishment urgency.

    Args:
        days_of_supply: Current days of supply
        critical_threshold: Threshold for critical
        high_threshold: Threshold for high
        medium_threshold: Threshold for medium

    Returns:
        Urgency classification
    """
    if days_of_supply <= critical_threshold:
        return "critical"
    elif days_of_supply <= high_threshold:
        return "high"
    elif days_of_supply <= medium_threshold:
        return "medium"
    else:
        return "low"


def round_to_pack_size(
    quantity: float,
    pack_size: int,
    round_up: bool = True,
) -> int:
    """Round quantity to pack size.

    Args:
        quantity: Quantity to round
        pack_size: Pack size
        round_up: Whether to round up (True) or down (False)

    Returns:
        Rounded quantity
    """
    if pack_size <= 0:
        return int(quantity)

    if round_up:
        return int(np.ceil(quantity / pack_size) * pack_size)
    else:
        return int(np.floor(quantity / pack_size) * pack_size)

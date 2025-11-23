"""
Validation utilities for Dynamic Pricing Engine
"""

import pandas as pd
import numpy as np
from typing import Union


def validate_price(price: Union[float, pd.Series]) -> bool:
    """
    Validate price value(s).
    
    Args:
        price: Price value or series to validate
        
    Returns:
        True if valid
        
    Raises:
        ValueError: If price is invalid
    """
    if isinstance(price, (int, float)):
        if price <= 0:
            raise ValueError(f"Price must be positive, got {price}")
        if price > 10000:  # Sanity check
            raise ValueError(f"Price seems unreasonably high: {price}")
    elif isinstance(price, pd.Series):
        if (price <= 0).any():
            raise ValueError("All prices must be positive")
        if (price > 10000).any():
            raise ValueError("Some prices seem unreasonably high")
    else:
        raise TypeError(f"Price must be numeric or Series, got {type(price)}")
    
    return True


def validate_elasticity(elasticity: Union[float, pd.Series]) -> bool:
    """
    Validate price elasticity value(s).
    
    Args:
        elasticity: Elasticity value or series to validate
        
    Returns:
        True if valid
        
    Raises:
        ValueError: If elasticity is invalid
    """
    if isinstance(elasticity, (int, float)):
        if elasticity > 0:
            raise ValueError(f"Price elasticity should typically be negative, got {elasticity}")
        if elasticity < -10:
            raise ValueError(f"Elasticity seems unreasonably high: {elasticity}")
    elif isinstance(elasticity, pd.Series):
        if (elasticity > 0).any():
            raise ValueError("Price elasticities should typically be negative")
        if (elasticity < -10).any():
            raise ValueError("Some elasticities seem unreasonably high")
    else:
        raise TypeError(f"Elasticity must be numeric or Series, got {type(elasticity)}")
    
    return True


def validate_dataframe(
    df: pd.DataFrame,
    required_columns: list = None,
    min_rows: int = 1
) -> bool:
    """
    Validate DataFrame structure and content.
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        min_rows: Minimum number of rows required
        
    Returns:
        True if valid
        
    Raises:
        ValueError: If DataFrame is invalid
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected DataFrame, got {type(df)}")
    
    if len(df) < min_rows:
        raise ValueError(f"DataFrame must have at least {min_rows} rows, got {len(df)}")
    
    if required_columns:
        missing = set(required_columns) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
    
    # Check for excessive missing values
    null_pct = df.isnull().sum() / len(df)
    high_null = null_pct[null_pct > 0.5]
    if len(high_null) > 0:
        raise ValueError(f"Columns with >50% missing values: {list(high_null.index)}")
    
    return True

"""Data validators for ensuring data quality.

This module provides validation classes to check data integrity,
required columns, and business rule compliance.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

from ..interfaces.base import IValidator

logger = logging.getLogger(__name__)


class DataValidator(IValidator):
    """Base validator with common validation logic."""
    
    def __init__(self, required_columns: Optional[List[str]] = None):
        """Initialize validator.
        
        Args:
            required_columns: List of required column names
        """
        self._required_columns = required_columns or []
    
    @property
    def required_columns(self) -> List[str]:
        """Get required columns."""
        return self._required_columns
    
    def validate(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate DataFrame.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Check for empty DataFrame
        if df.empty:
            errors.append("DataFrame is empty")
            return False, errors
        
        # Check required columns
        missing_cols = set(self._required_columns) - set(df.columns)
        if missing_cols:
            errors.append(f"Missing required columns: {missing_cols}")
        
        # Check for all-null columns among required
        for col in self._required_columns:
            if col in df.columns and df[col].isna().all():
                errors.append(f"Column '{col}' contains only null values")
        
        is_valid = len(errors) == 0
        return is_valid, errors
    
    def validate_numeric_positive(
        self,
        df: pd.DataFrame,
        columns: List[str],
    ) -> List[str]:
        """Validate numeric columns have positive values.
        
        Args:
            df: DataFrame to validate
            columns: Columns to check
            
        Returns:
            List of validation errors
        """
        errors = []
        for col in columns:
            if col in df.columns:
                if (df[col] < 0).any():
                    errors.append(f"Column '{col}' contains negative values")
        return errors


class InventoryValidator(DataValidator):
    """Validator for inventory data."""
    
    def __init__(self):
        """Initialize inventory validator."""
        super().__init__(required_columns=["item_id", "current_stock"])
        self.numeric_columns = ["current_stock", "on_order", "backorders"]
    
    def validate(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate inventory DataFrame.
        
        Args:
            df: Inventory DataFrame
            
        Returns:
            Tuple of (is_valid, errors)
        """
        is_valid, errors = super().validate(df)
        
        if not is_valid and "DataFrame is empty" in errors[0]:
            return False, errors
        
        # Check for duplicate item IDs
        if "item_id" in df.columns:
            if df["item_id"].duplicated().any():
                duplicates = df[df["item_id"].duplicated()]["item_id"].unique()
                errors.append(f"Duplicate item IDs found: {list(duplicates)[:5]}...")
        
        # Check numeric columns are positive
        positive_errors = self.validate_numeric_positive(df, self.numeric_columns)
        errors.extend(positive_errors)
        
        # Check current_stock is not negative
        if "current_stock" in df.columns:
            if (df["current_stock"] < 0).any():
                errors.append("current_stock contains negative values")
        
        is_valid = len(errors) == 0
        return is_valid, errors


class DemandValidator(DataValidator):
    """Validator for demand/sales data."""
    
    def __init__(self):
        """Initialize demand validator."""
        super().__init__(required_columns=["item_id", "date", "quantity"])
    
    def validate(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate demand DataFrame.
        
        Args:
            df: Demand DataFrame
            
        Returns:
            Tuple of (is_valid, errors)
        """
        is_valid, errors = super().validate(df)
        
        if not is_valid and errors and "DataFrame is empty" in errors[0]:
            return False, errors
        
        # Check date column is datetime
        if "date" in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df["date"]):
                errors.append("Column 'date' is not datetime type")
        
        # Check quantity is numeric and mostly positive
        if "quantity" in df.columns:
            if not pd.api.types.is_numeric_dtype(df["quantity"]):
                errors.append("Column 'quantity' is not numeric")
            elif (df["quantity"] < 0).mean() > 0.1:  # More than 10% negative
                errors.append("More than 10% of quantities are negative (returns?)")
        
        # Check for sufficient history per item
        if "item_id" in df.columns and "date" in df.columns:
            history_days = df.groupby("item_id")["date"].apply(
                lambda x: (x.max() - x.min()).days
            )
            short_history = (history_days < 30).sum()
            if short_history > 0:
                logger.warning(
                    "%d items have less than 30 days of history", 
                    short_history
                )
        
        is_valid = len(errors) == 0
        return is_valid, errors
    
    def check_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate data quality report.
        
        Args:
            df: Demand DataFrame
            
        Returns:
            Dictionary with quality metrics
        """
        report = {
            "total_rows": len(df),
            "unique_items": df["item_id"].nunique() if "item_id" in df.columns else 0,
            "date_range": None,
            "null_counts": df.isnull().sum().to_dict(),
            "negative_quantities": 0,
            "zero_quantities": 0,
        }
        
        if "date" in df.columns:
            report["date_range"] = {
                "min": df["date"].min(),
                "max": df["date"].max(),
                "days": (df["date"].max() - df["date"].min()).days,
            }
        
        if "quantity" in df.columns:
            report["negative_quantities"] = (df["quantity"] < 0).sum()
            report["zero_quantities"] = (df["quantity"] == 0).sum()
        
        return report


class SourceInventoryValidator(DataValidator):
    """Validator for source/supplier inventory data."""
    
    def __init__(self):
        """Initialize source inventory validator."""
        super().__init__(required_columns=["item_id", "available_quantity"])
    
    def validate(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate source inventory DataFrame.
        
        Args:
            df: Source inventory DataFrame
            
        Returns:
            Tuple of (is_valid, errors)
        """
        is_valid, errors = super().validate(df)
        
        if not is_valid and errors and "DataFrame is empty" in errors[0]:
            return False, errors
        
        # Check available_quantity is non-negative
        if "available_quantity" in df.columns:
            if (df["available_quantity"] < 0).any():
                errors.append("available_quantity contains negative values")
        
        is_valid = len(errors) == 0
        return is_valid, errors

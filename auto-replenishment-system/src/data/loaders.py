"""Data loaders for various data sources.

This module provides implementations for loading inventory and demand data
from CSV files, databases, and APIs.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from ..interfaces.base import ILoader

logger = logging.getLogger(__name__)


class DataLoader(ILoader):
    """Base data loader with common functionality."""
    
    def __init__(self, source_path: Optional[str] = None):
        """Initialize data loader.
        
        Args:
            source_path: Path to data source (file, connection string, etc.)
        """
        self.source_path = source_path
        self._data: Optional[pd.DataFrame] = None
    
    def load(self, **kwargs) -> pd.DataFrame:
        """Load data from source.
        
        Args:
            **kwargs: Additional loading parameters
            
        Returns:
            Loaded DataFrame
        """
        raise NotImplementedError("Subclasses must implement load()")
    
    def validate(self) -> bool:
        """Validate data source is accessible.
        
        Returns:
            True if source is valid and accessible
        """
        if self.source_path is None:
            return False
        return Path(self.source_path).exists()
    
    @property
    def data(self) -> Optional[pd.DataFrame]:
        """Get loaded data."""
        return self._data


class CSVLoader(DataLoader):
    """Load data from CSV files."""
    
    def __init__(
        self,
        source_path: str,
        date_columns: Optional[List[str]] = None,
        dtype: Optional[Dict[str, Any]] = None,
    ):
        """Initialize CSV loader.
        
        Args:
            source_path: Path to CSV file
            date_columns: Columns to parse as dates
            dtype: Column data types
        """
        super().__init__(source_path)
        self.date_columns = date_columns or []
        self.dtype = dtype
    
    def load(self, **kwargs) -> pd.DataFrame:
        """Load data from CSV file.
        
        Args:
            **kwargs: Additional pd.read_csv parameters
            
        Returns:
            Loaded DataFrame
            
        Raises:
            FileNotFoundError: If file doesn't exist
        """
        if not self.validate():
            raise FileNotFoundError(f"CSV file not found: {self.source_path}")
        
        parse_dates = self.date_columns if self.date_columns else False
        
        self._data = pd.read_csv(
            self.source_path,
            parse_dates=parse_dates,
            dtype=self.dtype,
            **kwargs,
        )
        
        logger.info("Loaded %d rows from %s", len(self._data), self.source_path)
        return self._data


class InventoryDataLoader(CSVLoader):
    """Specialized loader for inventory data.
    
    Expected columns:
        - item_id: Unique item identifier
        - current_stock: Current inventory level
        - unit_cost: Cost per unit
        - max_capacity: Maximum storage capacity (optional)
        - location: Storage location (optional)
    """
    
    REQUIRED_COLUMNS = ["item_id", "current_stock"]
    OPTIONAL_COLUMNS = ["unit_cost", "max_capacity", "location", "on_order", "backorders"]
    
    def __init__(
        self,
        source_path: str,
        item_id_column: str = "item_id",
        stock_column: str = "current_stock",
    ):
        """Initialize inventory data loader.
        
        Args:
            source_path: Path to inventory CSV
            item_id_column: Name of item ID column
            stock_column: Name of current stock column
        """
        super().__init__(source_path)
        self.item_id_column = item_id_column
        self.stock_column = stock_column
    
    def load(self, **kwargs) -> pd.DataFrame:
        """Load and preprocess inventory data.
        
        Returns:
            Preprocessed inventory DataFrame
        """
        df = super().load(**kwargs)
        
        # Standardize column names
        rename_map = {}
        if self.item_id_column != "item_id":
            rename_map[self.item_id_column] = "item_id"
        if self.stock_column != "current_stock":
            rename_map[self.stock_column] = "current_stock"
        
        if rename_map:
            df = df.rename(columns=rename_map)
        
        # Add default values for optional columns
        if "on_order" not in df.columns:
            df["on_order"] = 0
        if "backorders" not in df.columns:
            df["backorders"] = 0
        
        # Calculate inventory position
        df["inventory_position"] = df["current_stock"] + df["on_order"] - df["backorders"]
        
        self._data = df
        return df


class DemandDataLoader(CSVLoader):
    """Specialized loader for demand/sales history data.
    
    Expected columns:
        - item_id: Unique item identifier
        - date: Transaction date
        - quantity: Demand quantity
        - revenue: Revenue (optional, for ABC analysis)
    """
    
    REQUIRED_COLUMNS = ["item_id", "date", "quantity"]
    OPTIONAL_COLUMNS = ["revenue", "price", "customer_id", "order_id"]
    
    def __init__(
        self,
        source_path: str,
        item_id_column: str = "item_id",
        date_column: str = "date",
        quantity_column: str = "quantity",
    ):
        """Initialize demand data loader.
        
        Args:
            source_path: Path to demand CSV
            item_id_column: Name of item ID column
            date_column: Name of date column
            quantity_column: Name of quantity column
        """
        super().__init__(source_path, date_columns=[date_column])
        self.item_id_column = item_id_column
        self.date_column = date_column
        self.quantity_column = quantity_column
    
    def load(self, **kwargs) -> pd.DataFrame:
        """Load and preprocess demand data.
        
        Returns:
            Preprocessed demand DataFrame
        """
        df = super().load(**kwargs)
        
        # Standardize column names
        rename_map = {}
        if self.item_id_column != "item_id":
            rename_map[self.item_id_column] = "item_id"
        if self.date_column != "date":
            rename_map[self.date_column] = "date"
        if self.quantity_column != "quantity":
            rename_map[self.quantity_column] = "quantity"
        
        if rename_map:
            df = df.rename(columns=rename_map)
        
        # Ensure date is datetime
        if not pd.api.types.is_datetime64_any_dtype(df["date"]):
            df["date"] = pd.to_datetime(df["date"])
        
        # Sort by date
        df = df.sort_values(["item_id", "date"])
        
        # Calculate revenue if not present but price is
        if "revenue" not in df.columns and "price" in df.columns:
            df["revenue"] = df["quantity"] * df["price"]
        
        self._data = df
        return df


class SourceInventoryLoader(CSVLoader):
    """Loader for source/supplier inventory data.
    
    Used to know available inventory at source locations
    for replenishment planning.
    """
    
    REQUIRED_COLUMNS = ["item_id", "available_quantity"]
    
    def __init__(
        self,
        source_path: str,
        item_id_column: str = "item_id",
        available_column: str = "available_quantity",
    ):
        """Initialize source inventory loader.
        
        Args:
            source_path: Path to source inventory CSV
            item_id_column: Name of item ID column
            available_column: Name of available quantity column
        """
        super().__init__(source_path)
        self.item_id_column = item_id_column
        self.available_column = available_column
    
    def load(self, **kwargs) -> pd.DataFrame:
        """Load source inventory data.
        
        Returns:
            Source inventory DataFrame
        """
        df = super().load(**kwargs)
        
        # Standardize column names
        rename_map = {}
        if self.item_id_column != "item_id":
            rename_map[self.item_id_column] = "item_id"
        if self.available_column != "available_quantity":
            rename_map[self.available_column] = "available_quantity"
        
        if rename_map:
            df = df.rename(columns=rename_map)
        
        self._data = df
        return df

"""
Data loading module

Loads and prepares M5 Walmart data for pricing analysis.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)


class PricingDataLoader:
    """
    Load and prepare M5 data for pricing analysis.
    
    This class handles loading the M5 Walmart dataset and preparing it
    for price elasticity analysis and optimization.
    
    Attributes:
        data_path: Path to M5 raw data directory
    """
    
    def __init__(self, data_path: str = 'data/raw'):
        """
        Initialize data loader.
        
        Args:
            data_path: Path to M5 raw data directory
        """
        self.data_path = Path(data_path)
        logger.info(f"Initialized PricingDataLoader with path: {self.data_path}")
    
    def load_sales_data(self, validation: bool = True) -> pd.DataFrame:
        """
        Load M5 sales data.
        
        Args:
            validation: If True, load validation dataset (includes eval period)
        
        Returns:
            DataFrame with sales data in long format
        """
        filename = 'sales_train_validation.csv' if validation else 'sales_train_evaluation.csv'
        filepath = self.data_path / filename
        
        logger.info(f"Loading sales data from {filepath}")
        
        if not filepath.exists():
            raise FileNotFoundError(f"Sales data not found at {filepath}")
        
        # Load sales data
        sales_df = pd.read_csv(filepath)
        logger.info(f"Loaded sales data: {sales_df.shape}")
        
        # Melt from wide to long format
        id_cols = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
        value_cols = [col for col in sales_df.columns if col.startswith('d_')]
        
        sales_long = pd.melt(
            sales_df,
            id_vars=id_cols,
            value_vars=value_cols,
            var_name='d',
            value_name='sales'
        )
        
        logger.info(f"Melted sales data to long format: {sales_long.shape}")
        return sales_long
    
    def load_price_data(self) -> pd.DataFrame:
        """
        Load M5 price data.
        
        Returns:
            DataFrame with price data
        """
        filepath = self.data_path / 'sell_prices.csv'
        
        logger.info(f"Loading price data from {filepath}")
        
        if not filepath.exists():
            raise FileNotFoundError(f"Price data not found at {filepath}")
        
        prices_df = pd.read_csv(filepath)
        logger.info(f"Loaded price data: {prices_df.shape}")
        
        return prices_df
    
    def load_calendar_data(self) -> pd.DataFrame:
        """
        Load M5 calendar data.
        
        Returns:
            DataFrame with calendar/date information
        """
        filepath = self.data_path / 'calendar.csv'
        
        logger.info(f"Loading calendar data from {filepath}")
        
        if not filepath.exists():
            raise FileNotFoundError(f"Calendar data not found at {filepath}")
        
        calendar_df = pd.read_csv(filepath)
        logger.info(f"Loaded calendar data: {calendar_df.shape}")
        
        # Convert date to datetime
        calendar_df['date'] = pd.to_datetime(calendar_df['date'])
        
        return calendar_df
    
    def merge_all(
        self,
        sample_stores: Optional[List[str]] = None,
        sample_items: Optional[int] = None,
        date_range: Optional[tuple] = None
    ) -> pd.DataFrame:
        """
        Merge all datasets into unified pricing dataset.
        
        Args:
            sample_stores: List of store IDs to include (e.g., ['CA_1', 'TX_1'])
            sample_items: Number of items to sample (for faster processing)
            date_range: Tuple of (start_date, end_date) strings
        
        Returns:
            Merged DataFrame with sales, prices, and calendar data
        """
        logger.info("Starting data merge process")
        
        # Load all datasets
        sales = self.load_sales_data(validation=True)
        prices = self.load_price_data()
        calendar = self.load_calendar_data()
        
        # Apply sampling if specified
        if sample_stores:
            logger.info(f"Filtering for stores: {sample_stores}")
            sales = sales[sales['store_id'].isin(sample_stores)]
        
        if sample_items:
            logger.info(f"Sampling {sample_items} items")
            unique_items = sales['item_id'].unique()
            sampled_items = np.random.choice(unique_items, size=min(sample_items, len(unique_items)), replace=False)
            sales = sales[sales['item_id'].isin(sampled_items)]
        
        # Merge sales with calendar
        logger.info("Merging sales with calendar")
        df = sales.merge(calendar, on='d', how='left')
        
        # Merge with prices
        logger.info("Merging with prices")
        df = df.merge(
            prices,
            on=['store_id', 'item_id', 'wm_yr_wk'],
            how='left'
        )
        
        # Apply date range filter if specified
        if date_range:
            start_date, end_date = date_range
            logger.info(f"Filtering date range: {start_date} to {end_date}")
            df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
        
        # Handle missing prices
        logger.info(f"Missing prices before fill: {df['sell_price'].isna().sum()}")
        
        # Forward fill prices within each item-store group
        df = df.sort_values(['store_id', 'item_id', 'date'])
        df['sell_price'] = df.groupby(['store_id', 'item_id'])['sell_price'].ffill()
        
        # Drop rows with no price data
        df = df.dropna(subset=['sell_price'])
        
        logger.info(f"Final merged dataset shape: {df.shape}")
        logger.info(f"Date range: {df['date'].min()} to {df['date'].max()}")
        logger.info(f"Unique items: {df['item_id'].nunique()}")
        logger.info(f"Unique stores: {df['store_id'].nunique()}")
        
        return df
    
    def load_sample_for_demo(self) -> pd.DataFrame:
        """
        Load a small sample for quick demo/testing.
        
        Returns:
            Small sample dataset
        """
        logger.info("Loading sample dataset for demo")
        
        return self.merge_all(
            sample_stores=['CA_1'],
            sample_items=100,
            date_range=('2015-01-01', '2016-06-19')
        )

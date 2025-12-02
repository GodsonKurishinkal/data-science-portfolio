"""Data loading and preprocessing module."""

import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class DataLoader:
    """Load and prepare M5 Walmart data for inventory optimization."""

    def __init__(self, data_path: str):
        """
        Initialize DataLoader.

        Args:
            data_path: Path to raw data directory
        """
        self.data_path = Path(data_path)
        self.calendar = None
        self.sales_train = None
        self.sell_prices = None
        self.sales_data = None

    def load_raw_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load raw M5 data files.

        Returns:
            Tuple of (calendar, sales_train, sell_prices) DataFrames
        """
        logger.info("Loading raw data files...")

        # Load calendar
        calendar_path = self.data_path / "calendar.csv"
        self.calendar = pd.read_csv(calendar_path)
        logger.info("Loaded calendar: %s", self.calendar.shape)

        # Load sales training data
        sales_path = self.data_path / "sales_train_evaluation.csv"
        self.sales_train = pd.read_csv(sales_path)
        logger.info("Loaded sales data: %s", self.sales_train.shape)

        # Load sell prices
        prices_path = self.data_path / "sell_prices.csv"
        self.sell_prices = pd.read_csv(prices_path)
        logger.info("Loaded price data: %s", self.sell_prices.shape)

        return self.calendar, self.sales_train, self.sell_prices

    def melt_sales_data(self) -> pd.DataFrame:
        """
        Transform wide-format sales data to long format.

        Returns:
            Long-format sales DataFrame
        """
        logger.info("Melting sales data to long format...")

        # Get ID columns
        id_cols = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']

        # Melt day columns
        sales_melted = self.sales_train.melt(
            id_vars=id_cols,
            var_name='d',
            value_name='sales'
        )

        logger.info("Melted sales shape: %s", sales_melted.shape)
        return sales_melted

    def merge_data(self, sales_melted: pd.DataFrame) -> pd.DataFrame:
        """
        Merge sales, calendar, and price data.

        Args:
            sales_melted: Long-format sales data

        Returns:
            Merged DataFrame
        """
        logger.info("Merging datasets...")

        # Merge with calendar
        data = sales_melted.merge(
            self.calendar,
            on='d',
            how='left'
        )

        # Merge with prices
        data = data.merge(
            self.sell_prices,
            on=['store_id', 'item_id', 'wm_yr_wk'],
            how='left'
        )

        # Convert date
        data['date'] = pd.to_datetime(data['date'])

        # Calculate revenue
        data['revenue'] = data['sales'] * data['sell_price']

        logger.info("Merged data shape: %s", data.shape)
        logger.info("Date range: %s to %s", data['date'].min(), data['date'].max())

        return data

    def process_data(self) -> pd.DataFrame:
        """
        Complete data processing pipeline.

        Returns:
            Processed DataFrame ready for analysis
        """
        # Load raw data
        self.load_raw_data()

        # Melt sales data
        sales_melted = self.melt_sales_data()

        # Merge all data
        data = self.merge_data(sales_melted)

        # Store processed data
        self.sales_data = data

        return data

    def get_item_hierarchy(self) -> Dict[str, List[str]]:
        """
        Extract item hierarchy information.

        Returns:
            Dictionary with hierarchy levels
        """
        if self.sales_train is None:
            self.load_raw_data()

        hierarchy = {
            'states': sorted(self.sales_train['state_id'].unique()),
            'stores': sorted(self.sales_train['store_id'].unique()),
            'categories': sorted(self.sales_train['cat_id'].unique()),
            'departments': sorted(self.sales_train['dept_id'].unique()),
            'items': sorted(self.sales_train['item_id'].unique())
        }

        logger.info(
            "Hierarchy: %d states, %d stores, %d categories, %d departments, %d items",
            len(hierarchy['states']),
            len(hierarchy['stores']),
            len(hierarchy['categories']),
            len(hierarchy['departments']),
            len(hierarchy['items'])
        )

        return hierarchy

    def filter_data(
        self,
        data: pd.DataFrame,
        stores: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Filter data by various criteria.

        Args:
            data: Input DataFrame
            stores: List of store IDs to include
            categories: List of category IDs to include
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            Filtered DataFrame
        """
        filtered = data.copy()

        if stores:
            filtered = filtered[filtered['store_id'].isin(stores)]
            logger.info("Filtered to stores: %s", stores)

        if categories:
            filtered = filtered[filtered['cat_id'].isin(categories)]
            logger.info("Filtered to categories: %s", categories)

        if start_date:
            filtered = filtered[filtered['date'] >= start_date]
            logger.info("Filtered to start_date >= %s", start_date)

        if end_date:
            filtered = filtered[filtered['date'] <= end_date]
            logger.info("Filtered to end_date <= %s", end_date)

        logger.info("Filtered data shape: %s", filtered.shape)
        return filtered


class DemandCalculator:
    """Calculate demand statistics for inventory optimization."""

    @staticmethod
    def calculate_demand_statistics(
        data: pd.DataFrame,
        group_cols: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Calculate demand statistics by group.

        Args:
            data: Sales data
            group_cols: Columns to group by

        Returns:
            DataFrame with demand statistics
        """
        if group_cols is None:
            group_cols = ['store_id', 'item_id']
        logger.info("Calculating demand statistics grouped by: %s", group_cols)

        stats = data.groupby(group_cols).agg({
            'sales': [
                'sum',      # Total demand
                'mean',     # Average daily demand
                'std',      # Standard deviation
                'min',      # Minimum demand
                'max',      # Maximum demand
                'count'     # Number of observations
            ],
            'revenue': ['sum', 'mean'],
            'sell_price': ['mean', 'min', 'max']
        }).reset_index()

        # Flatten column names
        stats.columns = ['_'.join(col).strip('_') for col in stats.columns.values]

        # Calculate coefficient of variation
        stats['demand_cv'] = stats['sales_std'] / stats['sales_mean']
        stats['demand_cv'] = stats['demand_cv'].fillna(0)

        # Calculate fill rate (% of days with sales > 0)
        fill_rate = data.groupby(group_cols)['sales'].apply(
            lambda x: (x > 0).sum() / len(x)
        ).reset_index()
        fill_rate.columns = group_cols + ['fill_rate']

        stats = stats.merge(fill_rate, on=group_cols, how='left')

        logger.info("Calculated statistics for %d groups", len(stats))

        return stats

    @staticmethod
    def calculate_rolling_statistics(
        data: pd.DataFrame,
        windows: Optional[List[int]] = None,
        group_cols: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Calculate rolling demand statistics.

        Args:
            data: Sales data with date column
            windows: List of window sizes (in days)
            group_cols: Columns to group by

        Returns:
            DataFrame with rolling statistics
        """
        if windows is None:
            windows = [7, 14, 28]
        if group_cols is None:
            group_cols = ['store_id', 'item_id']
        logger.info("Calculating rolling statistics for windows: %s", windows)

        # Sort by group and date
        data_sorted = data.sort_values(group_cols + ['date'])

        # Calculate rolling stats for each window
        for window in windows:
            w = window  # Capture loop variable
            data_sorted[f'rolling_mean_{window}d'] = data_sorted.groupby(group_cols)['sales'].transform(
                lambda x, w=w: x.rolling(window=w, min_periods=1).mean()
            )
            data_sorted[f'rolling_std_{window}d'] = data_sorted.groupby(group_cols)['sales'].transform(
                lambda x, w=w: x.rolling(window=w, min_periods=1).std()
            )

        return data_sorted

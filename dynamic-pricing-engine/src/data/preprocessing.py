"""
Data preprocessing module

Preprocesses pricing data and engineers features for pricing analysis.
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict
import logging

logger = logging.getLogger(__name__)


class PricingDataPreprocessor:
    """
    Preprocess pricing data and engineer features.

    This class handles:
    - Price history extraction
    - Promotional period identification
    - Price feature engineering
    - Statistical calculations
    """

    def __init__(self):
        """Initialize data preprocessor."""
        logger.info("Initialized PricingDataPreprocessor")

    def extract_price_history(
        self,
        df: pd.DataFrame,
        group_cols: List[str] = ['store_id', 'item_id']
    ) -> pd.DataFrame:
        """
        Extract price change history over time.

        Args:
            df: DataFrame with price and sales data
            group_cols: Columns to group by

        Returns:
            DataFrame with price change indicators
        """
        logger.info("Extracting price change history")

        df = df.copy()
        df = df.sort_values(group_cols + ['date'])

        # Calculate price changes
        df['price_lag_1'] = df.groupby(group_cols)['sell_price'].shift(1)
        df['price_change'] = df['sell_price'] - df['price_lag_1']
        df['price_change_pct'] = (df['price_change'] / df['price_lag_1']) * 100

        # Identify price change events
        df['price_changed'] = (df['price_change'] != 0) & (df['price_change'].notna())

        # Days since last price change
        df['days_since_price_change'] = 0
        for name, group in df.groupby(group_cols):
            change_dates = group[group['price_changed']].index
            for idx in group.index:
                prior_changes = change_dates[change_dates < idx]
                if len(prior_changes) > 0:
                    days = (group.loc[idx, 'date'] - group.loc[prior_changes[-1], 'date']).days
                    df.loc[idx, 'days_since_price_change'] = days
                else:
                    df.loc[idx, 'days_since_price_change'] = 999  # No prior change

        logger.info(f"Price changes identified: {df['price_changed'].sum()}")

        return df

    def calculate_price_statistics(
        self,
        df: pd.DataFrame,
        group_cols: List[str] = ['store_id', 'item_id']
    ) -> pd.DataFrame:
        """
        Calculate price statistics by product.

        Args:
            df: DataFrame with price data
            group_cols: Columns to group by

        Returns:
            DataFrame with price statistics merged
        """
        logger.info("Calculating price statistics")

        # Calculate statistics
        price_stats = df.groupby(group_cols).agg({
            'sell_price': ['mean', 'std', 'min', 'max', 'median']
        }).reset_index()

        price_stats.columns = group_cols + [
            'price_mean', 'price_std', 'price_min', 'price_max', 'price_median'
        ]

        # Merge back to original dataframe
        df = df.merge(price_stats, on=group_cols, how='left')

        # Calculate relative price metrics
        df['price_vs_mean'] = (df['sell_price'] - df['price_mean']) / df['price_mean']
        df['price_vs_median'] = (df['sell_price'] - df['price_median']) / df['price_median']
        df['price_percentile'] = df.groupby(group_cols)['sell_price'].rank(pct=True)

        # Price range
        df['price_range'] = df['price_max'] - df['price_min']
        df['price_volatility'] = df['price_std'] / df['price_mean']  # Coefficient of variation

        logger.info("Price statistics calculated")

        return df

    def identify_promotions(
        self,
        df: pd.DataFrame,
        promotion_threshold: float = -0.05
    ) -> pd.DataFrame:
        """
        Identify promotional periods (price drops).

        Args:
            df: DataFrame with price data
            promotion_threshold: Threshold for price drop (e.g., -5%)

        Returns:
            DataFrame with promotion indicators
        """
        logger.info(f"Identifying promotions (threshold: {promotion_threshold})")

        df = df.copy()

        # Promotion is when price drops significantly below average
        df['is_promotion'] = df['price_vs_mean'] < promotion_threshold

        # Discount depth
        df['discount_depth'] = np.where(
            df['is_promotion'],
            abs(df['price_vs_mean']),
            0
        )

        # Count promotions
        n_promotions = df['is_promotion'].sum()
        pct_promotions = (n_promotions / len(df)) * 100

        logger.info(f"Promotions identified: {n_promotions} ({pct_promotions:.2f}% of data)")

        return df

    def engineer_pricing_features(
        self,
        df: pd.DataFrame,
        include_lags: bool = True,
        include_rolling: bool = True
    ) -> pd.DataFrame:
        """
        Engineer comprehensive pricing features.

        Args:
            df: DataFrame with base data
            include_lags: Whether to include lag features
            include_rolling: Whether to include rolling statistics

        Returns:
            DataFrame with engineered features
        """
        logger.info("Engineering pricing features")

        df = df.copy()
        df = df.sort_values(['store_id', 'item_id', 'date'])

        # Price momentum features
        if include_lags:
            logger.info("Creating lag features")
            for lag in [7, 14, 28]:
                df[f'price_lag_{lag}'] = df.groupby(['store_id', 'item_id'])['sell_price'].shift(lag)
                df[f'price_change_{lag}d'] = df['sell_price'] - df[f'price_lag_{lag}']
                df[f'price_change_pct_{lag}d'] = (df[f'price_change_{lag}d'] / df[f'price_lag_{lag}']) * 100

        # Rolling price features
        if include_rolling:
            logger.info("Creating rolling features")
            for window in [7, 14, 28]:
                df[f'price_rolling_mean_{window}'] = df.groupby(['store_id', 'item_id'])['sell_price'].transform(
                    lambda x: x.rolling(window, min_periods=1).mean()
                )
                df[f'price_rolling_std_{window}'] = df.groupby(['store_id', 'item_id'])['sell_price'].transform(
                    lambda x: x.rolling(window, min_periods=1).std()
                )

        # Price trend indicators
        df['price_trend_7d'] = df['sell_price'] - df.get('price_rolling_mean_7', df['sell_price'])
        df['price_trend_28d'] = df['sell_price'] - df.get('price_rolling_mean_28', df['sell_price'])

        # Competitive price proxies (store-level averages by category)
        logger.info("Creating competitive price proxies")
        store_cat_prices = df.groupby(['store_id', 'cat_id', 'date'])['sell_price'].mean().reset_index()
        store_cat_prices.columns = ['store_id', 'cat_id', 'date', 'store_cat_avg_price']
        df = df.merge(store_cat_prices, on=['store_id', 'cat_id', 'date'], how='left')

        df['price_vs_store_cat_avg'] = (df['sell_price'] - df['store_cat_avg_price']) / df['store_cat_avg_price']

        logger.info(f"Feature engineering complete. Final shape: {df.shape}")

        return df

    def create_pricing_dataset(
        self,
        df: pd.DataFrame,
        include_all_features: bool = True
    ) -> pd.DataFrame:
        """
        Create complete pricing dataset with all preprocessing steps.

        Args:
            df: Raw merged DataFrame
            include_all_features: Whether to include all engineered features

        Returns:
            Fully processed pricing dataset
        """
        logger.info("Creating complete pricing dataset")

        # Extract price history
        df = self.extract_price_history(df)

        # Calculate statistics
        df = self.calculate_price_statistics(df)

        # Identify promotions
        df = self.identify_promotions(df)

        # Engineer features if requested
        if include_all_features:
            df = self.engineer_pricing_features(df)

        logger.info(f"Pricing dataset created with {len(df.columns)} columns")

        return df

    def get_product_price_summary(
        self,
        df: pd.DataFrame,
        group_cols: List[str] = ['store_id', 'item_id']
    ) -> pd.DataFrame:
        """
        Get summary statistics for each product.

        Args:
            df: Processed pricing DataFrame
            group_cols: Columns to group by

        Returns:
            Summary statistics DataFrame
        """
        logger.info("Creating product price summary")

        summary = df.groupby(group_cols).agg({
            'sell_price': ['mean', 'std', 'min', 'max', 'count'],
            'sales': ['sum', 'mean', 'std'],
            'is_promotion': 'sum',
            'price_changed': 'sum'
        }).reset_index()

        summary.columns = ['_'.join(col).strip('_') for col in summary.columns.values]

        # Calculate additional metrics
        summary['price_changes_count'] = summary['price_changed_sum']
        summary['promotion_days'] = summary['is_promotion_sum']
        summary['total_sales'] = summary['sales_sum']
        summary['avg_daily_sales'] = summary['sales_mean']

        logger.info(f"Summary created for {len(summary)} products")

        return summary

"""
Feature engineering module for time series forecasting.

This module contains functions for creating time-based and lag features.
"""

import pandas as pd
import numpy as np
from typing import List, Optional


def create_time_features(df: pd.DataFrame, date_column: Optional[str] = None) -> pd.DataFrame:
    """
    Create time-based features from datetime index or column.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with datetime index or column.
    date_column : str, optional
        Name of date column if not using index.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with additional time-based features.
        
    Examples
    --------
    >>> df_with_features = create_time_features(df)
    >>> print(df_with_features.columns)
    Index(['original_cols', 'year', 'month', 'day', 'dayofweek', 'quarter'])
    """
    df_features = df.copy()
    
    # Get datetime series
    if date_column:
        dt_series = pd.to_datetime(df_features[date_column])
    else:
        if not isinstance(df_features.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have DatetimeIndex or provide date_column")
        dt_series = df_features.index
    
    # Create time features
    df_features['year'] = dt_series.year
    df_features['month'] = dt_series.month
    df_features['day'] = dt_series.day
    df_features['dayofweek'] = dt_series.dayofweek
    df_features['quarter'] = dt_series.quarter
    df_features['weekofyear'] = dt_series.isocalendar().week
    df_features['is_weekend'] = dt_series.dayofweek.isin([5, 6]).astype(int)
    df_features['is_month_start'] = dt_series.is_month_start.astype(int)
    df_features['is_month_end'] = dt_series.is_month_end.astype(int)
    
    return df_features


def create_lag_features(
    df: pd.DataFrame,
    target_column: str,
    lags: List[int] = [1, 7, 14, 30]
) -> pd.DataFrame:
    """
    Create lag features for time series forecasting.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with target column.
    target_column : str
        Name of the target column to create lags from.
    lags : List[int], default=[1, 7, 14, 30]
        List of lag periods to create.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with additional lag features.
        
    Examples
    --------
    >>> df_with_lags = create_lag_features(df, 'demand', lags=[1, 7, 30])
    >>> print(df_with_lags.columns)
    Index(['demand', 'demand_lag_1', 'demand_lag_7', 'demand_lag_30'])
    """
    df_lags = df.copy()
    
    if target_column not in df_lags.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame")
    
    for lag in lags:
        df_lags[f'{target_column}_lag_{lag}'] = df_lags[target_column].shift(lag)
    
    return df_lags


def create_rolling_features(
    df: pd.DataFrame,
    target_column: str,
    windows: List[int] = [7, 14, 30]
) -> pd.DataFrame:
    """
    Create rolling window statistics features.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with target column.
    target_column : str
        Name of the target column to calculate rolling statistics.
    windows : List[int], default=[7, 14, 30]
        List of window sizes for rolling calculations.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with additional rolling features.
        
    Examples
    --------
    >>> df_with_rolling = create_rolling_features(df, 'demand', windows=[7, 30])
    """
    df_rolling = df.copy()
    
    if target_column not in df_rolling.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame")
    
    for window in windows:
        df_rolling[f'{target_column}_rolling_mean_{window}'] = (
            df_rolling[target_column].rolling(window=window).mean()
        )
        df_rolling[f'{target_column}_rolling_std_{window}'] = (
            df_rolling[target_column].rolling(window=window).std()
        )
        df_rolling[f'{target_column}_rolling_min_{window}'] = (
            df_rolling[target_column].rolling(window=window).min()
        )
        df_rolling[f'{target_column}_rolling_max_{window}'] = (
            df_rolling[target_column].rolling(window=window).max()
        )
    
    return df_rolling


def create_all_features(
    df: pd.DataFrame,
    target_column: str,
    date_column: Optional[str] = None,
    lags: List[int] = [1, 7, 14, 30],
    windows: List[int] = [7, 14, 30]
) -> pd.DataFrame:
    """
    Create all time series features (time, lag, and rolling features).
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    target_column : str
        Name of the target column.
    date_column : str, optional
        Name of date column if not using index.
    lags : List[int], default=[1, 7, 14, 30]
        List of lag periods.
    windows : List[int], default=[7, 14, 30]
        List of rolling window sizes.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with all engineered features.
        
    Examples
    --------
    >>> df_final = create_all_features(df, 'demand')
    """
    df_features = df.copy()
    
    # Create time features
    df_features = create_time_features(df_features, date_column=date_column)
    
    # Create lag features
    df_features = create_lag_features(df_features, target_column, lags=lags)
    
    # Create rolling features
    df_features = create_rolling_features(df_features, target_column, windows=windows)
    
    return df_features


# M5-Specific feature engineering functions

def create_price_features(df: pd.DataFrame, inplace: bool = False) -> pd.DataFrame:
    """
    Create price-related features for M5 dataset.
    
    Parameters
    ----------
    df : pd.DataFrame
        M5 DataFrame with 'sell_price' column.
    inplace : bool, default=False
        If True, modify DataFrame in place and return None.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with additional price features (or None if inplace=True).
        
    Examples
    --------
    >>> df = create_price_features(df)
    """
    df_price = df if inplace else df.copy()
    
    if 'sell_price' not in df_price.columns:
        return None if inplace else df_price
    
    # Group by item and store for item-store specific features
    group_cols = ['item_id', 'store_id']
    
    # Price change features
    df_price['price_change'] = df_price.groupby(group_cols)['sell_price'].diff()
    df_price['price_change_pct'] = df_price.groupby(group_cols)['sell_price'].pct_change()
    
    # Price momentum (7-day and 28-day)
    df_price['price_momentum_7'] = df_price.groupby(group_cols)['sell_price'].diff(7)
    df_price['price_momentum_28'] = df_price.groupby(group_cols)['sell_price'].diff(28)
    
    # Rolling price statistics
    for window in [7, 14, 28]:
        df_price[f'price_rolling_mean_{window}'] = df_price.groupby(group_cols)['sell_price'].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean()
        )
        df_price[f'price_rolling_std_{window}'] = df_price.groupby(group_cols)['sell_price'].transform(
            lambda x: x.rolling(window=window, min_periods=1).std()
        )
    
    # Price relative to historical average
    df_price['price_vs_avg'] = df_price['sell_price'] / df_price.groupby(group_cols)['sell_price'].transform('mean')
    
    # Price quantile within item-store history
    df_price['price_rank'] = df_price.groupby(group_cols)['sell_price'].rank(pct=True)
    
    return None if inplace else df_price


def encode_calendar_features(df: pd.DataFrame, inplace: bool = False) -> pd.DataFrame:
    """
    Encode calendar event and SNAP features for M5 dataset.
    
    Parameters
    ----------
    df : pd.DataFrame
        M5 DataFrame with calendar event columns.
    inplace : bool, default=False
        If True, modify DataFrame in place and return None.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with encoded calendar features (or None if inplace=True).
        
    Examples
    --------
    >>> df = encode_calendar_features(df)
    """
    df_calendar = df if inplace else df.copy()
    
    # Event indicators
    if 'event_name_1' in df_calendar.columns:
        df_calendar['has_event'] = df_calendar['event_name_1'].notna().astype(int)
        
        # Encode event types
        if 'event_type_1' in df_calendar.columns:
            event_dummies = pd.get_dummies(df_calendar['event_type_1'], prefix='event_type', dummy_na=False)
            df_calendar = pd.concat([df_calendar, event_dummies], axis=1)
    
    # SNAP indicators by state
    snap_cols = ['snap_CA', 'snap_TX', 'snap_WI']
    for col in snap_cols:
        if col in df_calendar.columns:
            df_calendar[col] = df_calendar[col].fillna(0).astype(int)
    
    # Days since last event - optimized vectorized approach
    if 'has_event' in df_calendar.columns:
        # Sort by store_id, item_id, and date (assuming date is in index or needs to be sorted)
        # Calculate cumulative sum in reverse order more efficiently
        df_calendar['days_since_event'] = (
            df_calendar.groupby(['store_id', 'item_id'])['has_event']
            .transform(lambda x: x[::-1].cumsum()[::-1].shift(-1, fill_value=0))
        )
    
    return None if inplace else df_calendar


def create_sales_lag_features(
    df: pd.DataFrame,
    target_col: str = 'sales',
    lags: List[int] = [1, 7, 14, 21, 28],
    inplace: bool = False
) -> pd.DataFrame:
    """
    Create lag features for sales, grouped by product and store.
    
    Parameters
    ----------
    df : pd.DataFrame
        M5 DataFrame with sales data.
    target_col : str, default='sales'
        Name of the sales column.
    lags : List[int], default=[1, 7, 14, 21, 28]
        List of lag periods (in days).
    inplace : bool, default=False
        If True, modify DataFrame in place and return None.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with sales lag features (or None if inplace=True).
        
    Examples
    --------
    >>> df = create_sales_lag_features(df, 'sales', lags=[1, 7, 28])
    """
    df_lags = df if inplace else df.copy()
    
    if target_col not in df_lags.columns:
        return None if inplace else df_lags
    
    # Group by item and store
    group_cols = ['item_id', 'store_id']
    
    for lag in lags:
        df_lags[f'{target_col}_lag_{lag}'] = df_lags.groupby(group_cols)[target_col].shift(lag)
    
    return None if inplace else df_lags


def create_sales_rolling_features(
    df: pd.DataFrame,
    target_col: str = 'sales',
    windows: List[int] = [7, 14, 28, 90],
    inplace: bool = False
) -> pd.DataFrame:
    """
    Create rolling window statistics for sales, grouped by product and store.
    
    Parameters
    ----------
    df : pd.DataFrame
        M5 DataFrame with sales data.
    target_col : str, default='sales'
        Name of the sales column.
    windows : List[int], default=[7, 14, 28, 90]
        List of window sizes (in days).
    inplace : bool, default=False
        If True, modify DataFrame in place and return None.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with sales rolling features (or None if inplace=True).
        
    Examples
    --------
    >>> df = create_sales_rolling_features(df, 'sales', windows=[7, 28])
    """
    df_rolling = df if inplace else df.copy()
    
    if target_col not in df_rolling.columns:
        return None if inplace else df_rolling
    
    # Group by item and store
    group_cols = ['item_id', 'store_id']
    
    for window in windows:
        # Rolling mean
        df_rolling[f'{target_col}_rolling_mean_{window}'] = df_rolling.groupby(group_cols)[target_col].transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
        )
        
        # Rolling standard deviation
        df_rolling[f'{target_col}_rolling_std_{window}'] = df_rolling.groupby(group_cols)[target_col].transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=1).std()
        )
        
        # Rolling min and max
        df_rolling[f'{target_col}_rolling_min_{window}'] = df_rolling.groupby(group_cols)[target_col].transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=1).min()
        )
        
        df_rolling[f'{target_col}_rolling_max_{window}'] = df_rolling.groupby(group_cols)[target_col].transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=1).max()
        )
    
    return None if inplace else df_rolling


def create_hierarchical_features(df: pd.DataFrame, target_col: str = 'sales', inplace: bool = False) -> pd.DataFrame:
    """
    Create hierarchical aggregation features (state, store, category level).
    
    Parameters
    ----------
    df : pd.DataFrame
        M5 DataFrame with hierarchical columns.
    target_col : str, default='sales'
        Name of the sales column to aggregate.
    inplace : bool, default=False
        If True, modify DataFrame in place and return None.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with hierarchical aggregation features (or None if inplace=True).
        
    Examples
    --------
    >>> df = create_hierarchical_features(df, 'sales')
    """
    df_hier = df if inplace else df.copy()
    
    if target_col not in df_hier.columns:
        return df_hier
    
    # State-level aggregations
    if 'state_id' in df_hier.columns and 'date' in df_hier.columns:
        state_sales = df_hier.groupby(['state_id', 'date'])[target_col].transform('sum')
        df_hier['state_sales_total'] = state_sales
    
    # Store-level aggregations
    if 'store_id' in df_hier.columns and 'date' in df_hier.columns:
        store_sales = df_hier.groupby(['store_id', 'date'])[target_col].transform('sum')
        df_hier['store_sales_total'] = store_sales
    
    # Category-level aggregations
    if 'cat_id' in df_hier.columns and 'date' in df_hier.columns:
        cat_sales = df_hier.groupby(['cat_id', 'date'])[target_col].transform('sum')
        df_hier['cat_sales_total'] = cat_sales
    
    # Department-level aggregations
    if 'dept_id' in df_hier.columns and 'date' in df_hier.columns:
        dept_sales = df_hier.groupby(['dept_id', 'date'])[target_col].transform('sum')
        df_hier['dept_sales_total'] = dept_sales
    
    # Item share of store sales
    if 'store_sales_total' in df_hier.columns:
        df_hier['item_store_share'] = df_hier[target_col] / (df_hier['store_sales_total'] + 1e-6)
    
    # Item share of category sales
    if 'cat_sales_total' in df_hier.columns:
        df_hier['item_cat_share'] = df_hier[target_col] / (df_hier['cat_sales_total'] + 1e-6)
    
    return None if inplace else df_hier


def build_m5_features(
    df: pd.DataFrame,
    target_col: str = 'sales',
    include_price: bool = True,
    include_calendar: bool = True,
    include_lags: bool = True,
    include_rolling: bool = True,
    include_hierarchical: bool = True,
    lags: List[int] = [1, 7, 14, 21, 28],
    windows: List[int] = [7, 14, 28, 90],
    inplace: bool = False
) -> pd.DataFrame:
    """
    Complete M5 feature engineering pipeline.
    
    Parameters
    ----------
    df : pd.DataFrame
        M5 DataFrame with merged data.
    target_col : str, default='sales'
        Name of the target column.
    include_price : bool, default=True
        Whether to create price features.
    include_calendar : bool, default=True
        Whether to encode calendar features.
    include_lags : bool, default=True
        Whether to create lag features.
    include_rolling : bool, default=True
        Whether to create rolling features.
    include_hierarchical : bool, default=True
        Whether to create hierarchical features.
    lags : List[int], default=[1, 7, 14, 21, 28]
        Lag periods to use.
    windows : List[int], default=[7, 14, 28, 90]
        Rolling window sizes to use.
    inplace : bool, default=False
        If True, modify DataFrame in place for better performance.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with all M5 features (or None if inplace=True).
        
    Examples
    --------
    >>> df_features = build_m5_features(df, target_col='sales')
    """
    print("Building M5 features...")
    
    # Make a single copy at the beginning if not inplace
    df_features = df if inplace else df.copy()
    
    if include_price:
        print("  Creating price features...")
        create_price_features(df_features, inplace=True)
    
    if include_calendar:
        print("  Encoding calendar features...")
        encode_calendar_features(df_features, inplace=True)
    
    if include_lags:
        print(f"  Creating lag features (lags={lags})...")
        create_sales_lag_features(df_features, target_col, lags, inplace=True)
    
    if include_rolling:
        print(f"  Creating rolling features (windows={windows})...")
        create_sales_rolling_features(df_features, target_col, windows, inplace=True)
    
    if include_hierarchical:
        print("  Creating hierarchical features...")
        create_hierarchical_features(df_features, target_col, inplace=True)
    
    print(f"Feature engineering complete! Shape: {df_features.shape}")
    
    return None if inplace else df_features
    
    return df_features

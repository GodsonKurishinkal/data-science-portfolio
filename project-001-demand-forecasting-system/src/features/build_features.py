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

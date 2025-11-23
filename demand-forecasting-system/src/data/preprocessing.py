"""
Data preprocessing module for demand forecasting system.

This module contains functions for loading, cleaning, and preprocessing data.
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple, List
from pathlib import Path


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load data from a CSV file.
    
    Parameters
    ----------
    file_path : str
        Path to the CSV file containing the data.
        
    Returns
    -------
    pd.DataFrame
        Loaded data as a pandas DataFrame.
        
    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.
    ValueError
        If the file format is not supported.
        
    Examples
    --------
    >>> df = load_data('data/raw/sales_data.csv')
    >>> print(df.shape)
    (10000, 15)
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    if file_path.suffix == '.csv':
        return pd.read_csv(file_path)
    elif file_path.suffix in ['.xlsx', '.xls']:
        return pd.read_excel(file_path)
    elif file_path.suffix == '.parquet':
        return pd.read_parquet(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")


def clean_data(
    df: pd.DataFrame,
    drop_duplicates: bool = True,
    handle_missing: str = 'drop'
) -> pd.DataFrame:
    """
    Clean the input DataFrame by handling duplicates and missing values.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame to clean.
    drop_duplicates : bool, default=True
        Whether to drop duplicate rows.
    handle_missing : str, default='drop'
        Strategy for handling missing values. Options: 'drop', 'fill_mean', 'fill_median'.
        
    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame.
        
    Examples
    --------
    >>> df_clean = clean_data(df, drop_duplicates=True, handle_missing='fill_mean')
    """
    df_clean = df.copy()
    
    # Handle duplicates
    if drop_duplicates:
        df_clean = df_clean.drop_duplicates()
    
    # Handle missing values
    if handle_missing == 'drop':
        df_clean = df_clean.dropna()
    elif handle_missing == 'fill_mean':
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].mean())
    elif handle_missing == 'fill_median':
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].median())
    
    return df_clean


def preprocess_data(
    df: pd.DataFrame,
    date_column: str = 'date',
    target_column: str = 'demand',
    feature_columns: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Preprocess data for modeling by separating features and target.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing all data.
    date_column : str, default='date'
        Name of the date column to parse.
    target_column : str, default='demand'
        Name of the target variable column.
    feature_columns : List[str], optional
        List of feature column names. If None, uses all columns except target.
        
    Returns
    -------
    Tuple[pd.DataFrame, pd.Series]
        Tuple containing (features DataFrame, target Series).
        
    Examples
    --------
    >>> X, y = preprocess_data(df, date_column='date', target_column='sales')
    >>> print(X.shape, y.shape)
    (10000, 10) (10000,)
    """
    df_processed = df.copy()
    
    # Parse date column
    if date_column in df_processed.columns:
        df_processed[date_column] = pd.to_datetime(df_processed[date_column])
        df_processed = df_processed.set_index(date_column)
    
    # Separate features and target
    if target_column not in df_processed.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame")
    
    y = df_processed[target_column]
    
    if feature_columns is None:
        X = df_processed.drop(columns=[target_column])
    else:
        X = df_processed[feature_columns]
    
    return X, y


def load_and_preprocess_data(
    file_path: str,
    date_column: str = 'date',
    target_column: str = 'demand',
    clean: bool = True
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load and preprocess data in a single function call.
    
    Parameters
    ----------
    file_path : str
        Path to the data file.
    date_column : str, default='date'
        Name of the date column.
    target_column : str, default='demand'
        Name of the target column.
    clean : bool, default=True
        Whether to clean the data before preprocessing.
        
    Returns
    -------
    Tuple[pd.DataFrame, pd.Series]
        Tuple containing (features DataFrame, target Series).
        
    Examples
    --------
    >>> X, y = load_and_preprocess_data('data/raw/sales.csv')
    """
    df = load_data(file_path)
    
    if clean:
        df = clean_data(df)
    
    X, y = preprocess_data(df, date_column=date_column, target_column=target_column)
    
    return X, y


# M5-Specific preprocessing functions

def load_m5_data(data_path: str = 'data/raw') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load M5 Competition dataset files.
    
    Parameters
    ----------
    data_path : str, default='data/raw'
        Path to the directory containing M5 data files.
        
    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        Tuple containing (sales, calendar, prices) DataFrames.
        
    Examples
    --------
    >>> sales, calendar, prices = load_m5_data('data/raw')
    """
    data_path = Path(data_path)
    
    # Load datasets
    sales = pd.read_csv(data_path / 'sales_train_validation.csv')
    calendar = pd.read_csv(data_path / 'calendar.csv')
    prices = pd.read_csv(data_path / 'sell_prices.csv')
    
    # Parse calendar dates
    calendar['date'] = pd.to_datetime(calendar['date'])
    
    return sales, calendar, prices


def melt_sales_data(sales: pd.DataFrame) -> pd.DataFrame:
    """
    Melt M5 sales data from wide to long format.
    
    Parameters
    ----------
    sales : pd.DataFrame
        M5 sales DataFrame in wide format (d_1, d_2, ..., d_1913 columns).
        
    Returns
    -------
    pd.DataFrame
        Melted sales data in long format with columns: id, item_id, dept_id, 
        cat_id, store_id, state_id, d, sales.
        
    Examples
    --------
    >>> sales_long = melt_sales_data(sales)
    """
    # Identify columns
    id_cols = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
    sales_cols = [col for col in sales.columns if col.startswith('d_')]
    
    # Melt to long format
    sales_long = sales.melt(
        id_vars=id_cols,
        value_vars=sales_cols,
        var_name='d',
        value_name='sales'
    )
    
    return sales_long


def merge_m5_data(
    sales_long: pd.DataFrame,
    calendar: pd.DataFrame,
    prices: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge M5 sales, calendar, and price data into a single DataFrame.
    
    Parameters
    ----------
    sales_long : pd.DataFrame
        M5 sales data in long format.
    calendar : pd.DataFrame
        M5 calendar data with date, event, and SNAP information.
    prices : pd.DataFrame
        M5 price data with weekly prices per store-item.
        
    Returns
    -------
    pd.DataFrame
        Merged DataFrame with all M5 data.
        
    Examples
    --------
    >>> df_merged = merge_m5_data(sales_long, calendar, prices)
    """
    # Merge with calendar
    df = sales_long.merge(calendar, on='d', how='left')
    
    # Merge with prices
    df = df.merge(
        prices,
        on=['store_id', 'item_id', 'wm_yr_wk'],
        how='left'
    )
    
    return df


def create_datetime_features(df: pd.DataFrame, date_col: str = 'date') -> pd.DataFrame:
    """
    Create datetime-based features from a date column.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with a date column.
    date_col : str, default='date'
        Name of the date column.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with additional datetime features.
        
    Examples
    --------
    >>> df = create_datetime_features(df, date_col='date')
    """
    df = df.copy()
    
    if date_col not in df.columns:
        raise ValueError(f"Date column '{date_col}' not found in DataFrame")
    
    # Ensure datetime type
    df[date_col] = pd.to_datetime(df[date_col])
    
    # Extract datetime components
    df['year'] = df[date_col].dt.year
    df['month'] = df[date_col].dt.month
    df['day'] = df[date_col].dt.day
    df['dayofweek'] = df[date_col].dt.dayofweek
    df['dayofyear'] = df[date_col].dt.dayofyear
    df['week'] = df[date_col].dt.isocalendar().week
    df['quarter'] = df[date_col].dt.quarter
    
    # Boolean features
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
    df['is_month_start'] = df[date_col].dt.is_month_start.astype(int)
    df['is_month_end'] = df[date_col].dt.is_month_end.astype(int)
    df['is_quarter_start'] = df[date_col].dt.is_quarter_start.astype(int)
    df['is_quarter_end'] = df[date_col].dt.is_quarter_end.astype(int)
    
    return df


def preprocess_m5_data(
    data_path: str = 'data/raw',
    add_datetime_features: bool = True,
    save_processed: bool = True,
    output_path: str = 'data/processed/m5_merged.parquet'
) -> pd.DataFrame:
    """
    Complete M5 data preprocessing pipeline.
    
    Parameters
    ----------
    data_path : str, default='data/raw'
        Path to raw M5 data files.
    add_datetime_features : bool, default=True
        Whether to add datetime-based features.
    save_processed : bool, default=True
        Whether to save processed data to disk.
    output_path : str, default='data/processed/m5_merged.parquet'
        Path to save processed data.
        
    Returns
    -------
    pd.DataFrame
        Fully preprocessed M5 DataFrame.
        
    Examples
    --------
    >>> df = preprocess_m5_data(data_path='data/raw')
    """
    print("Loading M5 data...")
    sales, calendar, prices = load_m5_data(data_path)
    
    print("Melting sales data...")
    sales_long = melt_sales_data(sales)
    
    print("Merging datasets...")
    df = merge_m5_data(sales_long, calendar, prices)
    
    if add_datetime_features:
        print("Creating datetime features...")
        df = create_datetime_features(df, date_col='date')
    
    if save_processed:
        print(f"Saving processed data to {output_path}...")
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path, index=False)
    
    print(f"Preprocessing complete! Shape: {df.shape}")
    
    return df

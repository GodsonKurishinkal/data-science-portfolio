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

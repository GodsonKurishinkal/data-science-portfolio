"""
Model training module for demand forecasting.

This module contains functions for training various forecasting models.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib


def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    model_type: str = 'random_forest',
    test_size: float = 0.2,
    random_state: int = 42,
    **model_params
) -> Tuple[Any, Dict[str, float]]:
    """
    Train a forecasting model and return the trained model with metrics.
    
    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target variable.
    model_type : str, default='random_forest'
        Type of model to train. Options: 'random_forest', 'xgboost', 'lightgbm'.
    test_size : float, default=0.2
        Proportion of data to use for testing.
    random_state : int, default=42
        Random state for reproducibility.
    **model_params
        Additional parameters to pass to the model.
        
    Returns
    -------
    Tuple[Any, Dict[str, float]]
        Tuple containing (trained model, metrics dictionary).
        
    Examples
    --------
    >>> model, metrics = train_model(X, y, model_type='random_forest', n_estimators=100)
    >>> print(metrics['rmse'])
    45.23
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False, random_state=random_state
    )
    
    # Initialize model based on type
    if model_type == 'random_forest':
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(
            n_estimators=model_params.get('n_estimators', 100),
            max_depth=model_params.get('max_depth', 10),
            random_state=random_state,
            n_jobs=-1
        )
    elif model_type == 'xgboost':
        import xgboost as xgb
        model = xgb.XGBRegressor(
            n_estimators=model_params.get('n_estimators', 100),
            max_depth=model_params.get('max_depth', 6),
            learning_rate=model_params.get('learning_rate', 0.1),
            random_state=random_state
        )
    elif model_type == 'lightgbm':
        import lightgbm as lgb
        model = lgb.LGBMRegressor(
            n_estimators=model_params.get('n_estimators', 100),
            max_depth=model_params.get('max_depth', 6),
            learning_rate=model_params.get('learning_rate', 0.1),
            random_state=random_state
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Train model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics = calculate_metrics(y_test, y_pred)
    
    return model, metrics


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate evaluation metrics for predictions.
    
    Parameters
    ----------
    y_true : np.ndarray
        True labels.
    y_pred : np.ndarray
        Predicted labels.
        
    Returns
    -------
    Dict[str, float]
        Dictionary containing MAE, RMSE, MAPE, and R² score.
        
    Examples
    --------
    >>> metrics = calculate_metrics(y_test, y_pred)
    >>> print(f"RMSE: {metrics['rmse']:.2f}")
    RMSE: 45.23
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    # Calculate MAPE (avoiding division by zero)
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    
    return {
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'r2': r2
    }


def cross_validate_model(
    X: pd.DataFrame,
    y: pd.Series,
    model_type: str = 'random_forest',
    n_splits: int = 5,
    **model_params
) -> Dict[str, float]:
    """
    Perform time series cross-validation.
    
    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target variable.
    model_type : str, default='random_forest'
        Type of model to train.
    n_splits : int, default=5
        Number of cross-validation splits.
    **model_params
        Additional parameters to pass to the model.
        
    Returns
    -------
    Dict[str, float]
        Dictionary containing average metrics across folds.
        
    Examples
    --------
    >>> cv_metrics = cross_validate_model(X, y, model_type='xgboost', n_splits=5)
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    all_metrics = []
    
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Train model on fold
        model, _ = train_model(
            X_train, y_train,
            model_type=model_type,
            test_size=0,  # Already split
            **model_params
        )
        
        # Evaluate on test fold
        y_pred = model.predict(X_test)
        metrics = calculate_metrics(y_test, y_pred)
        all_metrics.append(metrics)
    
    # Average metrics across folds
    avg_metrics = {
        metric: np.mean([m[metric] for m in all_metrics])
        for metric in all_metrics[0].keys()
    }
    
    return avg_metrics


def save_model(model: Any, file_path: str) -> None:
    """
    Save trained model to disk.
    
    Parameters
    ----------
    model : Any
        Trained model to save.
    file_path : str
        Path where to save the model.
        
    Examples
    --------
    >>> save_model(model, 'models/demand_forecast_model.pkl')
    """
    joblib.dump(model, file_path)


def load_model(file_path: str) -> Any:
    """
    Load trained model from disk.
    
    Parameters
    ----------
    file_path : str
        Path to the saved model.
        
    Returns
    -------
    Any
        Loaded model.
        
    Examples
    --------
    >>> model = load_model('models/demand_forecast_model.pkl')
    """
    return joblib.load(file_path)

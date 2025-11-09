"""
Model prediction module for demand forecasting.

This module contains functions for making predictions with trained models.
"""

import pandas as pd
import numpy as np
from typing import Any, Union, Optional
import joblib


def make_prediction(
    model: Any,
    X: pd.DataFrame
) -> np.ndarray:
    """
    Make predictions using a trained model.
    
    Parameters
    ----------
    model : Any
        Trained forecasting model.
    X : pd.DataFrame
        Feature matrix for prediction.
        
    Returns
    -------
    np.ndarray
        Array of predictions.
        
    Examples
    --------
    >>> predictions = make_prediction(model, X_new)
    >>> print(predictions[:5])
    [123.4, 145.6, 132.1, 167.8, 154.3]
    """
    return model.predict(X)


def make_forecast(
    model: Any,
    last_known_data: pd.DataFrame,
    horizon: int = 30,
    freq: str = 'D'
) -> pd.DataFrame:
    """
    Generate future forecasts for a specified horizon.
    
    Parameters
    ----------
    model : Any
        Trained forecasting model.
    last_known_data : pd.DataFrame
        Most recent data with features for bootstrapping forecast.
    horizon : int, default=30
        Number of periods to forecast into the future.
    freq : str, default='D'
        Frequency of forecasts ('D' for daily, 'W' for weekly, etc.).
        
    Returns
    -------
    pd.DataFrame
        DataFrame with forecasted values and dates.
        
    Examples
    --------
    >>> forecast_df = make_forecast(model, last_data, horizon=30)
    >>> print(forecast_df.head())
                  date  forecast
    0       2025-11-10     145.6
    1       2025-11-11     132.1
    """
    # Create future dates
    last_date = last_known_data.index[-1] if isinstance(
        last_known_data.index, pd.DatetimeIndex
    ) else pd.Timestamp.now()
    
    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=horizon,
        freq=freq
    )
    
    # Initialize forecast DataFrame
    forecast_df = pd.DataFrame(index=future_dates)
    
    # Note: This is a simplified example
    # In practice, you'd need to generate features for future dates
    # and handle recursive forecasting for models that use lag features
    
    predictions = []
    for i in range(horizon):
        # Here you would create features for each future time step
        # For now, we'll use the last known features as a placeholder
        X_future = last_known_data.iloc[[-1]].copy()
        
        # Make prediction
        pred = model.predict(X_future)[0]
        predictions.append(pred)
    
    forecast_df['forecast'] = predictions
    
    return forecast_df


def predict_with_confidence_interval(
    model: Any,
    X: pd.DataFrame,
    confidence_level: float = 0.95
) -> pd.DataFrame:
    """
    Make predictions with confidence intervals (for ensemble models).
    
    Parameters
    ----------
    model : Any
        Trained ensemble model (e.g., RandomForest).
    X : pd.DataFrame
        Feature matrix for prediction.
    confidence_level : float, default=0.95
        Confidence level for intervals (0-1).
        
    Returns
    -------
    pd.DataFrame
        DataFrame with predictions and confidence intervals.
        
    Examples
    --------
    >>> pred_df = predict_with_confidence_interval(model, X_test)
    >>> print(pred_df.head())
         prediction  lower_bound  upper_bound
    0         145.6        130.2        161.0
    """
    # Get predictions from all trees (for RandomForest)
    if hasattr(model, 'estimators_'):
        # Get predictions from each tree
        tree_predictions = np.array([
            tree.predict(X) for tree in model.estimators_
        ])
        
        # Calculate mean and confidence intervals
        predictions = np.mean(tree_predictions, axis=0)
        std = np.std(tree_predictions, axis=0)
        
        # Calculate confidence interval
        alpha = 1 - confidence_level
        z_score = 1.96  # for 95% confidence
        margin = z_score * std
        
        result_df = pd.DataFrame({
            'prediction': predictions,
            'lower_bound': predictions - margin,
            'upper_bound': predictions + margin
        })
    else:
        # For models without ensemble, just return predictions
        predictions = model.predict(X)
        result_df = pd.DataFrame({
            'prediction': predictions,
            'lower_bound': predictions,
            'upper_bound': predictions
        })
    
    return result_df


def load_and_predict(
    model_path: str,
    X: pd.DataFrame
) -> np.ndarray:
    """
    Load a saved model and make predictions.
    
    Parameters
    ----------
    model_path : str
        Path to the saved model file.
    X : pd.DataFrame
        Feature matrix for prediction.
        
    Returns
    -------
    np.ndarray
        Array of predictions.
        
    Examples
    --------
    >>> predictions = load_and_predict('models/best_model.pkl', X_new)
    """
    model = joblib.load(model_path)
    return make_prediction(model, X)

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


# Evaluation Metrics

def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Root Mean Squared Error.

    Parameters
    ----------
    y_true : np.ndarray
        True values.
    y_pred : np.ndarray
        Predicted values.

    Returns
    -------
    float
        RMSE value.
    """
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Absolute Error.

    Parameters
    ----------
    y_true : np.ndarray
        True values.
    y_pred : np.ndarray
        Predicted values.

    Returns
    -------
    float
        MAE value.
    """
    return np.mean(np.abs(y_true - y_pred))


def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Absolute Percentage Error.

    Parameters
    ----------
    y_true : np.ndarray
        True values.
    y_pred : np.ndarray
        Predicted values.

    Returns
    -------
    float
        MAPE value (percentage).
    """
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def calculate_smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Symmetric Mean Absolute Percentage Error.

    Parameters
    ----------
    y_true : np.ndarray
        True values.
    y_pred : np.ndarray
        Predicted values.

    Returns
    -------
    float
        SMAPE value (percentage).
    """
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    mask = denominator != 0
    return np.mean(np.abs(y_true[mask] - y_pred[mask]) / denominator[mask]) * 100


def calculate_wrmsse(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scale: Optional[np.ndarray] = None
) -> float:
    """
    Calculate Weighted Root Mean Squared Scaled Error (M5 Competition metric).

    Parameters
    ----------
    y_true : np.ndarray
        True values.
    y_pred : np.ndarray
        Predicted values.
    scale : np.ndarray, optional
        Scaling factors for each series. If None, uses overall scale.

    Returns
    -------
    float
        WRMSSE value.

    Notes
    -----
    This is a simplified version. Full M5 WRMSSE requires series weights and
    hierarchical aggregation.
    """
    # Calculate squared errors
    squared_errors = (y_true - y_pred) ** 2

    # Calculate scale (MAE of naive forecast on training data)
    if scale is None:
        # Simplified: use mean absolute difference between consecutive values
        scale = np.mean(np.abs(np.diff(y_true)))
        if scale == 0:
            scale = 1.0

    # Calculate RMSSE
    rmsse = np.sqrt(np.mean(squared_errors)) / scale

    return rmsse


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metrics: Optional[list] = None
) -> dict:
    """
    Evaluate model performance using multiple metrics.

    Parameters
    ----------
    y_true : np.ndarray
        True values.
    y_pred : np.ndarray
        Predicted values.
    metrics : list, optional
        List of metrics to calculate. If None, calculates all.
        Options: 'rmse', 'mae', 'mape', 'smape', 'wrmsse', 'r2'

    Returns
    -------
    dict
        Dictionary containing all calculated metrics.

    Examples
    --------
    >>> metrics = evaluate_model(y_test, y_pred)
    >>> print(f"RMSE: {metrics['rmse']:.4f}")
    """
    if metrics is None:
        metrics = ['rmse', 'mae', 'mape', 'smape', 'wrmsse', 'r2']

    results = {}

    if 'rmse' in metrics:
        results['rmse'] = calculate_rmse(y_true, y_pred)

    if 'mae' in metrics:
        results['mae'] = calculate_mae(y_true, y_pred)

    if 'mape' in metrics:
        results['mape'] = calculate_mape(y_true, y_pred)

    if 'smape' in metrics:
        results['smape'] = calculate_smape(y_true, y_pred)

    if 'wrmsse' in metrics:
        results['wrmsse'] = calculate_wrmsse(y_true, y_pred)

    if 'r2' in metrics:
        from sklearn.metrics import r2_score
        results['r2'] = r2_score(y_true, y_pred)

    return results


def compare_model_predictions(
    y_true: np.ndarray,
    predictions_dict: dict,
    metrics: Optional[list] = None
) -> pd.DataFrame:
    """
    Compare predictions from multiple models.

    Parameters
    ----------
    y_true : np.ndarray
        True values.
    predictions_dict : dict
        Dictionary with model names as keys and predictions as values.
    metrics : list, optional
        List of metrics to calculate.

    Returns
    -------
    pd.DataFrame
        DataFrame comparing model performances.

    Examples
    --------
    >>> preds = {'rf': rf_pred, 'xgb': xgb_pred, 'lgbm': lgbm_pred}
    >>> comparison = compare_model_predictions(y_test, preds)
    """
    results = []

    for model_name, y_pred in predictions_dict.items():
        model_metrics = evaluate_model(y_true, y_pred, metrics)
        model_metrics['model'] = model_name
        results.append(model_metrics)

    comparison_df = pd.DataFrame(results)

    # Reorder columns to put model name first
    cols = ['model'] + [col for col in comparison_df.columns if col != 'model']
    comparison_df = comparison_df[cols]

    # Sort by RMSE (lower is better)
    if 'rmse' in comparison_df.columns:
        comparison_df = comparison_df.sort_values('rmse')

    return comparison_df


def plot_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    dates: Optional[pd.DatetimeIndex] = None,
    title: str = 'Actual vs Predicted'
) -> None:
    """
    Plot actual vs predicted values.

    Parameters
    ----------
    y_true : np.ndarray
        True values.
    y_pred : np.ndarray
        Predicted values.
    dates : pd.DatetimeIndex, optional
        Date index for x-axis.
    title : str, default='Actual vs Predicted'
        Plot title.

    Examples
    --------
    >>> plot_predictions(y_test, predictions, dates=test_dates)
    """
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

    # Time series plot
    x = dates if dates is not None else np.arange(len(y_true))
    ax1.plot(x, y_true, label='Actual', linewidth=1, alpha=0.8)
    ax1.plot(x, y_pred, label='Predicted', linewidth=1, alpha=0.8)
    ax1.set_title(title, fontsize=14, fontweight='bold')
    ax1.set_xlabel('Time' if dates is not None else 'Index')
    ax1.set_ylabel('Value')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Scatter plot
    ax2.scatter(y_true, y_pred, alpha=0.5, s=10)

    # Add perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')

    ax2.set_title('Actual vs Predicted Scatter', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Actual')
    ax2.set_ylabel('Predicted')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    dates: Optional[pd.DatetimeIndex] = None
) -> None:
    """
    Plot residuals analysis.

    Parameters
    ----------
    y_true : np.ndarray
        True values.
    y_pred : np.ndarray
        Predicted values.
    dates : pd.DatetimeIndex, optional
        Date index for x-axis.

    Examples
    --------
    >>> plot_residuals(y_test, predictions, dates=test_dates)
    """
    import matplotlib.pyplot as plt

    residuals = y_true - y_pred

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Residuals over time
    x = dates if dates is not None else np.arange(len(residuals))
    axes[0, 0].plot(x, residuals, linewidth=0.8, alpha=0.8)
    axes[0, 0].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[0, 0].set_title('Residuals Over Time', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Time' if dates is not None else 'Index')
    axes[0, 0].set_ylabel('Residual')
    axes[0, 0].grid(True, alpha=0.3)

    # Residuals histogram
    axes[0, 1].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    axes[0, 1].axvline(x=0, color='r', linestyle='--', linewidth=2)
    axes[0, 1].set_title('Residuals Distribution', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Residual')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].grid(True, alpha=0.3)

    # Residuals vs predicted
    axes[1, 0].scatter(y_pred, residuals, alpha=0.5, s=10)
    axes[1, 0].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[1, 0].set_title('Residuals vs Predicted', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Predicted')
    axes[1, 0].set_ylabel('Residual')
    axes[1, 0].grid(True, alpha=0.3)

    # Q-Q plot
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title('Q-Q Plot', fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def save_predictions(
    predictions: np.ndarray,
    file_path: str,
    dates: Optional[pd.DatetimeIndex] = None,
    additional_cols: Optional[dict] = None
) -> None:
    """
    Save predictions to a CSV file.

    Parameters
    ----------
    predictions : np.ndarray
        Array of predictions.
    file_path : str
        Path to save the CSV file.
    dates : pd.DatetimeIndex, optional
        Date index to include.
    additional_cols : dict, optional
        Additional columns to include (e.g., {'item_id': item_ids}).

    Examples
    --------
    >>> save_predictions(preds, 'predictions/forecast.csv', dates=future_dates)
    """
    df = pd.DataFrame({'prediction': predictions})

    if dates is not None:
        df.insert(0, 'date', dates)

    if additional_cols:
        for col_name, col_values in additional_cols.items():
            df[col_name] = col_values

    df.to_csv(file_path, index=False)
    print(f"Predictions saved to {file_path}")

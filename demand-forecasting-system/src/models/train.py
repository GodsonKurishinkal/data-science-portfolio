"""
Model training module for demand forecasting.

This module contains functions for training various forecasting models.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional
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


# Baseline Models

class NaiveForecast:
    """
    Naive forecasting model that uses the last observed value.

    Examples
    --------
    >>> model = NaiveForecast()
    >>> model.fit(y_train)
    >>> predictions = model.predict(horizon=7)
    """

    def __init__(self):
        self.last_value = None

    def fit(self, y: pd.Series) -> 'NaiveForecast':
        """Fit the model by storing the last value."""
        self.last_value = y.iloc[-1]
        return self

    def predict(self, horizon: int = 1) -> np.ndarray:
        """Predict by repeating the last value."""
        return np.full(horizon, self.last_value)


class MovingAverageForecast:
    """
    Moving average forecasting model.

    Parameters
    ----------
    window : int, default=7
        Window size for moving average.

    Examples
    --------
    >>> model = MovingAverageForecast(window=7)
    >>> model.fit(y_train)
    >>> predictions = model.predict(horizon=7)
    """

    def __init__(self, window: int = 7):
        self.window = window
        self.last_values = None

    def fit(self, y: pd.Series) -> 'MovingAverageForecast':
        """Fit the model by storing the last window values."""
        self.last_values = y.iloc[-self.window:].values
        return self

    def predict(self, horizon: int = 1) -> np.ndarray:
        """Predict by using the moving average of last window values."""
        ma = np.mean(self.last_values)
        return np.full(horizon, ma)


class SeasonalNaiveForecast:
    """
    Seasonal naive forecasting model (uses same day from last season).

    Parameters
    ----------
    seasonal_period : int, default=7
        Seasonal period (e.g., 7 for weekly seasonality).

    Examples
    --------
    >>> model = SeasonalNaiveForecast(seasonal_period=7)
    >>> model.fit(y_train)
    >>> predictions = model.predict(horizon=7)
    """

    def __init__(self, seasonal_period: int = 7):
        self.seasonal_period = seasonal_period
        self.last_season = None

    def fit(self, y: pd.Series) -> 'SeasonalNaiveForecast':
        """Fit the model by storing the last seasonal period."""
        self.last_season = y.iloc[-self.seasonal_period:].values
        return self

    def predict(self, horizon: int = 1) -> np.ndarray:
        """Predict by repeating the last seasonal pattern."""
        predictions = []
        for i in range(horizon):
            predictions.append(self.last_season[i % self.seasonal_period])
        return np.array(predictions)


def train_baseline_models(
    y_train: pd.Series,
    y_test: pd.Series,
    models: Optional[list] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Train and evaluate baseline forecasting models.

    Parameters
    ----------
    y_train : pd.Series
        Training target values.
    y_test : pd.Series
        Test target values.
    models : list, optional
        List of model names to train. If None, trains all baseline models.

    Returns
    -------
    Dict[str, Dict[str, Any]]
        Dictionary with model names as keys and {'model': model, 'metrics': metrics} as values.

    Examples
    --------
    >>> results = train_baseline_models(y_train, y_test)
    >>> print(results['naive']['metrics'])
    """
    if models is None:
        models = ['naive', 'moving_average_7', 'moving_average_28', 'seasonal_naive']

    results = {}
    horizon = len(y_test)

    for model_name in models:
        if model_name == 'naive':
            model = NaiveForecast()
        elif model_name.startswith('moving_average'):
            window = int(model_name.split('_')[-1])
            model = MovingAverageForecast(window=window)
        elif model_name == 'seasonal_naive':
            model = SeasonalNaiveForecast(seasonal_period=7)
        else:
            continue

        # Fit and predict
        model.fit(y_train)
        predictions = model.predict(horizon=horizon)

        # Calculate metrics
        metrics = calculate_metrics(y_test.values, predictions)

        results[model_name] = {
            'model': model,
            'predictions': predictions,
            'metrics': metrics
        }

    return results


# M5-Specific Training Functions

def prepare_m5_train_data(
    df: pd.DataFrame,
    target_col: str = 'sales',
    drop_cols: Optional[list] = None,
    fillna_value: float = 0.0
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare M5 data for model training.

    Parameters
    ----------
    df : pd.DataFrame
        M5 DataFrame with features.
    target_col : str, default='sales'
        Name of the target column.
    drop_cols : list, optional
        List of columns to drop from features.
    fillna_value : float, default=0.0
        Value to fill NaN values with.

    Returns
    -------
    Tuple[pd.DataFrame, pd.Series]
        Tuple containing (X, y) for training.

    Examples
    --------
    >>> X, y = prepare_m5_train_data(df, target_col='sales')
    """
    df_train = df.copy()

    # Define columns to drop
    if drop_cols is None:
        drop_cols = [
            target_col, 'id', 'd', 'date',
            'event_name_1', 'event_name_2',
            'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'
        ]

    # Ensure target exists
    if target_col not in df_train.columns:
        raise ValueError(f"Target column '{target_col}' not found")

    # Separate features and target
    y = df_train[target_col]

    # Drop specified columns from features
    X = df_train.drop(columns=[col for col in drop_cols if col in df_train.columns])

    # Fill NaN values
    X = X.fillna(fillna_value)

    # Ensure all columns are numeric
    X = X.select_dtypes(include=[np.number])

    return X, y


def train_m5_model(
    df: pd.DataFrame,
    target_col: str = 'sales',
    model_type: str = 'lightgbm',
    test_size: float = 0.2,
    validation_split: bool = True,
    **model_params
) -> Tuple[Any, Dict[str, float], pd.DataFrame]:
    """
    Train a model on M5 data with proper train/validation/test split.

    Parameters
    ----------
    df : pd.DataFrame
        M5 DataFrame with features.
    target_col : str, default='sales'
        Name of the target column.
    model_type : str, default='lightgbm'
        Type of model to train.
    test_size : float, default=0.2
        Proportion of data for testing.
    validation_split : bool, default=True
        Whether to use a validation set.
    **model_params
        Additional parameters for the model.

    Returns
    -------
    Tuple[Any, Dict[str, float], pd.DataFrame]
        Tuple containing (trained model, metrics, feature importance dataframe).

    Examples
    --------
    >>> model, metrics, importance = train_m5_model(df, model_type='lightgbm')
    """
    print(f"Training {model_type} model on M5 data...")

    # Prepare data
    X, y = prepare_m5_train_data(df, target_col=target_col)

    # Time-based split (no shuffle for time series)
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    print(f"Training set: {len(X_train):,} samples")
    print(f"Test set: {len(X_test):,} samples")

    # Initialize model
    if model_type == 'random_forest':
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(
            n_estimators=model_params.get('n_estimators', 100),
            max_depth=model_params.get('max_depth', 10),
            min_samples_split=model_params.get('min_samples_split', 10),
            n_jobs=-1,
            random_state=42,
            verbose=1
        )
        model.fit(X_train, y_train)

    elif model_type == 'xgboost':
        import xgboost as xgb
        model = xgb.XGBRegressor(
            n_estimators=model_params.get('n_estimators', 100),
            max_depth=model_params.get('max_depth', 6),
            learning_rate=model_params.get('learning_rate', 0.1),
            subsample=model_params.get('subsample', 0.8),
            colsample_bytree=model_params.get('colsample_bytree', 0.8),
            random_state=42,
            n_jobs=-1
        )

        if validation_split:
            val_idx = int(len(X_train) * 0.8)
            X_tr, X_val = X_train.iloc[:val_idx], X_train.iloc[val_idx:]
            y_tr, y_val = y_train.iloc[:val_idx], y_train.iloc[val_idx:]

            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=10,
                verbose=False
            )
        else:
            model.fit(X_train, y_train)

    elif model_type == 'lightgbm':
        import lightgbm as lgb
        model = lgb.LGBMRegressor(
            n_estimators=model_params.get('n_estimators', 100),
            max_depth=model_params.get('max_depth', 6),
            learning_rate=model_params.get('learning_rate', 0.1),
            num_leaves=model_params.get('num_leaves', 31),
            subsample=model_params.get('subsample', 0.8),
            colsample_bytree=model_params.get('colsample_bytree', 0.8),
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )

        if validation_split:
            val_idx = int(len(X_train) * 0.8)
            X_tr, X_val = X_train.iloc[:val_idx], X_train.iloc[val_idx:]
            y_tr, y_val = y_train.iloc[:val_idx], y_train.iloc[val_idx:]

            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)]
            )
        else:
            model.fit(X_train, y_train)

    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # Make predictions
    print("Making predictions...")
    y_pred = model.predict(X_test)

    # Calculate metrics
    metrics = calculate_metrics(y_test.values, y_pred)

    print("\nModel Performance:")
    print(f"  MAE:  {metrics['mae']:.4f}")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  MAPE: {metrics['mape']:.2f}%")
    print(f"  R²:   {metrics['r2']:.4f}")

    # Get feature importance
    if hasattr(model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
    else:
        importance_df = pd.DataFrame()

    return model, metrics, importance_df


def compare_models(
    df: pd.DataFrame,
    target_col: str = 'sales',
    models: Optional[list] = None,
    test_size: float = 0.2
) -> pd.DataFrame:
    """
    Train and compare multiple models on M5 data.

    Parameters
    ----------
    df : pd.DataFrame
        M5 DataFrame with features.
    target_col : str, default='sales'
        Name of the target column.
    models : list, optional
        List of model types to compare. If None, uses default set.
    test_size : float, default=0.2
        Proportion of data for testing.

    Returns
    -------
    pd.DataFrame
        DataFrame comparing model performances.

    Examples
    --------
    >>> comparison = compare_models(df, models=['random_forest', 'xgboost', 'lightgbm'])
    """
    if models is None:
        models = ['random_forest', 'xgboost', 'lightgbm']

    results = []

    for model_type in models:
        print(f"\n{'='*60}")
        print(f"Training {model_type.upper()}")
        print(f"{'='*60}")

        try:
            model, metrics, _ = train_m5_model(
                df,
                target_col=target_col,
                model_type=model_type,
                test_size=test_size
            )

            results.append({
                'model': model_type,
                'mae': metrics['mae'],
                'rmse': metrics['rmse'],
                'mape': metrics['mape'],
                'r2': metrics['r2']
            })
        except Exception as e:
            print(f"Error training {model_type}: {e}")

    comparison_df = pd.DataFrame(results).sort_values('rmse')

    print(f"\n{'='*60}")
    print("MODEL COMPARISON")
    print(f"{'='*60}")
    print(comparison_df.to_string(index=False))

    return comparison_df

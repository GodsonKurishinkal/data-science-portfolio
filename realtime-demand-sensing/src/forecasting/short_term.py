"""
Short-Term Forecasting - Real-Time Demand Prediction

Provides short-term forecasts for real-time demand sensing:
- Exponential Weighted Moving Average (EWM)
- Simple forecasters (naive, moving average)
- Prophet-based forecasting (when available)
- Ensemble methods

Author: Godson Kurishinkal
Date: December 2025
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging
import warnings

logger = logging.getLogger(__name__)


@dataclass
class Forecast:
    """Forecast result container."""
    timestamp: datetime
    value: float
    lower_bound: float
    upper_bound: float
    horizon: int  # Hours ahead
    model: str
    confidence: float = 0.95


class BaseForecaster(ABC):
    """Abstract base class for forecasters."""
    
    @abstractmethod
    def fit(self, data: pd.DataFrame) -> 'BaseForecaster':
        """Fit the model to historical data."""
        pass
    
    @abstractmethod
    def predict(self, horizon: int = 24) -> List[Forecast]:
        """Generate forecasts for specified horizon."""
        pass
    
    @abstractmethod
    def update(self, new_value: float, timestamp: datetime) -> None:
        """Update model with new observation."""
        pass


class EWMForecaster(BaseForecaster):
    """
    Exponential Weighted Moving Average Forecaster.
    
    Simple but effective for short-term forecasting:
    - Captures trend through EWM
    - Estimates volatility for prediction intervals
    - Fast to update with new observations
    
    Example:
        >>> forecaster = EWMForecaster(span=24)  # 24-hour window
        >>> forecaster.fit(history_df)
        >>> predictions = forecaster.predict(horizon=12)
    """
    
    def __init__(
        self,
        span: int = 24,
        trend_span: int = 48,
        confidence: float = 0.95
    ):
        """
        Initialize EWM forecaster.
        
        Args:
            span: Span for EWM (in hours)
            trend_span: Span for trend calculation
            confidence: Confidence level for intervals
        """
        self.span = span
        self.trend_span = trend_span
        self.confidence = confidence
        
        self.current_level: Optional[float] = None
        self.current_trend: Optional[float] = None
        self.std_estimate: Optional[float] = None
        self.residuals: List[float] = []
        
        self.alpha = 2.0 / (span + 1)
        self.trend_alpha = 2.0 / (trend_span + 1)
        
        self._history: List[Tuple[datetime, float]] = []
        
        logger.info(
            "Initialized EWMForecaster: span=%d, trend_span=%d",
            span, trend_span
        )
    
    def fit(self, data: pd.DataFrame) -> 'EWMForecaster':
        """
        Fit forecaster to historical data.
        
        Args:
            data: DataFrame with 'timestamp' and 'value' columns
        
        Returns:
            Self for method chaining
        """
        if len(data) < 2:
            raise ValueError("Need at least 2 data points to fit")
        
        df = data.sort_values('timestamp').copy()
        
        # Calculate EWM level and trend
        values = df['value'].values
        
        # Initialize with first values
        self.current_level = values[0]
        self.current_trend = 0.0
        
        fitted_values = []
        
        for i, val in enumerate(values):
            if i == 0:
                fitted_values.append(self.current_level)
                continue
            
            # Double exponential smoothing (Holt's method)
            prev_level = self.current_level
            self.current_level = self.alpha * val + (1 - self.alpha) * (prev_level + self.current_trend)
            self.current_trend = self.trend_alpha * (self.current_level - prev_level) + \
                                 (1 - self.trend_alpha) * self.current_trend
            
            fitted_values.append(self.current_level)
        
        # Calculate residuals for prediction intervals
        fitted = np.array(fitted_values)
        self.residuals = list(values - fitted)
        self.std_estimate = np.std(self.residuals) if len(self.residuals) > 1 else 0.0
        
        # Store recent history
        self._history = list(zip(df['timestamp'], values))[-self.span:]
        
        logger.info(
            "Fitted EWM forecaster: level=%.2f, trend=%.4f, std=%.2f",
            self.current_level, self.current_trend, self.std_estimate
        )
        
        return self
    
    def predict(self, horizon: int = 24) -> List[Forecast]:
        """
        Generate forecasts.
        
        Args:
            horizon: Hours ahead to forecast
        
        Returns:
            List of Forecast objects
        """
        if self.current_level is None:
            raise RuntimeError("Forecaster not fitted. Call fit() first.")
        
        forecasts = []
        base_time = datetime.now()
        
        # Z-score for confidence interval
        from scipy import stats
        z = stats.norm.ppf((1 + self.confidence) / 2)
        
        for h in range(1, horizon + 1):
            # Point forecast
            point = self.current_level + h * self.current_trend
            point = max(0, point)  # Non-negative
            
            # Prediction interval widens with horizon
            interval_width = z * self.std_estimate * np.sqrt(h)
            
            forecast = Forecast(
                timestamp=base_time + timedelta(hours=h),
                value=point,
                lower_bound=max(0, point - interval_width),
                upper_bound=point + interval_width,
                horizon=h,
                model='ewm',
                confidence=self.confidence
            )
            forecasts.append(forecast)
        
        return forecasts
    
    def update(self, new_value: float, timestamp: datetime) -> None:
        """
        Update model with new observation.
        
        Args:
            new_value: New observed value
            timestamp: Timestamp of observation
        """
        if self.current_level is None:
            self.current_level = new_value
            self.current_trend = 0.0
            self.std_estimate = 0.0
            return
        
        # Update with double exponential smoothing
        prev_level = self.current_level
        self.current_level = self.alpha * new_value + (1 - self.alpha) * (prev_level + self.current_trend)
        self.current_trend = self.trend_alpha * (self.current_level - prev_level) + \
                             (1 - self.trend_alpha) * self.current_trend
        
        # Update residuals (keep last span)
        residual = new_value - (prev_level + self.current_trend)
        self.residuals.append(residual)
        self.residuals = self.residuals[-self.span:]
        
        if len(self.residuals) > 1:
            self.std_estimate = np.std(self.residuals)
        
        # Update history
        self._history.append((timestamp, new_value))
        self._history = self._history[-self.span:]


class MovingAverageForecaster(BaseForecaster):
    """
    Simple Moving Average Forecaster.
    
    Uses moving average for level estimation:
    - Very robust to outliers
    - No trend component
    - Good for stationary series
    """
    
    def __init__(
        self,
        window: int = 24,
        confidence: float = 0.95
    ):
        """
        Initialize MA forecaster.
        
        Args:
            window: Window size for moving average
            confidence: Confidence level for intervals
        """
        self.window = window
        self.confidence = confidence
        
        self.values: List[float] = []
        self.std_estimate: float = 0.0
        
        logger.info("Initialized MovingAverageForecaster: window=%d", window)
    
    def fit(self, data: pd.DataFrame) -> 'MovingAverageForecaster':
        """Fit to historical data."""
        df = data.sort_values('timestamp').copy()
        self.values = list(df['value'].values[-self.window:])
        
        if len(self.values) > 1:
            self.std_estimate = np.std(self.values)
        
        return self
    
    def predict(self, horizon: int = 24) -> List[Forecast]:
        """Generate forecasts."""
        if not self.values:
            raise RuntimeError("Forecaster not fitted.")
        
        from scipy import stats
        z = stats.norm.ppf((1 + self.confidence) / 2)
        
        point = np.mean(self.values)
        base_time = datetime.now()
        
        forecasts = []
        for h in range(1, horizon + 1):
            interval = z * self.std_estimate / np.sqrt(min(len(self.values), self.window))
            
            forecast = Forecast(
                timestamp=base_time + timedelta(hours=h),
                value=max(0, point),
                lower_bound=max(0, point - interval),
                upper_bound=point + interval,
                horizon=h,
                model='moving_average',
                confidence=self.confidence
            )
            forecasts.append(forecast)
        
        return forecasts
    
    def update(self, new_value: float, timestamp: datetime) -> None:
        """Update with new observation."""
        self.values.append(new_value)
        self.values = self.values[-self.window:]
        
        if len(self.values) > 1:
            self.std_estimate = np.std(self.values)


class NaiveForecaster(BaseForecaster):
    """
    Naive Forecaster - uses last value.
    
    Useful as baseline comparison:
    - Last value as forecast
    - Historical std for intervals
    """
    
    def __init__(self, confidence: float = 0.95):
        """Initialize naive forecaster."""
        self.confidence = confidence
        self.last_value: Optional[float] = None
        self.std_estimate: float = 0.0
        self.values: List[float] = []
    
    def fit(self, data: pd.DataFrame) -> 'NaiveForecaster':
        """Fit to historical data."""
        df = data.sort_values('timestamp').copy()
        self.values = list(df['value'].values[-24:])  # Keep last 24
        self.last_value = self.values[-1] if self.values else 0.0
        
        if len(self.values) > 1:
            self.std_estimate = np.std(self.values)
        
        return self
    
    def predict(self, horizon: int = 24) -> List[Forecast]:
        """Generate forecasts."""
        if self.last_value is None:
            raise RuntimeError("Forecaster not fitted.")
        
        from scipy import stats
        z = stats.norm.ppf((1 + self.confidence) / 2)
        
        base_time = datetime.now()
        forecasts = []
        
        for h in range(1, horizon + 1):
            interval = z * self.std_estimate * np.sqrt(h)
            
            forecast = Forecast(
                timestamp=base_time + timedelta(hours=h),
                value=max(0, self.last_value),
                lower_bound=max(0, self.last_value - interval),
                upper_bound=self.last_value + interval,
                horizon=h,
                model='naive',
                confidence=self.confidence
            )
            forecasts.append(forecast)
        
        return forecasts
    
    def update(self, new_value: float, timestamp: datetime) -> None:
        """Update with new observation."""
        self.values.append(new_value)
        self.values = self.values[-24:]
        self.last_value = new_value
        
        if len(self.values) > 1:
            self.std_estimate = np.std(self.values)


class ProphetForecaster(BaseForecaster):
    """
    Prophet-based forecaster wrapper.
    
    Uses Facebook Prophet when available:
    - Handles seasonality automatically
    - Robust to missing data
    - Provides uncertainty intervals
    
    Falls back to EWM if Prophet not installed.
    """
    
    def __init__(
        self,
        yearly_seasonality: bool = False,
        weekly_seasonality: bool = True,
        daily_seasonality: bool = True,
        confidence: float = 0.95
    ):
        """
        Initialize Prophet forecaster.
        
        Args:
            yearly_seasonality: Include yearly patterns
            weekly_seasonality: Include weekly patterns  
            daily_seasonality: Include daily patterns
            confidence: Confidence interval width
        """
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.confidence = confidence
        
        self.model: Optional[Any] = None
        self._prophet_available = False
        self._fallback: Optional[EWMForecaster] = None
        self._last_train_data: Optional[pd.DataFrame] = None
        
        # Check if Prophet is available
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                from prophet import Prophet
            self._prophet_available = True
            logger.info("Prophet is available")
        except ImportError:
            logger.warning("Prophet not installed, using EWM fallback")
            self._fallback = EWMForecaster()
    
    def fit(self, data: pd.DataFrame) -> 'ProphetForecaster':
        """
        Fit Prophet model to data.
        
        Args:
            data: DataFrame with 'timestamp' and 'value' columns
        
        Returns:
            Self
        """
        if not self._prophet_available:
            self._fallback.fit(data)
            return self
        
        # Prepare data for Prophet
        df = data[['timestamp', 'value']].copy()
        df.columns = ['ds', 'y']
        df['ds'] = pd.to_datetime(df['ds'])
        
        # Create and fit Prophet model
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from prophet import Prophet
            
            self.model = Prophet(
                yearly_seasonality=self.yearly_seasonality,
                weekly_seasonality=self.weekly_seasonality,
                daily_seasonality=self.daily_seasonality,
                interval_width=self.confidence
            )
            self.model.fit(df)
        
        self._last_train_data = df
        
        logger.info("Fitted Prophet model on %d observations", len(df))
        return self
    
    def predict(self, horizon: int = 24) -> List[Forecast]:
        """Generate forecasts using Prophet."""
        if not self._prophet_available:
            return self._fallback.predict(horizon)
        
        if self.model is None:
            raise RuntimeError("Model not fitted.")
        
        # Create future dataframe
        future = self.model.make_future_dataframe(periods=horizon, freq='h')
        
        # Generate predictions
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            forecast_df = self.model.predict(future)
        
        # Extract recent forecasts
        forecasts = []
        recent = forecast_df.tail(horizon)
        
        for i, row in enumerate(recent.itertuples()):
            forecast = Forecast(
                timestamp=row.ds,
                value=max(0, row.yhat),
                lower_bound=max(0, row.yhat_lower),
                upper_bound=row.yhat_upper,
                horizon=i + 1,
                model='prophet',
                confidence=self.confidence
            )
            forecasts.append(forecast)
        
        return forecasts
    
    def update(self, new_value: float, timestamp: datetime) -> None:
        """
        Update with new observation.
        
        Note: Prophet requires refitting, so we accumulate data
        and refit periodically.
        """
        if not self._prophet_available:
            self._fallback.update(new_value, timestamp)
            return
        
        # Add to training data
        if self._last_train_data is not None:
            new_row = pd.DataFrame({'ds': [timestamp], 'y': [new_value]})
            self._last_train_data = pd.concat([self._last_train_data, new_row], ignore_index=True)
            
            # Keep last 7 days (168 hours) for efficiency
            if len(self._last_train_data) > 168:
                self._last_train_data = self._last_train_data.tail(168)


class EnsembleForecaster:
    """
    Ensemble forecaster combining multiple methods.
    
    Combines forecasts from multiple models:
    - Weighted average of predictions
    - Combined prediction intervals
    - Automatic weight optimization
    
    Example:
        >>> ensemble = EnsembleForecaster()
        >>> ensemble.fit(history_df)
        >>> forecasts = ensemble.predict(horizon=24)
    """
    
    def __init__(
        self,
        models: Optional[List[str]] = None,
        weights: Optional[Dict[str, float]] = None,
        confidence: float = 0.95
    ):
        """
        Initialize ensemble forecaster.
        
        Args:
            models: List of model names to include
            weights: Model weights for averaging
            confidence: Confidence level
        """
        self.confidence = confidence
        
        # Default models
        if models is None:
            models = ['ewm', 'moving_average']
        
        # Default equal weights
        if weights is None:
            weights = {m: 1.0 / len(models) for m in models}
        
        self.weights = weights
        self.forecasters: Dict[str, BaseForecaster] = {}
        
        # Initialize forecasters
        for model in models:
            if model == 'ewm':
                self.forecasters[model] = EWMForecaster(confidence=confidence)
            elif model == 'moving_average':
                self.forecasters[model] = MovingAverageForecaster(confidence=confidence)
            elif model == 'naive':
                self.forecasters[model] = NaiveForecaster(confidence=confidence)
            elif model == 'prophet':
                self.forecasters[model] = ProphetForecaster(confidence=confidence)
        
        # Normalize weights
        total = sum(self.weights.get(m, 0) for m in self.forecasters)
        if total > 0:
            self.weights = {m: self.weights.get(m, 0) / total for m in self.forecasters}
        
        logger.info("Initialized EnsembleForecaster with models: %s", list(self.forecasters.keys()))
    
    def fit(self, data: pd.DataFrame) -> 'EnsembleForecaster':
        """
        Fit all models to data.
        
        Args:
            data: DataFrame with 'timestamp' and 'value' columns
        
        Returns:
            Self
        """
        for name, forecaster in self.forecasters.items():
            try:
                forecaster.fit(data)
            except Exception as e:
                logger.warning("Failed to fit %s: %s", name, str(e))
        
        return self
    
    def predict(self, horizon: int = 24) -> List[Forecast]:
        """
        Generate ensemble forecasts.
        
        Args:
            horizon: Hours to forecast
        
        Returns:
            List of combined forecasts
        """
        # Get predictions from each model
        all_predictions: Dict[str, List[Forecast]] = {}
        
        for name, forecaster in self.forecasters.items():
            try:
                preds = forecaster.predict(horizon)
                all_predictions[name] = preds
            except Exception as e:
                logger.warning("Prediction failed for %s: %s", name, str(e))
        
        if not all_predictions:
            raise RuntimeError("No models could generate predictions")
        
        # Combine predictions
        ensemble_forecasts = []
        
        for h in range(horizon):
            values = []
            lower_bounds = []
            upper_bounds = []
            total_weight = 0
            
            for name, preds in all_predictions.items():
                if h < len(preds):
                    weight = self.weights.get(name, 0)
                    values.append(preds[h].value * weight)
                    lower_bounds.append(preds[h].lower_bound * weight)
                    upper_bounds.append(preds[h].upper_bound * weight)
                    total_weight += weight
            
            if total_weight > 0:
                ensemble_value = sum(values) / total_weight
                ensemble_lower = sum(lower_bounds) / total_weight
                ensemble_upper = sum(upper_bounds) / total_weight
            else:
                continue
            
            # Get timestamp from first available model
            first_preds = list(all_predictions.values())[0]
            
            forecast = Forecast(
                timestamp=first_preds[h].timestamp if h < len(first_preds) else datetime.now() + timedelta(hours=h+1),
                value=max(0, ensemble_value),
                lower_bound=max(0, ensemble_lower),
                upper_bound=ensemble_upper,
                horizon=h + 1,
                model='ensemble',
                confidence=self.confidence
            )
            ensemble_forecasts.append(forecast)
        
        return ensemble_forecasts
    
    def update(self, new_value: float, timestamp: datetime) -> None:
        """Update all models with new observation."""
        for name, forecaster in self.forecasters.items():
            try:
                forecaster.update(new_value, timestamp)
            except Exception as e:
                logger.warning("Update failed for %s: %s", name, str(e))


class ShortTermForecaster:
    """
    High-level short-term forecasting interface.
    
    Provides unified interface for real-time forecasting:
    - Automatic model selection
    - Forecast evaluation
    - Rolling forecast updates
    
    Example:
        >>> forecaster = ShortTermForecaster()
        >>> forecaster.fit(history_df)
        >>> predictions = forecaster.forecast(horizon=24)
        >>> forecaster.update(new_sales, timestamp)
    """
    
    def __init__(
        self,
        model: str = 'ensemble',
        confidence: float = 0.95
    ):
        """
        Initialize forecaster.
        
        Args:
            model: Model type ('ewm', 'moving_average', 'prophet', 'ensemble')
            confidence: Confidence level for intervals
        """
        self.model_type = model
        self.confidence = confidence
        
        if model == 'ensemble':
            self._forecaster = EnsembleForecaster(confidence=confidence)
        elif model == 'ewm':
            self._forecaster = EWMForecaster(confidence=confidence)
        elif model == 'moving_average':
            self._forecaster = MovingAverageForecaster(confidence=confidence)
        elif model == 'prophet':
            self._forecaster = ProphetForecaster(confidence=confidence)
        else:
            self._forecaster = EnsembleForecaster(confidence=confidence)
        
        self.forecast_history: List[Dict] = []
        self.actual_history: List[Tuple[datetime, float]] = []
        
        logger.info("Initialized ShortTermForecaster with model: %s", model)
    
    def fit(self, data: pd.DataFrame) -> 'ShortTermForecaster':
        """
        Fit forecaster to historical data.
        
        Args:
            data: DataFrame with 'timestamp' and 'value' columns
        
        Returns:
            Self
        """
        self._forecaster.fit(data)
        return self
    
    def forecast(self, horizon: int = 24) -> pd.DataFrame:
        """
        Generate forecasts.
        
        Args:
            horizon: Hours ahead to forecast
        
        Returns:
            DataFrame with forecasts
        """
        forecasts = self._forecaster.predict(horizon)
        
        # Convert to DataFrame
        df = pd.DataFrame([{
            'timestamp': f.timestamp,
            'forecast': f.value,
            'lower_bound': f.lower_bound,
            'upper_bound': f.upper_bound,
            'horizon': f.horizon,
            'model': f.model
        } for f in forecasts])
        
        return df
    
    def update(self, value: float, timestamp: Optional[datetime] = None) -> None:
        """
        Update model with new observation.
        
        Args:
            value: Observed value
            timestamp: Observation timestamp
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        self._forecaster.update(value, timestamp)
        self.actual_history.append((timestamp, value))
        
        # Keep last 168 hours (1 week)
        self.actual_history = self.actual_history[-168:]
    
    def evaluate(self, actuals: pd.DataFrame) -> Dict[str, float]:
        """
        Evaluate forecast accuracy.
        
        Args:
            actuals: DataFrame with 'timestamp' and 'value' columns
        
        Returns:
            Dictionary of evaluation metrics
        """
        if not self.forecast_history:
            return {}
        
        # Match forecasts to actuals
        matched_pairs = []
        
        for forecast_record in self.forecast_history:
            forecast_time = forecast_record['timestamp']
            forecast_val = forecast_record['forecast']
            
            # Find matching actual
            mask = actuals['timestamp'] == forecast_time
            if mask.any():
                actual_val = actuals.loc[mask, 'value'].values[0]
                matched_pairs.append((forecast_val, actual_val))
        
        if not matched_pairs:
            return {}
        
        forecasts = np.array([p[0] for p in matched_pairs])
        actuals = np.array([p[1] for p in matched_pairs])
        
        # Calculate metrics
        errors = actuals - forecasts
        abs_errors = np.abs(errors)
        pct_errors = np.abs(errors / np.where(actuals != 0, actuals, 1)) * 100
        
        return {
            'mae': np.mean(abs_errors),
            'rmse': np.sqrt(np.mean(errors ** 2)),
            'mape': np.mean(pct_errors),
            'bias': np.mean(errors),
            'n_samples': len(matched_pairs)
        }


def forecasts_to_dataframe(forecasts: List[Forecast]) -> pd.DataFrame:
    """
    Convert list of Forecast objects to DataFrame.
    
    Args:
        forecasts: List of Forecast objects
    
    Returns:
        DataFrame with forecast data
    """
    return pd.DataFrame([{
        'timestamp': f.timestamp,
        'value': f.value,
        'lower_bound': f.lower_bound,
        'upper_bound': f.upper_bound,
        'horizon': f.horizon,
        'model': f.model,
        'confidence': f.confidence
    } for f in forecasts])


if __name__ == "__main__":
    # Quick test
    logging.basicConfig(level=logging.INFO)
    
    # Generate sample data
    np.random.seed(42)
    n_hours = 168  # 1 week
    
    dates = pd.date_range(
        end=datetime.now(),
        periods=n_hours,
        freq='h'
    )
    
    # Create seasonal pattern with noise
    hour_of_day = dates.hour
    day_of_week = dates.dayofweek
    
    base = 100
    daily_pattern = 20 * np.sin(2 * np.pi * hour_of_day / 24 - np.pi / 2)
    weekly_pattern = 10 * np.where(day_of_week >= 5, 1.2, 1.0)
    noise = np.random.normal(0, 5, n_hours)
    
    values = base + daily_pattern + weekly_pattern + noise
    values = np.maximum(values, 0)
    
    data = pd.DataFrame({
        'timestamp': dates,
        'value': values
    })
    
    print("=" * 60)
    print("Short-Term Forecasting Test")
    print("=" * 60)
    print(f"History: {len(data)} hours")
    print(f"Mean: {data['value'].mean():.2f}, Std: {data['value'].std():.2f}")
    
    # Test EWM forecaster
    print("\n1. EWM Forecaster:")
    ewm = EWMForecaster(span=24)
    ewm.fit(data)
    ewm_forecasts = ewm.predict(horizon=12)
    print(f"   Next 12h forecasts: {[f'{f.value:.1f}' for f in ewm_forecasts[:5]]}...")
    
    # Test ensemble forecaster
    print("\n2. Ensemble Forecaster:")
    ensemble = EnsembleForecaster(models=['ewm', 'moving_average'])
    ensemble.fit(data)
    ens_forecasts = ensemble.predict(horizon=12)
    print(f"   Next 12h forecasts: {[f'{f.value:.1f}' for f in ens_forecasts[:5]]}...")
    
    # Test high-level interface
    print("\n3. ShortTermForecaster:")
    forecaster = ShortTermForecaster(model='ensemble')
    forecaster.fit(data)
    forecast_df = forecaster.forecast(horizon=24)
    print(forecast_df[['timestamp', 'forecast', 'lower_bound', 'upper_bound']].head())
    
    # Update with new value
    forecaster.update(120.0)
    print("\nUpdated with new value: 120.0")
    
    print("\nForecasting test complete!")

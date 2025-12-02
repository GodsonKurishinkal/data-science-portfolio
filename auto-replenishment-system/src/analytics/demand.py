"""Demand analysis and forecasting utilities.

This module provides demand analytics including weighted averages,
outlier detection, and demand statistics calculation.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
from scipy import stats

from ..interfaces.base import IAnalyzer

logger = logging.getLogger(__name__)


class DemandAnalyzer(IAnalyzer):
    """Comprehensive demand analyzer for replenishment planning.
    
    Calculates:
    - Daily/weekly/monthly demand rates
    - Demand variability (std, CV)
    - Demand trends
    - Outlier-adjusted statistics
    
    Examples:
        >>> analyzer = DemandAnalyzer()
        >>> df_analyzed = analyzer.analyze(demand_df)
        >>> stats = analyzer.get_statistics()
    """
    
    def __init__(
        self,
        item_column: str = "item_id",
        date_column: str = "date",
        quantity_column: str = "quantity",
        location_column: Optional[str] = None,
        aggregation_period: str = "D",
        outlier_method: str = "iqr",
        outlier_threshold: float = 1.5,
    ):
        """Initialize demand analyzer.
        
        Args:
            item_column: Column containing item identifiers
            date_column: Column containing dates
            quantity_column: Column containing demand quantity
            location_column: Optional column for location-specific analysis
            aggregation_period: Period for aggregation ('D', 'W', 'M')
            outlier_method: Method for outlier detection ('iqr', 'zscore')
            outlier_threshold: Threshold for outlier detection
        """
        self.item_column = item_column
        self.date_column = date_column
        self.quantity_column = quantity_column
        self.location_column = location_column
        self.aggregation_period = aggregation_period
        self.outlier_method = outlier_method
        self.outlier_threshold = outlier_threshold
        
        self._statistics: Dict[str, Any] = {}
    
    def analyze(self, df: pd.DataFrame) -> pd.DataFrame:
        """Analyze demand data and calculate statistics.
        
        Args:
            df: DataFrame with demand history
            
        Returns:
            DataFrame with demand statistics per item
        """
        if df.empty:
            logger.warning("Empty DataFrame provided for demand analysis")
            return df
        
        df = df.copy()
        
        # Ensure date is datetime
        if not pd.api.types.is_datetime64_any_dtype(df[self.date_column]):
            df[self.date_column] = pd.to_datetime(df[self.date_column])
        
        # Aggregate by period
        group_cols = [self.item_column]
        if self.location_column and self.location_column in df.columns:
            group_cols.append(self.location_column)
        
        # Calculate daily demand
        df["period"] = df[self.date_column].dt.to_period(self.aggregation_period)
        period_demand = (
            df.groupby(group_cols + ["period"])[self.quantity_column]
            .sum()
            .reset_index()
        )
        
        # Calculate demand statistics per item (and location if specified)
        demand_stats = self._calculate_demand_stats(period_demand, group_cols)
        
        # Detect outliers
        demand_stats = self._flag_outliers(demand_stats)
        
        # Store overall statistics
        self._statistics = self._calculate_overall_stats(demand_stats)
        
        return demand_stats
    
    def _calculate_demand_stats(
        self,
        period_demand: pd.DataFrame,
        group_cols: List[str],
    ) -> pd.DataFrame:
        """Calculate demand statistics per item."""
        stats_df = period_demand.groupby(group_cols)[self.quantity_column].agg([
            ("demand_mean", "mean"),
            ("demand_std", "std"),
            ("demand_min", "min"),
            ("demand_max", "max"),
            ("demand_median", "median"),
            ("total_demand", "sum"),
            ("period_count", "count"),
        ]).reset_index()
        
        # Calculate coefficient of variation
        stats_df["demand_cv"] = np.where(
            stats_df["demand_mean"] > 0,
            stats_df["demand_std"] / stats_df["demand_mean"],
            0
        )
        
        # Fill NaN std with 0 (single observation)
        stats_df["demand_std"] = stats_df["demand_std"].fillna(0)
        
        # Calculate demand rate (daily)
        if self.aggregation_period == "D":
            stats_df["daily_demand_rate"] = stats_df["demand_mean"]
        elif self.aggregation_period == "W":
            stats_df["daily_demand_rate"] = stats_df["demand_mean"] / 7
        elif self.aggregation_period == "M":
            stats_df["daily_demand_rate"] = stats_df["demand_mean"] / 30
        
        return stats_df
    
    def _flag_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Flag items with outlier demand patterns."""
        if self.outlier_method == "iqr":
            q1 = df["demand_cv"].quantile(0.25)
            q3 = df["demand_cv"].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - self.outlier_threshold * iqr
            upper = q3 + self.outlier_threshold * iqr
            df["is_outlier"] = (df["demand_cv"] < lower) | (df["demand_cv"] > upper)
        elif self.outlier_method == "zscore":
            z_scores = np.abs(stats.zscore(df["demand_cv"].fillna(0)))
            df["is_outlier"] = z_scores > self.outlier_threshold
        else:
            df["is_outlier"] = False
        
        return df
    
    def _calculate_overall_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate overall demand statistics."""
        return {
            "total_items": len(df),
            "total_demand": df["total_demand"].sum(),
            "avg_daily_demand": df["daily_demand_rate"].mean(),
            "demand_cv_distribution": {
                "mean": df["demand_cv"].mean(),
                "median": df["demand_cv"].median(),
                "std": df["demand_cv"].std(),
            },
            "outlier_count": df["is_outlier"].sum(),
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get overall statistics from last analysis."""
        return self._statistics
    
    def get_item_demand(
        self,
        df: pd.DataFrame,
        item_id: str,
        location_id: Optional[str] = None,
    ) -> pd.DataFrame:
        """Get demand history for a specific item.
        
        Args:
            df: DataFrame with demand data
            item_id: Item identifier
            location_id: Optional location identifier
            
        Returns:
            Filtered demand DataFrame
        """
        mask = df[self.item_column] == item_id
        if location_id and self.location_column in df.columns:
            mask &= df[self.location_column] == location_id
        return df[mask]


class WeightedDemandCalculator:
    """Calculate weighted demand with recency bias.
    
    Gives more weight to recent periods for demand forecasting,
    which is useful for capturing recent trends.
    
    Examples:
        >>> calculator = WeightedDemandCalculator(weights=[0.4, 0.3, 0.2, 0.1])
        >>> weighted_demand = calculator.calculate(demand_series)
    """
    
    def __init__(
        self,
        weights: Optional[List[float]] = None,
        n_periods: int = 4,
    ):
        """Initialize weighted demand calculator.
        
        Args:
            weights: Custom weights (most recent first). Must sum to 1.
            n_periods: Number of periods to use if weights not specified
        """
        if weights:
            if abs(sum(weights) - 1.0) > 0.001:
                raise ValueError("Weights must sum to 1.0")
            self.weights = weights
        else:
            # Generate exponentially decaying weights
            self.weights = self._generate_weights(n_periods)
        
        self.n_periods = len(self.weights)
    
    def _generate_weights(self, n: int) -> List[float]:
        """Generate exponentially decaying weights."""
        raw_weights = [2 ** (n - i - 1) for i in range(n)]
        total = sum(raw_weights)
        return [w / total for w in raw_weights]
    
    def calculate(
        self,
        demand_series: pd.Series,
        min_periods: int = 2,
    ) -> float:
        """Calculate weighted demand.
        
        Args:
            demand_series: Series of demand values (most recent last)
            min_periods: Minimum periods required
            
        Returns:
            Weighted demand value
        """
        # Get the last n_periods values
        recent = demand_series.tail(self.n_periods).values
        
        if len(recent) < min_periods:
            return demand_series.mean() if len(demand_series) > 0 else 0
        
        # Adjust weights if we have fewer periods
        if len(recent) < self.n_periods:
            adjusted_weights = self.weights[-len(recent):]
            total = sum(adjusted_weights)
            adjusted_weights = [w / total for w in adjusted_weights]
        else:
            adjusted_weights = self.weights
        
        # Reverse to match recent (which is oldest to newest)
        adjusted_weights = list(reversed(adjusted_weights))
        
        return sum(d * w for d, w in zip(recent, adjusted_weights))
    
    def calculate_for_items(
        self,
        df: pd.DataFrame,
        item_column: str = "item_id",
        date_column: str = "date",
        quantity_column: str = "quantity",
    ) -> pd.DataFrame:
        """Calculate weighted demand for all items.
        
        Args:
            df: DataFrame with demand history
            item_column: Item identifier column
            date_column: Date column
            quantity_column: Quantity column
            
        Returns:
            DataFrame with weighted demand per item
        """
        results = []
        
        for item_id in df[item_column].unique():
            item_data = df[df[item_column] == item_id].sort_values(date_column)
            weighted_demand = self.calculate(item_data[quantity_column])
            
            results.append({
                item_column: item_id,
                "weighted_demand": weighted_demand,
                "simple_mean": item_data[quantity_column].mean(),
                "periods_used": min(len(item_data), self.n_periods),
            })
        
        return pd.DataFrame(results)


class DemandForecaster:
    """Simple demand forecasting for replenishment planning.
    
    Provides multiple forecasting methods:
    - Simple moving average
    - Weighted moving average
    - Exponential smoothing
    """
    
    def __init__(
        self,
        method: str = "weighted_average",
        periods: int = 4,
        alpha: float = 0.3,
    ):
        """Initialize forecaster.
        
        Args:
            method: Forecasting method ('simple_average', 'weighted_average', 'exponential')
            periods: Number of periods for moving average
            alpha: Smoothing factor for exponential smoothing
        """
        self.method = method
        self.periods = periods
        self.alpha = alpha
        
        if method == "weighted_average":
            self.weighted_calc = WeightedDemandCalculator(n_periods=periods)
    
    def forecast(
        self,
        demand_history: pd.Series,
        horizon: int = 1,
    ) -> float:
        """Generate demand forecast.
        
        Args:
            demand_history: Historical demand (sorted by date)
            horizon: Forecast horizon (periods ahead)
            
        Returns:
            Forecasted demand per period
        """
        if len(demand_history) == 0:
            return 0
        
        if self.method == "simple_average":
            return demand_history.tail(self.periods).mean()
        
        elif self.method == "weighted_average":
            return self.weighted_calc.calculate(demand_history)
        
        elif self.method == "exponential":
            return self._exponential_smoothing(demand_history)
        
        else:
            return demand_history.mean()
    
    def _exponential_smoothing(self, series: pd.Series) -> float:
        """Apply simple exponential smoothing."""
        values = series.values
        result = values[0]
        
        for value in values[1:]:
            result = self.alpha * value + (1 - self.alpha) * result
        
        return result
    
    def forecast_all_items(
        self,
        df: pd.DataFrame,
        item_column: str = "item_id",
        date_column: str = "date",
        quantity_column: str = "quantity",
    ) -> pd.DataFrame:
        """Forecast demand for all items.
        
        Args:
            df: DataFrame with demand history
            item_column: Item identifier column
            date_column: Date column
            quantity_column: Quantity column
            
        Returns:
            DataFrame with forecasts per item
        """
        results = []
        
        for item_id in df[item_column].unique():
            item_data = df[df[item_column] == item_id].sort_values(date_column)
            forecast = self.forecast(item_data[quantity_column])
            
            results.append({
                item_column: item_id,
                "forecast_demand": forecast,
                "forecast_method": self.method,
                "history_periods": len(item_data),
            })
        
        return pd.DataFrame(results)

"""XYZ Classification based on demand variability.

XYZ analysis categorizes items by the predictability of their demand,
using coefficient of variation (CV) as the primary metric.
"""

import logging
from typing import Any, Dict, Optional

import pandas as pd
import numpy as np

from ..interfaces.base import IClassifier

logger = logging.getLogger(__name__)


class XYZClassifier(IClassifier):
    """XYZ Classification based on demand variability.

    Items are classified as:
    - X: Stable demand (low CV < x_threshold)
    - Y: Moderate variability (CV between thresholds)
    - Z: Highly variable demand (high CV >= y_threshold)

    Coefficient of Variation (CV) = Standard Deviation / Mean

    Examples:
        >>> classifier = XYZClassifier(x_threshold=0.5, y_threshold=1.0)
        >>> df_classified = classifier.classify(demand_df)
        >>> print(df_classified['xyz_class'].value_counts())
        X    200
        Y    400
        Z    400
    """

    def __init__(
        self,
        x_threshold: float = 0.5,
        y_threshold: float = 1.0,
        quantity_column: str = "quantity",
        item_column: str = "item_id",
        date_column: str = "date",
        location_column: Optional[str] = None,
        min_periods: int = 4,
        aggregation_period: str = "W",
    ):
        """Initialize XYZ classifier.

        Args:
            x_threshold: CV threshold below which items are class X (default: 0.5)
            y_threshold: CV threshold below which items are class Y (default: 1.0)
            quantity_column: Column containing demand quantity
            item_column: Column containing item identifiers
            date_column: Column containing dates
            location_column: Optional column for location-specific classification
            min_periods: Minimum periods required for classification
            aggregation_period: Time period for aggregation ('D', 'W', 'M')
        """
        if not (0 < x_threshold < y_threshold):
            raise ValueError("Thresholds must satisfy: 0 < x_threshold < y_threshold")

        self.x_threshold = x_threshold
        self.y_threshold = y_threshold
        self.quantity_column = quantity_column
        self.item_column = item_column
        self.date_column = date_column
        self.location_column = location_column
        self.min_periods = min_periods
        self.aggregation_period = aggregation_period

        self._classification_stats: Dict[str, Any] = {}

    @property
    def classification_column(self) -> str:
        """Name of the classification column added."""
        return "xyz_class"

    def classify(self, df: pd.DataFrame) -> pd.DataFrame:
        """Classify items using XYZ analysis.

        Args:
            df: DataFrame with demand history

        Returns:
            DataFrame with 'xyz_class' column added
        """
        if df.empty:
            logger.warning("Empty DataFrame provided for XYZ classification")
            return df

        # Validate required columns
        required_cols = [self.item_column, self.quantity_column, self.date_column]
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # If location column specified, classify per location
        if self.location_column and self.location_column in df.columns:
            return self._classify_by_location(df)

        return self._classify_global(df)

    def _classify_global(self, df: pd.DataFrame) -> pd.DataFrame:
        """Classify items globally."""
        # Ensure date column is datetime
        df = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(df[self.date_column]):
            df[self.date_column] = pd.to_datetime(df[self.date_column])

        # Aggregate demand by period
        df["period"] = df[self.date_column].dt.to_period(self.aggregation_period)

        period_demand = (
            df.groupby([self.item_column, "period"])[self.quantity_column]
            .sum()
            .reset_index()
        )

        # Calculate CV for each item
        item_stats = (
            period_demand.groupby(self.item_column)[self.quantity_column]
            .agg(["mean", "std", "count"])
            .reset_index()
        )

        # Handle items with insufficient data
        item_stats["cv"] = np.where(
            (item_stats["mean"] > 0) & (item_stats["count"] >= self.min_periods),
            item_stats["std"] / item_stats["mean"],
            np.nan
        )

        # Assign XYZ classes
        def assign_xyz(row):
            if pd.isna(row["cv"]) or row["count"] < self.min_periods:
                return "Z"  # Default to Z for insufficient data (conservative)
            elif row["cv"] < self.x_threshold:
                return "X"
            elif row["cv"] < self.y_threshold:
                return "Y"
            else:
                return "Z"

        item_stats["xyz_class"] = item_stats.apply(assign_xyz, axis=1)

        # Store statistics
        self._classification_stats = self._calculate_stats(item_stats)

        # Merge back to original DataFrame
        result = df.merge(
            item_stats[[self.item_column, "xyz_class", "cv"]],
            on=self.item_column,
            how="left"
        )

        # Remove temporary period column
        result = result.drop(columns=["period"])

        logger.info(
            "XYZ classification complete: X=%d, Y=%d, Z=%d items",
            (result[self.item_column].isin(
                item_stats[item_stats["xyz_class"] == "X"][self.item_column]
            )).sum() // max(1, result.groupby(self.item_column).ngroups),
            (result[self.item_column].isin(
                item_stats[item_stats["xyz_class"] == "Y"][self.item_column]
            )).sum() // max(1, result.groupby(self.item_column).ngroups),
            (result[self.item_column].isin(
                item_stats[item_stats["xyz_class"] == "Z"][self.item_column]
            )).sum() // max(1, result.groupby(self.item_column).ngroups),
        )

        return result

    def _classify_by_location(self, df: pd.DataFrame) -> pd.DataFrame:
        """Classify items separately for each location."""
        results = []

        for location in df[self.location_column].unique():
            location_df = df[df[self.location_column] == location].copy()
            classified = self._classify_global(location_df)
            results.append(classified)

        return pd.concat(results, ignore_index=True)

    def _calculate_stats(self, item_stats: pd.DataFrame) -> Dict[str, Any]:
        """Calculate classification statistics."""
        stats = {
            "total_items": len(item_stats),
            "class_counts": item_stats["xyz_class"].value_counts().to_dict(),
            "cv_stats": {
                "mean": item_stats["cv"].mean(),
                "median": item_stats["cv"].median(),
                "min": item_stats["cv"].min(),
                "max": item_stats["cv"].max(),
            },
            "insufficient_data_count": (item_stats["count"] < self.min_periods).sum(),
        }

        return stats

    def get_statistics(self) -> Dict[str, Any]:
        """Get classification statistics from last run.

        Returns:
            Dictionary with classification stats
        """
        return self._classification_stats

    def calculate_cv(
        self,
        df: pd.DataFrame,
        item_id: str,
    ) -> Optional[float]:
        """Calculate CV for a specific item.

        Args:
            df: DataFrame with demand history
            item_id: Item identifier

        Returns:
            Coefficient of variation or None if insufficient data
        """
        item_data = df[df[self.item_column] == item_id]

        if len(item_data) < self.min_periods:
            return None

        mean = item_data[self.quantity_column].mean()
        std = item_data[self.quantity_column].std()

        if mean == 0:
            return None

        return std / mean


class DemandVariabilityAnalyzer:
    """Advanced demand variability analysis."""

    def __init__(self, classifier: XYZClassifier):
        """Initialize analyzer.

        Args:
            classifier: Configured XYZClassifier instance
        """
        self.classifier = classifier

    def analyze_variability_trends(
        self,
        df: pd.DataFrame,
        rolling_periods: int = 4,
    ) -> pd.DataFrame:
        """Analyze how variability changes over time.

        Args:
            df: DataFrame with demand history
            rolling_periods: Number of periods for rolling calculation

        Returns:
            DataFrame with rolling CV values
        """
        df = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(df[self.classifier.date_column]):
            df[self.classifier.date_column] = pd.to_datetime(
                df[self.classifier.date_column]
            )

        df["period"] = df[self.classifier.date_column].dt.to_period(
            self.classifier.aggregation_period
        )

        period_demand = (
            df.groupby([self.classifier.item_column, "period"])
            [self.classifier.quantity_column]
            .sum()
            .reset_index()
        )

        # Calculate rolling CV
        def rolling_cv(x):
            if len(x) < 2:
                return np.nan
            return x.std() / x.mean() if x.mean() > 0 else np.nan

        period_demand["rolling_cv"] = (
            period_demand.groupby(self.classifier.item_column)
            [self.classifier.quantity_column]
            .transform(lambda x: x.rolling(rolling_periods, min_periods=2).apply(rolling_cv))
        )

        return period_demand

    def get_cv_distribution(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get distribution of CV values across items.

        Args:
            df: DataFrame with demand history

        Returns:
            Dictionary with distribution statistics
        """
        # Classify first to get CV values
        classified = self.classifier.classify(df)

        cv_values = classified.groupby(self.classifier.item_column)["cv"].first()

        return {
            "percentiles": cv_values.quantile([0.1, 0.25, 0.5, 0.75, 0.9]).to_dict(),
            "mean": cv_values.mean(),
            "std": cv_values.std(),
            "histogram": np.histogram(cv_values.dropna(), bins=20),
        }

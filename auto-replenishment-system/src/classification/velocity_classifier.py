"""Velocity Classification (FMR - Fast/Medium/Slow Moving).

Velocity classification categorizes items by their sales frequency/velocity,
useful for warehouse slotting and replenishment frequency decisions.
"""

import logging
from typing import Any, Dict, Optional

import pandas as pd

from ..interfaces.base import IClassifier

logger = logging.getLogger(__name__)


class VelocityClassifier(IClassifier):
    """Velocity-based classification (Fast/Medium/Slow movers).

    Items are classified based on:
    - Pick frequency (orders per period)
    - Units sold per period
    - Days with sales

    Classes:
    - F (Fast): High velocity items (top performers)
    - M (Medium): Moderate velocity items
    - R (Rare/Slow): Low velocity items

    Examples:
        >>> classifier = VelocityClassifier(f_threshold=0.6, m_threshold=0.85)
        >>> df_classified = classifier.classify(demand_df)
    """

    def __init__(
        self,
        f_threshold: float = 0.60,
        m_threshold: float = 0.85,
        quantity_column: str = "quantity",
        item_column: str = "item_id",
        date_column: str = "date",
        location_column: Optional[str] = None,
        velocity_metric: str = "units",
    ):
        """Initialize velocity classifier.

        Args:
            f_threshold: Cumulative units threshold for Fast class (default: 0.60)
            m_threshold: Cumulative units threshold for Medium class (default: 0.85)
            quantity_column: Column containing quantity
            item_column: Column containing item identifiers
            date_column: Column containing dates
            location_column: Optional column for location-specific classification
            velocity_metric: Metric to use ('units', 'orders', 'days_with_sales')
        """
        if not (0 < f_threshold < m_threshold <= 1.0):
            raise ValueError("Thresholds must satisfy: 0 < f_threshold < m_threshold <= 1.0")

        self.f_threshold = f_threshold
        self.m_threshold = m_threshold
        self.quantity_column = quantity_column
        self.item_column = item_column
        self.date_column = date_column
        self.location_column = location_column
        self.velocity_metric = velocity_metric

        self._classification_stats: Dict[str, Any] = {}

    @property
    def classification_column(self) -> str:
        """Name of the classification column added."""
        return "velocity_class"

    def classify(self, df: pd.DataFrame) -> pd.DataFrame:
        """Classify items using velocity analysis.

        Args:
            df: DataFrame with demand history

        Returns:
            DataFrame with 'velocity_class' column added
        """
        if df.empty:
            logger.warning("Empty DataFrame provided for velocity classification")
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
        # Calculate velocity metrics per item
        item_metrics = self._calculate_velocity_metrics(df)

        # Determine which metric to use for classification
        metric_column = self._get_metric_column()

        # Sort and calculate cumulative percentage
        item_metrics = item_metrics.sort_values(metric_column, ascending=False)
        total_metric = item_metrics[metric_column].sum()

        if total_metric == 0:
            logger.warning("Total velocity metric is zero, assigning all to R class")
            item_metrics["velocity_class"] = "R"
        else:
            item_metrics["cumulative"] = item_metrics[metric_column].cumsum()
            item_metrics["cumulative_pct"] = item_metrics["cumulative"] / total_metric

            # Assign velocity classes
            item_metrics["velocity_class"] = item_metrics["cumulative_pct"].apply(
                lambda x: "F" if x <= self.f_threshold
                else ("M" if x <= self.m_threshold else "R")
            )

        # Store statistics
        self._classification_stats = self._calculate_stats(item_metrics, metric_column)

        # Merge back to original DataFrame
        result = df.merge(
            item_metrics[[self.item_column, "velocity_class", "velocity_score"]],
            on=self.item_column,
            how="left"
        )

        logger.info(
            "Velocity classification complete: F=%d, M=%d, R=%d items",
            (item_metrics["velocity_class"] == "F").sum(),
            (item_metrics["velocity_class"] == "M").sum(),
            (item_metrics["velocity_class"] == "R").sum(),
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

    def _calculate_velocity_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate velocity metrics for each item."""
        # Ensure date is datetime
        df = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(df[self.date_column]):
            df[self.date_column] = pd.to_datetime(df[self.date_column])

        # Calculate various velocity metrics
        item_metrics = df.groupby(self.item_column).agg({
            self.quantity_column: ["sum", "count"],
            self.date_column: ["min", "max", "nunique"],
        })

        item_metrics.columns = [
            "total_units",
            "order_count",
            "first_sale",
            "last_sale",
            "days_with_sales"
        ]
        item_metrics = item_metrics.reset_index()

        # Calculate days in range
        item_metrics["days_in_range"] = (
            item_metrics["last_sale"] - item_metrics["first_sale"]
        ).dt.days + 1

        # Calculate daily velocity
        item_metrics["units_per_day"] = (
            item_metrics["total_units"] / item_metrics["days_in_range"].clip(lower=1)
        )

        # Calculate sale frequency (% of days with sales)
        item_metrics["sale_frequency"] = (
            item_metrics["days_with_sales"] / item_metrics["days_in_range"].clip(lower=1)
        )

        # Composite velocity score (normalized)
        item_metrics["velocity_score"] = (
            item_metrics["units_per_day"] * item_metrics["sale_frequency"]
        )

        return item_metrics

    def _get_metric_column(self) -> str:
        """Get the column to use for velocity classification."""
        metric_map = {
            "units": "total_units",
            "orders": "order_count",
            "days_with_sales": "days_with_sales",
            "velocity_score": "velocity_score",
        }
        return metric_map.get(self.velocity_metric, "total_units")

    def _calculate_stats(
        self,
        item_metrics: pd.DataFrame,
        metric_column: str,
    ) -> Dict[str, Any]:
        """Calculate classification statistics."""
        stats = {
            "total_items": len(item_metrics),
            "metric_used": self.velocity_metric,
            "class_counts": item_metrics["velocity_class"].value_counts().to_dict(),
            "class_metrics": {},
        }

        for cls in ["F", "M", "R"]:
            class_data = item_metrics[item_metrics["velocity_class"] == cls]
            stats["class_metrics"][cls] = {
                "count": len(class_data),
                "total_units": class_data["total_units"].sum(),
                "avg_velocity_score": class_data["velocity_score"].mean(),
            }

        return stats

    def get_statistics(self) -> Dict[str, Any]:
        """Get classification statistics from last run."""
        return self._classification_stats


class SlottingRecommender:
    """Recommend warehouse slotting based on velocity classification."""

    # Zone recommendations by velocity class
    ZONE_RECOMMENDATIONS = {
        "F": {
            "zone": "golden_zone",
            "description": "Eye-level, easy access for frequent picks",
            "pick_location": "forward_pick",
            "replenishment_priority": 1,
        },
        "M": {
            "zone": "standard_zone",
            "description": "Standard racking, moderate access",
            "pick_location": "forward_pick",
            "replenishment_priority": 2,
        },
        "R": {
            "zone": "bulk_storage",
            "description": "High shelves or back of warehouse",
            "pick_location": "reserve_pick",
            "replenishment_priority": 3,
        },
    }

    def __init__(self, classifier: VelocityClassifier):
        """Initialize recommender.

        Args:
            classifier: Configured VelocityClassifier
        """
        self.classifier = classifier

    def get_slotting_recommendations(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Get slotting recommendations for items.

        Args:
            df: Classified DataFrame with velocity_class

        Returns:
            DataFrame with slotting recommendations
        """
        if "velocity_class" not in df.columns:
            df = self.classifier.classify(df)

        # Get unique items with their velocity class
        items = df.groupby(self.classifier.item_column).agg({
            "velocity_class": "first",
            "velocity_score": "first" if "velocity_score" in df.columns else lambda x: None,
        }).reset_index()

        # Add recommendations
        items["recommended_zone"] = items["velocity_class"].map(
            lambda x: self.ZONE_RECOMMENDATIONS[x]["zone"]
        )
        items["zone_description"] = items["velocity_class"].map(
            lambda x: self.ZONE_RECOMMENDATIONS[x]["description"]
        )
        items["pick_location"] = items["velocity_class"].map(
            lambda x: self.ZONE_RECOMMENDATIONS[x]["pick_location"]
        )
        items["replenishment_priority"] = items["velocity_class"].map(
            lambda x: self.ZONE_RECOMMENDATIONS[x]["replenishment_priority"]
        )

        return items.sort_values("replenishment_priority")

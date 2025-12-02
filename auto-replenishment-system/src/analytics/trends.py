"""Trend detection for demand patterns.

Identifies increasing, decreasing, or stable demand trends
for proactive inventory planning.
"""

import logging
from typing import Any, Dict

import pandas as pd
import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


class TrendDetector:
    """Detect demand trends over time.

    Identifies:
    - Increasing trends (growing demand)
    - Decreasing trends (declining demand)
    - Stable patterns (no significant trend)
    - Trend strength and confidence

    Examples:
        >>> detector = TrendDetector(threshold=0.1, min_periods=4)
        >>> trends = detector.detect_trends(demand_df)
    """

    def __init__(
        self,
        item_column: str = "item_id",
        date_column: str = "date",
        quantity_column: str = "quantity",
        threshold: float = 0.1,
        min_periods: int = 4,
        confidence_level: float = 0.95,
    ):
        """Initialize trend detector.

        Args:
            item_column: Item identifier column
            date_column: Date column
            quantity_column: Quantity column
            threshold: Minimum % change to consider a trend
            min_periods: Minimum periods required for trend detection
            confidence_level: Statistical confidence level
        """
        self.item_column = item_column
        self.date_column = date_column
        self.quantity_column = quantity_column
        self.threshold = threshold
        self.min_periods = min_periods
        self.confidence_level = confidence_level

    def detect_trends(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect trends for all items.

        Args:
            df: DataFrame with demand history

        Returns:
            DataFrame with trend analysis per item
        """
        if df.empty:
            logger.warning("Empty DataFrame provided for trend detection")
            return pd.DataFrame()

        df = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(df[self.date_column]):
            df[self.date_column] = pd.to_datetime(df[self.date_column])

        results = []

        for item_id in df[self.item_column].unique():
            item_data = df[df[self.item_column] == item_id].sort_values(
                self.date_column
            )
            trend_info = self._analyze_item_trend(item_id, item_data)
            results.append(trend_info)

        return pd.DataFrame(results)

    def _analyze_item_trend(
        self,
        item_id: str,
        item_data: pd.DataFrame,
    ) -> Dict[str, Any]:
        """Analyze trend for a single item."""
        result = {
            self.item_column: item_id,
            "trend_direction": "stable",
            "trend_strength": 0.0,
            "trend_pct_change": 0.0,
            "is_significant": False,
            "p_value": 1.0,
            "periods_analyzed": len(item_data),
        }

        if len(item_data) < self.min_periods:
            result["trend_direction"] = "insufficient_data"
            return result

        # Calculate linear regression
        x = np.arange(len(item_data))
        y = item_data[self.quantity_column].values

        if np.std(y) == 0:
            return result  # No variation, stable

        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

        # Calculate percent change
        start_value = intercept
        end_value = intercept + slope * (len(item_data) - 1)

        if start_value > 0:
            pct_change = (end_value - start_value) / start_value
        else:
            pct_change = 0

        # Determine trend direction
        is_significant = p_value < (1 - self.confidence_level)

        if is_significant and abs(pct_change) > self.threshold:
            if pct_change > 0:
                result["trend_direction"] = "increasing"
            else:
                result["trend_direction"] = "decreasing"

        result["trend_strength"] = abs(r_value)  # R-squared would be r_value**2
        result["trend_pct_change"] = pct_change
        result["is_significant"] = is_significant
        result["p_value"] = p_value
        result["slope"] = slope

        return result

    def detect_sudden_changes(
        self,
        df: pd.DataFrame,
        change_threshold: float = 0.5,
        window: int = 2,
    ) -> pd.DataFrame:
        """Detect sudden demand changes (spikes or drops).

        Args:
            df: DataFrame with demand history
            change_threshold: % change threshold to flag
            window: Comparison window size

        Returns:
            DataFrame with sudden change flags
        """
        df = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(df[self.date_column]):
            df[self.date_column] = pd.to_datetime(df[self.date_column])

        results = []

        for item_id in df[self.item_column].unique():
            item_data = df[df[self.item_column] == item_id].sort_values(
                self.date_column
            )

            if len(item_data) < window * 2:
                continue

            # Compare recent window to previous window
            recent = item_data[self.quantity_column].tail(window).mean()
            previous = item_data[self.quantity_column].iloc[-(window*2):-window].mean()

            if previous > 0:
                pct_change = (recent - previous) / previous
            else:
                pct_change = 0

            change_type = "stable"
            if abs(pct_change) > change_threshold:
                change_type = "spike" if pct_change > 0 else "drop"

            results.append({
                self.item_column: item_id,
                "change_type": change_type,
                "pct_change": pct_change,
                "recent_avg": recent,
                "previous_avg": previous,
            })

        return pd.DataFrame(results)

    def get_trending_items(
        self,
        df: pd.DataFrame,
        direction: str = "increasing",
        top_n: int = 10,
    ) -> pd.DataFrame:
        """Get top trending items in a specific direction.

        Args:
            df: DataFrame with demand history
            direction: 'increasing' or 'decreasing'
            top_n: Number of items to return

        Returns:
            Top trending items
        """
        trends = self.detect_trends(df)

        filtered = trends[
            (trends["trend_direction"] == direction) &
            (trends["is_significant"])
        ]

        return filtered.nlargest(top_n, "trend_strength")

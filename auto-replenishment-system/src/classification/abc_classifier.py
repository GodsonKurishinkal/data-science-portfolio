"""ABC Classification based on revenue/volume contribution.

ABC analysis categorizes items by their contribution to total revenue/value,
enabling differentiated inventory management strategies.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

from ..interfaces.base import IClassifier

logger = logging.getLogger(__name__)


class ABCClassifier(IClassifier):
    """ABC Classification based on Pareto principle.

    Items are classified as:
    - A: High value items (top X% of cumulative revenue)
    - B: Medium value items (next Y% of cumulative revenue)
    - C: Low value items (remaining items)

    Examples:
        >>> classifier = ABCClassifier(a_threshold=0.67, b_threshold=0.90)
        >>> df_classified = classifier.classify(demand_df)
        >>> print(df_classified['abc_class'].value_counts())
        A    150
        B    350
        C    500
    """

    def __init__(
        self,
        a_threshold: float = 0.67,
        b_threshold: float = 0.90,
        value_column: str = "revenue",
        item_column: str = "item_id",
        location_column: Optional[str] = None,
    ):
        """Initialize ABC classifier.

        Args:
            a_threshold: Cumulative value threshold for A class (default: 0.67)
            b_threshold: Cumulative value threshold for B class (default: 0.90)
            value_column: Column containing revenue/value data
            item_column: Column containing item identifiers
            location_column: Optional column for location-specific classification
        """
        if not (0 < a_threshold < b_threshold <= 1.0):
            raise ValueError("Thresholds must satisfy: 0 < a_threshold < b_threshold <= 1.0")

        self.a_threshold = a_threshold
        self.b_threshold = b_threshold
        self.value_column = value_column
        self.item_column = item_column
        self.location_column = location_column

        self._classification_stats: Dict[str, Any] = {}

    @property
    def classification_column(self) -> str:
        """Name of the classification column added."""
        return "abc_class"

    def classify(self, df: pd.DataFrame) -> pd.DataFrame:
        """Classify items using ABC analysis.

        Args:
            df: DataFrame with item and value data

        Returns:
            DataFrame with 'abc_class' column added
        """
        if df.empty:
            logger.warning("Empty DataFrame provided for ABC classification")
            return df

        # Validate required columns
        required_cols = [self.item_column, self.value_column]
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # If location column specified, classify per location
        if self.location_column and self.location_column in df.columns:
            return self._classify_by_location(df)

        return self._classify_global(df)

    def _classify_global(self, df: pd.DataFrame) -> pd.DataFrame:
        """Classify items globally (across all locations)."""
        # Aggregate value per item
        item_values = (
            df.groupby(self.item_column)[self.value_column]
            .sum()
            .reset_index()
        )

        # Sort by value descending and calculate cumulative percentage
        item_values = item_values.sort_values(self.value_column, ascending=False)
        total_value = item_values[self.value_column].sum()

        if total_value == 0:
            logger.warning("Total value is zero, assigning all items to class C")
            item_values["abc_class"] = "C"
        else:
            item_values["cumulative_value"] = item_values[self.value_column].cumsum()
            item_values["cumulative_pct"] = item_values["cumulative_value"] / total_value

            # Assign ABC classes
            item_values["abc_class"] = item_values["cumulative_pct"].apply(
                lambda x: "A" if x <= self.a_threshold
                else ("B" if x <= self.b_threshold else "C")
            )

        # Store classification statistics
        self._classification_stats = self._calculate_stats(item_values)

        # Merge back to original DataFrame
        result = df.merge(
            item_values[[self.item_column, "abc_class"]],
            on=self.item_column,
            how="left"
        )

        logger.info(
            "ABC classification complete: A=%d, B=%d, C=%d items",
            (result["abc_class"] == "A").sum(),
            (result["abc_class"] == "B").sum(),
            (result["abc_class"] == "C").sum(),
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

    def _calculate_stats(self, item_values: pd.DataFrame) -> Dict[str, Any]:
        """Calculate classification statistics."""
        stats = {
            "total_items": len(item_values),
            "total_value": item_values[self.value_column].sum(),
            "class_counts": item_values["abc_class"].value_counts().to_dict(),
            "class_value_pct": {},
        }

        for cls in ["A", "B", "C"]:
            class_value = item_values[
                item_values["abc_class"] == cls
            ][self.value_column].sum()
            stats["class_value_pct"][cls] = (
                class_value / stats["total_value"] * 100
                if stats["total_value"] > 0 else 0
            )

        return stats

    def get_statistics(self) -> Dict[str, Any]:
        """Get classification statistics from last run.

        Returns:
            Dictionary with classification stats
        """
        return self._classification_stats

    def get_class_items(
        self,
        df: pd.DataFrame,
        abc_class: str
    ) -> pd.DataFrame:
        """Get items belonging to a specific ABC class.

        Args:
            df: Classified DataFrame
            abc_class: Class to filter ('A', 'B', or 'C')

        Returns:
            Filtered DataFrame
        """
        if "abc_class" not in df.columns:
            raise ValueError("DataFrame not classified. Run classify() first.")

        return df[df["abc_class"] == abc_class]


class ABCAnalyzer:
    """Advanced ABC analysis with additional insights."""

    def __init__(self, classifier: ABCClassifier):
        """Initialize analyzer with a classifier.

        Args:
            classifier: Configured ABCClassifier instance
        """
        self.classifier = classifier

    def analyze_transitions(
        self,
        current_df: pd.DataFrame,
        previous_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Analyze class transitions between periods.

        Args:
            current_df: Current period classified data
            previous_df: Previous period classified data

        Returns:
            DataFrame showing class transitions
        """
        # Ensure both are classified
        if "abc_class" not in current_df.columns:
            current_df = self.classifier.classify(current_df)
        if "abc_class" not in previous_df.columns:
            previous_df = self.classifier.classify(previous_df)

        # Get unique item classifications
        current_classes = current_df.groupby(
            self.classifier.item_column
        )["abc_class"].first().reset_index()
        current_classes.columns = [self.classifier.item_column, "current_class"]

        previous_classes = previous_df.groupby(
            self.classifier.item_column
        )["abc_class"].first().reset_index()
        previous_classes.columns = [self.classifier.item_column, "previous_class"]

        # Merge and identify transitions
        transitions = current_classes.merge(
            previous_classes,
            on=self.classifier.item_column,
            how="outer"
        )

        transitions["transition"] = (
            transitions["previous_class"].fillna("NEW") +
            " → " +
            transitions["current_class"].fillna("REMOVED")
        )

        return transitions

    def get_pareto_curve(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate data for Pareto curve visualization.

        Args:
            df: DataFrame with item and value data

        Returns:
            DataFrame with cumulative percentages for plotting
        """
        item_values = (
            df.groupby(self.classifier.item_column)[self.classifier.value_column]
            .sum()
            .reset_index()
            .sort_values(self.classifier.value_column, ascending=False)
        )

        total_value = item_values[self.classifier.value_column].sum()
        n_items = len(item_values)

        item_values["item_rank"] = range(1, n_items + 1)
        item_values["item_pct"] = item_values["item_rank"] / n_items * 100
        item_values["cumulative_value"] = item_values[self.classifier.value_column].cumsum()
        item_values["value_pct"] = item_values["cumulative_value"] / total_value * 100

        return item_values[["item_rank", "item_pct", "value_pct"]]

"""ABC-XYZ Classification Matrix.

Combines ABC and XYZ classifications to create a 9-cell matrix
for differentiated inventory management strategies.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

from .abc_classifier import ABCClassifier
from .xyz_classifier import XYZClassifier
from .velocity_classifier import VelocityClassifier

logger = logging.getLogger(__name__)


class ABCXYZMatrix:
    """Combined ABC-XYZ classification matrix.

    Creates a 9-cell matrix combining ABC (value) and XYZ (variability)
    classifications for differentiated inventory policies.

    Matrix:
           |    X (Stable)    |   Y (Variable)   |   Z (Erratic)   |
    -------|------------------|------------------|-----------------|
    A (High)|  AX: Predictable | AY: Variable     | AZ: Sporadic    |
           |  High Value      | High Value       | High Value      |
    -------|------------------|------------------|-----------------|
    B (Med) |  BX: Predictable | BY: Variable     | BZ: Sporadic    |
           |  Medium Value    | Medium Value     | Medium Value    |
    -------|------------------|------------------|-----------------|
    C (Low) |  CX: Predictable | CY: Variable     | CZ: Sporadic    |
           |  Low Value       | Low Value        | Low Value       |

    Examples:
        >>> matrix = ABCXYZMatrix()
        >>> df_classified = matrix.classify(demand_df)
        >>> service_level = matrix.get_service_level('A', 'X')
        0.99
    """

    # Default service level matrix
    DEFAULT_SERVICE_LEVELS = {
        "AX": 0.99,  # High value, stable - highest service
        "AY": 0.97,
        "AZ": 0.95,
        "BX": 0.97,
        "BY": 0.95,
        "BZ": 0.92,
        "CX": 0.95,
        "CY": 0.92,
        "CZ": 0.90,  # Low value, erratic - lowest service
    }

    # Default policy recommendations
    DEFAULT_POLICIES = {
        "AX": {
            "policy": "continuous_review",
            "forecast_method": "moving_average",
            "review_frequency": "daily",
            "safety_stock_method": "standard",
            "automation_level": "full",
        },
        "AY": {
            "policy": "periodic_review",
            "forecast_method": "exponential_smoothing",
            "review_frequency": "weekly",
            "safety_stock_method": "dynamic",
            "automation_level": "full",
        },
        "AZ": {
            "policy": "periodic_review",
            "forecast_method": "manual_override",
            "review_frequency": "weekly",
            "safety_stock_method": "dynamic_high",
            "automation_level": "supervised",
        },
        "BX": {
            "policy": "periodic_review",
            "forecast_method": "moving_average",
            "review_frequency": "weekly",
            "safety_stock_method": "standard",
            "automation_level": "full",
        },
        "BY": {
            "policy": "periodic_review",
            "forecast_method": "exponential_smoothing",
            "review_frequency": "weekly",
            "safety_stock_method": "standard",
            "automation_level": "full",
        },
        "BZ": {
            "policy": "periodic_review",
            "forecast_method": "exponential_smoothing",
            "review_frequency": "bi-weekly",
            "safety_stock_method": "dynamic",
            "automation_level": "supervised",
        },
        "CX": {
            "policy": "min_max",
            "forecast_method": "simple_average",
            "review_frequency": "weekly",
            "safety_stock_method": "minimal",
            "automation_level": "full",
        },
        "CY": {
            "policy": "min_max",
            "forecast_method": "simple_average",
            "review_frequency": "bi-weekly",
            "safety_stock_method": "minimal",
            "automation_level": "full",
        },
        "CZ": {
            "policy": "make_to_order",
            "forecast_method": "none",
            "review_frequency": "monthly",
            "safety_stock_method": "minimal",
            "automation_level": "manual",
        },
    }

    def __init__(
        self,
        abc_classifier: Optional[ABCClassifier] = None,
        xyz_classifier: Optional[XYZClassifier] = None,
        service_levels: Optional[Dict[str, float]] = None,
        policies: Optional[Dict[str, Dict[str, Any]]] = None,
    ):
        """Initialize ABC-XYZ matrix.

        Args:
            abc_classifier: Configured ABC classifier (creates default if None)
            xyz_classifier: Configured XYZ classifier (creates default if None)
            service_levels: Custom service level matrix
            policies: Custom policy recommendations
        """
        self.abc_classifier = abc_classifier or ABCClassifier()
        self.xyz_classifier = xyz_classifier or XYZClassifier()
        self.service_levels = service_levels or self.DEFAULT_SERVICE_LEVELS.copy()
        self.policies = policies or self.DEFAULT_POLICIES.copy()

        self._matrix_stats: Dict[str, Any] = {}

    def classify(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply combined ABC-XYZ classification.

        Args:
            df: DataFrame with demand data (must have revenue and quantity)

        Returns:
            DataFrame with abc_class, xyz_class, and matrix_class columns
        """
        # Apply ABC classification
        df_abc = self.abc_classifier.classify(df)

        # Apply XYZ classification
        df_xyz = self.xyz_classifier.classify(df_abc)

        # Combine classifications
        df_xyz["matrix_class"] = df_xyz["abc_class"] + df_xyz["xyz_class"]

        # Add service level and policy recommendations
        df_xyz["target_service_level"] = df_xyz["matrix_class"].map(self.service_levels)
        df_xyz["recommended_policy"] = df_xyz["matrix_class"].map(
            lambda x: self.policies.get(x, {}).get("policy", "periodic_review")
        )

        # Calculate matrix statistics
        self._matrix_stats = self._calculate_matrix_stats(df_xyz)

        logger.info("ABC-XYZ matrix classification complete")
        return df_xyz

    def _calculate_matrix_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate matrix statistics."""
        item_col = self.abc_classifier.item_column

        # Get unique item classifications
        item_classes = df.groupby(item_col).agg({
            "matrix_class": "first",
            "abc_class": "first",
            "xyz_class": "first",
        }).reset_index()

        stats = {
            "total_items": len(item_classes),
            "matrix_distribution": item_classes["matrix_class"].value_counts().to_dict(),
            "abc_distribution": item_classes["abc_class"].value_counts().to_dict(),
            "xyz_distribution": item_classes["xyz_class"].value_counts().to_dict(),
        }

        return stats

    def get_service_level(self, abc_class: str, xyz_class: str) -> float:
        """Get service level for a matrix cell.

        Args:
            abc_class: ABC class (A, B, or C)
            xyz_class: XYZ class (X, Y, or Z)

        Returns:
            Target service level
        """
        key = f"{abc_class}{xyz_class}"
        return self.service_levels.get(key, 0.90)

    def get_policy(self, abc_class: str, xyz_class: str) -> Dict[str, Any]:
        """Get policy recommendation for a matrix cell.

        Args:
            abc_class: ABC class (A, B, or C)
            xyz_class: XYZ class (X, Y, or Z)

        Returns:
            Policy recommendation dictionary
        """
        key = f"{abc_class}{xyz_class}"
        return self.policies.get(key, self.DEFAULT_POLICIES["BY"])

    def get_matrix_summary(self) -> pd.DataFrame:
        """Get summary of the classification matrix.

        Returns:
            DataFrame with matrix cell summaries
        """
        rows = []
        for abc in ["A", "B", "C"]:
            for xyz in ["X", "Y", "Z"]:
                key = f"{abc}{xyz}"
                policy = self.policies.get(key, {})
                rows.append({
                    "cell": key,
                    "abc_class": abc,
                    "xyz_class": xyz,
                    "service_level": self.service_levels.get(key, 0.90),
                    "policy": policy.get("policy", "N/A"),
                    "review_frequency": policy.get("review_frequency", "N/A"),
                    "automation": policy.get("automation_level", "N/A"),
                    "item_count": self._matrix_stats.get(
                        "matrix_distribution", {}
                    ).get(key, 0),
                })

        return pd.DataFrame(rows)

    def get_statistics(self) -> Dict[str, Any]:
        """Get matrix statistics from last classification."""
        return self._matrix_stats


class ClassificationMatrix:
    """Extended classification matrix supporting multiple classification schemes.

    Combines ABC, XYZ, and Velocity (FMR) classifications for comprehensive
    inventory segmentation.
    """

    def __init__(
        self,
        abc_classifier: Optional[ABCClassifier] = None,
        xyz_classifier: Optional[XYZClassifier] = None,
        velocity_classifier: Optional[VelocityClassifier] = None,
    ):
        """Initialize extended classification matrix.

        Args:
            abc_classifier: ABC classifier
            xyz_classifier: XYZ classifier
            velocity_classifier: Velocity classifier
        """
        self.abc_classifier = abc_classifier or ABCClassifier()
        self.xyz_classifier = xyz_classifier or XYZClassifier()
        self.velocity_classifier = velocity_classifier or VelocityClassifier()

        self.abc_xyz_matrix = ABCXYZMatrix(
            abc_classifier=self.abc_classifier,
            xyz_classifier=self.xyz_classifier,
        )

    def classify_full(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all classification schemes.

        Args:
            df: DataFrame with demand data

        Returns:
            DataFrame with all classification columns
        """
        # ABC-XYZ classification
        df_classified = self.abc_xyz_matrix.classify(df)

        # Add velocity classification
        df_classified = self.velocity_classifier.classify(df_classified)

        # Create combined segmentation
        df_classified["full_segment"] = (
            df_classified["matrix_class"] + "-" + df_classified["velocity_class"]
        )

        return df_classified

    def get_segment_recommendations(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Get replenishment recommendations by segment.

        Args:
            df: Classified DataFrame

        Returns:
            DataFrame with segment-level recommendations
        """
        if "full_segment" not in df.columns:
            df = self.classify_full(df)

        # Get unique segments
        item_col = self.abc_classifier.item_column
        segments = df.groupby(item_col).agg({
            "full_segment": "first",
            "matrix_class": "first",
            "velocity_class": "first",
            "target_service_level": "first",
            "recommended_policy": "first",
        }).reset_index()

        # Add velocity-based recommendations
        segments["pick_priority"] = segments["velocity_class"].map({
            "F": 1, "M": 2, "R": 3
        })

        segments["replenishment_frequency"] = segments.apply(
            self._get_replenishment_frequency, axis=1
        )

        return segments.sort_values(["pick_priority", "matrix_class"])

    def _get_replenishment_frequency(self, row: pd.Series) -> str:
        """Determine replenishment frequency based on segment."""
        abc = row["matrix_class"][0]  # First character is ABC class
        velocity = row["velocity_class"]

        # High velocity items need frequent replenishment
        if velocity == "F":
            if abc == "A":
                return "multiple_daily"
            else:
                return "daily"
        elif velocity == "M":
            if abc in ["A", "B"]:
                return "daily"
            else:
                return "every_other_day"
        else:  # R (slow movers)
            return "weekly"

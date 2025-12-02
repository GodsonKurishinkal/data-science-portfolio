"""ABC/XYZ Analysis for inventory classification."""

import pandas as pd
import numpy as np
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class ABCAnalyzer:
    """
    Perform ABC and XYZ analysis for inventory classification.

    ABC Analysis: Classifies items by value/revenue (Pareto principle)
    XYZ Analysis: Classifies items by demand variability
    """

    def __init__(
        self,
        abc_thresholds: Dict[str, float] = None,
        xyz_thresholds: Dict[str, float] = None
    ):
        """
        Initialize ABCAnalyzer.

        Args:
            abc_thresholds: Cumulative revenue thresholds for A, B, C classes
            xyz_thresholds: CV thresholds for X, Y, Z classes
        """
        self.abc_thresholds = abc_thresholds or {'A': 0.80, 'B': 0.95, 'C': 1.00}
        self.xyz_thresholds = xyz_thresholds or {'X': 0.5, 'Y': 1.0, 'Z': 999}

    def perform_abc_analysis(
        self,
        data: pd.DataFrame,
        value_col: str = 'revenue_sum',
        group_cols: list = None
    ) -> pd.DataFrame:
        """
        Perform ABC analysis based on revenue contribution.

        Args:
            data: DataFrame with item-level statistics
            value_col: Column containing value metric
            group_cols: Optional grouping columns (e.g., ['store_id'])

        Returns:
            DataFrame with ABC classification
        """
        logger.info("Performing ABC analysis...")

        result = data.copy()

        if group_cols:
            # Perform ABC analysis within each group
            def classify_group(group):
                return self._classify_abc(group, value_col)

            result = result.groupby(group_cols).apply(classify_group).reset_index(drop=True)
        else:
            # Perform ABC analysis on entire dataset
            result = self._classify_abc(result, value_col)

        # Count items in each class
        abc_counts = result['abc_class'].value_counts().sort_index()
        logger.info(f"ABC distribution: {abc_counts.to_dict()}")

        return result

    def _classify_abc(self, data: pd.DataFrame, value_col: str) -> pd.DataFrame:
        """
        Internal method to classify items into ABC categories.

        Args:
            data: DataFrame to classify
            value_col: Column containing value metric

        Returns:
            DataFrame with ABC class added
        """
        # Sort by value descending
        data_sorted = data.sort_values(value_col, ascending=False).copy()

        # Calculate cumulative percentage
        data_sorted['cumulative_value'] = data_sorted[value_col].cumsum()
        total_value = data_sorted[value_col].sum()
        data_sorted['cumulative_pct'] = data_sorted['cumulative_value'] / total_value

        # Classify
        conditions = [
            data_sorted['cumulative_pct'] <= self.abc_thresholds['A'],
            data_sorted['cumulative_pct'] <= self.abc_thresholds['B'],
            data_sorted['cumulative_pct'] <= self.abc_thresholds['C']
        ]
        choices = ['A', 'B', 'C']

        data_sorted['abc_class'] = np.select(conditions, choices, default='C')

        return data_sorted

    def perform_xyz_analysis(
        self,
        data: pd.DataFrame,
        cv_col: str = 'demand_cv'
    ) -> pd.DataFrame:
        """
        Perform XYZ analysis based on demand variability.

        Args:
            data: DataFrame with demand statistics
            cv_col: Column containing coefficient of variation

        Returns:
            DataFrame with XYZ classification
        """
        logger.info("Performing XYZ analysis...")

        result = data.copy()

        # Handle infinite or NaN CV values
        result[cv_col] = result[cv_col].replace([np.inf, -np.inf], np.nan)
        result[cv_col] = result[cv_col].fillna(result[cv_col].median())

        # Classify based on CV thresholds
        conditions = [
            result[cv_col] < self.xyz_thresholds['X'],
            result[cv_col] < self.xyz_thresholds['Y'],
            result[cv_col] >= self.xyz_thresholds['Y']
        ]
        choices = ['X', 'Y', 'Z']

        result['xyz_class'] = np.select(conditions, choices, default='Z')

        # Count items in each class
        xyz_counts = result['xyz_class'].value_counts().sort_index()
        logger.info(f"XYZ distribution: {xyz_counts.to_dict()}")

        return result

    def perform_combined_analysis(
        self,
        data: pd.DataFrame,
        value_col: str = 'revenue_sum',
        cv_col: str = 'demand_cv',
        group_cols: list = None
    ) -> pd.DataFrame:
        """
        Perform both ABC and XYZ analysis and create combined matrix.

        Args:
            data: DataFrame with item statistics
            value_col: Column containing value metric
            cv_col: Column containing coefficient of variation
            group_cols: Optional grouping columns

        Returns:
            DataFrame with ABC, XYZ, and combined classification
        """
        logger.info("Performing combined ABC-XYZ analysis...")

        # Perform ABC analysis
        result = self.perform_abc_analysis(data, value_col, group_cols)

        # Perform XYZ analysis
        result = self.perform_xyz_analysis(result, cv_col)

        # Create combined classification
        result['abc_xyz_class'] = result['abc_class'] + result['xyz_class']

        # Create priority score (1-9, where 1 is highest priority)
        priority_map = {
            'AX': 1, 'AY': 2, 'AZ': 3,
            'BX': 4, 'BY': 5, 'BZ': 6,
            'CX': 7, 'CY': 8, 'CZ': 9
        }
        result['priority_score'] = result['abc_xyz_class'].map(priority_map)

        # Summary statistics
        matrix_counts = result.groupby(['abc_class', 'xyz_class']).size().unstack(fill_value=0)
        logger.info(f"\nABC-XYZ Matrix:\n{matrix_counts}")

        return result

    def get_class_statistics(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate statistics for each ABC-XYZ class.

        Args:
            data: DataFrame with ABC-XYZ classification

        Returns:
            DataFrame with class-level statistics
        """
        stats = data.groupby('abc_xyz_class').agg({
            'revenue_sum': ['sum', 'mean', 'count'],
            'sales_sum': ['sum', 'mean'],
            'demand_cv': 'mean',
            'fill_rate': 'mean'
        }).round(2)

        # Flatten column names
        stats.columns = ['_'.join(col).strip('_') for col in stats.columns.values]
        stats = stats.reset_index()

        # Calculate percentage of total items and revenue
        stats['item_pct'] = (stats['revenue_sum_count'] / stats['revenue_sum_count'].sum() * 100).round(2)
        stats['revenue_pct'] = (stats['revenue_sum_sum'] / stats['revenue_sum_sum'].sum() * 100).round(2)

        return stats

    def recommend_inventory_policy(self, abc_xyz_class: str) -> Dict[str, str]:
        """
        Recommend inventory management policy based on ABC-XYZ class.

        Args:
            abc_xyz_class: Combined ABC-XYZ classification (e.g., 'AX')

        Returns:
            Dictionary with policy recommendations
        """
        policies = {
            'AX': {
                'policy': 'Continuous Review',
                'service_level': '99%',
                'safety_stock': 'High',
                'review_frequency': 'Daily',
                'attention': 'High - Critical items'
            },
            'AY': {
                'policy': 'Continuous Review',
                'service_level': '99%',
                'safety_stock': 'Very High',
                'review_frequency': 'Daily',
                'attention': 'High - Monitor variability'
            },
            'AZ': {
                'policy': 'Make-to-Order/VMI',
                'service_level': '95%',
                'safety_stock': 'Very High',
                'review_frequency': 'Daily',
                'attention': 'Very High - Critical & volatile'
            },
            'BX': {
                'policy': 'Periodic Review',
                'service_level': '95%',
                'safety_stock': 'Medium',
                'review_frequency': 'Weekly',
                'attention': 'Medium - Standard monitoring'
            },
            'BY': {
                'policy': 'Periodic Review',
                'service_level': '95%',
                'safety_stock': 'High',
                'review_frequency': 'Weekly',
                'attention': 'Medium - Watch variability'
            },
            'BZ': {
                'policy': 'Periodic Review with Buffer',
                'service_level': '90%',
                'safety_stock': 'Very High',
                'review_frequency': 'Weekly',
                'attention': 'High - Manage volatility'
            },
            'CX': {
                'policy': 'Periodic Review',
                'service_level': '90%',
                'safety_stock': 'Low',
                'review_frequency': 'Monthly',
                'attention': 'Low - Optimize costs'
            },
            'CY': {
                'policy': 'Min-Max System',
                'service_level': '85%',
                'safety_stock': 'Medium',
                'review_frequency': 'Monthly',
                'attention': 'Low - Cost focus'
            },
            'CZ': {
                'policy': 'Make-to-Order',
                'service_level': '85%',
                'safety_stock': 'Low/None',
                'review_frequency': 'Ad-hoc',
                'attention': 'Low - Consider discontinuation'
            }
        }

        return policies.get(abc_xyz_class, policies['CZ'])

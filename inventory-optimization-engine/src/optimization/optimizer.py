"""Main inventory optimization engine."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging
from scipy.optimize import minimize

from ..inventory.abc_analysis import ABCAnalyzer
from ..inventory.safety_stock import SafetyStockCalculator
from ..inventory.reorder_point import ReorderPointCalculator
from ..inventory.eoq import EOQCalculator
from .cost_calculator import CostCalculator

logger = logging.getLogger(__name__)


class InventoryOptimizer:
    """
    Main inventory optimization engine integrating all components.
    """

    def __init__(self, config: Dict):
        """
        Initialize InventoryOptimizer with configuration.

        Args:
            config: Configuration dictionary
        """
        self.config = config

        # Initialize components
        self.abc_analyzer = ABCAnalyzer(
            abc_thresholds=config.get('inventory', {}).get('abc_thresholds'),
            xyz_thresholds=config.get('inventory', {}).get('xyz_thresholds')
        )

        costs = config.get('inventory', {}).get('costs', {})
        self.cost_calculator = CostCalculator(
            holding_cost_rate=costs.get('holding_cost_rate', 0.25),
            ordering_cost=costs.get('ordering_cost', 100),
            stockout_cost_rate=costs.get('stockout_cost_rate', 2.0)
        )

        self.eoq_calculator = EOQCalculator(
            ordering_cost=costs.get('ordering_cost', 100),
            holding_cost_rate=costs.get('holding_cost_rate', 0.25)
        )

    def optimize_inventory_policy(
        self,
        data: pd.DataFrame,
        objective: str = 'minimize_cost'
    ) -> pd.DataFrame:
        """
        Optimize inventory policy for all items.

        Args:
            data: DataFrame with demand statistics
            objective: Optimization objective

        Returns:
            DataFrame with optimized inventory parameters
        """
        logger.info(f"Optimizing inventory policy with objective: {objective}")

        # Step 1: ABC-XYZ Classification
        classified_data = self.abc_analyzer.perform_combined_analysis(
            data,
            value_col='revenue_sum',
            cv_col='demand_cv'
        )

        # Step 2: Assign service levels based on classification
        service_levels = self.config.get('inventory', {}).get('service_levels', {})
        classified_data['target_service_level'] = classified_data['abc_class'].map({
            'A': service_levels.get('high_value', 0.99),
            'B': service_levels.get('medium_value', 0.95),
            'C': service_levels.get('low_value', 0.90)
        })

        # Step 3: Calculate EOQ
        eoq_data = self.eoq_calculator.calculate_for_dataframe(
            classified_data,
            demand_col='sales_sum',
            price_col='sell_price_mean'
        )

        # Step 4: Calculate safety stock
        lead_time = self.config.get('inventory', {}).get('lead_time', {}).get('default', 7)

        optimized_data = eoq_data.copy()
        optimized_data['safety_stock'] = 0
        optimized_data['reorder_point'] = 0

        for sl in optimized_data['target_service_level'].unique():
            mask = optimized_data['target_service_level'] == sl
            ss_calc = SafetyStockCalculator(service_level=sl, lead_time=lead_time)

            subset = ss_calc.calculate_for_dataframe(
                optimized_data[mask],
                method='basic'
            )
            optimized_data.loc[mask, 'safety_stock'] = subset['safety_stock']

        # Step 5: Calculate reorder points
        rop_calc = ReorderPointCalculator(lead_time=lead_time)
        optimized_data = rop_calc.calculate_for_dataframe(optimized_data)

        # Step 6: Calculate costs
        optimized_data = self._calculate_inventory_costs(optimized_data)

        logger.info("Optimization complete")
        return optimized_data

    def _calculate_inventory_costs(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate total inventory costs for each item."""
        result = data.copy()

        # Average inventory = (EOQ/2) + Safety Stock
        result['avg_inventory_level'] = (result['eoq'] / 2) + result['safety_stock']

        # Calculate individual cost components (already in data from EOQ calculation)
        # Just ensure all costs are present

        return result

    def generate_recommendations(
        self,
        optimized_data: pd.DataFrame,
        top_n: int = 20
    ) -> pd.DataFrame:
        """
        Generate actionable recommendations.

        Args:
            optimized_data: Optimized inventory data
            top_n: Number of top recommendations to generate

        Returns:
            DataFrame with recommendations
        """
        recommendations = []

        # High-priority items (AX, AY, AZ)
        high_priority = optimized_data[
            optimized_data['abc_class'] == 'A'
        ].nlargest(top_n, 'revenue_sum')

        for _, item in high_priority.iterrows():
            policy = self.abc_analyzer.recommend_inventory_policy(
                item['abc_xyz_class']
            )

            recommendations.append({
                'item_id': item['item_id'],
                'store_id': item['store_id'],
                'abc_xyz_class': item['abc_xyz_class'],
                'priority_score': item['priority_score'],
                'recommended_policy': policy['policy'],
                'target_service_level': f"{item['target_service_level']*100:.0f}%",
                'eoq': int(item['eoq']),
                'reorder_point': int(item['reorder_point']),
                'safety_stock': int(item['safety_stock']),
                'review_frequency': policy['review_frequency'],
                'annual_cost': item['total_annual_cost']
            })

        return pd.DataFrame(recommendations)

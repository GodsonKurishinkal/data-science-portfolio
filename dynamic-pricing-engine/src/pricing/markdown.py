"""
Markdown Optimization Module

This module implements markdown strategies for inventory clearance,
optimizing the trade-off between clearance speed and revenue maximization.

Author: Godson Kurishinkal
Date: November 12, 2025
"""

import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MarkdownOptimizer:
    """
    Optimize markdown strategies for inventory clearance.

    This class implements sophisticated markdown optimization algorithms that
    balance the trade-off between:
    - Speed of inventory clearance
    - Revenue maximization
    - Holding cost minimization
    - Salvage value protection

    Key Features:
    - Progressive markdown schedules (15% → 30% → 50%)
    - Elasticity-based clearance simulation
    - Multi-strategy comparison
    - Optimal timing recommendations
    - Profit maximization under clearance constraints

    Methods:
        calculate_optimal_markdown: Determine optimal discount schedule
        simulate_clearance: Simulate inventory trajectory
        compare_strategies: Compare different markdown approaches
        progressive_markdown: Generate standard progressive schedule
        emergency_clearance: Aggressive clearance for urgent situations

    Example:
        >>> optimizer = MarkdownOptimizer(holding_cost_per_day=0.001)
        >>> result = optimizer.calculate_optimal_markdown(
        ...     product_id='PROD_001',
        ...     current_inventory=500,
        ...     days_remaining=30,
        ...     current_price=50.0,
        ...     elasticity=-2.0,
        ...     baseline_demand=10
        ... )
        >>> print(f"Week 1 discount: {result['schedule'][0]['discount_pct']:.1f}%")
    """

    def __init__(
        self,
        holding_cost_per_day: float = 0.001,
        salvage_value_pct: float = 0.30
    ):
        """
        Initialize markdown optimizer.

        Args:
            holding_cost_per_day: Daily holding cost as fraction of price (default: 0.1%)
            salvage_value_pct: Salvage value as percentage of original price (default: 30%)

        Raises:
            ValueError: If costs are negative or exceed 100%
        """
        if holding_cost_per_day < 0 or holding_cost_per_day > 1:
            raise ValueError(f"Holding cost must be between 0 and 1, got {holding_cost_per_day}")

        if salvage_value_pct < 0 or salvage_value_pct > 1:
            raise ValueError(f"Salvage value must be between 0 and 1, got {salvage_value_pct}")

        self.holding_cost_per_day = holding_cost_per_day
        self.salvage_value_pct = salvage_value_pct
        self.clearance_history = []

        logger.info(
            "Initialized MarkdownOptimizer: "
            "holding_cost=%.4f, "
            "salvage_value=%.2f%%",
            holding_cost_per_day,
            salvage_value_pct * 100
        )

    def calculate_optimal_markdown(
        self,
        product_id: str,
        current_inventory: int,
        days_remaining: int,
        current_price: float,
        elasticity: float,
        baseline_demand: float,
        cost_per_unit: Optional[float] = None,
        target_clearance_pct: float = 0.95
    ) -> Dict:
        """
        Calculate optimal markdown schedule to maximize profit while clearing inventory.

        Args:
            product_id: Product identifier
            current_inventory: Current inventory units
            days_remaining: Days until end of season/clearance deadline
            current_price: Current selling price
            elasticity: Price elasticity of demand (should be negative)
            baseline_demand: Daily demand at current price
            cost_per_unit: Unit cost for profit calculation (optional)
            target_clearance_pct: Target percentage of inventory to clear (default: 95%)

        Returns:
            Dictionary containing:
                - schedule: List of markdown stages with prices and timing
                - expected_revenue: Projected total revenue
                - expected_profit: Projected profit (if cost provided)
                - clearance_rate: Expected percentage cleared
                - total_holding_cost: Total holding costs
                - recommendation: Strategic recommendation

        Raises:
            ValueError: If inputs are invalid
        """
        # Validate inputs
        if current_inventory <= 0:
            raise ValueError(f"Inventory must be positive, got {current_inventory}")
        if days_remaining <= 0:
            raise ValueError(f"Days remaining must be positive, got {days_remaining}")
        if current_price <= 0:
            raise ValueError(f"Price must be positive, got {current_price}")
        if elasticity >= 0:
            raise ValueError(f"Elasticity must be negative, got {elasticity}")
        if baseline_demand <= 0:
            raise ValueError(f"Baseline demand must be positive, got {baseline_demand}")
        if target_clearance_pct <= 0 or target_clearance_pct > 1:
            raise ValueError(f"Target clearance must be between 0 and 1, got {target_clearance_pct}")

        # Target units to clear: int(current_inventory * target_clearance_pct)
        # Determine markdown strategy based on inventory pressure
        days_of_supply = current_inventory / baseline_demand

        if days_of_supply > days_remaining * 2:
            # High inventory pressure - aggressive markdowns needed
            strategy = 'aggressive'
            markdown_stages = self._create_aggressive_schedule(
                days_remaining, current_price, elasticity
            )
        elif days_of_supply > days_remaining:
            # Moderate pressure - standard progressive markdowns
            strategy = 'progressive'
            markdown_stages = self._create_progressive_schedule(
                days_remaining, current_price, elasticity
            )
        else:
            # Low pressure - conservative markdowns
            strategy = 'conservative'
            markdown_stages = self._create_conservative_schedule(
                days_remaining, current_price, elasticity
            )

        # Simulate clearance with this schedule
        simulation = self.simulate_clearance(
            product_id=product_id,
            initial_inventory=current_inventory,
            markdown_schedule=markdown_stages,
            elasticity=elasticity,
            baseline_demand=baseline_demand,
            cost_per_unit=cost_per_unit
        )

        # Calculate metrics
        total_revenue = simulation['total_revenue']
        total_profit = simulation['total_profit'] if cost_per_unit else None
        clearance_rate = simulation['clearance_pct']
        total_holding_cost = simulation['total_holding_cost']

        # Generate recommendation
        recommendation = self._generate_recommendation(
            strategy=strategy,
            clearance_rate=clearance_rate,
            target_clearance_pct=target_clearance_pct,
            days_of_supply=days_of_supply,
            days_remaining=days_remaining
        )

        result = {
            'product_id': product_id,
            'strategy': strategy,
            'schedule': markdown_stages,
            'expected_revenue': total_revenue,
            'expected_profit': total_profit,
            'clearance_rate': clearance_rate,
            'total_holding_cost': total_holding_cost,
            'simulation': simulation,
            'recommendation': recommendation,
            'timestamp': datetime.now().isoformat()
        }

        self.clearance_history.append(result)

        logger.info(
            "Calculated markdown for %s: "
            "strategy=%s, "
            "clearance=%.1f%%, "
            "revenue=$%.2f",
            product_id,
            strategy,
            clearance_rate * 100,
            total_revenue
        )

        return result

    def simulate_clearance(
        self,
        product_id: str,
        initial_inventory: int,
        markdown_schedule: List[Dict],
        elasticity: float,
        baseline_demand: float,
        cost_per_unit: Optional[float] = None
    ) -> Dict:
        """
        Simulate inventory clearance over time with given markdown schedule.

        Args:
            product_id: Product identifier
            initial_inventory: Starting inventory units
            markdown_schedule: List of markdown stages with prices and timing
            elasticity: Price elasticity of demand
            baseline_demand: Daily demand at base price
            cost_per_unit: Unit cost for profit calculation (optional)

        Returns:
            Dictionary with simulation results including daily trajectory
        """
        base_price = markdown_schedule[0]['original_price']

        # Initialize simulation
        inventory = initial_inventory
        total_revenue = 0
        total_units_sold = 0
        total_holding_cost = 0
        daily_data = []

        current_day = 0

        for stage in markdown_schedule:
            stage_price = stage['price']
            stage_duration = stage['duration_days']

            # Calculate demand at this price using elasticity
            price_ratio = stage_price / base_price
            demand_multiplier = price_ratio ** elasticity
            daily_demand = baseline_demand * demand_multiplier

            # Simulate each day in this stage
            for _day in range(stage_duration):
                if inventory <= 0:
                    break

                # Units sold today (can't sell more than inventory)
                units_sold = min(daily_demand, inventory)
                revenue = units_sold * stage_price
                holding_cost = inventory * stage_price * self.holding_cost_per_day

                # Update inventory
                inventory -= units_sold
                total_revenue += revenue
                total_units_sold += units_sold
                total_holding_cost += holding_cost

                # Record daily data
                daily_data.append({
                    'day': current_day,
                    'stage': stage['stage_name'],
                    'price': stage_price,
                    'discount_pct': stage['discount_pct'],
                    'inventory': inventory,
                    'units_sold': units_sold,
                    'revenue': revenue,
                    'holding_cost': holding_cost
                })

                current_day += 1

                if inventory <= 0:
                    break

        # Calculate salvage value for remaining inventory
        salvage_revenue = inventory * base_price * self.salvage_value_pct
        total_revenue += salvage_revenue

        # Calculate profit if cost provided
        total_profit = None
        if cost_per_unit is not None:
            total_cost = initial_inventory * cost_per_unit
            total_profit = total_revenue - total_cost - total_holding_cost

        # Calculate clearance percentage
        clearance_pct = total_units_sold / initial_inventory

        return {
            'product_id': product_id,
            'initial_inventory': initial_inventory,
            'final_inventory': inventory,
            'total_units_sold': total_units_sold,
            'clearance_pct': clearance_pct,
            'total_revenue': total_revenue,
            'total_profit': total_profit,
            'total_holding_cost': total_holding_cost,
            'salvage_revenue': salvage_revenue,
            'daily_trajectory': pd.DataFrame(daily_data),
            'simulation_days': current_day
        }

    def compare_strategies(
        self,
        product_id: str,
        initial_inventory: int,
        days_remaining: int,
        current_price: float,
        elasticity: float,
        baseline_demand: float,
        cost_per_unit: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Compare different markdown strategies side-by-side.

        Args:
            product_id: Product identifier
            initial_inventory: Starting inventory
            days_remaining: Days until clearance deadline
            current_price: Current selling price
            elasticity: Price elasticity of demand
            baseline_demand: Daily demand at current price
            cost_per_unit: Unit cost for profit calculation (optional)

        Returns:
            DataFrame comparing strategies with metrics
        """
        strategies = ['conservative', 'progressive', 'aggressive']
        results = []

        for strategy in strategies:
            # Create schedule for this strategy
            if strategy == 'conservative':
                schedule = self._create_conservative_schedule(
                    days_remaining, current_price, elasticity
                )
            elif strategy == 'progressive':
                schedule = self._create_progressive_schedule(
                    days_remaining, current_price, elasticity
                )
            else:  # aggressive
                schedule = self._create_aggressive_schedule(
                    days_remaining, current_price, elasticity
                )

            # Simulate clearance
            simulation = self.simulate_clearance(
                product_id=product_id,
                initial_inventory=initial_inventory,
                markdown_schedule=schedule,
                elasticity=elasticity,
                baseline_demand=baseline_demand,
                cost_per_unit=cost_per_unit
            )

            # Collect results
            results.append({
                'strategy': strategy,
                'clearance_pct': simulation['clearance_pct'],
                'total_revenue': simulation['total_revenue'],
                'total_profit': simulation['total_profit'],
                'holding_cost': simulation['total_holding_cost'],
                'salvage_revenue': simulation['salvage_revenue'],
                'final_inventory': simulation['final_inventory'],
                'max_discount': max([s['discount_pct'] for s in schedule]),
                'avg_price': sum([s['price'] * s['duration_days'] for s in schedule]) / days_remaining
            })

        comparison_df = pd.DataFrame(results)

        logger.info("Compared %d strategies for %s", len(strategies), product_id)

        return comparison_df

    def progressive_markdown(
        self,
        days_remaining: int,
        current_price: float,
        _stages: int = 3
    ) -> List[Dict]:
        """
        Generate standard progressive markdown schedule.

        Args:
            days_remaining: Days until clearance deadline
            current_price: Current selling price
            stages: Number of markdown stages (default: 3)

        Returns:
            List of markdown stages
        """
        return self._create_progressive_schedule(days_remaining, current_price, -1.5)

    def emergency_clearance(
        self,
        days_remaining: int,
        current_price: float
    ) -> List[Dict]:
        """
        Generate aggressive emergency clearance schedule.

        Args:
            days_remaining: Days until clearance deadline
            current_price: Current selling price

        Returns:
            List of aggressive markdown stages
        """
        return self._create_aggressive_schedule(days_remaining, current_price, -2.0)

    # Private helper methods

    def _create_conservative_schedule(
        self,
        days_remaining: int,
        base_price: float,
        _elasticity: float
    ) -> List[Dict]:
        """Create conservative markdown schedule (10% → 20% → 35%)."""
        stage_duration = max(1, days_remaining // 3)

        return [
            {
                'stage_name': 'Stage 1',
                'original_price': base_price,
                'discount_pct': 10,
                'price': base_price * 0.90,
                'duration_days': stage_duration,
                'start_day': 0
            },
            {
                'stage_name': 'Stage 2',
                'original_price': base_price,
                'discount_pct': 20,
                'price': base_price * 0.80,
                'duration_days': stage_duration,
                'start_day': stage_duration
            },
            {
                'stage_name': 'Stage 3',
                'original_price': base_price,
                'discount_pct': 35,
                'price': base_price * 0.65,
                'duration_days': days_remaining - 2 * stage_duration,
                'start_day': 2 * stage_duration
            }
        ]

    def _create_progressive_schedule(
        self,
        days_remaining: int,
        base_price: float,
        _elasticity: float
    ) -> List[Dict]:
        """Create standard progressive markdown schedule (15% → 30% → 50%)."""
        stage_duration = max(1, days_remaining // 3)

        return [
            {
                'stage_name': 'Stage 1',
                'original_price': base_price,
                'discount_pct': 15,
                'price': base_price * 0.85,
                'duration_days': stage_duration,
                'start_day': 0
            },
            {
                'stage_name': 'Stage 2',
                'original_price': base_price,
                'discount_pct': 30,
                'price': base_price * 0.70,
                'duration_days': stage_duration,
                'start_day': stage_duration
            },
            {
                'stage_name': 'Stage 3',
                'original_price': base_price,
                'discount_pct': 50,
                'price': base_price * 0.50,
                'duration_days': days_remaining - 2 * stage_duration,
                'start_day': 2 * stage_duration
            }
        ]

    def _create_aggressive_schedule(
        self,
        days_remaining: int,
        base_price: float,
        _elasticity: float
    ) -> List[Dict]:
        """Create aggressive markdown schedule (25% → 40% → 60%)."""
        stage_duration = max(1, days_remaining // 3)

        return [
            {
                'stage_name': 'Stage 1',
                'original_price': base_price,
                'discount_pct': 25,
                'price': base_price * 0.75,
                'duration_days': stage_duration,
                'start_day': 0
            },
            {
                'stage_name': 'Stage 2',
                'original_price': base_price,
                'discount_pct': 40,
                'price': base_price * 0.60,
                'duration_days': stage_duration,
                'start_day': stage_duration
            },
            {
                'stage_name': 'Stage 3',
                'original_price': base_price,
                'discount_pct': 60,
                'price': base_price * 0.40,
                'duration_days': days_remaining - 2 * stage_duration,
                'start_day': 2 * stage_duration
            }
        ]

    def _generate_recommendation(
        self,
        strategy: str,
        clearance_rate: float,
        target_clearance_pct: float,
        days_of_supply: float,
        days_remaining: int
    ) -> str:
        """Generate strategic recommendation based on simulation results."""
        if clearance_rate >= target_clearance_pct:
            status = "✅ Target clearance achieved"
        elif clearance_rate >= target_clearance_pct * 0.9:
            status = "⚠️ Near target clearance"
        else:
            status = "❌ Below target clearance"

        if days_of_supply > days_remaining * 2:
            urgency = "HIGH"
            action = "Consider more aggressive markdowns or bundle deals"
        elif days_of_supply > days_remaining:
            urgency = "MEDIUM"
            action = "Current strategy appropriate, monitor closely"
        else:
            urgency = "LOW"
            action = "Conservative markdowns sufficient"

        recommendation = (
            f"{status}\n"
            f"Strategy: {strategy.upper()}\n"
            f"Urgency: {urgency}\n"
            f"Clearance Rate: {clearance_rate:.1%}\n"
            f"Target: {target_clearance_pct:.1%}\n"
            f"Action: {action}"
        )

        return recommendation

    def get_clearance_summary(self) -> pd.DataFrame:
        """
        Get summary of all clearance optimizations performed.

        Returns:
            DataFrame with summary statistics
        """
        if not self.clearance_history:
            return pd.DataFrame()

        summary_data = []
        for result in self.clearance_history:
            summary_data.append({
                'product_id': result['product_id'],
                'strategy': result['strategy'],
                'clearance_rate': result['clearance_rate'],
                'total_revenue': result['expected_revenue'],
                'total_profit': result['expected_profit'],
                'holding_cost': result['total_holding_cost'],
                'timestamp': result['timestamp']
            })

        return pd.DataFrame(summary_data)

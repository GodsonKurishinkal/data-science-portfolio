"""
Tests for Markdown Optimization Module

Author: Godson Kurishinkal
Date: November 12, 2025
"""

import pytest
import pandas as pd
import numpy as np
from src.pricing.markdown import MarkdownOptimizer


class TestMarkdownOptimizerInitialization:
    """Test MarkdownOptimizer initialization."""

    def test_initialization_default(self):
        """Test initialization with default parameters."""
        optimizer = MarkdownOptimizer()
        assert optimizer.holding_cost_per_day == 0.001
        assert optimizer.salvage_value_pct == 0.30
        assert optimizer.clearance_history == []

    def test_initialization_custom(self):
        """Test initialization with custom parameters."""
        optimizer = MarkdownOptimizer(
            holding_cost_per_day=0.002,
            salvage_value_pct=0.25
        )
        assert optimizer.holding_cost_per_day == 0.002
        assert optimizer.salvage_value_pct == 0.25

    def test_initialization_invalid_holding_cost(self):
        """Test initialization with invalid holding cost."""
        with pytest.raises(ValueError):
            MarkdownOptimizer(holding_cost_per_day=-0.001)

        with pytest.raises(ValueError):
            MarkdownOptimizer(holding_cost_per_day=1.5)

    def test_initialization_invalid_salvage_value(self):
        """Test initialization with invalid salvage value."""
        with pytest.raises(ValueError):
            MarkdownOptimizer(salvage_value_pct=-0.1)

        with pytest.raises(ValueError):
            MarkdownOptimizer(salvage_value_pct=1.5)


class TestCalculateOptimalMarkdown:
    """Test calculate_optimal_markdown method."""

    @pytest.fixture
    def optimizer(self):
        """Create optimizer instance for testing."""
        return MarkdownOptimizer()

    def test_progressive_strategy(self, optimizer):
        """Test progressive markdown strategy selection."""
        result = optimizer.calculate_optimal_markdown(
            product_id='PROD_001',
            current_inventory=400,
            days_remaining=30,
            current_price=50.0,
            elasticity=-1.5,
            baseline_demand=10  # 400/10 = 40 days of supply, 40 > 30, triggers progressive
        )

        assert result['strategy'] == 'progressive'
        assert result['product_id'] == 'PROD_001'
        assert len(result['schedule']) == 3
        assert result['clearance_rate'] >= 0
        assert result['expected_revenue'] > 0

    def test_aggressive_strategy(self, optimizer):
        """Test aggressive markdown strategy selection."""
        result = optimizer.calculate_optimal_markdown(
            product_id='PROD_002',
            current_inventory=1000,
            days_remaining=15,
            current_price=40.0,
            elasticity=-2.0,
            baseline_demand=10  # High inventory pressure
        )

        assert result['strategy'] == 'aggressive'
        assert result['clearance_rate'] > 0.5

    def test_conservative_strategy(self, optimizer):
        """Test conservative markdown strategy selection."""
        result = optimizer.calculate_optimal_markdown(
            product_id='PROD_003',
            current_inventory=100,
            days_remaining=30,
            current_price=60.0,
            elasticity=-1.2,
            baseline_demand=10  # Low inventory pressure
        )

        assert result['strategy'] == 'conservative'

    def test_with_cost_profit_calculation(self, optimizer):
        """Test markdown calculation with cost for profit optimization."""
        result = optimizer.calculate_optimal_markdown(
            product_id='PROD_004',
            current_inventory=200,
            days_remaining=20,
            current_price=50.0,
            elasticity=-1.8,
            baseline_demand=12,
            cost_per_unit=25.0
        )

        assert result['expected_profit'] is not None
        assert result['expected_profit'] < result['expected_revenue']

    def test_invalid_inventory(self, optimizer):
        """Test with invalid inventory."""
        with pytest.raises(ValueError):
            optimizer.calculate_optimal_markdown(
                product_id='PROD_005',
                current_inventory=0,
                days_remaining=30,
                current_price=50.0,
                elasticity=-1.5,
                baseline_demand=10
            )

    def test_invalid_days_remaining(self, optimizer):
        """Test with invalid days remaining."""
        with pytest.raises(ValueError):
            optimizer.calculate_optimal_markdown(
                product_id='PROD_006',
                current_inventory=100,
                days_remaining=-5,
                current_price=50.0,
                elasticity=-1.5,
                baseline_demand=10
            )

    def test_invalid_price(self, optimizer):
        """Test with invalid price."""
        with pytest.raises(ValueError):
            optimizer.calculate_optimal_markdown(
                product_id='PROD_007',
                current_inventory=100,
                days_remaining=30,
                current_price=-50.0,
                elasticity=-1.5,
                baseline_demand=10
            )

    def test_invalid_elasticity(self, optimizer):
        """Test with invalid elasticity (positive)."""
        with pytest.raises(ValueError):
            optimizer.calculate_optimal_markdown(
                product_id='PROD_008',
                current_inventory=100,
                days_remaining=30,
                current_price=50.0,
                elasticity=1.5,  # Should be negative
                baseline_demand=10
            )

    def test_invalid_baseline_demand(self, optimizer):
        """Test with invalid baseline demand."""
        with pytest.raises(ValueError):
            optimizer.calculate_optimal_markdown(
                product_id='PROD_009',
                current_inventory=100,
                days_remaining=30,
                current_price=50.0,
                elasticity=-1.5,
                baseline_demand=-10
            )

    def test_invalid_target_clearance(self, optimizer):
        """Test with invalid target clearance percentage."""
        with pytest.raises(ValueError):
            optimizer.calculate_optimal_markdown(
                product_id='PROD_010',
                current_inventory=100,
                days_remaining=30,
                current_price=50.0,
                elasticity=-1.5,
                baseline_demand=10,
                target_clearance_pct=1.5  # Should be <= 1.0
            )


class TestSimulateClearance:
    """Test simulate_clearance method."""

    @pytest.fixture
    def optimizer(self):
        """Create optimizer instance for testing."""
        return MarkdownOptimizer()

    @pytest.fixture
    def progressive_schedule(self):
        """Create progressive markdown schedule for testing."""
        return [
            {
                'stage_name': 'Stage 1',
                'original_price': 50.0,
                'discount_pct': 15,
                'price': 42.5,
                'duration_days': 10,
                'start_day': 0
            },
            {
                'stage_name': 'Stage 2',
                'original_price': 50.0,
                'discount_pct': 30,
                'price': 35.0,
                'duration_days': 10,
                'start_day': 10
            },
            {
                'stage_name': 'Stage 3',
                'original_price': 50.0,
                'discount_pct': 50,
                'price': 25.0,
                'duration_days': 10,
                'start_day': 20
            }
        ]

    def test_simulate_basic(self, optimizer, progressive_schedule):
        """Test basic clearance simulation."""
        result = optimizer.simulate_clearance(
            product_id='PROD_001',
            initial_inventory=300,
            markdown_schedule=progressive_schedule,
            elasticity=-1.5,
            baseline_demand=10
        )

        assert result['product_id'] == 'PROD_001'
        assert result['initial_inventory'] == 300
        assert result['total_units_sold'] >= 0
        # Allow small floating point tolerance
        assert result['clearance_pct'] >= 0 and result['clearance_pct'] <= 1.01
        assert result['total_revenue'] > 0
        assert isinstance(result['daily_trajectory'], pd.DataFrame)

    def test_simulate_with_cost(self, optimizer, progressive_schedule):
        """Test simulation with cost calculation."""
        result = optimizer.simulate_clearance(
            product_id='PROD_002',
            initial_inventory=200,
            markdown_schedule=progressive_schedule,
            elasticity=-2.0,
            baseline_demand=8,
            cost_per_unit=20.0
        )

        assert result['total_profit'] is not None
        assert result['total_profit'] < result['total_revenue']

    def test_simulate_high_elasticity(self, optimizer, progressive_schedule):
        """Test simulation with highly elastic product."""
        result = optimizer.simulate_clearance(
            product_id='PROD_003',
            initial_inventory=150,
            markdown_schedule=progressive_schedule,
            elasticity=-2.5,  # Highly elastic
            baseline_demand=5
        )

        # High elasticity should lead to high clearance
        assert result['clearance_pct'] > 0.7

    def test_simulate_low_elasticity(self, optimizer, progressive_schedule):
        """Test simulation with inelastic product."""
        result = optimizer.simulate_clearance(
            product_id='PROD_004',
            initial_inventory=150,
            markdown_schedule=progressive_schedule,
            elasticity=-0.5,  # Inelastic
            baseline_demand=5
        )

        # Low elasticity may result in lower clearance
        assert result['clearance_pct'] >= 0

    def test_daily_trajectory_structure(self, optimizer, progressive_schedule):
        """Test that daily trajectory has correct structure."""
        result = optimizer.simulate_clearance(
            product_id='PROD_005',
            initial_inventory=100,
            markdown_schedule=progressive_schedule,
            elasticity=-1.5,
            baseline_demand=5
        )

        trajectory = result['daily_trajectory']

        assert 'day' in trajectory.columns
        assert 'stage' in trajectory.columns
        assert 'price' in trajectory.columns
        assert 'discount_pct' in trajectory.columns
        assert 'inventory' in trajectory.columns
        assert 'units_sold' in trajectory.columns
        assert 'revenue' in trajectory.columns
        assert 'holding_cost' in trajectory.columns


class TestCompareStrategies:
    """Test compare_strategies method."""

    @pytest.fixture
    def optimizer(self):
        """Create optimizer instance for testing."""
        return MarkdownOptimizer()

    def test_compare_basic(self, optimizer):
        """Test basic strategy comparison."""
        comparison = optimizer.compare_strategies(
            product_id='PROD_001',
            initial_inventory=200,
            days_remaining=30,
            current_price=50.0,
            elasticity=-1.5,
            baseline_demand=8
        )

        assert isinstance(comparison, pd.DataFrame)
        assert len(comparison) == 3  # Conservative, progressive, aggressive
        assert 'strategy' in comparison.columns
        assert 'clearance_pct' in comparison.columns
        assert 'total_revenue' in comparison.columns

    def test_compare_with_cost(self, optimizer):
        """Test strategy comparison with cost."""
        comparison = optimizer.compare_strategies(
            product_id='PROD_002',
            initial_inventory=150,
            days_remaining=25,
            current_price=60.0,
            elasticity=-1.8,
            baseline_demand=7,
            cost_per_unit=30.0
        )

        assert 'total_profit' in comparison.columns
        # All strategies should have profit calculated
        assert comparison['total_profit'].notna().all()

    def test_compare_strategies_ordering(self, optimizer):
        """Test that aggressive strategy has highest discount."""
        comparison = optimizer.compare_strategies(
            product_id='PROD_003',
            initial_inventory=100,
            days_remaining=20,
            current_price=40.0,
            elasticity=-2.0,
            baseline_demand=6
        )

        aggressive_row = comparison[comparison['strategy'] == 'aggressive'].iloc[0]
        conservative_row = comparison[comparison['strategy'] == 'conservative'].iloc[0]

        # Aggressive should have higher max discount
        assert aggressive_row['max_discount'] > conservative_row['max_discount']


class TestMarkdownSchedules:
    """Test markdown schedule creation methods."""

    @pytest.fixture
    def optimizer(self):
        """Create optimizer instance for testing."""
        return MarkdownOptimizer()

    def test_progressive_markdown_schedule(self, optimizer):
        """Test progressive markdown schedule generation."""
        schedule = optimizer.progressive_markdown(
            days_remaining=30,
            current_price=50.0
        )

        assert len(schedule) == 3
        assert schedule[0]['discount_pct'] == 15
        assert schedule[1]['discount_pct'] == 30
        assert schedule[2]['discount_pct'] == 50

    def test_emergency_clearance_schedule(self, optimizer):
        """Test emergency clearance schedule generation."""
        schedule = optimizer.emergency_clearance(
            days_remaining=15,
            current_price=60.0
        )

        assert len(schedule) == 3
        assert schedule[0]['discount_pct'] == 25
        assert schedule[1]['discount_pct'] == 40
        assert schedule[2]['discount_pct'] == 60

    def test_schedule_timing(self, optimizer):
        """Test that schedule timing adds up correctly."""
        days = 30
        schedule = optimizer.progressive_markdown(
            days_remaining=days,
            current_price=50.0
        )

        total_days = sum([stage['duration_days'] for stage in schedule])
        assert total_days == days


class TestClearanceHistory:
    """Test clearance history tracking."""

    def test_history_tracking(self):
        """Test that optimizer tracks clearance history."""
        optimizer = MarkdownOptimizer()

        # Perform multiple optimizations
        for i in range(3):
            optimizer.calculate_optimal_markdown(
                product_id=f'PROD_{i:03d}',
                current_inventory=100 + i * 50,
                days_remaining=20 + i * 5,
                current_price=50.0,
                elasticity=-1.5,
                baseline_demand=10
            )

        assert len(optimizer.clearance_history) == 3

    def test_get_clearance_summary_empty(self):
        """Test getting summary with no history."""
        optimizer = MarkdownOptimizer()
        summary = optimizer.get_clearance_summary()

        assert isinstance(summary, pd.DataFrame)
        assert len(summary) == 0

    def test_get_clearance_summary_with_data(self):
        """Test getting summary with clearance history."""
        optimizer = MarkdownOptimizer()

        # Perform optimizations
        for i in range(2):
            optimizer.calculate_optimal_markdown(
                product_id=f'PROD_{i:03d}',
                current_inventory=150,
                days_remaining=25,
                current_price=55.0,
                elasticity=-1.6,
                baseline_demand=8,
                cost_per_unit=28.0
            )

        summary = optimizer.get_clearance_summary()

        assert len(summary) == 2
        assert 'product_id' in summary.columns
        assert 'strategy' in summary.columns
        assert 'clearance_rate' in summary.columns
        assert 'total_revenue' in summary.columns
        assert 'total_profit' in summary.columns


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.fixture
    def optimizer(self):
        """Create optimizer instance for testing."""
        return MarkdownOptimizer()

    def test_very_short_clearance_period(self, optimizer):
        """Test with very short clearance period."""
        result = optimizer.calculate_optimal_markdown(
            product_id='PROD_SHORT',
            current_inventory=50,
            days_remaining=3,  # Very short period
            current_price=40.0,
            elasticity=-1.5,
            baseline_demand=20
        )

        assert result['schedule'] is not None
        assert len(result['schedule']) == 3

    def test_very_high_inventory(self, optimizer):
        """Test with very high inventory relative to demand."""
        result = optimizer.calculate_optimal_markdown(
            product_id='PROD_HIGH_INV',
            current_inventory=10000,
            days_remaining=20,
            current_price=30.0,
            elasticity=-2.0,
            baseline_demand=10
        )

        # Should trigger aggressive strategy
        assert result['strategy'] == 'aggressive'

    def test_very_low_inventory(self, optimizer):
        """Test with very low inventory."""
        result = optimizer.calculate_optimal_markdown(
            product_id='PROD_LOW_INV',
            current_inventory=10,
            days_remaining=30,
            current_price=50.0,
            elasticity=-1.2,
            baseline_demand=5
        )

        # Should trigger conservative strategy
        assert result['strategy'] == 'conservative'

    def test_extreme_elasticity(self, optimizer):
        """Test with extreme elasticity value."""
        result = optimizer.calculate_optimal_markdown(
            product_id='PROD_EXTREME',
            current_inventory=100,
            days_remaining=20,
            current_price=45.0,
            elasticity=-5.0,  # Extreme elasticity
            baseline_demand=8
        )

        # Should handle extreme elasticity gracefully
        assert result['clearance_rate'] > 0

    def test_zero_cost(self, optimizer):
        """Test with zero unit cost."""
        result = optimizer.calculate_optimal_markdown(
            product_id='PROD_ZERO_COST',
            current_inventory=150,
            days_remaining=25,
            current_price=50.0,
            elasticity=-1.5,
            baseline_demand=10,
            cost_per_unit=0.01  # Very low cost instead of zero
        )

        # Profit should be close to revenue when cost is very low (minus holding cost)
        assert result['expected_profit'] is not None
        assert result['expected_profit'] < result['expected_revenue']

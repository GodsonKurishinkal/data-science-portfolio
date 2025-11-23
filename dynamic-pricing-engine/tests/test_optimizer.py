"""
Test suite for Price Optimization Module - Phase 5

Tests cover:
- Initialization and configuration
- Single product optimization (revenue and profit)
- Multiple optimization methods (scipy, grid, gradient)
- Business constraints (price bounds, margins, discounts)
- Portfolio optimization
- Scenario simulation
- Sensitivity analysis
- Edge cases and error handling

Author: Godson Kurishinkal
Date: November 11, 2025
"""

import pytest
import pandas as pd
import numpy as np
from src.pricing.optimizer import PriceOptimizer, create_standard_scenarios


class TestPriceOptimizerInitialization:
    """Test optimizer initialization and configuration."""
    
    def test_initialization_default(self):
        """Test default initialization."""
        optimizer = PriceOptimizer()
        assert optimizer.objective == 'revenue'
        assert optimizer.method == 'scipy'
        assert optimizer.demand_model is None
        assert optimizer.optimization_history == []
    
    def test_initialization_with_parameters(self):
        """Test initialization with custom parameters."""
        optimizer = PriceOptimizer(objective='profit', method='grid')
        assert optimizer.objective == 'profit'
        assert optimizer.method == 'grid'
    
    def test_initialization_invalid_objective(self):
        """Test that invalid objective raises error."""
        with pytest.raises(ValueError, match="Objective must be 'revenue' or 'profit'"):
            PriceOptimizer(objective='invalid')
    
    def test_initialization_invalid_method(self):
        """Test that invalid method raises error."""
        with pytest.raises(ValueError, match="Method must be 'scipy', 'grid', or 'gradient'"):
            PriceOptimizer(method='invalid')


class TestSingleProductOptimization:
    """Test single product optimization functionality."""
    
    def test_optimize_elastic_product_revenue(self):
        """Test optimization of elastic product for revenue maximization."""
        optimizer = PriceOptimizer(objective='revenue', method='scipy')
        
        result = optimizer.optimize_single_product(
            product_id='ELASTIC_001',
            current_price=50.0,
            baseline_demand=1000.0,
            elasticity=-1.5
        )
        
        # Elastic product should reduce price to increase revenue
        assert result['optimal_price'] < result['current_price']
        assert result['optimal_revenue'] >= result['current_revenue']
        assert result['product_id'] == 'ELASTIC_001'
        assert 'timestamp' in result
        assert result['method'] == 'scipy'
        assert result['objective'] == 'revenue'
    
    def test_optimize_inelastic_product_revenue(self):
        """Test optimization of inelastic product for revenue maximization."""
        optimizer = PriceOptimizer(objective='revenue', method='scipy')
        
        result = optimizer.optimize_single_product(
            product_id='INELASTIC_001',
            current_price=50.0,
            baseline_demand=1000.0,
            elasticity=-0.5
        )
        
        # Inelastic product should increase price to increase revenue
        assert result['optimal_price'] > result['current_price']
        assert result['optimal_revenue'] >= result['current_revenue']
    
    def test_optimize_with_profit_objective(self):
        """Test profit optimization."""
        optimizer = PriceOptimizer(objective='profit', method='scipy')
        
        result = optimizer.optimize_single_product(
            product_id='PROFIT_001',
            current_price=50.0,
            baseline_demand=1000.0,
            elasticity=-1.5,
            cost_per_unit=20.0
        )
        
        assert 'current_profit' in result
        assert 'optimal_profit' in result
        assert 'profit_change_pct' in result
        assert 'current_margin_pct' in result
        assert 'optimal_margin_pct' in result
        assert result['optimal_profit'] >= result['current_profit']
    
    def test_optimize_with_price_constraints(self):
        """Test optimization with price bounds."""
        optimizer = PriceOptimizer(objective='revenue', method='scipy')
        
        result = optimizer.optimize_single_product(
            product_id='CONSTRAINED_001',
            current_price=50.0,
            baseline_demand=1000.0,
            elasticity=-1.5,
            constraints={'min_price': 45.0, 'max_price': 55.0}
        )
        
        # Optimal price should respect bounds
        assert result['optimal_price'] >= 45.0
        assert result['optimal_price'] <= 55.0
    
    def test_optimize_with_margin_constraint(self):
        """Test optimization with minimum margin constraint."""
        optimizer = PriceOptimizer(objective='profit', method='scipy')
        
        result = optimizer.optimize_single_product(
            product_id='MARGIN_001',
            current_price=50.0,
            baseline_demand=1000.0,
            elasticity=-1.5,
            cost_per_unit=20.0,
            constraints={'min_margin_pct': 40.0}  # Minimum 40% margin
        )
        
        # Optimal price should maintain minimum margin
        optimal_margin = ((result['optimal_price'] - 20.0) / result['optimal_price']) * 100
        assert optimal_margin >= 39.9  # Allow small floating point error
    
    def test_optimize_with_discount_constraint(self):
        """Test optimization with maximum discount constraint."""
        optimizer = PriceOptimizer(objective='revenue', method='scipy')
        
        result = optimizer.optimize_single_product(
            product_id='DISCOUNT_001',
            current_price=50.0,
            baseline_demand=1000.0,
            elasticity=-1.5,
            constraints={'max_discount_pct': 10.0}  # Max 10% discount
        )
        
        # Optimal price should not be more than 10% below current price
        assert result['optimal_price'] >= 45.0  # 50 * 0.9
    
    def test_optimize_with_seasonality(self):
        """Test optimization with seasonality factor."""
        optimizer = PriceOptimizer(objective='revenue', method='scipy')
        
        result = optimizer.optimize_single_product(
            product_id='SEASONAL_001',
            current_price=50.0,
            baseline_demand=1000.0,
            elasticity=-1.5,
            seasonality_factor=1.5  # 50% higher demand
        )
        
        assert result['seasonality_factor'] == 1.5
        assert result['current_demand'] == 1500.0  # baseline * seasonality
    
    def test_optimize_with_promotion(self):
        """Test optimization with promotional lift."""
        optimizer = PriceOptimizer(objective='revenue', method='scipy')
        
        result = optimizer.optimize_single_product(
            product_id='PROMO_001',
            current_price=50.0,
            baseline_demand=1000.0,
            elasticity=-1.5,
            promotion_lift=1.3  # 30% promotional lift
        )
        
        assert result['promotion_lift'] == 1.3
        assert result['current_demand'] == 1300.0  # baseline * promotion
    
    def test_optimization_history_tracking(self):
        """Test that optimization history is tracked."""
        optimizer = PriceOptimizer(objective='revenue', method='scipy')
        
        # First optimization
        optimizer.optimize_single_product(
            product_id='PROD_001',
            current_price=50.0,
            baseline_demand=1000.0,
            elasticity=-1.5
        )
        
        assert len(optimizer.optimization_history) == 1
        
        # Second optimization
        optimizer.optimize_single_product(
            product_id='PROD_002',
            current_price=30.0,
            baseline_demand=2000.0,
            elasticity=-0.8
        )
        
        assert len(optimizer.optimization_history) == 2
        assert optimizer.optimization_history[0]['product_id'] == 'PROD_001'
        assert optimizer.optimization_history[1]['product_id'] == 'PROD_002'


class TestOptimizationMethods:
    """Test different optimization methods."""
    
    def test_scipy_method(self):
        """Test scipy bounded optimization."""
        optimizer = PriceOptimizer(objective='revenue', method='scipy')
        
        result = optimizer.optimize_single_product(
            product_id='SCIPY_001',
            current_price=50.0,
            baseline_demand=1000.0,
            elasticity=-1.5
        )
        
        assert result['method'] == 'scipy'
        assert result['optimal_revenue'] >= result['current_revenue']
    
    def test_grid_search_method(self):
        """Test grid search optimization."""
        optimizer = PriceOptimizer(objective='revenue', method='grid')
        
        result = optimizer.optimize_single_product(
            product_id='GRID_001',
            current_price=50.0,
            baseline_demand=1000.0,
            elasticity=-1.5
        )
        
        assert result['method'] == 'grid'
        assert result['optimal_revenue'] >= result['current_revenue']
    
    def test_gradient_descent_method(self):
        """Test custom gradient descent optimization."""
        optimizer = PriceOptimizer(objective='revenue', method='gradient')
        
        result = optimizer.optimize_single_product(
            product_id='GRADIENT_001',
            current_price=50.0,
            baseline_demand=1000.0,
            elasticity=-1.5
        )
        
        assert result['method'] == 'gradient'
        assert result['optimal_revenue'] >= result['current_revenue']
    
    def test_methods_produce_similar_results(self):
        """Test that different methods produce similar optimal prices."""
        product_params = {
            'product_id': 'COMPARE_001',
            'current_price': 50.0,
            'baseline_demand': 1000.0,
            'elasticity': -1.5
        }
        
        scipy_optimizer = PriceOptimizer(objective='revenue', method='scipy')
        grid_optimizer = PriceOptimizer(objective='revenue', method='grid')
        gradient_optimizer = PriceOptimizer(objective='revenue', method='gradient')
        
        scipy_result = scipy_optimizer.optimize_single_product(**product_params)
        grid_result = grid_optimizer.optimize_single_product(**product_params)
        gradient_result = gradient_optimizer.optimize_single_product(**product_params)
        
        # All methods should produce similar optimal prices (within 5%)
        prices = [
            scipy_result['optimal_price'],
            grid_result['optimal_price'],
            gradient_result['optimal_price']
        ]
        
        avg_price = np.mean(prices)
        for price in prices:
            assert abs(price - avg_price) / avg_price < 0.05


class TestPortfolioOptimization:
    """Test portfolio optimization functionality."""
    
    def test_optimize_portfolio_basic(self):
        """Test basic portfolio optimization."""
        optimizer = PriceOptimizer(objective='revenue', method='scipy')
        
        products_df = pd.DataFrame({
            'product_id': ['A', 'B', 'C'],
            'current_price': [50.0, 30.0, 100.0],
            'baseline_demand': [1000.0, 2000.0, 500.0],
            'elasticity': [-1.5, -0.8, -2.0]
        })
        
        results = optimizer.optimize_portfolio(products_df)
        
        assert len(results) == 3
        assert set(results['product_id']) == {'A', 'B', 'C'}
        assert all(results['optimal_revenue'] >= results['current_revenue'])
    
    def test_optimize_portfolio_with_costs(self):
        """Test portfolio optimization with cost data."""
        optimizer = PriceOptimizer(objective='profit', method='scipy')
        
        products_df = pd.DataFrame({
            'product_id': ['A', 'B', 'C'],
            'current_price': [50.0, 30.0, 100.0],
            'baseline_demand': [1000.0, 2000.0, 500.0],
            'elasticity': [-1.5, -0.8, -2.0],
            'cost_per_unit': [20.0, 15.0, 40.0]
        })
        
        results = optimizer.optimize_portfolio(products_df)
        
        assert 'current_profit' in results.columns
        assert 'optimal_profit' in results.columns
        assert all(results['optimal_profit'] >= results['current_profit'])
    
    def test_optimize_portfolio_with_constraints(self):
        """Test portfolio optimization with shared constraints."""
        optimizer = PriceOptimizer(objective='revenue', method='scipy')
        
        products_df = pd.DataFrame({
            'product_id': ['A', 'B', 'C'],
            'current_price': [50.0, 30.0, 100.0],
            'baseline_demand': [1000.0, 2000.0, 500.0],
            'elasticity': [-1.5, -0.8, -2.0]
        })
        
        constraints = {'max_discount_pct': 15.0}
        results = optimizer.optimize_portfolio(products_df, constraints=constraints)
        
        # All products should respect the discount constraint
        for _, row in results.iterrows():
            discount_pct = ((row['current_price'] - row['optimal_price']) / row['current_price']) * 100
            assert discount_pct <= 15.1  # Small tolerance for floating point


class TestScenarioSimulation:
    """Test scenario simulation functionality."""
    
    def test_simulate_scenarios_default(self):
        """Test scenario simulation with default scenarios."""
        optimizer = PriceOptimizer(objective='revenue', method='scipy')
        
        results = optimizer.simulate_scenarios(
            product_id='SIM_001',
            current_price=50.0,
            baseline_demand=1000.0,
            elasticity=-1.5
        )
        
        assert len(results) == 9  # Default scenarios
        assert 'scenario' in results.columns
        assert 'price' in results.columns
        assert 'predicted_demand' in results.columns
        assert 'revenue' in results.columns
        assert 'Current Price' in results['scenario'].values
    
    def test_simulate_scenarios_custom(self):
        """Test scenario simulation with custom scenarios."""
        optimizer = PriceOptimizer(objective='revenue', method='scipy')
        
        custom_scenarios = [
            {'name': 'Low', 'price': 40.0},
            {'name': 'Medium', 'price': 50.0},
            {'name': 'High', 'price': 60.0}
        ]
        
        results = optimizer.simulate_scenarios(
            product_id='CUSTOM_001',
            current_price=50.0,
            baseline_demand=1000.0,
            elasticity=-1.5,
            scenarios=custom_scenarios
        )
        
        assert len(results) == 3
        assert set(results['scenario']) == {'Low', 'Medium', 'High'}
    
    def test_simulate_scenarios_with_cost(self):
        """Test scenario simulation with profit calculations."""
        optimizer = PriceOptimizer(objective='profit', method='scipy')
        
        results = optimizer.simulate_scenarios(
            product_id='PROFIT_SIM_001',
            current_price=50.0,
            baseline_demand=1000.0,
            elasticity=-1.5,
            cost_per_unit=20.0
        )
        
        assert 'profit' in results.columns
        assert 'margin_pct' in results.columns


class TestSensitivityAnalysis:
    """Test sensitivity analysis functionality."""
    
    def test_sensitivity_analysis_default(self):
        """Test sensitivity analysis with default price range."""
        optimizer = PriceOptimizer(objective='revenue', method='scipy')
        
        results = optimizer.sensitivity_analysis(
            product_id='SENS_001',
            current_price=50.0,
            baseline_demand=1000.0,
            elasticity=-1.5
        )
        
        assert len(results) == 50  # Default n_points
        assert 'price' in results.columns
        assert 'predicted_demand' in results.columns
        assert 'revenue' in results.columns
        assert results['price'].min() == 25.0  # 50% of current
        assert results['price'].max() == 100.0  # 200% of current
    
    def test_sensitivity_analysis_custom_range(self):
        """Test sensitivity analysis with custom price range."""
        optimizer = PriceOptimizer(objective='revenue', method='scipy')
        
        results = optimizer.sensitivity_analysis(
            product_id='CUSTOM_SENS_001',
            current_price=50.0,
            baseline_demand=1000.0,
            elasticity=-1.5,
            price_range=(40.0, 60.0),
            n_points=20
        )
        
        assert len(results) == 20
        assert results['price'].min() == 40.0
        assert results['price'].max() == 60.0
    
    def test_sensitivity_analysis_with_profit(self):
        """Test sensitivity analysis with profit calculations."""
        optimizer = PriceOptimizer(objective='profit', method='scipy')
        
        results = optimizer.sensitivity_analysis(
            product_id='PROFIT_SENS_001',
            current_price=50.0,
            baseline_demand=1000.0,
            elasticity=-1.5,
            cost_per_unit=20.0
        )
        
        assert 'profit' in results.columns
        assert 'margin_pct' in results.columns


class TestElasticitySegmentation:
    """Test elasticity segmentation functionality."""
    
    def test_elasticity_segmentation_default(self):
        """Test elasticity segmentation with default thresholds."""
        optimizer = PriceOptimizer(objective='revenue', method='scipy')
        
        products_df = pd.DataFrame({
            'product_id': ['A', 'B', 'C', 'D', 'E'],
            'elasticity': [-2.5, -1.8, -1.2, -0.7, -0.3]
        })
        
        result = optimizer.calculate_price_elasticity_segments(products_df)
        
        assert 'segments' in result
        assert 'summary' in result
        assert 'recommendations' in result
        assert len(result['segments']) == 5
        assert 'elasticity_segment' in result['segments'].columns
    
    def test_elasticity_segmentation_recommendations(self):
        """Test that recommendations are provided for each segment."""
        optimizer = PriceOptimizer(objective='revenue', method='scipy')
        
        products_df = pd.DataFrame({
            'product_id': ['A', 'B'],
            'elasticity': [-2.5, -0.3]
        })
        
        result = optimizer.calculate_price_elasticity_segments(products_df)
        recommendations = result['recommendations']
        
        assert 'Highly Elastic' in recommendations
        assert 'Highly Inelastic' in recommendations
        assert isinstance(recommendations['Highly Elastic'], str)


class TestOptimizationSummary:
    """Test optimization summary functionality."""
    
    def test_optimization_summary_empty(self):
        """Test summary when no optimizations performed."""
        optimizer = PriceOptimizer(objective='revenue', method='scipy')
        summary = optimizer.get_optimization_summary()
        
        assert 'message' in summary
        assert summary['message'] == 'No optimizations performed yet'
    
    def test_optimization_summary_with_data(self):
        """Test summary after multiple optimizations."""
        optimizer = PriceOptimizer(objective='revenue', method='scipy')
        
        # Perform multiple optimizations
        for i in range(3):
            optimizer.optimize_single_product(
                product_id=f'PROD_{i:03d}',
                current_price=50.0,
                baseline_demand=1000.0,
                elasticity=-1.5
            )
        
        summary = optimizer.get_optimization_summary()
        
        assert summary['total_optimizations'] == 3
        assert 'avg_price_change_pct' in summary
        assert 'avg_revenue_change_pct' in summary
        assert 'total_current_revenue' in summary
        assert 'total_optimal_revenue' in summary
        assert 'total_revenue_gain' in summary
        assert summary['products_optimized'] == 3


class TestInputValidation:
    """Test input validation and error handling."""
    
    def test_negative_price_raises_error(self):
        """Test that negative current price raises error."""
        optimizer = PriceOptimizer(objective='revenue', method='scipy')
        
        with pytest.raises(ValueError, match="Current price must be positive"):
            optimizer.optimize_single_product(
                product_id='INVALID_001',
                current_price=-50.0,
                baseline_demand=1000.0,
                elasticity=-1.5
            )
    
    def test_negative_demand_raises_error(self):
        """Test that negative baseline demand raises error."""
        optimizer = PriceOptimizer(objective='revenue', method='scipy')
        
        with pytest.raises(ValueError, match="Baseline demand must be positive"):
            optimizer.optimize_single_product(
                product_id='INVALID_002',
                current_price=50.0,
                baseline_demand=-1000.0,
                elasticity=-1.5
            )
    
    def test_positive_elasticity_raises_error(self):
        """Test that positive elasticity raises error."""
        optimizer = PriceOptimizer(objective='revenue', method='scipy')
        
        with pytest.raises(ValueError, match="Elasticity must be negative"):
            optimizer.optimize_single_product(
                product_id='INVALID_003',
                current_price=50.0,
                baseline_demand=1000.0,
                elasticity=1.5  # Should be negative
            )
    
    def test_profit_optimization_without_cost_raises_error(self):
        """Test that profit optimization without cost raises error."""
        optimizer = PriceOptimizer(objective='profit', method='scipy')
        
        with pytest.raises(ValueError, match="cost_per_unit is required for profit optimization"):
            optimizer.optimize_single_product(
                product_id='INVALID_004',
                current_price=50.0,
                baseline_demand=1000.0,
                elasticity=-1.5
                # Missing cost_per_unit
            )
    
    def test_negative_cost_raises_error(self):
        """Test that negative cost raises error."""
        optimizer = PriceOptimizer(objective='profit', method='scipy')
        
        with pytest.raises(ValueError, match="Cost per unit must be non-negative"):
            optimizer.optimize_single_product(
                product_id='INVALID_005',
                current_price=50.0,
                baseline_demand=1000.0,
                elasticity=-1.5,
                cost_per_unit=-20.0
            )


class TestHelperFunctions:
    """Test helper functions."""
    
    def test_create_standard_scenarios(self):
        """Test standard scenario creation."""
        scenarios = create_standard_scenarios(current_price=50.0)
        
        assert len(scenarios) == 9
        assert all('name' in s and 'price' in s for s in scenarios)
        
        # Check specific scenarios
        current_scenario = next(s for s in scenarios if s['name'] == 'Current Price')
        assert current_scenario['price'] == 50.0
        
        discount_20_scenario = next(s for s in scenarios if s['name'] == '20% Discount')
        assert discount_20_scenario['price'] == 40.0
        
        increase_20_scenario = next(s for s in scenarios if s['name'] == '20% Increase')
        assert increase_20_scenario['price'] == 60.0


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_unit_elastic_product(self):
        """Test optimization of unit elastic product (elasticity = -1.0)."""
        optimizer = PriceOptimizer(objective='revenue', method='scipy')
        
        result = optimizer.optimize_single_product(
            product_id='UNIT_001',
            current_price=50.0,
            baseline_demand=1000.0,
            elasticity=-1.0
        )
        
        # Unit elastic: revenue is constant regardless of price
        # Optimal price should be close to current price or at a boundary
        assert result is not None
        assert result['revenue_change_pct'] <= 1.0  # Minimal change expected
    
    def test_very_high_elasticity(self):
        """Test optimization with very high elasticity."""
        optimizer = PriceOptimizer(objective='revenue', method='scipy')
        
        result = optimizer.optimize_single_product(
            product_id='HIGH_ELAST_001',
            current_price=50.0,
            baseline_demand=1000.0,
            elasticity=-5.0,
            constraints={'min_price': 10.0, 'max_price': 100.0}
        )
        
        # Very elastic: should significantly reduce price
        assert result['optimal_price'] < result['current_price']
    
    def test_very_low_elasticity(self):
        """Test optimization with very low elasticity."""
        optimizer = PriceOptimizer(objective='revenue', method='scipy')
        
        result = optimizer.optimize_single_product(
            product_id='LOW_ELAST_001',
            current_price=50.0,
            baseline_demand=1000.0,
            elasticity=-0.1,
            constraints={'min_price': 10.0, 'max_price': 100.0}
        )
        
        # Very inelastic: should increase price significantly
        assert result['optimal_price'] > result['current_price']
    
    def test_conflicting_constraints(self):
        """Test handling of conflicting constraints."""
        optimizer = PriceOptimizer(objective='revenue', method='scipy')
        
        result = optimizer.optimize_single_product(
            product_id='CONFLICT_001',
            current_price=50.0,
            baseline_demand=1000.0,
            elasticity=-1.5,
            constraints={
                'min_price': 60.0,  # Min > current
                'max_price': 55.0   # Max < min (conflict!)
            }
        )
        
        # Should handle gracefully by keeping current price
        assert result['optimal_price'] == result['current_price']
        assert result['price_change_pct'] == 0.0
        assert 'no_change' in result['constraints_applied']

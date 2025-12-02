"""
Price Optimization Module - Phase 5

Implements various optimization algorithms to find optimal prices that maximize
revenue or profit while respecting business constraints.

Author: Godson Kurishinkal
Date: November 11, 2025
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from scipy.optimize import minimize_scalar
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PriceOptimizer:
    """
    Advanced price optimization engine using multiple algorithmic approaches.

    This class implements sophisticated optimization algorithms to find optimal
    prices that maximize revenue or profit while respecting business constraints.

    Key Features:
    - Multiple optimization methods (scipy, grid search, gradient descent)
    - Business constraint handling (price bounds, margins, discounts)
    - Revenue or profit maximization
    - Batch optimization for product portfolios
    - Scenario simulation and comparison
    - Sensitivity analysis with visualizations
    - Elasticity-based segmentation and recommendations

    Methods:
        optimize_single_product: Find optimal price for one product
        optimize_portfolio: Optimize multiple products simultaneously
        simulate_scenarios: Compare different pricing strategies
        sensitivity_analysis: Generate price-demand-revenue curves
        calculate_price_elasticity_segments: Segment products by elasticity
        get_optimization_summary: Get aggregate statistics

    Example:
        >>> optimizer = PriceOptimizer(objective='revenue', method='scipy')
        >>> result = optimizer.optimize_single_product(
        ...     product_id='PROD_001',
        ...     current_price=50.0,
        ...     baseline_demand=1000,
        ...     elasticity=-1.5,
        ...     constraints={'min_price': 40, 'max_price': 70}
        ... )
        >>> print(f"Optimal price: ${result['optimal_price']:.2f}")
    """

    def __init__(
        self,
        objective: str = 'revenue',
        method: str = 'scipy',
        demand_model=None
    ):
        """
        Initialize the price optimizer.

        Args:
            objective: Optimization objective - 'revenue' or 'profit'
            method: Optimization method - 'scipy', 'grid', or 'gradient'
            demand_model: Optional trained demand response model for advanced predictions

        Raises:
            ValueError: If objective or method is invalid
        """
        if objective not in ['revenue', 'profit']:
            raise ValueError(f"Objective must be 'revenue' or 'profit', got '{objective}'")

        if method not in ['scipy', 'grid', 'gradient']:
            raise ValueError(f"Method must be 'scipy', 'grid', or 'gradient', got '{method}'")

        self.objective = objective
        self.method = method
        self.demand_model = demand_model
        self.optimization_history = []

        logger.info(f"Initialized PriceOptimizer with objective='{objective}', method='{method}'")

    def optimize_single_product(
        self,
        product_id: str,
        current_price: float,
        baseline_demand: float,
        elasticity: float,
        constraints: Optional[Dict] = None,
        cost_per_unit: Optional[float] = None,
        seasonality_factor: float = 1.0,
        promotion_lift: float = 1.0
    ) -> Dict:
        """
        Find optimal price for a single product using elasticity-based demand modeling.

        This method uses the power-law demand model:
            predicted_demand = baseline_demand * (price / current_price) ^ elasticity

        And optimizes either:
            Revenue = price * predicted_demand
            Profit = (price - cost) * predicted_demand

        Args:
            product_id: Unique product identifier
            current_price: Current price of the product
            baseline_demand: Baseline demand at current price
            elasticity: Price elasticity of demand (negative value)
            constraints: Dict with optional keys:
                - min_price: Minimum allowed price
                - max_price: Maximum allowed price
                - min_margin_pct: Minimum profit margin percentage
                - max_discount_pct: Maximum discount percentage from current price
            cost_per_unit: Cost per unit (required for profit optimization)
            seasonality_factor: Seasonal demand multiplier (default 1.0)
            promotion_lift: Promotional demand lift (default 1.0)

        Returns:
            Dict containing:
                - product_id: Product identifier
                - current_price: Original price
                - optimal_price: Recommended optimal price
                - current_demand: Predicted demand at current price
                - optimal_demand: Predicted demand at optimal price
                - current_revenue: Revenue at current price
                - optimal_revenue: Revenue at optimal price
                - revenue_change_pct: Percentage change in revenue
                - price_change_pct: Percentage change in price
                - method: Optimization method used
                - elasticity: Price elasticity used
                - constraints_applied: List of active constraints
                - objective: Optimization objective used

        Raises:
            ValueError: If inputs are invalid or cost is missing for profit optimization

        Example:
            >>> result = optimizer.optimize_single_product(
            ...     product_id='PROD_001',
            ...     current_price=50.0,
            ...     baseline_demand=1000,
            ...     elasticity=-1.5,
            ...     constraints={'min_price': 40, 'max_price': 70}
            ... )
        """
        # Input validation
        if current_price <= 0:
            raise ValueError(f"Current price must be positive, got {current_price}")
        if baseline_demand <= 0:
            raise ValueError(f"Baseline demand must be positive, got {baseline_demand}")
        if elasticity >= 0:
            raise ValueError(f"Elasticity must be negative, got {elasticity}")
        if self.objective == 'profit' and cost_per_unit is None:
            raise ValueError("cost_per_unit is required for profit optimization")
        if cost_per_unit is not None and cost_per_unit < 0:
            raise ValueError(f"Cost per unit must be non-negative, got {cost_per_unit}")

        # Set default constraints
        if constraints is None:
            constraints = {}

        # Calculate price bounds
        min_price = constraints.get('min_price', current_price * 0.5)
        max_price = constraints.get('max_price', current_price * 2.0)

        # Apply margin constraint
        if 'min_margin_pct' in constraints and cost_per_unit is not None:
            min_margin = constraints['min_margin_pct'] / 100.0
            min_price_from_margin = cost_per_unit / (1 - min_margin)
            min_price = max(min_price, min_price_from_margin)

        # Apply discount constraint
        if 'max_discount_pct' in constraints:
            max_discount = constraints['max_discount_pct'] / 100.0
            min_price_from_discount = current_price * (1 - max_discount)
            min_price = max(min_price, min_price_from_discount)

        # Ensure min_price <= max_price
        if min_price > max_price:
            logger.warning(f"Constraints conflict: min_price ({min_price:.2f}) > max_price ({max_price:.2f}). Using current_price.")
            return self._create_no_change_result(
                product_id, current_price, baseline_demand, elasticity,
                seasonality_factor, promotion_lift, cost_per_unit
            )

        # Define objective function to minimize (negative for maximization)
        def objective_func(price):
            price_ratio = price / current_price
            demand_multiplier = price_ratio ** elasticity
            predicted_demand = baseline_demand * demand_multiplier * seasonality_factor * promotion_lift

            if self.objective == 'revenue':
                return -(price * predicted_demand)  # Negative for minimization
            elif self.objective == 'profit':
                return -((price - cost_per_unit) * predicted_demand)

        # Optimize based on method
        if self.method == 'scipy':
            result = minimize_scalar(
                objective_func,
                bounds=(min_price, max_price),
                method='bounded'
            )
            optimal_price = result.x

        elif self.method == 'grid':
            # Grid search over price range
            prices = np.linspace(min_price, max_price, 100)
            objectives = [objective_func(p) for p in prices]
            optimal_idx = np.argmin(objectives)
            optimal_price = prices[optimal_idx]

        elif self.method == 'gradient':
            # Custom gradient descent
            optimal_price = self._gradient_descent(
                objective_func,
                initial_price=(min_price + max_price) / 2,
                bounds=(min_price, max_price),
                learning_rate=0.1,
                max_iterations=1000,
                tolerance=1e-6
            )

        # Calculate results
        optimal_price = np.clip(optimal_price, min_price, max_price)

        # Current metrics
        current_demand = baseline_demand * seasonality_factor * promotion_lift
        current_revenue = current_price * current_demand
        current_profit = (current_price - (cost_per_unit or 0)) * current_demand if cost_per_unit else None

        # Optimal metrics
        price_ratio = optimal_price / current_price
        demand_multiplier = price_ratio ** elasticity
        optimal_demand = baseline_demand * demand_multiplier * seasonality_factor * promotion_lift
        optimal_revenue = optimal_price * optimal_demand
        optimal_profit = (optimal_price - (cost_per_unit or 0)) * optimal_demand if cost_per_unit else None

        # Calculate changes
        revenue_change_pct = ((optimal_revenue - current_revenue) / current_revenue) * 100
        price_change_pct = ((optimal_price - current_price) / current_price) * 100
        demand_change_pct = ((optimal_demand - current_demand) / current_demand) * 100

        # Track constraints applied
        constraints_applied = []
        if optimal_price == min_price:
            constraints_applied.append('min_price')
        if optimal_price == max_price:
            constraints_applied.append('max_price')

        # Build result dictionary
        result = {
            'product_id': product_id,
            'timestamp': datetime.now().isoformat(),
            'current_price': round(current_price, 2),
            'optimal_price': round(optimal_price, 2),
            'price_change_pct': round(price_change_pct, 2),
            'current_demand': round(current_demand, 2),
            'optimal_demand': round(optimal_demand, 2),
            'demand_change_pct': round(demand_change_pct, 2),
            'current_revenue': round(current_revenue, 2),
            'optimal_revenue': round(optimal_revenue, 2),
            'revenue_change_pct': round(revenue_change_pct, 2),
            'elasticity': round(elasticity, 3),
            'method': self.method,
            'objective': self.objective,
            'constraints_applied': constraints_applied,
            'seasonality_factor': round(seasonality_factor, 3),
            'promotion_lift': round(promotion_lift, 3)
        }

        # Add profit metrics if available
        if cost_per_unit is not None:
            result['cost_per_unit'] = round(cost_per_unit, 2)
            result['current_profit'] = round(current_profit, 2)
            result['optimal_profit'] = round(optimal_profit, 2)
            result['profit_change_pct'] = round(((optimal_profit - current_profit) / current_profit) * 100, 2)
            result['current_margin_pct'] = round(((current_price - cost_per_unit) / current_price) * 100, 2)
            result['optimal_margin_pct'] = round(((optimal_price - cost_per_unit) / optimal_price) * 100, 2)

        # Store in history
        self.optimization_history.append(result)

        logger.info(f"Optimized {product_id}: ${current_price:.2f} -> ${optimal_price:.2f} "
                   f"({price_change_pct:+.1f}%), Revenue: ${current_revenue:.2f} -> ${optimal_revenue:.2f} "
                   f"({revenue_change_pct:+.1f}%)")

        return result

    def optimize_portfolio(
        self,
        products_df: pd.DataFrame,
        constraints: Optional[Dict] = None
    ) -> pd.DataFrame:
        """
        Optimize prices for a portfolio of products simultaneously.

        Args:
            products_df: DataFrame with columns:
                - product_id: Unique identifier
                - current_price: Current price
                - baseline_demand: Baseline demand
                - elasticity: Price elasticity
                - cost_per_unit: (Optional) Cost per unit
                - seasonality_factor: (Optional) Seasonality multiplier
                - promotion_lift: (Optional) Promotion lift
            constraints: Optional dict of constraints to apply to all products

        Returns:
            DataFrame with optimization results for each product

        Example:
            >>> products = pd.DataFrame({
            ...     'product_id': ['A', 'B', 'C'],
            ...     'current_price': [50, 30, 100],
            ...     'baseline_demand': [1000, 2000, 500],
            ...     'elasticity': [-1.5, -0.8, -2.0]
            ... })
            >>> results = optimizer.optimize_portfolio(products)
        """
        logger.info(f"Starting portfolio optimization for {len(products_df)} products")

        results = []
        for _, row in products_df.iterrows():
            try:
                result = self.optimize_single_product(
                    product_id=row['product_id'],
                    current_price=row['current_price'],
                    baseline_demand=row['baseline_demand'],
                    elasticity=row['elasticity'],
                    constraints=constraints,
                    cost_per_unit=row.get('cost_per_unit'),
                    seasonality_factor=row.get('seasonality_factor', 1.0),
                    promotion_lift=row.get('promotion_lift', 1.0)
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to optimize {row['product_id']}: {str(e)}")
                continue

        results_df = pd.DataFrame(results)

        # Calculate portfolio-level metrics
        if len(results_df) > 0:
            total_current_revenue = results_df['current_revenue'].sum()
            total_optimal_revenue = results_df['optimal_revenue'].sum()
            portfolio_revenue_change = ((total_optimal_revenue - total_current_revenue) / total_current_revenue) * 100

            logger.info(f"Portfolio optimization complete: {len(results_df)} products optimized")
            logger.info(f"Total revenue: ${total_current_revenue:,.2f} -> ${total_optimal_revenue:,.2f} "
                       f"({portfolio_revenue_change:+.1f}%)")

        return results_df

    def simulate_scenarios(
        self,
        product_id: str,
        current_price: float,
        baseline_demand: float,
        elasticity: float,
        cost_per_unit: Optional[float] = None,
        scenarios: Optional[List[Dict]] = None
    ) -> pd.DataFrame:
        """
        Simulate multiple pricing scenarios and compare outcomes.

        Args:
            product_id: Product identifier
            current_price: Current price
            baseline_demand: Baseline demand
            elasticity: Price elasticity
            cost_per_unit: Optional cost per unit
            scenarios: List of scenario dicts with 'name' and 'price' keys.
                      If None, uses standard scenarios.

        Returns:
            DataFrame with scenario comparison results

        Example:
            >>> scenarios = [
            ...     {'name': 'Current Price', 'price': 50},
            ...     {'name': '10% Discount', 'price': 45},
            ...     {'name': '20% Premium', 'price': 60}
            ... ]
            >>> results = optimizer.simulate_scenarios('PROD_001', 50, 1000, -1.5, scenarios=scenarios)
        """
        if scenarios is None:
            scenarios = create_standard_scenarios(current_price)

        results = []
        for scenario in scenarios:
            price = scenario['price']
            price_ratio = price / current_price
            demand_multiplier = price_ratio ** elasticity
            predicted_demand = baseline_demand * demand_multiplier
            revenue = price * predicted_demand

            result = {
                'scenario': scenario['name'],
                'price': round(price, 2),
                'price_change_pct': round(((price - current_price) / current_price) * 100, 2),
                'predicted_demand': round(predicted_demand, 2),
                'demand_change_pct': round(((predicted_demand - baseline_demand) / baseline_demand) * 100, 2),
                'revenue': round(revenue, 2),
                'revenue_vs_current_pct': round(((revenue - (current_price * baseline_demand)) /
                                                 (current_price * baseline_demand)) * 100, 2)
            }

            if cost_per_unit is not None:
                profit = (price - cost_per_unit) * predicted_demand
                current_profit = (current_price - cost_per_unit) * baseline_demand
                result['profit'] = round(profit, 2)
                result['profit_vs_current_pct'] = round(((profit - current_profit) / current_profit) * 100, 2)
                result['margin_pct'] = round(((price - cost_per_unit) / price) * 100, 2)

            results.append(result)

        results_df = pd.DataFrame(results)
        logger.info(f"Simulated {len(scenarios)} scenarios for {product_id}")

        return results_df

    def sensitivity_analysis(
        self,
        product_id: str,
        current_price: float,
        baseline_demand: float,
        elasticity: float,
        price_range: Optional[Tuple[float, float]] = None,
        n_points: int = 50,
        cost_per_unit: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Analyze sensitivity of demand, revenue, and profit to price changes.

        Generates a detailed price-response curve by evaluating metrics across
        a range of price points.

        Args:
            product_id: Product identifier
            current_price: Current price
            baseline_demand: Baseline demand at current price
            elasticity: Price elasticity of demand
            price_range: Optional tuple of (min_price, max_price).
                        Defaults to (50% below, 100% above current price)
            n_points: Number of price points to evaluate (default 50)
            cost_per_unit: Optional cost per unit for profit calculations

        Returns:
            DataFrame with columns:
                - price: Price point
                - price_change_pct: Percentage change from current price
                - predicted_demand: Expected demand at this price
                - revenue: Expected revenue
                - profit: Expected profit (if cost provided)
                - margin_pct: Profit margin percentage (if cost provided)

        Example:
            >>> analysis = optimizer.sensitivity_analysis(
            ...     product_id='PROD_001',
            ...     current_price=50.0,
            ...     baseline_demand=1000,
            ...     elasticity=-1.5,
            ...     price_range=(30, 80),
            ...     n_points=50
            ... )
        """
        if price_range is None:
            min_price = current_price * 0.5
            max_price = current_price * 2.0
        else:
            min_price, max_price = price_range

        prices = np.linspace(min_price, max_price, n_points)

        results = []
        for price in prices:
            price_ratio = price / current_price
            demand_multiplier = price_ratio ** elasticity
            predicted_demand = baseline_demand * demand_multiplier
            revenue = price * predicted_demand

            result = {
                'price': round(price, 2),
                'price_change_pct': round(((price - current_price) / current_price) * 100, 2),
                'predicted_demand': round(predicted_demand, 2),
                'revenue': round(revenue, 2)
            }

            if cost_per_unit is not None:
                profit = (price - cost_per_unit) * predicted_demand
                result['profit'] = round(profit, 2)
                result['margin_pct'] = round(((price - cost_per_unit) / price) * 100, 2) if price > 0 else 0

            results.append(result)

        results_df = pd.DataFrame(results)

        # Find optimal points
        max_revenue_idx = results_df['revenue'].idxmax()
        optimal_revenue_price = results_df.loc[max_revenue_idx, 'price']

        logger.info(f"Sensitivity analysis for {product_id}: Optimal revenue at ${optimal_revenue_price:.2f}")

        if cost_per_unit is not None:
            max_profit_idx = results_df['profit'].idxmax()
            optimal_profit_price = results_df.loc[max_profit_idx, 'price']
            logger.info(f"  Optimal profit at ${optimal_profit_price:.2f}")

        return results_df

    def calculate_price_elasticity_segments(
        self,
        products_df: pd.DataFrame,
        elasticity_thresholds: Optional[Dict[str, float]] = None
    ) -> Dict:
        """
        Segment products by price elasticity and provide recommendations.

        Args:
            products_df: DataFrame with 'product_id' and 'elasticity' columns
            elasticity_thresholds: Optional dict with 'elastic' and 'inelastic' thresholds
                                  Default: elastic < -1.5, inelastic > -0.5

        Returns:
            Dict with:
                - segments: DataFrame with product segments
                - summary: Summary statistics by segment
                - recommendations: Strategic recommendations by segment

        Example:
            >>> segments = optimizer.calculate_price_elasticity_segments(products_df)
            >>> print(segments['recommendations']['highly_elastic'])
        """
        if elasticity_thresholds is None:
            elasticity_thresholds = {
                'highly_elastic': -2.0,
                'elastic': -1.5,
                'unit_elastic': -1.0,
                'inelastic': -0.5
            }

        def categorize_elasticity(e):
            if e < elasticity_thresholds['highly_elastic']:
                return 'Highly Elastic'
            elif e < elasticity_thresholds['elastic']:
                return 'Elastic'
            elif e < elasticity_thresholds['unit_elastic']:
                return 'Unit Elastic'
            elif e < elasticity_thresholds['inelastic']:
                return 'Inelastic'
            else:
                return 'Highly Inelastic'

        products_df = products_df.copy()
        products_df['elasticity_segment'] = products_df['elasticity'].apply(categorize_elasticity)

        # Summary by segment
        summary = products_df.groupby('elasticity_segment').agg({
            'product_id': 'count',
            'elasticity': ['mean', 'min', 'max']
        }).round(3)
        summary.columns = ['count', 'avg_elasticity', 'min_elasticity', 'max_elasticity']

        # Recommendations
        recommendations = {
            'Highly Elastic': 'Price reductions drive large demand increases. Consider aggressive discounting and promotions.',
            'Elastic': 'Demand is price-sensitive. Small price cuts can boost revenue significantly.',
            'Unit Elastic': 'Revenue is relatively stable across price changes. Focus on volume or margin based on costs.',
            'Inelastic': 'Demand is price-insensitive. Price increases can boost revenue with minimal demand loss.',
            'Highly Inelastic': 'Strong pricing power. Consider premium pricing strategies to maximize margins.'
        }

        logger.info(f"Segmented {len(products_df)} products into {len(summary)} elasticity segments")

        return {
            'segments': products_df,
            'summary': summary,
            'recommendations': recommendations
        }

    def get_optimization_summary(self) -> Dict:
        """
        Get summary statistics of all optimizations performed.

        Returns:
            Dict with aggregate statistics across optimization history

        Example:
            >>> summary = optimizer.get_optimization_summary()
            >>> print(f"Average revenue lift: {summary['avg_revenue_change_pct']:.1f}%")
        """
        if not self.optimization_history:
            return {'message': 'No optimizations performed yet'}

        df = pd.DataFrame(self.optimization_history)

        summary = {
            'total_optimizations': len(df),
            'avg_price_change_pct': round(df['price_change_pct'].mean(), 2),
            'avg_revenue_change_pct': round(df['revenue_change_pct'].mean(), 2),
            'total_current_revenue': round(df['current_revenue'].sum(), 2),
            'total_optimal_revenue': round(df['optimal_revenue'].sum(), 2),
            'total_revenue_gain': round((df['optimal_revenue'] - df['current_revenue']).sum(), 2),
            'products_optimized': df['product_id'].nunique(),
            'method_used': self.method,
            'objective_used': self.objective
        }

        if 'profit_change_pct' in df.columns:
            summary['avg_profit_change_pct'] = round(df['profit_change_pct'].mean(), 2)
            summary['total_current_profit'] = round(df['current_profit'].sum(), 2)
            summary['total_optimal_profit'] = round(df['optimal_profit'].sum(), 2)
            summary['total_profit_gain'] = round((df['optimal_profit'] - df['current_profit']).sum(), 2)

        return summary

    def _gradient_descent(
        self,
        objective_func,
        initial_price: float,
        bounds: Tuple[float, float],
        learning_rate: float = 0.1,
        max_iterations: int = 1000,
        tolerance: float = 1e-6
    ) -> float:
        """
        Custom gradient descent implementation for price optimization.

        Args:
            objective_func: Function to minimize
            initial_price: Starting price
            bounds: (min_price, max_price) tuple
            learning_rate: Step size for gradient descent
            max_iterations: Maximum number of iterations
            tolerance: Convergence tolerance

        Returns:
            Optimal price found by gradient descent
        """
        price = initial_price
        min_price, max_price = bounds

        for iteration in range(max_iterations):
            # Numerical gradient (finite difference)
            epsilon = 0.01
            grad = (objective_func(price + epsilon) - objective_func(price - epsilon)) / (2 * epsilon)

            # Update price
            new_price = price - learning_rate * grad

            # Apply bounds
            new_price = np.clip(new_price, min_price, max_price)

            # Check convergence
            if abs(new_price - price) < tolerance:
                logger.debug(f"Gradient descent converged in {iteration + 1} iterations")
                break

            price = new_price

        return price

    def _create_no_change_result(
        self,
        product_id: str,
        current_price: float,
        baseline_demand: float,
        elasticity: float,
        seasonality_factor: float,
        promotion_lift: float,
        cost_per_unit: Optional[float]
    ) -> Dict:
        """
        Create a result dict when no price change is recommended.

        Used when constraints conflict or optimal price equals current price.
        """
        current_demand = baseline_demand * seasonality_factor * promotion_lift
        current_revenue = current_price * current_demand

        result = {
            'product_id': product_id,
            'timestamp': datetime.now().isoformat(),
            'current_price': round(current_price, 2),
            'optimal_price': round(current_price, 2),
            'price_change_pct': 0.0,
            'current_demand': round(current_demand, 2),
            'optimal_demand': round(current_demand, 2),
            'demand_change_pct': 0.0,
            'current_revenue': round(current_revenue, 2),
            'optimal_revenue': round(current_revenue, 2),
            'revenue_change_pct': 0.0,
            'elasticity': round(elasticity, 3),
            'method': self.method,
            'objective': self.objective,
            'constraints_applied': ['no_change'],
            'seasonality_factor': round(seasonality_factor, 3),
            'promotion_lift': round(promotion_lift, 3)
        }

        if cost_per_unit is not None:
            current_profit = (current_price - cost_per_unit) * current_demand
            result['cost_per_unit'] = round(cost_per_unit, 2)
            result['current_profit'] = round(current_profit, 2)
            result['optimal_profit'] = round(current_profit, 2)
            result['profit_change_pct'] = 0.0
            result['current_margin_pct'] = round(((current_price - cost_per_unit) / current_price) * 100, 2)
            result['optimal_margin_pct'] = round(((current_price - cost_per_unit) / current_price) * 100, 2)

        return result


def create_standard_scenarios(current_price: float) -> List[Dict]:
    """
    Create a standard set of pricing scenarios for comparison.

    Args:
        current_price: Current price to base scenarios on

    Returns:
        List of scenario dicts with 'name' and 'price' keys
    """
    return [
        {'name': 'Current Price', 'price': current_price},
        {'name': '20% Discount', 'price': current_price * 0.80},
        {'name': '10% Discount', 'price': current_price * 0.90},
        {'name': '5% Discount', 'price': current_price * 0.95},
        {'name': '5% Increase', 'price': current_price * 1.05},
        {'name': '10% Increase', 'price': current_price * 1.10},
        {'name': '20% Increase', 'price': current_price * 1.20},
        {'name': '50% Discount', 'price': current_price * 0.50},
        {'name': '100% Premium', 'price': current_price * 2.00}
    ]

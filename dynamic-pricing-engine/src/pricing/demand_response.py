"""
Demand response modeling module

Forecasts demand at different price points using elasticity estimates,
seasonality, and promotional effects.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging
from scipy import stats

from .elasticity import ElasticityAnalyzer

logger = logging.getLogger(__name__)


class DemandResponseModel:
    """
    Predict demand changes based on price adjustments.

    Uses price elasticity estimates to forecast how demand will respond
    to price changes, incorporating:
    - Price elasticity (from elasticity analyzer)
    - Baseline demand (historical average or forecast)
    - Seasonality effects (day of week, month, holidays)
    - Promotional impacts
    - Competitor pricing effects (optional)

    The core model is:
        Q_new = Q_base * (P_new / P_base) ^ elasticity * seasonality * promotion

    where:
        Q_new = predicted demand at new price
        Q_base = baseline demand
        P_new = proposed new price
        P_base = current/baseline price
        elasticity = price elasticity of demand
    """

    def __init__(
        self,
        elasticity_analyzer: Optional[ElasticityAnalyzer] = None,
        use_confidence_intervals: bool = True,
        confidence_level: float = 0.95
    ):
        """
        Initialize demand response model.

        Args:
            elasticity_analyzer: Pre-configured ElasticityAnalyzer instance
            use_confidence_intervals: Whether to compute prediction intervals
            confidence_level: Confidence level for prediction intervals (default 95%)
        """
        self.elasticity_analyzer = elasticity_analyzer or ElasticityAnalyzer()
        self.use_confidence_intervals = use_confidence_intervals
        self.confidence_level = confidence_level
        self.z_score = stats.norm.ppf((1 + confidence_level) / 2)

        # Cache for elasticity estimates
        self.elasticity_cache = {}

        # Seasonality patterns (learned from data)
        self.seasonality_patterns = {}

        # Promotional lift factors
        self.promotion_lifts = {}

        logger.info("Initialized DemandResponseModel with %s%% confidence intervals", confidence_level*100)

    def predict_demand_at_price(
        self,
        product_id: str,
        new_price: float,
        baseline_demand: float,
        current_price: float,
        elasticity: Optional[float] = None,
        date: Optional[datetime] = None,
        is_promotion: bool = False,
        promotion_type: Optional[str] = None
    ) -> Dict:
        """
        Predict demand at a new price point.

        Args:
            product_id: Product identifier
            new_price: Proposed new price
            baseline_demand: Current/baseline demand level
            current_price: Current/baseline price
            elasticity: Price elasticity (if None, uses cached value)
            date: Date for seasonality adjustment
            is_promotion: Whether this is a promotional period
            promotion_type: Type of promotion ('BOGO', 'discount', 'bundle', etc.)

        Returns:
            Dictionary with predicted demand, change %, confidence intervals
        """
        # Get elasticity estimate
        if elasticity is None:
            if product_id in self.elasticity_cache:
                elasticity = self.elasticity_cache[product_id]['elasticity']
            else:
                logger.warning("No elasticity estimate for %s, using default -1.0", product_id)
                elasticity = -1.0

        # Validate inputs
        if new_price <= 0 or current_price <= 0 or baseline_demand < 0:
            raise ValueError("Prices must be positive and demand must be non-negative")

        # Calculate price ratio
        price_ratio = new_price / current_price
        price_change_pct = (price_ratio - 1) * 100

        # Core elasticity-based demand adjustment
        # Q_new = Q_base * (P_new / P_base) ^ elasticity
        demand_multiplier = price_ratio ** elasticity
        predicted_demand = baseline_demand * demand_multiplier

        # Apply seasonality adjustment if date provided
        seasonality_factor = 1.0
        if date is not None and product_id in self.seasonality_patterns:
            seasonality_factor = self._get_seasonality_factor(product_id, date)
            predicted_demand *= seasonality_factor

        # Apply promotional lift if applicable
        promotion_lift = 1.0
        if is_promotion:
            promotion_lift = self._get_promotion_lift(product_id, promotion_type)
            predicted_demand *= promotion_lift

        # Calculate demand change
        demand_change = predicted_demand - baseline_demand
        demand_change_pct = (demand_change / baseline_demand * 100) if baseline_demand > 0 else 0

        # Calculate confidence intervals if requested
        lower_bound, upper_bound = None, None
        if self.use_confidence_intervals:
            # Standard error increases with price change magnitude
            std_error = baseline_demand * 0.15 * abs(price_change_pct) / 10  # Heuristic
            margin_of_error = self.z_score * std_error
            lower_bound = max(0, predicted_demand - margin_of_error)
            upper_bound = predicted_demand + margin_of_error

        # Calculate revenue impact
        revenue_base = baseline_demand * current_price
        revenue_new = predicted_demand * new_price
        revenue_change = revenue_new - revenue_base
        revenue_change_pct = (revenue_change / revenue_base * 100) if revenue_base > 0 else 0

        return {
            'product_id': product_id,
            'baseline_demand': baseline_demand,
            'predicted_demand': predicted_demand,
            'demand_change': demand_change,
            'demand_change_pct': demand_change_pct,
            'current_price': current_price,
            'new_price': new_price,
            'price_change_pct': price_change_pct,
            'elasticity': elasticity,
            'demand_multiplier': demand_multiplier,
            'seasonality_factor': seasonality_factor,
            'promotion_lift': promotion_lift,
            'revenue_base': revenue_base,
            'revenue_new': revenue_new,
            'revenue_change': revenue_change,
            'revenue_change_pct': revenue_change_pct,
            'confidence_lower': lower_bound,
            'confidence_upper': upper_bound,
            'confidence_level': self.confidence_level if self.use_confidence_intervals else None
        }

    def predict_demand_curve(
        self,
        product_id: str,
        baseline_demand: float,
        current_price: float,
        price_range: Optional[Tuple[float, float]] = None,
        num_points: int = 20,
        elasticity: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Generate a demand curve across a range of prices.

        Args:
            product_id: Product identifier
            baseline_demand: Current/baseline demand level
            current_price: Current/baseline price
            price_range: (min_price, max_price) tuple. If None, uses ±30% of current
            num_points: Number of price points to evaluate
            elasticity: Price elasticity (if None, uses cached value)

        Returns:
            DataFrame with columns: price, demand, revenue, elasticity
        """
        # Define price range
        if price_range is None:
            min_price = current_price * 0.70  # -30%
            max_price = current_price * 1.30  # +30%
        else:
            min_price, max_price = price_range

        # Generate price points
        prices = np.linspace(min_price, max_price, num_points)

        # Predict demand at each price point
        results = []
        for price in prices:
            prediction = self.predict_demand_at_price(
                product_id=product_id,
                new_price=price,
                baseline_demand=baseline_demand,
                current_price=current_price,
                elasticity=elasticity
            )
            results.append({
                'price': price,
                'demand': prediction['predicted_demand'],
                'revenue': prediction['revenue_new'],
                'demand_change_pct': prediction['demand_change_pct'],
                'revenue_change_pct': prediction['revenue_change_pct'],
                'elasticity': prediction['elasticity'],
                'confidence_lower': prediction['confidence_lower'],
                'confidence_upper': prediction['confidence_upper']
            })

        df = pd.DataFrame(results)

        # Find optimal price (maximum revenue)
        optimal_idx = df['revenue'].idxmax()
        df['is_optimal'] = False
        df.loc[optimal_idx, 'is_optimal'] = True

        return df

    def predict_bulk(
        self,
        predictions_df: pd.DataFrame,
        baseline_demand_col: str = 'baseline_demand',
        current_price_col: str = 'current_price',
        new_price_col: str = 'new_price',
        product_id_col: str = 'product_id',
        elasticity_col: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Make bulk demand predictions for multiple products.

        Args:
            predictions_df: DataFrame with columns for product_id, prices, baseline demand
            baseline_demand_col: Column name for baseline demand
            current_price_col: Column name for current price
            new_price_col: Column name for new/proposed price
            product_id_col: Column name for product identifier
            elasticity_col: Column name for elasticity (optional)

        Returns:
            DataFrame with prediction results
        """
        results = []

        for _idx, row in predictions_df.iterrows():
            try:
                elasticity = row[elasticity_col] if elasticity_col and elasticity_col in row else None

                prediction = self.predict_demand_at_price(
                    product_id=row[product_id_col],
                    new_price=row[new_price_col],
                    baseline_demand=row[baseline_demand_col],
                    current_price=row[current_price_col],
                    elasticity=elasticity
                )
                results.append(prediction)

            except (ValueError, KeyError, TypeError) as e:
                logger.error("Error predicting for %s: %s", row[product_id_col], str(e))
                continue

        return pd.DataFrame(results)

    def cache_elasticity(
        self,
        product_id: str,
        elasticity: float,
        metadata: Optional[Dict] = None
    ):
        """
        Cache elasticity estimate for a product.

        Args:
            product_id: Product identifier
            elasticity: Elasticity value
            metadata: Additional information (method, r_squared, etc.)
        """
        self.elasticity_cache[product_id] = {
            'elasticity': elasticity,
            'metadata': metadata or {},
            'cached_at': datetime.now()
        }
        logger.debug("Cached elasticity %.3f for %s", elasticity, product_id)

    def load_elasticity_from_analyzer(
        self,
        price_data: pd.DataFrame,
        sales_data: pd.DataFrame,
        product_id_col: str = 'product_id',
        price_col: str = 'sell_price',
        sales_col: str = 'sales'
    ):
        """
        Calculate and cache elasticities for all products in dataset.

        Args:
            price_data: DataFrame with price data
            sales_data: DataFrame with sales data
            product_id_col: Column name for product identifier
            price_col: Column name for price
            sales_col: Column name for sales/demand
        """
        # Merge price and sales data
        data = pd.merge(price_data, sales_data, on=[product_id_col], how='inner')

        # Calculate elasticity for each product
        products = data[product_id_col].unique()
        logger.info("Calculating elasticity for %d products", len(products))

        for product_id in products:
            product_data = data[data[product_id_col] == product_id]

            result = self.elasticity_analyzer.calculate_own_price_elasticity(
                product_id=product_id,
                price_series=product_data[price_col],
                sales_series=product_data[sales_col]
            )

            if result['valid']:
                self.cache_elasticity(
                    product_id=product_id,
                    elasticity=result['elasticity'],
                    metadata={
                        'method': result['method'],
                        'r_squared': result['r_squared'],
                        'observations': result['observations']
                    }
                )

        logger.info("Cached elasticity for %d products", len(self.elasticity_cache))

    def learn_seasonality(
        self,
        data: pd.DataFrame,
        product_id_col: str = 'product_id',
        date_col: str = 'date',
        sales_col: str = 'sales'
    ):
        """
        Learn seasonality patterns from historical data.

        Args:
            data: DataFrame with sales history
            product_id_col: Column name for product identifier
            date_col: Column name for date
            sales_col: Column name for sales
        """
        logger.info("Learning seasonality patterns...")

        # Ensure date column is datetime
        data[date_col] = pd.to_datetime(data[date_col])

        products = data[product_id_col].unique()

        for product_id in products:
            product_data = data[data[product_id_col] == product_id].copy()

            if len(product_data) < 90:  # Need at least 3 months
                continue

            # Extract temporal features
            product_data['dayofweek'] = product_data[date_col].dt.dayofweek
            product_data['month'] = product_data[date_col].dt.month
            product_data['is_weekend'] = product_data['dayofweek'].isin([5, 6]).astype(int)

            # Calculate average sales by day of week and month
            overall_mean = product_data[sales_col].mean()

            dow_pattern = (product_data.groupby('dayofweek')[sales_col].mean() / overall_mean).to_dict()
            month_pattern = (product_data.groupby('month')[sales_col].mean() / overall_mean).to_dict()
            weekend_lift = product_data[product_data['is_weekend'] == 1][sales_col].mean() / overall_mean

            self.seasonality_patterns[product_id] = {
                'day_of_week': dow_pattern,
                'month': month_pattern,
                'weekend_lift': weekend_lift,
                'overall_mean': overall_mean
            }

        logger.info("Learned seasonality for %d products", len(self.seasonality_patterns))

    def set_promotion_lift(
        self,
        product_id: str,
        promotion_type: str,
        lift_factor: float
    ):
        """
        Set promotional lift factor for a product.

        Args:
            product_id: Product identifier
            promotion_type: Type of promotion
            lift_factor: Multiplicative lift factor (e.g., 1.3 = 30% increase)
        """
        if product_id not in self.promotion_lifts:
            self.promotion_lifts[product_id] = {}

        self.promotion_lifts[product_id][promotion_type] = lift_factor
        logger.debug("Set %s lift to %.2fx for %s", promotion_type, lift_factor, product_id)

    def _get_seasonality_factor(self, product_id: str, date: datetime) -> float:
        """Get seasonality adjustment factor for a given date."""
        if product_id not in self.seasonality_patterns:
            return 1.0

        patterns = self.seasonality_patterns[product_id]

        # Day of week effect
        dow_factor = patterns['day_of_week'].get(date.weekday(), 1.0)

        # Month effect
        month_factor = patterns['month'].get(date.month, 1.0)

        # Weekend effect (overrides day of week if more significant)
        if date.weekday() in [5, 6]:  # Weekend
            dow_factor = patterns.get('weekend_lift', dow_factor)

        # Combine effects (weighted average)
        seasonality = 0.6 * dow_factor + 0.4 * month_factor

        return seasonality

    def _get_promotion_lift(self, product_id: str, promotion_type: Optional[str]) -> float:
        """Get promotional lift factor."""
        if not promotion_type:
            return 1.0

        if product_id in self.promotion_lifts:
            return self.promotion_lifts[product_id].get(promotion_type, 1.0)

        # Default promotion lifts if not learned
        default_lifts = {
            'BOGO': 1.50,  # 50% lift
            'discount': 1.30,  # 30% lift
            'bundle': 1.25,  # 25% lift
            'clearance': 1.40  # 40% lift
        }

        return default_lifts.get(promotion_type, 1.0)

    def get_elasticity_summary(self) -> pd.DataFrame:
        """
        Get summary of cached elasticity estimates.

        Returns:
            DataFrame with elasticity statistics by product
        """
        if not self.elasticity_cache:
            return pd.DataFrame()

        summary = []
        for product_id, data in self.elasticity_cache.items():
            summary.append({
                'product_id': product_id,
                'elasticity': data['elasticity'],
                'method': data['metadata'].get('method'),
                'r_squared': data['metadata'].get('r_squared'),
                'observations': data['metadata'].get('observations'),
                'cached_at': data['cached_at']
            })

        return pd.DataFrame(summary)

    def simulate_price_scenarios(
        self,
        product_id: str,
        baseline_demand: float,
        current_price: float,
        scenarios: List[Dict],
        elasticity: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Simulate multiple pricing scenarios.

        Args:
            product_id: Product identifier
            baseline_demand: Current/baseline demand
            current_price: Current price
            scenarios: List of scenario dicts with 'name', 'price_change_pct', etc.
            elasticity: Price elasticity

        Returns:
            DataFrame comparing scenario outcomes
        """
        results = []

        for scenario in scenarios:
            price_change_pct = scenario.get('price_change_pct', 0)
            new_price = current_price * (1 + price_change_pct / 100)

            prediction = self.predict_demand_at_price(
                product_id=product_id,
                new_price=new_price,
                baseline_demand=baseline_demand,
                current_price=current_price,
                elasticity=elasticity,
                is_promotion=scenario.get('is_promotion', False),
                promotion_type=scenario.get('promotion_type')
            )

            results.append({
                'scenario_name': scenario.get('name', f'{price_change_pct:+.0f}% price change'),
                **prediction
            })

        return pd.DataFrame(results)


def create_standard_scenarios() -> List[Dict]:
    """
    Create standard price change scenarios for testing.

    Returns:
        List of scenario dictionaries
    """
    return [
        {'name': 'Current Price', 'price_change_pct': 0},
        {'name': 'Small Decrease (-5%)', 'price_change_pct': -5},
        {'name': 'Medium Decrease (-10%)', 'price_change_pct': -10},
        {'name': 'Large Decrease (-20%)', 'price_change_pct': -20},
        {'name': 'Small Increase (+5%)', 'price_change_pct': 5},
        {'name': 'Medium Increase (+10%)', 'price_change_pct': 10},
        {'name': 'Large Increase (+20%)', 'price_change_pct': 20},
        {'name': 'Promotional Discount (-15%)', 'price_change_pct': -15, 'is_promotion': True, 'promotion_type': 'discount'},
        {'name': 'Clearance (-30%)', 'price_change_pct': -30, 'is_promotion': True, 'promotion_type': 'clearance'},
    ]

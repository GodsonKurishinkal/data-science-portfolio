"""
Price elasticity analysis module

Calculates price elasticity of demand using econometric methods.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple, List
from sklearn.linear_model import LinearRegression
from scipy import stats
import logging

logger = logging.getLogger(__name__)


class ElasticityAnalyzer:
    """
    Calculate and analyze price elasticity of demand.

    Price elasticity measures how sensitive demand is to price changes:
    - Elastic (|e| > 1): Demand highly sensitive to price
    - Unit elastic (|e| = 1): Proportional response
    - Inelastic (|e| < 1): Demand relatively insensitive to price

    Methods:
        - log-log: ln(Q) = a + b*ln(P) → elasticity = b
        - arc: (ΔQ/Q_avg) / (ΔP/P_avg)
        - point: (dQ/dP) * (P/Q)
    """

    def __init__(self, method: str = 'log-log', min_observations: int = 30):
        """
        Initialize elasticity analyzer.

        Args:
            method: Elasticity calculation method ('log-log', 'arc', 'point')
            min_observations: Minimum data points required for reliable elasticity
        """
        self.method = method
        self.min_observations = min_observations
        logger.info(f"Initialized ElasticityAnalyzer with method: {method}")

    def calculate_own_price_elasticity(
        self,
        product_id: str,
        price_series: pd.Series,
        sales_series: pd.Series,
        method: Optional[str] = None
    ) -> Dict:
        """
        Calculate own-price elasticity for a product.

        Args:
            product_id: Product identifier
            price_series: Series of prices
            sales_series: Series of sales quantities
            method: Override default elasticity method

        Returns:
            Dictionary with elasticity, statistics, and diagnostics
        """
        method = method or self.method

        # Remove NaN and zero values
        mask = (price_series > 0) & (sales_series >= 0) & price_series.notna() & sales_series.notna()
        prices = price_series[mask]
        sales = sales_series[mask]

        if len(prices) < self.min_observations:
            logger.warning(f"Product {product_id}: Insufficient data ({len(prices)} < {self.min_observations})")
            return {
                'product_id': product_id,
                'elasticity': np.nan,
                'method': method,
                'observations': len(prices),
                'r_squared': np.nan,
                'p_value': np.nan,
                'std_error': np.nan,
                'valid': False,
                'message': 'Insufficient observations'
            }

        # Calculate based on method
        if method == 'log-log':
            result = self._elasticity_log_log(prices, sales)
        elif method == 'arc':
            result = self._elasticity_arc(prices, sales)
        elif method == 'point':
            result = self._elasticity_point(prices, sales)
        else:
            raise ValueError(f"Unknown method: {method}")

        result['product_id'] = product_id
        result['method'] = method
        result['observations'] = len(prices)

        return result

    def _elasticity_log_log(
        self,
        prices: pd.Series,
        sales: pd.Series
    ) -> Dict:
        """
        Calculate elasticity using log-log regression.

        Model: ln(Q) = a + b*ln(P)
        Elasticity = b (constant across all price levels)

        Args:
            prices: Price series
            sales: Sales series

        Returns:
            Dictionary with elasticity and statistics
        """
        # Add small constant to handle zero sales
        sales_adj = sales + 0.1

        # Log transform
        log_prices = np.log(prices)
        log_sales = np.log(sales_adj)

        # Remove inf values
        mask = np.isfinite(log_prices) & np.isfinite(log_sales)
        X = log_prices[mask].values.reshape(-1, 1)
        y = log_sales[mask].values

        if len(X) < 10:
            return {
                'elasticity': np.nan,
                'r_squared': np.nan,
                'p_value': np.nan,
                'std_error': np.nan,
                'valid': False,
                'message': 'Insufficient valid data after transformation'
            }

        # Fit regression
        model = LinearRegression()
        model.fit(X, y)

        # Predictions and statistics
        y_pred = model.predict(X)
        residuals = y - y_pred

        # R-squared
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        # Standard error and p-value
        n = len(X)
        dof = n - 2
        mse = ss_res / dof if dof > 0 else np.nan

        # Standard error of coefficient
        x_mean = np.mean(X)
        se_squared = mse / np.sum((X.flatten() - x_mean)**2) if np.sum((X.flatten() - x_mean)**2) > 0 else np.nan
        std_error = np.sqrt(se_squared) if not np.isnan(se_squared) else np.nan

        # T-statistic and p-value
        elasticity = model.coef_[0]
        t_stat = elasticity / std_error if not np.isnan(std_error) and std_error > 0 else np.nan
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), dof)) if not np.isnan(t_stat) else np.nan

        return {
            'elasticity': elasticity,
            'r_squared': r_squared,
            'p_value': p_value,
            'std_error': std_error,
            'intercept': model.intercept_,
            'valid': True,
            'message': 'Success'
        }

    def _elasticity_arc(
        self,
        prices: pd.Series,
        sales: pd.Series
    ) -> Dict:
        """
        Calculate elasticity using arc elasticity formula.

        Arc elasticity = (ΔQ/Q_avg) / (ΔP/P_avg)
        Average of all consecutive price-quantity changes.

        Args:
            prices: Price series
            sales: Sales series

        Returns:
            Dictionary with elasticity and statistics
        """
        # Calculate changes
        price_changes = prices.diff()
        sales_changes = sales.diff()

        # Average prices and sales for each pair
        price_avg = (prices + prices.shift(1)) / 2
        sales_avg = (sales + sales.shift(1)) / 2

        # Calculate arc elasticities
        mask = (price_changes != 0) & (price_avg > 0) & (sales_avg > 0)
        mask = mask & price_changes.notna() & sales_changes.notna()

        if mask.sum() < 5:
            return {
                'elasticity': np.nan,
                'r_squared': np.nan,
                'p_value': np.nan,
                'std_error': np.nan,
                'valid': False,
                'message': 'Insufficient price variation'
            }

        arc_elasticities = (sales_changes[mask] / sales_avg[mask]) / (price_changes[mask] / price_avg[mask])

        # Remove outliers (beyond 3 std)
        elasticity_mean = arc_elasticities.mean()
        elasticity_std = arc_elasticities.std()
        outlier_mask = np.abs(arc_elasticities - elasticity_mean) <= 3 * elasticity_std

        clean_elasticities = arc_elasticities[outlier_mask]

        if len(clean_elasticities) < 3:
            return {
                'elasticity': np.nan,
                'r_squared': np.nan,
                'p_value': np.nan,
                'std_error': np.nan,
                'valid': False,
                'message': 'Too many outliers'
            }

        elasticity = clean_elasticities.mean()
        std_error = clean_elasticities.std() / np.sqrt(len(clean_elasticities))

        return {
            'elasticity': elasticity,
            'r_squared': np.nan,  # Not applicable for arc method
            'p_value': np.nan,
            'std_error': std_error,
            'valid': True,
            'message': 'Success'
        }

    def _elasticity_point(
        self,
        prices: pd.Series,
        sales: pd.Series
    ) -> Dict:
        """
        Calculate point elasticity.

        Point elasticity = (dQ/dP) * (P/Q)
        Uses linear regression to estimate dQ/dP, then calculates at mean price.

        Args:
            prices: Price series
            sales: Sales series

        Returns:
            Dictionary with elasticity and statistics
        """
        # Fit linear model: Q = a + b*P
        X = prices.values.reshape(-1, 1)
        y = sales.values

        model = LinearRegression()
        model.fit(X, y)

        # dQ/dP is the slope
        dq_dp = model.coef_[0]

        # Calculate at mean price and mean quantity
        mean_price = prices.mean()
        mean_sales = sales.mean()

        if mean_sales == 0:
            return {
                'elasticity': np.nan,
                'r_squared': np.nan,
                'p_value': np.nan,
                'std_error': np.nan,
                'valid': False,
                'message': 'Zero mean sales'
            }

        elasticity = dq_dp * (mean_price / mean_sales)

        # R-squared
        y_pred = model.predict(X)
        ss_res = np.sum((y - y_pred)**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        return {
            'elasticity': elasticity,
            'r_squared': r_squared,
            'p_value': np.nan,
            'std_error': np.nan,
            'valid': True,
            'message': 'Success'
        }

    def calculate_elasticities_batch(
        self,
        df: pd.DataFrame,
        group_cols: List[str] = ['store_id', 'item_id']
    ) -> pd.DataFrame:
        """
        Calculate elasticities for multiple products in batch.

        Args:
            df: DataFrame with price and sales data
            group_cols: Columns to group by (identifies unique products)

        Returns:
            DataFrame with elasticity results for each product
        """
        logger.info(f"Calculating elasticities for {df.groupby(group_cols).ngroups} products")

        results = []

        for name, group in df.groupby(group_cols):
            if isinstance(name, tuple):
                product_id = '_'.join(str(x) for x in name)
            else:
                product_id = str(name)

            result = self.calculate_own_price_elasticity(
                product_id=product_id,
                price_series=group['sell_price'],
                sales_series=group['sales']
            )

            # Add group identifiers
            for i, col in enumerate(group_cols):
                result[col] = name[i] if isinstance(name, tuple) else name

            results.append(result)

        results_df = pd.DataFrame(results)
        logger.info(f"Calculated {len(results_df)} elasticities, {results_df['valid'].sum()} valid")

        return results_df

    def calculate_cross_elasticity(
        self,
        product_a_id: str,
        product_b_id: str,
        data: pd.DataFrame
    ) -> Dict:
        """
        Calculate cross-price elasticity between two products.

        Cross elasticity = % change in demand for A / % change in price of B
        Positive: Substitutes, Negative: Complements

        Args:
            product_a_id: First product identifier
            product_b_id: Second product identifier
            data: DataFrame with both products' data

        Returns:
            Dictionary with cross-elasticity and interpretation
        """
        logger.info(f"Calculating cross-elasticity: {product_a_id} vs {product_b_id}")

        # Get data for both products
        mask_a = data['item_id'] == product_a_id
        mask_b = data['item_id'] == product_b_id

        data_a = data[mask_a].sort_values('date')
        data_b = data[mask_b].sort_values('date')

        # Merge on date
        merged = data_a[['date', 'sales']].merge(
            data_b[['date', 'sell_price']],
            on='date',
            suffixes=('_a', '_b')
        )

        if len(merged) < self.min_observations:
            return {
                'product_a': product_a_id,
                'product_b': product_b_id,
                'cross_elasticity': np.nan,
                'valid': False,
                'relationship': 'Unknown',
                'message': 'Insufficient overlapping data'
            }

        # Log-log regression: ln(Q_a) = a + b*ln(P_b)
        sales_a = merged['sales'] + 0.1
        price_b = merged['sell_price']

        log_sales_a = np.log(sales_a)
        log_price_b = np.log(price_b)

        mask = np.isfinite(log_sales_a) & np.isfinite(log_price_b)
        X = log_price_b[mask].values.reshape(-1, 1)
        y = log_sales_a[mask].values

        if len(X) < 10:
            return {
                'product_a': product_a_id,
                'product_b': product_b_id,
                'cross_elasticity': np.nan,
                'valid': False,
                'relationship': 'Unknown',
                'message': 'Insufficient valid data'
            }

        model = LinearRegression()
        model.fit(X, y)

        cross_elasticity = model.coef_[0]

        # Determine relationship
        if abs(cross_elasticity) < 0.1:
            relationship = 'Independent'
        elif cross_elasticity > 0:
            relationship = 'Substitutes'
        else:
            relationship = 'Complements'

        return {
            'product_a': product_a_id,
            'product_b': product_b_id,
            'cross_elasticity': cross_elasticity,
            'valid': True,
            'relationship': relationship,
            'message': 'Success'
        }

    def segment_by_elasticity(
        self,
        elasticities: pd.DataFrame,
        elastic_threshold: float = -1.0
    ) -> pd.DataFrame:
        """
        Classify products by elasticity category.

        Categories:
        - Highly elastic: |e| > 2.0
        - Elastic: 1.0 < |e| <= 2.0
        - Unit elastic: 0.9 <= |e| <= 1.1
        - Inelastic: 0.5 < |e| < 0.9
        - Highly inelastic: |e| <= 0.5

        Args:
            elasticities: DataFrame with elasticity values
            elastic_threshold: Threshold for elastic/inelastic split

        Returns:
            DataFrame with elasticity categories added
        """
        logger.info("Segmenting products by elasticity")

        df = elasticities.copy()

        # Take absolute value for classification
        df['elasticity_abs'] = df['elasticity'].abs()

        def classify_elasticity(e_abs):
            if pd.isna(e_abs):
                return 'Unknown'
            elif e_abs > 2.0:
                return 'Highly Elastic'
            elif e_abs > 1.0:
                return 'Elastic'
            elif e_abs >= 0.9 and e_abs <= 1.1:
                return 'Unit Elastic'
            elif e_abs > 0.5:
                return 'Inelastic'
            else:
                return 'Highly Inelastic'

        df['elasticity_category'] = df['elasticity_abs'].apply(classify_elasticity)

        # Pricing recommendations
        def pricing_recommendation(row):
            if pd.isna(row['elasticity']):
                return 'Need more data'

            category = row['elasticity_category']

            if category == 'Highly Elastic':
                return 'Lower prices to boost volume significantly'
            elif category == 'Elastic':
                return 'Price reductions drive higher revenue'
            elif category == 'Unit Elastic':
                return 'Revenue unchanged by price changes'
            elif category == 'Inelastic':
                return 'Raise prices to increase revenue'
            else:  # Highly Inelastic
                return 'Significant price increases viable'

        df['pricing_recommendation'] = df.apply(pricing_recommendation, axis=1)

        # Distribution summary
        category_counts = df['elasticity_category'].value_counts()
        logger.info(f"Elasticity distribution:\n{category_counts}")

        return df

    def get_elasticity_summary(
        self,
        elasticities: pd.DataFrame
    ) -> Dict:
        """
        Get summary statistics for elasticity results.

        Args:
            elasticities: DataFrame with elasticity calculations

        Returns:
            Dictionary with summary statistics
        """
        valid = elasticities[elasticities['valid'] == True]

        if len(valid) == 0:
            return {
                'total_products': len(elasticities),
                'valid_results': 0,
                'mean_elasticity': np.nan,
                'median_elasticity': np.nan,
                'std_elasticity': np.nan
            }

        return {
            'total_products': len(elasticities),
            'valid_results': len(valid),
            'valid_percentage': len(valid) / len(elasticities) * 100,
            'mean_elasticity': valid['elasticity'].mean(),
            'median_elasticity': valid['elasticity'].median(),
            'std_elasticity': valid['elasticity'].std(),
            'min_elasticity': valid['elasticity'].min(),
            'max_elasticity': valid['elasticity'].max(),
            'mean_r_squared': valid['r_squared'].mean(),
            'category_distribution': valid['elasticity_category'].value_counts().to_dict() if 'elasticity_category' in valid.columns else {}
        }

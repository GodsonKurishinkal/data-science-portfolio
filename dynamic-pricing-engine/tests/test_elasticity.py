"""
Tests for elasticity analysis module
"""

import pytest
import pandas as pd
import numpy as np
from src.pricing.elasticity import ElasticityAnalyzer


@pytest.fixture
def sample_elasticity_data():
    """
    Create sample data with known elasticity.

    Generate data where ln(Q) = 10 - 1.5 * ln(P)
    This gives elasticity = -1.5 (elastic)
    """
    np.random.seed(42)
    n = 100

    # Generate prices with variation
    prices = np.random.uniform(5, 15, n)

    # Generate sales with elasticity = -1.5
    log_sales = 10 - 1.5 * np.log(prices) + np.random.normal(0, 0.1, n)
    sales = np.exp(log_sales)

    # Create date range
    dates = pd.date_range('2023-01-01', periods=n, freq='D')

    df = pd.DataFrame({
        'date': dates,
        'store_id': 'CA_1',
        'item_id': 'TEST_ITEM_001',
        'sell_price': prices,
        'sales': sales
    })

    return df


@pytest.fixture
def multi_product_data():
    """Create data for multiple products with different elasticities."""
    np.random.seed(42)
    data = []

    # Product 1: Elastic (e = -1.8)
    prices1 = np.random.uniform(10, 20, 50)
    log_sales1 = 12 - 1.8 * np.log(prices1) + np.random.normal(0, 0.1, 50)
    sales1 = np.exp(log_sales1)
    dates1 = pd.date_range('2023-01-01', periods=50, freq='D')

    for i in range(50):
        data.append({
            'date': dates1[i],
            'store_id': 'CA_1',
            'item_id': 'FOODS_1_001',
            'sell_price': prices1[i],
            'sales': sales1[i]
        })

    # Product 2: Inelastic (e = -0.5)
    prices2 = np.random.uniform(3, 8, 50)
    log_sales2 = 8 - 0.5 * np.log(prices2) + np.random.normal(0, 0.1, 50)
    sales2 = np.exp(log_sales2)
    dates2 = pd.date_range('2023-01-01', periods=50, freq='D')

    for i in range(50):
        data.append({
            'date': dates2[i],
            'store_id': 'CA_1',
            'item_id': 'HOUSEHOLD_1_001',
            'sell_price': prices2[i],
            'sales': sales2[i]
        })

    return pd.DataFrame(data)


class TestElasticityAnalyzer:
    """Test ElasticityAnalyzer class."""

    def test_initialization(self):
        """Test analyzer initialization."""
        analyzer = ElasticityAnalyzer(method='log-log', min_observations=30)

        assert analyzer.method == 'log-log'
        assert analyzer.min_observations == 30

    def test_log_log_elasticity(self, sample_elasticity_data):
        """Test log-log elasticity calculation."""
        analyzer = ElasticityAnalyzer(method='log-log')

        result = analyzer.calculate_own_price_elasticity(
            product_id='TEST_ITEM_001',
            price_series=sample_elasticity_data['sell_price'],
            sales_series=sample_elasticity_data['sales']
        )

        # Check result structure
        assert 'product_id' in result
        assert 'elasticity' in result
        assert 'method' in result
        assert 'valid' in result
        assert 'r_squared' in result

        # Check values
        assert result['product_id'] == 'TEST_ITEM_001'
        assert result['method'] == 'log-log'
        assert result['valid'] is True

        # Elasticity should be close to -1.5 (generated value)
        assert -2.0 < result['elasticity'] < -1.0

        # Good fit expected
        assert result['r_squared'] > 0.8

    def test_arc_elasticity(self, sample_elasticity_data):
        """Test arc elasticity calculation."""
        analyzer = ElasticityAnalyzer(method='arc')

        result = analyzer.calculate_own_price_elasticity(
            product_id='TEST_ITEM_001',
            price_series=sample_elasticity_data['sell_price'],
            sales_series=sample_elasticity_data['sales']
        )

        assert result['valid'] is True
        assert result['method'] == 'arc'
        # Arc elasticity should be negative for normal goods
        assert result['elasticity'] < 0

    def test_point_elasticity(self, sample_elasticity_data):
        """Test point elasticity calculation."""
        analyzer = ElasticityAnalyzer(method='point')

        result = analyzer.calculate_own_price_elasticity(
            product_id='TEST_ITEM_001',
            price_series=sample_elasticity_data['sell_price'],
            sales_series=sample_elasticity_data['sales']
        )

        assert result['valid'] is True
        assert result['method'] == 'point'
        assert result['elasticity'] < 0

    def test_insufficient_data(self):
        """Test handling of insufficient observations."""
        analyzer = ElasticityAnalyzer(min_observations=50)

        # Only 10 observations
        prices = pd.Series([10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
        sales = pd.Series([100, 95, 90, 85, 80, 75, 70, 65, 60, 55])

        result = analyzer.calculate_own_price_elasticity(
            product_id='TEST',
            price_series=prices,
            sales_series=sales
        )

        assert result['valid'] is False
        assert 'Insufficient' in result['message']
        assert np.isnan(result['elasticity'])

    def test_batch_calculation(self, multi_product_data):
        """Test batch elasticity calculation."""
        analyzer = ElasticityAnalyzer(method='log-log')

        results = analyzer.calculate_elasticities_batch(
            df=multi_product_data,
            group_cols=['store_id', 'item_id']
        )

        # Should have 2 products
        assert len(results) == 2

        # Check results structure
        assert 'product_id' in results.columns
        assert 'elasticity' in results.columns
        assert 'valid' in results.columns
        assert 'store_id' in results.columns
        assert 'item_id' in results.columns

        # Both should be valid
        assert results['valid'].sum() == 2

        # Check elasticity values are reasonable
        assert all(results['elasticity'] < 0)  # Negative elasticities
        assert all(results['elasticity'] > -5)  # Not too extreme

    def test_cross_elasticity(self, multi_product_data):
        """Test cross-price elasticity calculation."""
        analyzer = ElasticityAnalyzer()

        result = analyzer.calculate_cross_elasticity(
            product_a_id='FOODS_1_001',
            product_b_id='HOUSEHOLD_1_001',
            data=multi_product_data
        )

        # Check result structure
        assert 'product_a' in result
        assert 'product_b' in result
        assert 'cross_elasticity' in result
        assert 'relationship' in result
        assert 'valid' in result

        # Should be valid with sufficient data
        assert result['valid'] is True

        # Relationship should be classified
        assert result['relationship'] in ['Substitutes', 'Complements', 'Independent']

    def test_segmentation(self, multi_product_data):
        """Test product segmentation by elasticity."""
        analyzer = ElasticityAnalyzer()

        # Calculate elasticities first
        elasticities = analyzer.calculate_elasticities_batch(
            df=multi_product_data,
            group_cols=['store_id', 'item_id']
        )

        # Segment
        segmented = analyzer.segment_by_elasticity(elasticities)

        # Check new columns added
        assert 'elasticity_category' in segmented.columns
        assert 'elasticity_abs' in segmented.columns
        assert 'pricing_recommendation' in segmented.columns

        # Check categories are valid
        valid_categories = [
            'Highly Elastic', 'Elastic', 'Unit Elastic',
            'Inelastic', 'Highly Inelastic', 'Unknown'
        ]
        assert all(segmented['elasticity_category'].isin(valid_categories))

        # Check recommendations exist
        assert all(segmented['pricing_recommendation'].notna())

    def test_elasticity_summary(self, multi_product_data):
        """Test elasticity summary statistics."""
        analyzer = ElasticityAnalyzer()

        elasticities = analyzer.calculate_elasticities_batch(
            df=multi_product_data,
            group_cols=['store_id', 'item_id']
        )

        segmented = analyzer.segment_by_elasticity(elasticities)

        summary = analyzer.get_elasticity_summary(segmented)

        # Check summary keys
        assert 'total_products' in summary
        assert 'valid_results' in summary
        assert 'mean_elasticity' in summary
        assert 'median_elasticity' in summary
        assert 'std_elasticity' in summary

        # Check values
        assert summary['total_products'] == 2
        assert summary['valid_results'] == 2
        assert summary['mean_elasticity'] < 0  # Negative elasticities

    def test_zero_sales_handling(self):
        """Test handling of zero sales values."""
        analyzer = ElasticityAnalyzer()

        # Data with some zero sales
        prices = pd.Series([10, 11, 12, 13, 14] * 10)
        sales = pd.Series([0, 5, 0, 8, 10] * 10)  # Some zeros

        result = analyzer.calculate_own_price_elasticity(
            product_id='TEST',
            price_series=prices,
            sales_series=sales
        )

        # Should handle zeros gracefully (adds small constant)
        assert result['valid'] is True
        assert not np.isnan(result['elasticity'])

    def test_constant_price_handling(self):
        """Test handling of constant prices (no variation)."""
        analyzer = ElasticityAnalyzer()

        # Constant price
        prices = pd.Series([10.0] * 50)
        sales = pd.Series(np.random.uniform(80, 120, 50))

        result = analyzer.calculate_own_price_elasticity(
            product_id='TEST',
            price_series=prices,
            sales_series=sales,
            method='arc'
        )

        # Arc method should fail with no price variation
        assert result['valid'] is False
        assert 'variation' in result['message'].lower()

    def test_invalid_method(self, sample_elasticity_data):
        """Test handling of invalid elasticity method."""
        analyzer = ElasticityAnalyzer(method='invalid_method')

        with pytest.raises(ValueError, match="Unknown method"):
            analyzer.calculate_own_price_elasticity(
                product_id='TEST',
                price_series=sample_elasticity_data['sell_price'],
                sales_series=sample_elasticity_data['sales']
            )

    def test_negative_prices_handling(self):
        """Test handling of negative prices."""
        analyzer = ElasticityAnalyzer()

        # Some negative prices (should be filtered)
        prices = pd.Series([-5, 10, 11, -2, 13, 14, 15] * 10)
        sales = pd.Series([100, 95, 90, 85, 80, 75, 70] * 10)

        result = analyzer.calculate_own_price_elasticity(
            product_id='TEST',
            price_series=prices,
            sales_series=sales
        )

        # Should filter out negative prices and still calculate
        assert result['valid'] is True
        assert result['observations'] < len(prices)  # Some filtered

    def test_elasticity_magnitude_categories(self):
        """Test correct categorization of elasticity magnitudes."""
        analyzer = ElasticityAnalyzer()

        # Create test data with known elasticities
        test_elasticities = pd.DataFrame({
            'product_id': ['P1', 'P2', 'P3', 'P4', 'P5'],
            'elasticity': [-2.5, -1.2, -1.0, -0.7, -0.3],
            'valid': [True, True, True, True, True],
            'r_squared': [0.8, 0.7, 0.6, 0.5, 0.4]
        })

        segmented = analyzer.segment_by_elasticity(test_elasticities)

        # Check correct categorization
        assert segmented.loc[0, 'elasticity_category'] == 'Highly Elastic'  # -2.5
        assert segmented.loc[1, 'elasticity_category'] == 'Elastic'  # -1.2
        assert segmented.loc[2, 'elasticity_category'] == 'Unit Elastic'  # -1.0
        assert segmented.loc[3, 'elasticity_category'] == 'Inelastic'  # -0.7
        assert segmented.loc[4, 'elasticity_category'] == 'Highly Inelastic'  # -0.3

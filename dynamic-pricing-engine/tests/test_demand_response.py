"""
Tests for demand response modeling module
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.pricing.demand_response import (
    DemandResponseModel,
    create_standard_scenarios
)
from src.pricing.elasticity import ElasticityAnalyzer


@pytest.fixture
def sample_product():
    """Sample product data for testing."""
    return {
        'product_id': 'FOODS_3_001',
        'baseline_demand': 100.0,
        'current_price': 5.00,
        'elasticity': -1.2
    }


@pytest.fixture
def sample_price_data():
    """Sample price and sales data."""
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=180, freq='D')

    data = []
    for date in dates:
        # Base price with some variation
        price = 5.0 + np.random.normal(0, 0.2)
        # Demand inversely related to price with elasticity ~-1.2
        demand = 100 * (price / 5.0) ** (-1.2) + np.random.normal(0, 5)
        demand = max(0, demand)

        data.append({
            'product_id': 'FOODS_3_001',
            'date': date,
            'sell_price': price,
            'sales': demand
        })

    return pd.DataFrame(data)


@pytest.fixture
def demand_model():
    """Initialize demand response model."""
    return DemandResponseModel(
        elasticity_analyzer=ElasticityAnalyzer(method='log-log'),
        use_confidence_intervals=True,
        confidence_level=0.95
    )


class TestDemandResponseModel:
    """Test suite for DemandResponseModel."""

    def test_initialization(self):
        """Test model initialization."""
        model = DemandResponseModel()
        assert model is not None
        assert model.elasticity_analyzer is not None
        assert model.use_confidence_intervals is True
        assert model.confidence_level == 0.95
        assert len(model.elasticity_cache) == 0

    def test_predict_demand_at_price_decrease(self, demand_model, sample_product):
        """Test demand prediction for price decrease."""
        result = demand_model.predict_demand_at_price(
            product_id=sample_product['product_id'],
            new_price=4.50,  # -10% price decrease
            baseline_demand=sample_product['baseline_demand'],
            current_price=sample_product['current_price'],
            elasticity=sample_product['elasticity']
        )

        # With elasticity of -1.2, a 10% price decrease should increase demand by ~12%
        assert result['predicted_demand'] > sample_product['baseline_demand']
        assert result['demand_change_pct'] > 0
        assert result['price_change_pct'] < 0
        assert result['elasticity'] == sample_product['elasticity']

        # Check confidence intervals
        assert result['confidence_lower'] is not None
        assert result['confidence_upper'] is not None
        assert result['confidence_lower'] <= result['predicted_demand'] <= result['confidence_upper']

    def test_predict_demand_at_price_increase(self, demand_model, sample_product):
        """Test demand prediction for price increase."""
        result = demand_model.predict_demand_at_price(
            product_id=sample_product['product_id'],
            new_price=5.50,  # +10% price increase
            baseline_demand=sample_product['baseline_demand'],
            current_price=sample_product['current_price'],
            elasticity=sample_product['elasticity']
        )

        # With elasticity of -1.2, a 10% price increase should decrease demand by ~12%
        assert result['predicted_demand'] < sample_product['baseline_demand']
        assert result['demand_change_pct'] < 0
        assert result['price_change_pct'] > 0

    def test_predict_demand_elastic_product(self, demand_model):
        """Test prediction for elastic product (|e| > 1)."""
        result = demand_model.predict_demand_at_price(
            product_id='ELASTIC_001',
            new_price=4.00,  # -20% decrease
            baseline_demand=100.0,
            current_price=5.00,
            elasticity=-2.0  # Highly elastic
        )

        # With elasticity of -2.0, 20% price decrease → ~48.8% demand increase
        assert result['demand_change_pct'] > 40  # Significant increase
        assert result['revenue_change'] > 0  # Revenue should increase

    def test_predict_demand_inelastic_product(self, demand_model):
        """Test prediction for inelastic product (|e| < 1)."""
        result = demand_model.predict_demand_at_price(
            product_id='INELASTIC_001',
            new_price=6.00,  # +20% increase
            baseline_demand=100.0,
            current_price=5.00,
            elasticity=-0.5  # Inelastic
        )

        # With elasticity of -0.5, 20% price increase → ~10% demand decrease
        assert abs(result['demand_change_pct']) < 15  # Small change
        assert result['revenue_change'] > 0  # Revenue should increase (price effect dominates)

    def test_predict_demand_with_seasonality(self, demand_model, sample_product):
        """Test demand prediction with seasonality adjustment."""
        # Set up seasonality pattern
        demand_model.seasonality_patterns[sample_product['product_id']] = {
            'day_of_week': {0: 0.9, 1: 0.95, 2: 1.0, 3: 1.05, 4: 1.1, 5: 1.3, 6: 1.2},
            'month': {i: 1.0 for i in range(1, 13)},
            'weekend_lift': 1.25,
            'overall_mean': 100.0
        }

        # Predict for a Saturday (high demand day)
        saturday = datetime(2024, 11, 16)  # A Saturday
        result = demand_model.predict_demand_at_price(
            product_id=sample_product['product_id'],
            new_price=sample_product['current_price'],
            baseline_demand=sample_product['baseline_demand'],
            current_price=sample_product['current_price'],
            elasticity=sample_product['elasticity'],
            date=saturday
        )

        # Weekend should have higher predicted demand due to seasonality
        assert result['seasonality_factor'] > 1.0
        assert result['predicted_demand'] > sample_product['baseline_demand']

    def test_predict_demand_with_promotion(self, demand_model, sample_product):
        """Test demand prediction with promotional lift."""
        # Set promotion lift
        demand_model.set_promotion_lift(
            sample_product['product_id'],
            'discount',
            1.30  # 30% lift
        )

        result = demand_model.predict_demand_at_price(
            product_id=sample_product['product_id'],
            new_price=4.50,  # -10% discount
            baseline_demand=sample_product['baseline_demand'],
            current_price=sample_product['current_price'],
            elasticity=sample_product['elasticity'],
            is_promotion=True,
            promotion_type='discount'
        )

        # Should have promotional lift on top of price elasticity effect
        assert result['promotion_lift'] == 1.30
        assert result['predicted_demand'] > sample_product['baseline_demand'] * 1.2

    def test_predict_demand_curve(self, demand_model, sample_product):
        """Test generation of demand curve."""
        curve = demand_model.predict_demand_curve(
            product_id=sample_product['product_id'],
            baseline_demand=sample_product['baseline_demand'],
            current_price=sample_product['current_price'],
            price_range=(3.50, 6.50),
            num_points=10,
            elasticity=sample_product['elasticity']
        )

        assert len(curve) == 10
        assert 'price' in curve.columns
        assert 'demand' in curve.columns
        assert 'revenue' in curve.columns
        assert 'is_optimal' in curve.columns

        # Check that demand decreases as price increases (elastic product)
        assert curve['demand'].iloc[0] > curve['demand'].iloc[-1]

        # Should have exactly one optimal price point
        assert curve['is_optimal'].sum() == 1

        # Optimal price should maximize revenue
        optimal_row = curve[curve['is_optimal']].iloc[0]
        assert optimal_row['revenue'] == curve['revenue'].max()

    def test_predict_bulk(self, demand_model):
        """Test bulk prediction for multiple products."""
        bulk_data = pd.DataFrame({
            'product_id': ['P001', 'P002', 'P003'],
            'baseline_demand': [100, 150, 80],
            'current_price': [5.0, 8.0, 3.5],
            'new_price': [4.5, 8.5, 3.0],
            'elasticity': [-1.2, -0.8, -1.5]
        })

        results = demand_model.predict_bulk(
            bulk_data,
            elasticity_col='elasticity'
        )

        assert len(results) == 3
        assert all(col in results.columns for col in [
            'product_id', 'predicted_demand', 'revenue_change'
        ])

        # All predictions should have valid values
        assert results['predicted_demand'].notna().all()
        assert results['revenue_change'].notna().all()

    def test_cache_elasticity(self, demand_model):
        """Test elasticity caching."""
        demand_model.cache_elasticity(
            product_id='TEST_001',
            elasticity=-1.5,
            metadata={'method': 'log-log', 'r_squared': 0.85}
        )

        assert 'TEST_001' in demand_model.elasticity_cache
        assert demand_model.elasticity_cache['TEST_001']['elasticity'] == -1.5
        assert demand_model.elasticity_cache['TEST_001']['metadata']['r_squared'] == 0.85

        # Test prediction using cached elasticity
        result = demand_model.predict_demand_at_price(
            product_id='TEST_001',
            new_price=4.5,
            baseline_demand=100.0,
            current_price=5.0
        )

        assert result['elasticity'] == -1.5

    def test_load_elasticity_from_analyzer(self, demand_model, sample_price_data):
        """Test loading elasticities from analyzer."""
        # Split data into price and sales
        price_data = sample_price_data[['product_id', 'sell_price']].copy()
        sales_data = sample_price_data[['product_id', 'sales']].copy()

        demand_model.load_elasticity_from_analyzer(
            price_data=price_data,
            sales_data=sales_data
        )

        assert len(demand_model.elasticity_cache) > 0
        assert 'FOODS_3_001' in demand_model.elasticity_cache

        # Elasticity should be negative (law of demand)
        elasticity = demand_model.elasticity_cache['FOODS_3_001']['elasticity']
        assert elasticity < 0

    def test_learn_seasonality(self, demand_model, sample_price_data):
        """Test seasonality learning."""
        demand_model.learn_seasonality(sample_price_data)

        assert len(demand_model.seasonality_patterns) > 0
        assert 'FOODS_3_001' in demand_model.seasonality_patterns

        patterns = demand_model.seasonality_patterns['FOODS_3_001']
        assert 'day_of_week' in patterns
        assert 'month' in patterns
        assert 'overall_mean' in patterns

        # Day of week pattern should have 7 entries
        assert len(patterns['day_of_week']) <= 7

    def test_get_elasticity_summary(self, demand_model):
        """Test elasticity summary generation."""
        # Cache some elasticities
        demand_model.cache_elasticity('P001', -1.2, {'method': 'log-log'})
        demand_model.cache_elasticity('P002', -0.8, {'method': 'arc'})

        summary = demand_model.get_elasticity_summary()

        assert len(summary) == 2
        assert 'product_id' in summary.columns
        assert 'elasticity' in summary.columns
        assert 'method' in summary.columns

        # Check values
        assert summary[summary['product_id'] == 'P001']['elasticity'].iloc[0] == -1.2
        assert summary[summary['product_id'] == 'P002']['elasticity'].iloc[0] == -0.8

    def test_simulate_price_scenarios(self, demand_model, sample_product):
        """Test scenario simulation."""
        scenarios = [
            {'name': 'Current', 'price_change_pct': 0},
            {'name': 'Decrease 10%', 'price_change_pct': -10},
            {'name': 'Increase 10%', 'price_change_pct': 10},
            {'name': 'Promo -15%', 'price_change_pct': -15, 'is_promotion': True, 'promotion_type': 'discount'}
        ]

        results = demand_model.simulate_price_scenarios(
            product_id=sample_product['product_id'],
            baseline_demand=sample_product['baseline_demand'],
            current_price=sample_product['current_price'],
            scenarios=scenarios,
            elasticity=sample_product['elasticity']
        )

        assert len(results) == 4
        assert 'scenario_name' in results.columns
        assert 'predicted_demand' in results.columns
        assert 'revenue_change' in results.columns

        # Current scenario should have zero change
        current = results[results['scenario_name'] == 'Current'].iloc[0]
        assert abs(current['demand_change_pct']) < 0.1
        assert abs(current['revenue_change_pct']) < 0.1

        # Promo scenario should have highest demand
        promo = results[results['scenario_name'] == 'Promo -15%'].iloc[0]
        assert promo['predicted_demand'] > sample_product['baseline_demand']

    def test_revenue_optimization(self, demand_model):
        """Test that revenue optimization works correctly."""
        # Elastic product: revenue maximized at lower price
        elastic_curve = demand_model.predict_demand_curve(
            product_id='ELASTIC',
            baseline_demand=100.0,
            current_price=10.0,
            elasticity=-2.0,  # Elastic
            num_points=50
        )

        optimal_elastic = elastic_curve[elastic_curve['is_optimal']].iloc[0]
        # For elastic products, optimal price is usually lower
        assert optimal_elastic['price'] < 10.0

        # Inelastic product: revenue maximized at higher price
        inelastic_curve = demand_model.predict_demand_curve(
            product_id='INELASTIC',
            baseline_demand=100.0,
            current_price=10.0,
            elasticity=-0.5,  # Inelastic
            num_points=50
        )

        optimal_inelastic = inelastic_curve[inelastic_curve['is_optimal']].iloc[0]
        # For inelastic products, optimal price is usually higher
        assert optimal_inelastic['price'] > 10.0

    def test_confidence_intervals(self, demand_model, sample_product):
        """Test confidence interval calculation."""
        result = demand_model.predict_demand_at_price(
            product_id=sample_product['product_id'],
            new_price=4.0,
            baseline_demand=sample_product['baseline_demand'],
            current_price=sample_product['current_price'],
            elasticity=sample_product['elasticity']
        )

        assert result['confidence_lower'] is not None
        assert result['confidence_upper'] is not None
        assert result['confidence_lower'] < result['predicted_demand']
        assert result['confidence_upper'] > result['predicted_demand']
        assert result['confidence_level'] == 0.95

        # Interval should be wider for larger price changes
        small_change = demand_model.predict_demand_at_price(
            product_id=sample_product['product_id'],
            new_price=4.95,  # -1% change
            baseline_demand=sample_product['baseline_demand'],
            current_price=sample_product['current_price'],
            elasticity=sample_product['elasticity']
        )

        large_change = demand_model.predict_demand_at_price(
            product_id=sample_product['product_id'],
            new_price=3.50,  # -30% change
            baseline_demand=sample_product['baseline_demand'],
            current_price=sample_product['current_price'],
            elasticity=sample_product['elasticity']
        )

        small_interval = small_change['confidence_upper'] - small_change['confidence_lower']
        large_interval = large_change['confidence_upper'] - large_change['confidence_lower']

        assert large_interval > small_interval

    def test_input_validation(self, demand_model):
        """Test input validation."""
        # Negative price should raise error
        with pytest.raises(ValueError):
            demand_model.predict_demand_at_price(
                product_id='TEST',
                new_price=-1.0,
                baseline_demand=100.0,
                current_price=5.0
            )

        # Zero price should raise error
        with pytest.raises(ValueError):
            demand_model.predict_demand_at_price(
                product_id='TEST',
                new_price=0.0,
                baseline_demand=100.0,
                current_price=5.0
            )

        # Negative demand should raise error
        with pytest.raises(ValueError):
            demand_model.predict_demand_at_price(
                product_id='TEST',
                new_price=5.0,
                baseline_demand=-10.0,
                current_price=5.0
            )


def test_create_standard_scenarios():
    """Test standard scenarios creation."""
    scenarios = create_standard_scenarios()

    assert len(scenarios) >= 7  # At least 7 scenarios
    assert all('name' in s for s in scenarios)
    assert all('price_change_pct' in s for s in scenarios)

    # Should have both increases and decreases
    price_changes = [s['price_change_pct'] for s in scenarios]
    assert any(pc < 0 for pc in price_changes)  # Decreases
    assert any(pc > 0 for pc in price_changes)  # Increases
    assert any(pc == 0 for pc in price_changes)  # Baseline


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

"""
Tests for Supply Chain Planning System KPI and utility modules.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.loader import PlanningConfig
from src.kpi.calculator import KPICalculator
from src.data.models import PlanningResult, DemandResult, InventoryResult
from src.utils.helpers import (
    format_currency,
    format_percentage,
    validate_config,
    ensure_directory,
    flatten_dict,
)
import pandas as pd


class TestKPICalculator:
    """Tests for KPICalculator class."""

    def test_initialization(self):
        """Test KPI calculator initialization."""
        config = PlanningConfig()
        calculator = KPICalculator(config)

        assert calculator is not None
        assert calculator.kpi_definitions is not None

    def test_calculate_all(self):
        """Test calculating all KPIs."""
        config = PlanningConfig()
        calculator = KPICalculator(config)

        # Create mock planning result
        planning_result = PlanningResult(
            demand=DemandResult(
                forecast=pd.DataFrame(),
                mape=0.12,
                rmse=100.0,
                bias=0.02
            ),
            inventory=InventoryResult(
                positions=pd.DataFrame(),
                classifications=pd.DataFrame(),
                service_level=0.96,
                total_inventory_value=1000000
            )
        )

        kpis = calculator.calculate_all(planning_result)

        assert isinstance(kpis, dict)
        assert len(kpis) > 0

    def test_evaluate_against_targets(self):
        """Test evaluating KPIs against targets."""
        config = PlanningConfig()
        calculator = KPICalculator(config)

        sample_kpis = {
            'forecast_accuracy': 0.88,
            'service_level': 0.96
        }

        evaluation = calculator.evaluate_against_targets(sample_kpis)

        assert isinstance(evaluation, dict)
        for _, result in evaluation.items():
            assert 'value' in result
            assert 'status' in result

    def test_get_summary(self):
        """Test getting KPI summary."""
        config = PlanningConfig()
        calculator = KPICalculator(config)

        sample_kpis = {
            'forecast_accuracy': 0.88,
            'service_level': 0.96
        }

        summary = calculator.get_summary(sample_kpis)

        assert isinstance(summary, dict)
        assert 'total_kpis' in summary
        assert 'health_score' in summary


class TestHelperFunctions:
    """Tests for utility helper functions."""

    def test_format_currency(self):
        """Test currency formatting."""
        # Test millions
        result = format_currency(1234567.89)
        assert "M" in result or "$" in result

        # Test thousands
        result = format_currency(1000)
        assert "K" in result or "$" in result

        # Test small values
        result = format_currency(50)
        assert "$" in result

    def test_format_percentage(self):
        """Test percentage formatting."""
        assert format_percentage(0.5) == "50.0%"
        assert format_percentage(0.123) == "12.3%"
        assert format_percentage(1.0) == "100.0%"

    def test_validate_config(self):
        """Test configuration validation."""
        # Valid config
        config = {'key1': 'value1', 'key2': 'value2'}
        assert validate_config(config, ['key1', 'key2']) is True

        # Invalid config (missing required key) - should raise ValueError
        try:
            validate_config({'key1': 'value1'}, ['key1', 'key2'])
            assert False, "Should have raised ValueError"
        except ValueError:
            pass  # Expected

    def test_ensure_directory(self, tmp_path):
        """Test directory creation."""
        new_dir = tmp_path / "test_dir" / "nested"

        result = ensure_directory(str(new_dir))

        assert result.exists()
        assert result.is_dir()

    def test_flatten_dict(self):
        """Test dictionary flattening."""
        nested = {
            'a': 1,
            'b': {
                'c': 2,
                'd': {
                    'e': 3
                }
            }
        }

        flattened = flatten_dict(nested)

        assert flattened['a'] == 1
        assert flattened['b.c'] == 2
        assert flattened['b.d.e'] == 3

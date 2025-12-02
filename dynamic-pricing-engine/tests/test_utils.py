"""
Test suite for utility functions

These tests verify the helper and validator utilities.
"""

import pytest
import pandas as pd
from pathlib import Path
import tempfile
import yaml

from src.utils.helpers import load_config, save_results, load_results
from src.utils.validators import (
    validate_price,
    validate_elasticity,
    validate_dataframe
)


class TestHelpers:
    """Test helper functions."""

    def test_load_config(self):
        """Test configuration loading."""
        config = load_config('config/config.yaml')
        assert isinstance(config, dict)
        assert 'elasticity' in config
        assert 'optimization' in config

    def test_save_load_results_pickle(self):
        """Test saving and loading results with pickle."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data = {'test': 'data', 'value': 123}
            filepath = Path(tmpdir) / 'test.pkl'

            save_results(data, str(filepath), format='pickle')
            loaded = load_results(str(filepath), format='pickle')

            assert loaded == data

    def test_save_load_results_csv(self):
        """Test saving and loading results with CSV."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
            filepath = Path(tmpdir) / 'test.csv'

            save_results(data, str(filepath), format='csv')
            loaded = load_results(str(filepath), format='csv')

            pd.testing.assert_frame_equal(loaded, data)


class TestValidators:
    """Test validation functions."""

    def test_validate_price_positive(self):
        """Test price validation with valid price."""
        assert validate_price(5.99) is True
        assert validate_price(pd.Series([1.0, 2.0, 3.0])) is True

    def test_validate_price_negative(self):
        """Test price validation with negative price."""
        with pytest.raises(ValueError, match="must be positive"):
            validate_price(-5.0)

    def test_validate_price_too_high(self):
        """Test price validation with unreasonably high price."""
        with pytest.raises(ValueError, match="unreasonably high"):
            validate_price(15000.0)

    def test_validate_elasticity_negative(self):
        """Test elasticity validation with valid negative value."""
        assert validate_elasticity(-1.5) is True
        assert validate_elasticity(pd.Series([-0.8, -1.2, -2.0])) is True

    def test_validate_elasticity_positive(self):
        """Test elasticity validation with invalid positive value."""
        with pytest.raises(ValueError, match="should typically be negative"):
            validate_elasticity(1.5)

    def test_validate_dataframe_valid(self):
        """Test DataFrame validation with valid data."""
        df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        assert validate_dataframe(df, required_columns=['a', 'b']) is True

    def test_validate_dataframe_missing_columns(self):
        """Test DataFrame validation with missing columns."""
        df = pd.DataFrame({'a': [1, 2, 3]})
        with pytest.raises(ValueError, match="Missing required columns"):
            validate_dataframe(df, required_columns=['a', 'b'])

    def test_validate_dataframe_insufficient_rows(self):
        """Test DataFrame validation with insufficient rows."""
        df = pd.DataFrame({'a': [1], 'b': [2]})
        with pytest.raises(ValueError, match="must have at least"):
            validate_dataframe(df, min_rows=10)

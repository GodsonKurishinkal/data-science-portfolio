"""Tests for ABC/XYZ analysis module."""

import pytest
import pandas as pd
import numpy as np
from src.inventory.abc_analysis import ABCAnalyzer


def test_abc_analysis_basic(sample_demand_stats):
    """Test basic ABC analysis."""
    analyzer = ABCAnalyzer()
    result = analyzer.perform_abc_analysis(sample_demand_stats, value_col='revenue_sum')

    assert 'abc_class' in result.columns
    assert set(result['abc_class'].unique()).issubset({'A', 'B', 'C'})
    assert len(result) == len(sample_demand_stats)


def test_xyz_analysis_basic(sample_demand_stats):
    """Test basic XYZ analysis."""
    analyzer = ABCAnalyzer()
    result = analyzer.perform_xyz_analysis(sample_demand_stats, cv_col='demand_cv')

    assert 'xyz_class' in result.columns
    assert set(result['xyz_class'].unique()).issubset({'X', 'Y', 'Z'})


def test_combined_analysis(sample_demand_stats):
    """Test combined ABC-XYZ analysis."""
    analyzer = ABCAnalyzer()
    result = analyzer.perform_combined_analysis(
        sample_demand_stats,
        value_col='revenue_sum',
        cv_col='demand_cv'
    )

    assert 'abc_class' in result.columns
    assert 'xyz_class' in result.columns
    assert 'abc_xyz_class' in result.columns
    assert 'priority_score' in result.columns

    # Check priority scores are in valid range
    assert result['priority_score'].min() >= 1
    assert result['priority_score'].max() <= 9


def test_recommend_policy():
    """Test inventory policy recommendations."""
    analyzer = ABCAnalyzer()

    # Test AX class
    policy_ax = analyzer.recommend_inventory_policy('AX')
    assert policy_ax['service_level'] == '99%'
    assert 'Continuous Review' in policy_ax['policy']

    # Test CZ class
    policy_cz = analyzer.recommend_inventory_policy('CZ')
    assert policy_cz['service_level'] == '85%'


def test_class_statistics(sample_demand_stats):
    """Test class statistics calculation."""
    analyzer = ABCAnalyzer()
    classified = analyzer.perform_combined_analysis(sample_demand_stats)

    stats = analyzer.get_class_statistics(classified)

    assert 'abc_xyz_class' in stats.columns
    assert 'revenue_sum_sum' in stats.columns
    assert 'item_pct' in stats.columns
    assert 'revenue_pct' in stats.columns

    # Check percentages sum to approximately 100
    assert abs(stats['item_pct'].sum() - 100) < 0.1
    assert abs(stats['revenue_pct'].sum() - 100) < 0.1

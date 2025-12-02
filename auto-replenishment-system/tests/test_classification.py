"""Tests for ABC/XYZ classification modules."""

import pytest
import pandas as pd
import numpy as np

from src.classification.abc_classifier import ABCClassifier
from src.classification.xyz_classifier import XYZClassifier
from src.classification.velocity_classifier import VelocityClassifier
from src.classification.matrix import ABCXYZMatrix


class TestABCClassifier:
    """Tests for ABC classification."""
    
    def test_classify_basic(self, abc_test_data):
        """Test basic ABC classification."""
        classifier = ABCClassifier(
            value_column="revenue",
            thresholds=(0.80, 0.95),
        )
        
        result = classifier.classify(abc_test_data)
        
        assert "abc_class" in result.columns
        assert "cumulative_value_pct" in result.columns
        
        # Check class distribution
        class_counts = result["abc_class"].value_counts()
        assert "A" in class_counts.index
        assert "B" in class_counts.index or "C" in class_counts.index
    
    def test_classify_pareto_principle(self, abc_test_data):
        """Test that A items represent ~80% of value."""
        classifier = ABCClassifier(
            value_column="revenue",
            thresholds=(0.80, 0.95),
        )
        
        result = classifier.classify(abc_test_data)
        
        total_value = abc_test_data["revenue"].sum()
        a_items = result[result["abc_class"] == "A"]
        a_value = abc_test_data.loc[a_items.index, "revenue"].sum()
        
        # A items should be close to 80% of value
        a_pct = a_value / total_value
        assert 0.70 <= a_pct <= 0.90
    
    def test_classify_empty_dataframe(self):
        """Test classification with empty DataFrame."""
        classifier = ABCClassifier()
        result = classifier.classify(pd.DataFrame())
        
        assert result.empty
    
    def test_classify_single_item(self):
        """Test classification with single item."""
        classifier = ABCClassifier(value_column="revenue")
        single_item = pd.DataFrame({
            "item_id": ["SKU001"],
            "revenue": [10000],
        })
        
        result = classifier.classify(single_item)
        
        assert len(result) == 1
        assert result["abc_class"].iloc[0] == "A"


class TestXYZClassifier:
    """Tests for XYZ classification."""
    
    def test_classify_basic(self, xyz_test_data):
        """Test basic XYZ classification."""
        classifier = XYZClassifier(cv_thresholds=(0.5, 1.0))
        
        result = classifier.classify(xyz_test_data)
        
        assert "xyz_class" in result.columns
        assert "cv" in result.columns
    
    def test_classify_variability_levels(self, xyz_test_data):
        """Test that X/Y/Z reflect variability levels."""
        classifier = XYZClassifier(cv_thresholds=(0.5, 1.0))
        
        result = classifier.classify(xyz_test_data)
        
        # X1 should be X (low CV)
        x1_class = result[result["item_id"] == "X1"]["xyz_class"].iloc[0]
        assert x1_class == "X"
        
        # Y1 should be Y (medium CV)
        y1_class = result[result["item_id"] == "Y1"]["xyz_class"].iloc[0]
        assert y1_class == "Y"
        
        # Z1 should be Z (high CV)
        z1_class = result[result["item_id"] == "Z1"]["xyz_class"].iloc[0]
        assert z1_class == "Z"
    
    def test_classify_empty_dataframe(self):
        """Test classification with empty DataFrame."""
        classifier = XYZClassifier()
        result = classifier.classify(pd.DataFrame())
        
        assert result.empty


class TestVelocityClassifier:
    """Tests for velocity/FMR classification."""
    
    def test_classify_basic(self, sample_inventory_data):
        """Test basic velocity classification."""
        classifier = VelocityClassifier(
            quantity_column="daily_demand_rate",
            thresholds=(0.60, 0.90),
        )
        
        result = classifier.classify(sample_inventory_data)
        
        assert "velocity_class" in result.columns
        assert set(result["velocity_class"].unique()).issubset({"F", "M", "R"})
    
    def test_classify_distribution(self, sample_inventory_data):
        """Test that velocity classes follow expected distribution."""
        classifier = VelocityClassifier(
            quantity_column="daily_demand_rate",
            thresholds=(0.60, 0.90),
        )
        
        result = classifier.classify(sample_inventory_data)
        
        class_counts = result["velocity_class"].value_counts(normalize=True)
        
        # F should have ~60% of movement
        # M should have ~30% of movement
        # R should have ~10% of movement
        assert "F" in class_counts.index


class TestABCXYZMatrix:
    """Tests for ABC-XYZ matrix."""
    
    def test_get_service_level(self):
        """Test getting service level from matrix."""
        matrix = ABCXYZMatrix()
        
        # AX should have highest service level
        ax_sl = matrix.get_service_level("A", "X")
        by_sl = matrix.get_service_level("B", "Y")
        cz_sl = matrix.get_service_level("C", "Z")
        
        assert ax_sl >= by_sl >= cz_sl
    
    def test_get_strategy(self):
        """Test getting strategy from matrix."""
        matrix = ABCXYZMatrix()
        
        ax_strategy = matrix.get_strategy("A", "X")
        cz_strategy = matrix.get_strategy("C", "Z")
        
        assert ax_strategy != cz_strategy
        assert "continuous" in ax_strategy.lower() or "periodic" in ax_strategy.lower()
    
    def test_set_service_levels(self):
        """Test setting custom service levels."""
        matrix = ABCXYZMatrix()
        
        custom_levels = {
            "AX": 0.999,
            "BY": 0.90,
        }
        
        matrix.set_service_levels(custom_levels)
        
        assert matrix.get_service_level("A", "X") == 0.999
        assert matrix.get_service_level("B", "Y") == 0.90
    
    def test_generate_matrix_dataframe(self):
        """Test generating matrix as DataFrame."""
        matrix = ABCXYZMatrix()
        
        df = matrix.to_dataframe()
        
        assert "abc_class" in df.columns
        assert "xyz_class" in df.columns
        assert "service_level" in df.columns
        assert len(df) == 9  # 3x3 matrix

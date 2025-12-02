"""
Tests for Anomaly Detector module.

Tests the AnomalyDetector ensemble for detecting anomalies
in demand data using multiple methods.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.detection.anomaly_detector import (
    AnomalyDetector,
    ZScoreDetector,
    IQRDetector,
    BusinessRuleDetector,
    AnomalySeverity,
    AnomalyType,
    Anomaly
)


class TestAnomaly:
    """Tests for Anomaly dataclass."""
    
    def test_anomaly_creation(self):
        """Test creating an Anomaly."""
        anomaly = Anomaly(
            product_id="TEST_001",
            timestamp=datetime.now(),
            anomaly_type=AnomalyType.DEMAND_SPIKE,
            severity=AnomalySeverity.WARNING,
            value=200.0,
            expected_value=100.0,
            deviation=100.0,
            z_score=3.5,
            message="Test anomaly",
            detection_method="test"
        )
        
        assert anomaly.product_id == "TEST_001"
        assert anomaly.value == 200.0
        assert anomaly.severity == AnomalySeverity.WARNING
    
    def test_anomaly_to_dict(self):
        """Test converting anomaly to dictionary."""
        anomaly = Anomaly(
            product_id="TEST_001",
            timestamp=datetime.now(),
            anomaly_type=AnomalyType.DEMAND_SPIKE,
            severity=AnomalySeverity.WARNING,
            value=200.0,
            expected_value=100.0,
            deviation=100.0,
            z_score=3.5,
            message="Test anomaly",
            detection_method="test"
        )
        
        d = anomaly.to_dict()
        
        assert isinstance(d, dict)
        assert d['product_id'] == "TEST_001"
        assert d['severity'] == 'warning'


class TestZScoreDetector:
    """Tests for ZScoreDetector class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data with clear anomaly."""
        np.random.seed(42)
        n = 168
        dates = pd.date_range('2024-01-01', periods=n, freq='h')
        values = np.random.normal(100, 10, n)
        # Insert clear anomaly
        values[100] = 300
        return pd.Series(values, index=dates)
    
    def test_initialization(self):
        """Test ZScoreDetector initialization."""
        detector = ZScoreDetector(threshold=3.0)
        assert detector.threshold == 3.0
    
    def test_detect_finds_anomaly(self, sample_data):
        """Test detection finds the inserted anomaly."""
        detector = ZScoreDetector(threshold=3.0)
        
        anomalies = detector.detect(sample_data, "TEST_PROD")
        
        # Should find at least one anomaly
        assert len(anomalies) >= 1
        
        # Check anomaly properties
        for a in anomalies:
            assert isinstance(a, Anomaly)
            assert a.detection_method == "z_score"
    
    def test_detect_normal_data(self):
        """Test detection on normal data."""
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=100, freq='h')
        # Very consistent data
        values = np.ones(100) * 100 + np.random.randn(100) * 0.1
        data = pd.Series(values, index=dates)
        
        detector = ZScoreDetector(threshold=3.0)
        anomalies = detector.detect(data, "TEST_PROD")
        
        # Should find no anomalies in very consistent data
        assert len(anomalies) == 0


class TestIQRDetector:
    """Tests for IQRDetector class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data with outliers."""
        np.random.seed(42)
        n = 100
        dates = pd.date_range('2024-01-01', periods=n, freq='h')
        values = np.random.normal(100, 10, n)
        # Insert outliers
        values[50] = 250
        values[75] = 20
        return pd.Series(values, index=dates)
    
    def test_initialization(self):
        """Test IQRDetector initialization."""
        detector = IQRDetector(multiplier=1.5)
        assert detector.multiplier == 1.5
    
    def test_detect_finds_outliers(self, sample_data):
        """Test detection finds outliers."""
        detector = IQRDetector(multiplier=1.5)
        
        anomalies = detector.detect(sample_data, "TEST_PROD")
        
        # Should find at least the inserted outliers
        assert len(anomalies) >= 1
        
        for a in anomalies:
            assert a.detection_method == "iqr"


class TestBusinessRuleDetector:
    """Tests for BusinessRuleDetector class."""
    
    def test_initialization(self):
        """Test BusinessRuleDetector initialization."""
        detector = BusinessRuleDetector(
            stockout_threshold_days=3.0,
            excess_inventory_days=45.0
        )
        assert detector.stockout_threshold_days == 3.0
        assert detector.excess_inventory_days == 45.0
    
    def test_detect_stockout_risk(self):
        """Test detection of stockout risk."""
        detector = BusinessRuleDetector(
            stockout_threshold_days=3.0,
            critical_stockout_days=1.5
        )
        
        # Low inventory, high demand = stockout risk
        anomalies = detector.detect(
            product_id="TEST_PROD",
            current_inventory=50,
            daily_demand=50,  # 1 day of stock
            baseline_demand=40
        )
        
        # Should detect stockout risk
        stockout_anomalies = [
            a for a in anomalies
            if a.anomaly_type == AnomalyType.STOCKOUT_RISK
        ]
        assert len(stockout_anomalies) >= 1
        
        # Critical severity for < 1.5 days
        assert stockout_anomalies[0].severity == AnomalySeverity.CRITICAL
    
    def test_detect_excess_inventory(self):
        """Test detection of excess inventory."""
        detector = BusinessRuleDetector(excess_inventory_days=45.0)
        
        # High inventory, low demand = excess
        anomalies = detector.detect(
            product_id="TEST_PROD",
            current_inventory=5000,
            daily_demand=50,  # 100 days of stock
            baseline_demand=50
        )
        
        # Should detect excess inventory
        excess_anomalies = [
            a for a in anomalies
            if a.anomaly_type == AnomalyType.EXCESS_INVENTORY
        ]
        assert len(excess_anomalies) >= 1
    
    def test_detect_demand_spike(self):
        """Test detection of demand spike."""
        detector = BusinessRuleDetector(demand_spike_multiplier=2.0)
        
        # Demand much higher than baseline
        anomalies = detector.detect(
            product_id="TEST_PROD",
            current_inventory=1000,
            daily_demand=300,  # 3x baseline
            baseline_demand=100
        )
        
        # Should detect demand spike
        spike_anomalies = [
            a for a in anomalies
            if a.anomaly_type == AnomalyType.DEMAND_SPIKE
        ]
        assert len(spike_anomalies) >= 1
    
    def test_no_anomalies_for_normal_state(self):
        """Test no anomalies for normal inventory state."""
        detector = BusinessRuleDetector()
        
        # Normal state
        anomalies = detector.detect(
            product_id="TEST_PROD",
            current_inventory=500,
            daily_demand=50,  # 10 days of stock
            baseline_demand=50
        )
        
        # No critical anomalies expected
        critical = [a for a in anomalies if a.severity == AnomalySeverity.CRITICAL]
        assert len(critical) == 0


class TestAnomalyDetector:
    """Tests for AnomalyDetector ensemble class."""
    
    @pytest.fixture
    def sample_series(self):
        """Create sample time series with anomaly."""
        np.random.seed(42)
        n = 168
        dates = pd.date_range('2024-01-01', periods=n, freq='h')
        values = 100 + 20 * np.sin(np.arange(n) * 2 * np.pi / 24) + np.random.normal(0, 5, n)
        values[50] = 300  # Spike
        return pd.Series(values, index=dates)
    
    def test_initialization(self):
        """Test AnomalyDetector initialization."""
        detector = AnomalyDetector()
        assert detector is not None
    
    def test_initialization_with_methods(self):
        """Test initialization with specific methods."""
        detector = AnomalyDetector(
            methods=['z_score', 'iqr', 'business_rules'],
            sensitivity='high'
        )
        
        assert 'z_score' in detector.detectors
        assert 'iqr' in detector.detectors
        assert 'business_rules' in detector.detectors
    
    def test_detect_from_series(self, sample_series):
        """Test detection from time series."""
        detector = AnomalyDetector(methods=['z_score', 'iqr'])
        
        anomalies = detector.detect_from_series(sample_series, "TEST_PROD")
        
        # Should find anomalies
        assert len(anomalies) >= 1
    
    def test_detect_operational(self):
        """Test operational anomaly detection."""
        detector = AnomalyDetector(methods=['business_rules'])
        
        anomalies = detector.detect_operational(
            product_id="TEST_PROD",
            current_inventory=50,
            daily_demand=100,
            baseline_demand=80
        )
        
        # Should detect stockout risk
        assert len(anomalies) >= 1
    
    def test_get_summary(self, sample_series):
        """Test summary statistics."""
        detector = AnomalyDetector(methods=['z_score'])
        anomalies = detector.detect_from_series(sample_series, "TEST_PROD")
        
        summary = detector.get_summary(anomalies)
        
        assert 'total' in summary
        assert 'by_severity' in summary
        assert 'by_type' in summary
        assert 'by_method' in summary
    
    def test_empty_summary(self):
        """Test summary with no anomalies."""
        detector = AnomalyDetector()
        
        summary = detector.get_summary([])
        
        assert summary['total'] == 0
    
    def test_sensitivity_levels(self):
        """Test different sensitivity levels."""
        for sensitivity in ['low', 'medium', 'high']:
            detector = AnomalyDetector(sensitivity=sensitivity)
            assert detector.sensitivity == sensitivity


class TestAnomalyDetectorEdgeCases:
    """Test edge cases for AnomalyDetector."""
    
    def test_empty_data(self):
        """Test with empty data."""
        detector = AnomalyDetector(methods=['z_score'])
        
        empty_series = pd.Series([], dtype=float)
        anomalies = detector.detect_from_series(empty_series, "TEST")
        
        assert len(anomalies) == 0
    
    def test_small_dataset(self):
        """Test with small dataset."""
        detector = AnomalyDetector(methods=['z_score'])
        
        small_series = pd.Series([100, 102, 98])
        anomalies = detector.detect_from_series(small_series, "TEST")
        
        # Should handle without error
        assert isinstance(anomalies, list)
    
    def test_constant_values(self):
        """Test with constant values."""
        detector = AnomalyDetector(methods=['z_score'])
        
        constant_series = pd.Series([100.0] * 100)
        anomalies = detector.detect_from_series(constant_series, "TEST")
        
        # Constant data should have no anomalies
        # (may have some due to std=0 handling)
        assert isinstance(anomalies, list)
    
    def test_zero_daily_demand(self):
        """Test business rules with zero demand."""
        detector = AnomalyDetector(methods=['business_rules'])
        
        # Zero demand - should handle gracefully
        anomalies = detector.detect_operational(
            product_id="TEST",
            current_inventory=1000,
            daily_demand=0,
            baseline_demand=50
        )
        
        assert isinstance(anomalies, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

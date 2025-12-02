"""
Tests for Supply Chain Planning System KPI and utility modules.
"""

import pytest
import pandas as pd
from datetime import datetime

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.kpi.calculator import KPICalculator
from src.kpi.dashboard import KPIDashboard
from src.kpi.alerts import AlertManager, AlertSeverity
from src.utils.helpers import (
    format_currency,
    format_percentage,
    validate_config,
    ensure_directory,
    flatten_dict,
)


class TestKPICalculator:
    """Tests for KPICalculator class."""
    
    def test_initialization(self):
        """Test KPI calculator initialization."""
        calculator = KPICalculator()
        
        assert calculator is not None
        assert calculator.targets is not None
    
    def test_calculate_all(self, sample_kpis):
        """Test calculating all KPIs."""
        calculator = KPICalculator()
        
        # Create mock data
        planning_result = type('obj', (object,), {
            'demand': type('obj', (object,), {'mape': 0.12})(),
            'inventory': type('obj', (object,), {'service_level': 0.96, 'total_inventory_value': 1000000})(),
            'pricing': type('obj', (object,), {'revenue_lift': 0.08, 'margin_improvement': 0.03})(),
            'replenishment': type('obj', (object,), {'automation_rate': 0.80})(),
        })()
        
        kpis = calculator.calculate_all(planning_result)
        
        assert isinstance(kpis, dict)
        assert len(kpis) > 0
    
    def test_evaluate_against_targets(self, sample_kpis):
        """Test evaluating KPIs against targets."""
        calculator = KPICalculator()
        
        evaluation = calculator.evaluate_against_targets(sample_kpis)
        
        assert isinstance(evaluation, dict)
        for kpi_name, result in evaluation.items():
            assert 'value' in result
            assert 'target' in result
            assert 'status' in result
    
    def test_get_summary(self, sample_kpis):
        """Test getting KPI summary."""
        calculator = KPICalculator()
        
        # Set KPIs first
        calculator.calculate_all(type('obj', (object,), {
            'demand': type('obj', (object,), {'mape': 0.12})(),
            'inventory': type('obj', (object,), {'service_level': 0.96, 'total_inventory_value': 1000000})(),
        })())
        
        summary = calculator.get_summary()
        
        assert isinstance(summary, dict)


class TestKPIDashboard:
    """Tests for KPIDashboard class."""
    
    def test_initialization(self):
        """Test dashboard initialization."""
        dashboard = KPIDashboard()
        
        assert dashboard is not None
    
    def test_update(self, sample_kpis):
        """Test updating dashboard with KPIs."""
        dashboard = KPIDashboard()
        
        dashboard.update(sample_kpis)
        
        assert dashboard.last_update is not None
    
    def test_get_display_data(self, sample_kpis):
        """Test getting display data."""
        dashboard = KPIDashboard()
        dashboard.update(sample_kpis)
        
        display_data = dashboard.get_display_data()
        
        assert isinstance(display_data, dict)
        assert 'kpis' in display_data
        assert 'last_update' in display_data
    
    def test_get_strategic_kpis(self, sample_kpis):
        """Test getting strategic KPIs."""
        dashboard = KPIDashboard()
        dashboard.update(sample_kpis)
        
        strategic = dashboard.get_strategic_kpis()
        
        assert isinstance(strategic, dict)
    
    def test_get_operational_kpis(self, sample_kpis):
        """Test getting operational KPIs."""
        dashboard = KPIDashboard()
        dashboard.update(sample_kpis)
        
        operational = dashboard.get_operational_kpis()
        
        assert isinstance(operational, dict)


class TestAlertManager:
    """Tests for AlertManager class."""
    
    def test_initialization(self):
        """Test alert manager initialization."""
        manager = AlertManager()
        
        assert manager is not None
        assert len(manager.alerts) == 0
    
    def test_create_alert(self):
        """Test creating an alert."""
        manager = AlertManager()
        
        alert = manager.create_alert(
            severity=AlertSeverity.HIGH,
            alert_type='stockout_risk',
            source='inventory',
            message='Low inventory on SKU001',
            details={'item_id': 'SKU001', 'current_stock': 5}
        )
        
        assert alert is not None
        assert alert.severity == AlertSeverity.HIGH
        assert alert.alert_type == 'stockout_risk'
    
    def test_get_active_alerts(self, sample_alerts):
        """Test getting active alerts."""
        manager = AlertManager()
        
        # Create some alerts
        for alert_data in sample_alerts:
            manager.create_alert(
                severity=AlertSeverity[alert_data['severity']],
                alert_type=alert_data['type'],
                source='test',
                message=alert_data['message']
            )
        
        active = manager.get_active_alerts()
        
        assert len(active) == len(sample_alerts)
    
    def test_get_alerts_by_severity(self):
        """Test filtering alerts by severity."""
        manager = AlertManager()
        
        manager.create_alert(AlertSeverity.CRITICAL, 'test', 'test', 'Critical alert')
        manager.create_alert(AlertSeverity.HIGH, 'test', 'test', 'High alert')
        manager.create_alert(AlertSeverity.MEDIUM, 'test', 'test', 'Medium alert')
        
        critical_alerts = manager.get_alerts_by_severity(AlertSeverity.CRITICAL)
        
        assert len(critical_alerts) == 1
        assert critical_alerts[0].severity == AlertSeverity.CRITICAL
    
    def test_acknowledge_alert(self):
        """Test acknowledging an alert."""
        manager = AlertManager()
        
        alert = manager.create_alert(
            AlertSeverity.HIGH, 'test', 'test', 'Test alert'
        )
        
        manager.acknowledge_alert(alert.alert_id, 'test_user')
        
        assert alert.acknowledged
        assert alert.acknowledged_by == 'test_user'
    
    def test_resolve_alert(self):
        """Test resolving an alert."""
        manager = AlertManager()
        
        alert = manager.create_alert(
            AlertSeverity.HIGH, 'test', 'test', 'Test alert'
        )
        
        manager.resolve_alert(alert.alert_id, 'Issue fixed')
        
        assert alert.resolved
        assert alert.resolution == 'Issue fixed'


class TestHelperFunctions:
    """Tests for utility helper functions."""
    
    def test_format_currency(self):
        """Test currency formatting."""
        assert format_currency(1000) == "$1,000.00"
        assert format_currency(1234567.89) == "$1,234,567.89"
        assert format_currency(0) == "$0.00"
    
    def test_format_percentage(self):
        """Test percentage formatting."""
        assert format_percentage(0.5) == "50.0%"
        assert format_percentage(0.123) == "12.3%"
        assert format_percentage(1.0) == "100.0%"
    
    def test_validate_config(self, sample_config):
        """Test configuration validation."""
        # Valid config
        assert validate_config(sample_config) is True
        
        # Invalid config (missing required key)
        invalid_config = {'demand': {}}
        assert validate_config(invalid_config) is False
    
    def test_ensure_directory(self, tmp_path):
        """Test directory creation."""
        new_dir = tmp_path / "test_dir" / "nested"
        
        result = ensure_directory(new_dir)
        
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

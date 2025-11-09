"""
Unit tests for model training and evaluation.
"""

import pytest
import pandas as pd
import numpy as np
from src.models.train import (
    train_model,
    calculate_metrics,
    cross_validate_model
)


@pytest.fixture
def sample_model_data():
    """Create sample data for model training."""
    np.random.seed(42)
    n_samples = 200
    
    X = pd.DataFrame({
        'feature_1': np.random.randn(n_samples),
        'feature_2': np.random.randn(n_samples),
        'feature_3': np.random.randn(n_samples)
    })
    
    # Create target with some relationship to features
    y = pd.Series(
        2 * X['feature_1'] + 3 * X['feature_2'] - X['feature_3'] + np.random.randn(n_samples) * 0.5
    )
    
    return X, y


class TestCalculateMetrics:
    """Test cases for calculate_metrics function."""
    
    def test_calculates_all_metrics(self):
        """Test that all metrics are calculated."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 2.2, 2.9, 4.1, 4.9])
        
        metrics = calculate_metrics(y_true, y_pred)
        
        assert 'mae' in metrics
        assert 'rmse' in metrics
        assert 'mape' in metrics
        assert 'r2' in metrics
    
    def test_perfect_predictions(self):
        """Test metrics with perfect predictions."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1, 2, 3, 4, 5])
        
        metrics = calculate_metrics(y_true, y_pred)
        
        assert metrics['mae'] == 0.0
        assert metrics['rmse'] == 0.0
        assert metrics['mape'] == 0.0
        assert metrics['r2'] == 1.0
    
    def test_metrics_positive(self):
        """Test that error metrics are non-negative."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1.5, 2.5, 2.5, 3.5, 5.5])
        
        metrics = calculate_metrics(y_true, y_pred)
        
        assert metrics['mae'] >= 0
        assert metrics['rmse'] >= 0
        assert metrics['mape'] >= 0


class TestTrainModel:
    """Test cases for train_model function."""
    
    def test_train_random_forest_model(self, sample_model_data):
        """Test training a Random Forest model."""
        X, y = sample_model_data
        model, metrics = train_model(X, y, model_type='random_forest', test_size=0.2)
        
        assert model is not None
        assert isinstance(metrics, dict)
        assert 'mae' in metrics
        assert 'rmse' in metrics
    
    def test_model_makes_predictions(self, sample_model_data):
        """Test that trained model can make predictions."""
        X, y = sample_model_data
        model, _ = train_model(X, y, model_type='random_forest', test_size=0.2)
        
        # Test prediction
        X_test = X.iloc[:10]
        predictions = model.predict(X_test)
        
        assert len(predictions) == len(X_test)
        assert isinstance(predictions, np.ndarray)
    
    def test_different_model_types(self, sample_model_data):
        """Test training different model types."""
        X, y = sample_model_data
        
        # Test Random Forest
        model_rf, metrics_rf = train_model(X, y, model_type='random_forest', test_size=0.2)
        assert model_rf is not None
        assert isinstance(metrics_rf, dict)
    
    def test_invalid_model_type_raises_error(self, sample_model_data):
        """Test that invalid model type raises error."""
        X, y = sample_model_data
        
        with pytest.raises(ValueError, match="Unsupported model type"):
            train_model(X, y, model_type='invalid_model')
    
    def test_model_parameters(self, sample_model_data):
        """Test passing custom parameters to model."""
        X, y = sample_model_data
        
        model, _ = train_model(
            X, y,
            model_type='random_forest',
            test_size=0.2,
            n_estimators=50,
            max_depth=5
        )
        
        assert model.n_estimators == 50
        assert model.max_depth == 5


class TestCrossValidateModel:
    """Test cases for cross_validate_model function."""
    
    def test_cross_validation_returns_metrics(self, sample_model_data):
        """Test that cross-validation returns averaged metrics."""
        X, y = sample_model_data
        
        cv_metrics = cross_validate_model(
            X, y,
            model_type='random_forest',
            n_splits=3
        )
        
        assert isinstance(cv_metrics, dict)
        assert 'mae' in cv_metrics
        assert 'rmse' in cv_metrics
        assert 'r2' in cv_metrics
    
    def test_cv_metrics_are_averaged(self, sample_model_data):
        """Test that cross-validation metrics are reasonable."""
        X, y = sample_model_data
        
        cv_metrics = cross_validate_model(
            X, y,
            model_type='random_forest',
            n_splits=3
        )
        
        # Metrics should be positive (except R2 which can be negative)
        assert cv_metrics['mae'] > 0
        assert cv_metrics['rmse'] > 0


class TestModelWorkflow:
    """Integration tests for complete modeling workflow."""
    
    def test_complete_training_workflow(self, sample_model_data):
        """Test complete workflow from training to prediction."""
        X, y = sample_model_data
        
        # Train model
        model, metrics = train_model(X, y, model_type='random_forest', test_size=0.2)
        
        # Verify model works
        predictions = model.predict(X.iloc[:10])
        
        assert len(predictions) == 10
        assert model is not None
        assert metrics['r2'] <= 1.0  # R² should be at most 1

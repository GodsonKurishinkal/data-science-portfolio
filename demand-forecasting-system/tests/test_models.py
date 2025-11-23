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


# M5-Specific Model Tests

@pytest.fixture
def sample_m5_model_data():
    """Create sample M5 data for model testing."""
    np.random.seed(42)
    n_samples = 100
    
    df = pd.DataFrame({
        'item_id': ['FOODS_1_001'] * n_samples,
        'store_id': ['CA_1'] * n_samples,
        'sales': np.random.randint(0, 20, n_samples),
        'sales_lag_1': np.random.randint(0, 20, n_samples),
        'sales_lag_7': np.random.randint(0, 20, n_samples),
        'price': np.random.uniform(1.0, 5.0, n_samples),
        'dayofweek': np.random.randint(0, 7, n_samples),
        'is_weekend': np.random.choice([0, 1], n_samples)
    })
    
    return df


class TestM5Models:
    """Test cases for M5-specific model functions."""
    
    def test_baseline_naive_forecast(self):
        """Test Naive forecast baseline model."""
        from src.models.train import NaiveForecast
        
        y_train = pd.Series([10, 15, 20, 25, 30])
        model = NaiveForecast()
        model.fit(y_train)
        
        predictions = model.predict(horizon=3)
        
        assert len(predictions) == 3
        assert all(p == 30 for p in predictions)  # Should repeat last value
    
    def test_baseline_moving_average(self):
        """Test Moving Average baseline model."""
        from src.models.train import MovingAverageForecast
        
        y_train = pd.Series([10, 12, 14, 16, 18, 20])
        model = MovingAverageForecast(window=3)
        model.fit(y_train)
        
        predictions = model.predict(horizon=2)
        
        assert len(predictions) == 2
        expected_avg = (16 + 18 + 20) / 3
        assert abs(predictions[0] - expected_avg) < 0.01
    
    def test_baseline_seasonal_naive(self):
        """Test Seasonal Naive baseline model."""
        from src.models.train import SeasonalNaiveForecast
        
        y_train = pd.Series([10, 20, 30, 40, 50, 60, 70])
        model = SeasonalNaiveForecast(seasonal_period=7)
        model.fit(y_train)
        
        predictions = model.predict(horizon=7)
        
        assert len(predictions) == 7
        # Should repeat the last 7 values
        assert predictions[0] == 10
        assert predictions[6] == 70
    
    def test_train_baseline_models(self):
        """Test training multiple baseline models."""
        from src.models.train import train_baseline_models
        
        y_train = pd.Series(range(50, 100))
        y_test = pd.Series(range(100, 120))
        
        results = train_baseline_models(y_train, y_test)
        
        assert 'naive' in results
        assert 'moving_average_7' in results
        assert 'predictions' in results['naive']
        assert 'metrics' in results['naive']
        assert len(results['naive']['predictions']) == len(y_test)
    
    def test_prepare_m5_train_data(self, sample_m5_model_data):
        """Test M5 data preparation for training."""
        from src.models.train import prepare_m5_train_data
        
        X, y = prepare_m5_train_data(sample_m5_model_data, target_col='sales')
        
        assert 'sales' not in X.columns  # Target should be removed
        assert 'item_id' not in X.columns  # ID columns should be removed
        assert 'store_id' not in X.columns
        assert len(X) == len(y)
        assert y.name == 'sales'
    
    def test_train_m5_model_returns_importance(self, sample_m5_model_data):
        """Test that M5 model training returns feature importance."""
        from src.models.train import train_m5_model
        
        model, metrics, importance = train_m5_model(
            sample_m5_model_data,
            target_col='sales',
            model_type='random_forest',
            test_size=0.2
        )
        
        assert model is not None
        assert isinstance(metrics, dict)
        assert isinstance(importance, pd.DataFrame)
        if len(importance) > 0:
            assert 'feature' in importance.columns
            assert 'importance' in importance.columns


class TestModelEvaluation:
    """Test cases for model evaluation metrics."""
    
    def test_evaluate_model_all_metrics(self):
        """Test evaluate_model function with all metrics."""
        from src.models.predict import evaluate_model
        
        y_true = np.array([10, 20, 30, 40, 50])
        y_pred = np.array([11, 19, 31, 39, 51])
        
        metrics = evaluate_model(y_true, y_pred)
        
        expected_metrics = ['rmse', 'mae', 'mape', 'smape', 'wrmsse', 'r2']
        for metric in expected_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], (int, float))
    
    def test_compare_model_predictions(self):
        """Test comparing predictions from multiple models."""
        from src.models.predict import compare_model_predictions
        
        y_true = np.array([10, 20, 30, 40, 50])
        predictions_dict = {
            'model_a': np.array([11, 19, 31, 39, 51]),
            'model_b': np.array([12, 18, 32, 38, 52]),
            'model_c': np.array([10, 20, 30, 40, 50])
        }
        
        comparison = compare_model_predictions(y_true, predictions_dict)
        
        assert isinstance(comparison, pd.DataFrame)
        assert 'model' in comparison.columns
        assert len(comparison) == 3
        assert 'rmse' in comparison.columns
        # Best model (model_c) should have RMSE of 0
        assert comparison.iloc[0]['model'] == 'model_c'

"""
Helper utility functions for the demand forecasting system.
"""

import pandas as pd
import numpy as np
from typing import Any, Dict, List
import json
import yaml


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.

    Parameters
    ----------
    config_path : str
        Path to the configuration file.

    Returns
    -------
    Dict[str, Any]
        Configuration dictionary.

    Examples
    --------
    >>> config = load_config('config/config.yaml')
    >>> print(config['model']['type'])
    'random_forest'
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: Dict[str, Any], config_path: str) -> None:
    """
    Save configuration to a YAML file.

    Parameters
    ----------
    config : Dict[str, Any]
        Configuration dictionary.
    config_path : str
        Path where to save the configuration.

    Examples
    --------
    >>> save_config(config, 'config/config.yaml')
    """
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def save_metrics(metrics: Dict[str, float], output_path: str) -> None:
    """
    Save model metrics to a JSON file.

    Parameters
    ----------
    metrics : Dict[str, float]
        Dictionary of metrics to save.
    output_path : str
        Path where to save the metrics.

    Examples
    --------
    >>> metrics = {'rmse': 45.23, 'mae': 32.15, 'r2': 0.89}
    >>> save_metrics(metrics, 'results/metrics.json')
    """
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=4)


def load_metrics(metrics_path: str) -> Dict[str, float]:
    """
    Load metrics from a JSON file.

    Parameters
    ----------
    metrics_path : str
        Path to the metrics file.

    Returns
    -------
    Dict[str, float]
        Dictionary of metrics.

    Examples
    --------
    >>> metrics = load_metrics('results/metrics.json')
    """
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    return metrics


def print_metrics(metrics: Dict[str, float], title: str = "Model Metrics") -> None:
    """
    Print metrics in a formatted way.

    Parameters
    ----------
    metrics : Dict[str, float]
        Dictionary of metrics to print.
    title : str, default="Model Metrics"
        Title for the metrics display.

    Examples
    --------
    >>> print_metrics(metrics, "Random Forest Results")
    """
    print(f"\n{'='*50}")
    print(f"{title:^50}")
    print(f"{'='*50}")
    for metric, value in metrics.items():
        print(f"{metric.upper():>15}: {value:>10.4f}")
    print(f"{'='*50}\n")


def set_random_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility.

    Parameters
    ----------
    seed : int, default=42
        Random seed value.

    Examples
    --------
    >>> set_random_seed(42)
    """
    np.random.seed(seed)
    # If using other libraries, set their seeds too
    try:
        import random
        random.seed(seed)
    except ImportError:
        pass


def create_submission_file(
    predictions: np.ndarray,
    ids: List[Any],
    output_path: str,
    columns: List[str] = ['id', 'prediction']
) -> None:
    """
    Create a submission file for competitions or model outputs.

    Parameters
    ----------
    predictions : np.ndarray
        Array of predictions.
    ids : List[Any]
        List of identifiers corresponding to predictions.
    output_path : str
        Path where to save the submission file.
    columns : List[str], default=['id', 'prediction']
        Column names for the submission file.

    Examples
    --------
    >>> create_submission_file(predictions, test_ids, 'submissions/submission.csv')
    """
    submission_df = pd.DataFrame({
        columns[0]: ids,
        columns[1]: predictions
    })
    submission_df.to_csv(output_path, index=False)
    print(f"Submission file saved to: {output_path}")

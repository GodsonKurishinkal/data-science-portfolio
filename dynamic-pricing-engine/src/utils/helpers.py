"""
Helper utilities for Dynamic Pricing Engine
"""

import yaml
import logging
import pickle
from pathlib import Path
from typing import Any, Dict


def load_config(config_path: str = 'config/config.yaml') -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to configuration file

    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_logging(config: Dict[str, Any] = None) -> logging.Logger:
    """
    Set up logging configuration.

    Args:
        config: Configuration dictionary with logging settings

    Returns:
        Configured logger instance
    """
    if config is None:
        config = load_config()

    log_config = config.get('logging', {})
    level = getattr(logging, log_config.get('level', 'INFO'))
    format_str = log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Create logs directory if specified
    log_file = log_config.get('file')
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=level,
        format=format_str,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file) if log_file else logging.NullHandler()
        ]
    )

    return logging.getLogger(__name__)


def save_results(data: Any, filepath: str, format: str = 'pickle') -> None:
    """
    Save results to file.

    Args:
        data: Data to save
        filepath: Path to save file
        format: Format to use ('pickle', 'csv', 'json')
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)

    if format == 'pickle':
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    elif format == 'csv':
        data.to_csv(filepath, index=False)
    elif format == 'json':
        data.to_json(filepath, orient='records', indent=2)
    else:
        raise ValueError(f"Unsupported format: {format}")


def load_results(filepath: str, format: str = 'pickle') -> Any:
    """
    Load results from file.

    Args:
        filepath: Path to load file
        format: Format to use ('pickle', 'csv', 'json')

    Returns:
        Loaded data
    """
    if format == 'pickle':
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    elif format == 'csv':
        import pandas as pd
        return pd.read_csv(filepath)
    elif format == 'json':
        import pandas as pd
        return pd.read_json(filepath, orient='records')
    else:
        raise ValueError(f"Unsupported format: {format}")

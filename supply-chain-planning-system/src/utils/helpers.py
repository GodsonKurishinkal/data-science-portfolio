"""
Helper utilities for Supply Chain Planning System.
"""

from typing import Dict, Any, List
from pathlib import Path


def validate_config(config: Dict[str, Any], required_keys: List[str]) -> bool:
    """
    Validate configuration has required keys.
    
    Args:
        config: Configuration dictionary
        required_keys: List of required key names
        
    Returns:
        True if valid, raises ValueError if not
    """
    missing = [k for k in required_keys if k not in config]
    
    if missing:
        raise ValueError(f"Missing required configuration keys: {missing}")
    
    return True


def format_currency(value: float, currency: str = "USD") -> str:
    """
    Format a value as currency.
    
    Args:
        value: Numeric value
        currency: Currency code
        
    Returns:
        Formatted currency string
    """
    symbols = {"USD": "$", "EUR": "€", "GBP": "£", "AED": "AED "}
    symbol = symbols.get(currency, currency + " ")
    
    if abs(value) >= 1_000_000:
        return f"{symbol}{value/1_000_000:.1f}M"
    elif abs(value) >= 1_000:
        return f"{symbol}{value/1_000:.1f}K"
    else:
        return f"{symbol}{value:.2f}"


def format_percentage(value: float, decimals: int = 1) -> str:
    """
    Format a value as percentage.
    
    Args:
        value: Decimal value (0.15 = 15%)
        decimals: Number of decimal places
        
    Returns:
        Formatted percentage string
    """
    return f"{value * 100:.{decimals}f}%"


def ensure_directory(path: str) -> Path:
    """
    Ensure a directory exists.
    
    Args:
        path: Directory path
        
    Returns:
        Path object
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def flatten_dict(d: Dict[str, Any], parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
    """
    Flatten a nested dictionary.
    
    Args:
        d: Dictionary to flatten
        parent_key: Parent key prefix
        sep: Separator for keys
        
    Returns:
        Flattened dictionary
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

"""
Pytest configuration file.
"""

import pytest
import numpy as np


@pytest.fixture(autouse=True)
def reset_random_seed():
    """Reset random seed before each test for reproducibility."""
    np.random.seed(42)

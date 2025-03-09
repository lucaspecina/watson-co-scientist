"""
Configuration for pytest in the Watson Co-Scientist project.
"""

import os
import sys
import pytest

# Add the root directory to the Python path to allow imports from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Define common fixtures here if needed
@pytest.fixture
def test_data_dir():
    """Return the path to the test data directory."""
    return os.path.join(os.path.dirname(__file__), "data", "small_dataset")

@pytest.fixture
def full_test_data_dir():
    """Return the path to the full test data directory."""
    return os.path.join(os.path.dirname(__file__), "data", "full_dataset")

@pytest.fixture
def test_fixtures_dir():
    """Return the path to the test fixtures directory."""
    return os.path.join(os.path.dirname(__file__), "data", "test_fixtures")
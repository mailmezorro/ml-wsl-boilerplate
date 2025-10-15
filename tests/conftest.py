"""
Test configuration and fixtures for ML project tests.
"""

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# Test data fixtures
@pytest.fixture
def sample_classification_data() -> tuple[pd.DataFrame, pd.Series]:
    """Generate sample classification dataset."""
    np.random.seed(42)
    n_samples = 1000
    n_features = 10

    # Generate features
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f"feature_{i}" for i in range(n_features)],
    )

    # Generate target with some correlation to features
    y = pd.Series(
        (
            X["feature_0"] + X["feature_1"] * 0.5 + np.random.randn(n_samples) * 0.1 > 0
        ).astype(int),
        name="target",
    )

    return X, y


@pytest.fixture
def sample_regression_data() -> tuple[pd.DataFrame, pd.Series]:
    """Generate sample regression dataset."""
    np.random.seed(42)
    n_samples = 1000
    n_features = 5

    # Generate features
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f"feature_{i}" for i in range(n_features)],
    )

    # Generate target with linear relationship
    y = pd.Series(
        X["feature_0"] * 2
        + X["feature_1"] * -1.5
        + X["feature_2"] * 0.8
        + np.random.randn(n_samples) * 0.1,
        name="target",
    )

    return X, y


@pytest.fixture
def temp_data_dir():
    """Create temporary directory for test data."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_csv_file(temp_data_dir, sample_classification_data):
    """Create sample CSV file for testing."""
    X, y = sample_classification_data
    df = X.copy()
    df["target"] = y

    csv_path = temp_data_dir / "sample_data.csv"
    df.to_csv(csv_path, index=False)

    return csv_path


@pytest.fixture
def config_for_testing():
    """Test configuration."""
    from omegaconf import OmegaConf

    test_config = OmegaConf.create(
        {
            "data": {
                "train_test_split": 0.8,
                "validation_split": 0.1,
                "random_state": 42,
            },
            "model": {
                "type": "sklearn",
                "sklearn": {
                    "n_estimators": 10,  # Small for fast testing
                    "random_state": 42,
                    "n_jobs": 1,
                },
            },
            "training": {
                "cv_folds": 3,  # Small for fast testing
                "cv_strategy": "stratified",
            },
            "mlflow": {
                "enabled": False  # Disable for testing
            },
            "logging": {
                "level": "WARNING"  # Reduce log noise in tests
            },
        }
    )

    return test_config

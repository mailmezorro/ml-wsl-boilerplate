"""
Tests for model training functionality.
"""

from unittest.mock import MagicMock, patch

import pytest
from src.models.trainer import BaseTrainer, SklearnTrainer, get_trainer


class MockTrainer(BaseTrainer):
    """Mock trainer for testing abstract base class."""

    def create_model(self):
        """Create a mock model."""
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(n_estimators=10, random_state=42)


class TestBaseTrainer:
    """Test cases for BaseTrainer abstract class."""

    def test_trainer_initialization(self):
        """Test trainer initialization."""
        trainer = MockTrainer("test_model")
        assert trainer.model_name == "test_model"
        assert trainer.model is None
        assert not trainer.is_fitted

    def test_fit_basic(self, sample_classification_data):
        """Test basic model fitting."""
        X, y = sample_classification_data
        trainer = MockTrainer("test_model")

        # Mock the config to disable MLflow
        with patch.object(trainer, "config") as mock_config:
            mock_config.get.return_value = False

            metrics = trainer.fit(X, y)

            assert trainer.is_fitted
            assert "train_accuracy" in metrics
            assert "train_precision" in metrics
            assert "train_recall" in metrics
            assert "train_f1" in metrics

    def test_fit_with_validation(self, sample_classification_data):
        """Test model fitting with validation data."""
        X, y = sample_classification_data

        # Split data for validation
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

        trainer = MockTrainer("test_model")

        with patch.object(trainer, "config") as mock_config:
            mock_config.get.return_value = False

            metrics = trainer.fit(X_train, y_train, validation_data=(X_val, y_val))

            assert "train_accuracy" in metrics
            assert "val_accuracy" in metrics

    def test_evaluate_not_fitted(self, sample_classification_data):
        """Test evaluation without fitting first."""
        X, y = sample_classification_data
        trainer = MockTrainer("test_model")

        with pytest.raises(ValueError, match="Model must be fitted"):
            trainer.evaluate(X, y)

    def test_evaluate_fitted(self, sample_classification_data):
        """Test model evaluation after fitting."""
        X, y = sample_classification_data
        trainer = MockTrainer("test_model")

        with patch.object(trainer, "config") as mock_config:
            mock_config.get.return_value = False
            trainer.fit(X, y)

            metrics = trainer.evaluate(X, y)

            assert "accuracy" in metrics
            assert "precision" in metrics
            assert "recall" in metrics
            assert "f1" in metrics
            assert 0 <= metrics["accuracy"] <= 1

    def test_cross_validate(self, sample_classification_data):
        """Test cross-validation functionality."""
        X, y = sample_classification_data
        trainer = MockTrainer("test_model")

        with patch.object(trainer, "config") as mock_config:
            mock_config.get.side_effect = lambda key, default=None: {
                "training.cv_strategy": "stratified"
            }.get(key, default)

            cv_results = trainer.cross_validate(X, y, cv_folds=3)

            assert "cv_mean" in cv_results
            assert "cv_std" in cv_results
            assert "cv_scores" in cv_results
            assert "cv_folds" in cv_results
            assert len(cv_results["cv_scores"]) == 3

    def test_save_load_model(self, sample_classification_data, temp_data_dir):
        """Test model saving and loading."""
        X, y = sample_classification_data
        trainer = MockTrainer("test_model")

        with patch.object(trainer, "config") as mock_config:
            mock_config.get.side_effect = lambda key, default=None: {
                "mlflow.enabled": False,
                "paths.models_dir": str(temp_data_dir),
            }.get(key, default)

            # Train and save model
            trainer.fit(X, y)
            model_path = trainer.save_model()

            assert model_path.endswith(".joblib")

            # Create new trainer and load model
            new_trainer = MockTrainer("loaded_model")
            new_trainer.load_model(model_path)

            assert new_trainer.is_fitted
            assert new_trainer.model is not None

    def test_save_model_not_fitted(self):
        """Test saving model that hasn't been fitted."""
        trainer = MockTrainer("test_model")

        with pytest.raises(ValueError, match="Model must be fitted"):
            trainer.save_model()


class TestSklearnTrainer:
    """Test cases for SklearnTrainer."""

    def test_create_model(self):
        """Test sklearn model creation."""
        trainer = SklearnTrainer("sklearn_model")
        model = trainer.create_model()

        from sklearn.ensemble import RandomForestClassifier

        assert isinstance(model, RandomForestClassifier)

    def test_full_training_pipeline(self, sample_classification_data):
        """Test complete training pipeline."""
        X, y = sample_classification_data
        trainer = SklearnTrainer("sklearn_model")

        with patch.object(trainer, "config") as mock_config:
            mock_config.get.side_effect = lambda key, default=None: {
                "mlflow.enabled": False,
                "training.cv_strategy": "stratified",
            }.get(key, default)
            mock_config.get_model_config.return_value = {
                "n_estimators": 10,
                "random_state": 42,
                "n_jobs": 1,
            }

            # Train model
            train_metrics = trainer.fit(X, y)

            # Cross-validate
            cv_results = trainer.cross_validate(X, y, cv_folds=3)

            # Evaluate
            test_metrics = trainer.evaluate(X, y)

            assert trainer.is_fitted
            assert "train_accuracy" in train_metrics
            assert "cv_mean" in cv_results
            assert "accuracy" in test_metrics


class TestGetTrainer:
    """Test cases for trainer factory function."""

    def test_get_trainer_sklearn(self):
        """Test getting sklearn trainer."""
        trainer = get_trainer("sklearn")
        assert isinstance(trainer, SklearnTrainer)
        assert trainer.model_name == "sklearn"

    def test_get_trainer_invalid_type(self):
        """Test getting trainer with invalid type."""
        with pytest.raises(ValueError, match="Unknown model type"):
            get_trainer("invalid_model_type")

    def test_get_trainer_default(self):
        """Test getting trainer with default config."""
        with patch("src.models.trainer.get_config") as mock_get_config:
            mock_config = MagicMock()
            mock_config.get.return_value = "sklearn"
            mock_get_config.return_value = mock_config

            trainer = get_trainer()
            assert isinstance(trainer, SklearnTrainer)

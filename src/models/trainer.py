"""
Base model trainer with MLflow integration and cross-validation.
"""

import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score

# Try to import MLflow, fallback if not available
try:
    import mlflow
    from mlflow import sklearn as mlflow_sklearn

    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

from ..utils.config import get_config
from ..utils.logging_config import get_logger

logger = get_logger(component="training")


class BaseTrainer(ABC):
    """
    Abstract base class for model training with MLflow integration.

    Provides:
    - Cross-validation
    - Metrics calculation
    - Model persistence
    - Experiment tracking
    """

    def __init__(self, model_name: str = "base_model"):
        """
        Initialize trainer.

        Args:
            model_name: Name for the model
        """
        self.model_name = model_name
        self.config = get_config()
        self.model = None
        self.is_fitted = False

        # Setup MLflow if available
        if MLFLOW_AVAILABLE and self.config.get("mlflow.enabled", False):
            self._setup_mlflow()

    def _setup_mlflow(self) -> None:
        """Setup MLflow tracking."""
        tracking_uri = self.config.get("mlflow.tracking_uri", "file:./mlruns")
        experiment_name = self.config.get("mlflow.experiment_name", "ml-experiment")

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

        logger.info(f"MLflow tracking setup: {tracking_uri}")

    @abstractmethod
    def create_model(self) -> Any:
        """Create and return the model instance."""

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        validation_data: tuple | None = None,
    ) -> dict[str, Any]:
        """
        Train the model.

        Args:
            X: Training features
            y: Training targets
            validation_data: Optional (X_val, y_val) tuple

        Returns:
            Training metrics
        """
        start_time = time.time()

        # Create model if not exists
        if self.model is None:
            self.model = self.create_model()

        logger.info(f"Starting training for {self.model_name}")

        # Start MLflow run if available
        if MLFLOW_AVAILABLE and self.config.get("mlflow.enabled", False):
            mlflow.start_run(run_name=f"{self.model_name}_{int(time.time())}")

        try:
            # Train the model
            self.model.fit(X, y)
            self.is_fitted = True

            # Calculate training metrics
            train_metrics = self.evaluate(X, y, stage="training")

            # Validation metrics if provided
            val_metrics = {}
            if validation_data:
                X_val, y_val = validation_data
                val_metrics = self.evaluate(X_val, y_val, stage="validation")

            # Log training time
            training_time = time.time() - start_time
            logger.info(f"Model training completed in {training_time:.2f} seconds")

            # Log to MLflow
            if MLFLOW_AVAILABLE and self.config.get("mlflow.enabled", False):
                self._log_to_mlflow(train_metrics, val_metrics, training_time)

            # Combine metrics
            all_metrics = {"train_" + k: v for k, v in train_metrics.items()}
            if val_metrics:
                all_metrics.update({"val_" + k: v for k, v in val_metrics.items()})

            logger.info("Training completed successfully")
            return all_metrics

        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
        finally:
            if MLFLOW_AVAILABLE and mlflow.active_run():
                mlflow.end_run()

    def cross_validate(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        cv_folds: int = 5,
        scoring: str = "accuracy",
    ) -> dict[str, Any]:
        """
        Perform cross-validation.

        Args:
            X: Features
            y: Targets
            cv_folds: Number of CV folds
            scoring: Scoring metric

        Returns:
            Cross-validation results
        """
        logger.info(f"Starting {cv_folds}-fold cross-validation")

        # Create model if not exists
        if self.model is None:
            self.model = self.create_model()

        # Choose CV strategy
        cv_strategy = self.config.get("training.cv_strategy", "stratified")
        if cv_strategy == "stratified":
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        else:
            cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)

        # Perform cross-validation
        cv_scores = cross_val_score(self.model, X, y, cv=cv, scoring=scoring, n_jobs=-1)

        cv_results = {
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(),
            "cv_scores": cv_scores.tolist(),
            "cv_folds": cv_folds,
        }

        logger.info(
            f"CV {scoring}: {cv_results['cv_mean']:.4f} (±{cv_results['cv_std']:.4f})"
        )

        return cv_results

    def evaluate(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        stage: str = "test",
    ) -> dict[str, float]:
        """
        Evaluate model performance.

        Args:
            X: Features
            y: True targets
            stage: Evaluation stage name

        Returns:
            Evaluation metrics
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before evaluation")

        # Make predictions
        if self.model is None:
            raise ValueError("Model must be created and fitted before evaluation")
        y_pred = self.model.predict(X)

        # Calculate metrics
        metrics = {
            "accuracy": accuracy_score(y, y_pred),
            "precision": precision_score(
                y, y_pred, average="weighted", zero_division=0
            ),
            "recall": recall_score(y, y_pred, average="weighted", zero_division=0),
            "f1": f1_score(y, y_pred, average="weighted", zero_division=0),
        }

        # Add ROC-AUC for binary classification
        if len(np.unique(y)) == 2:
            try:
                if self.model is None:
                    raise ValueError(
                        "Model must be created before calling predict_proba"
                    )
                y_proba = self.model.predict_proba(X)[:, 1]
                metrics["roc_auc"] = roc_auc_score(y, y_proba)
            except AttributeError:
                pass  # Model doesn't support predict_proba

        # Log metrics
        logger.info(f"Metrics for {stage}: {metrics}")

        return metrics

    def save_model(self, filepath: str | None = None) -> str:
        """
        Save the trained model.

        Args:
            filepath: Path to save model. If None, auto-generate.

        Returns:
            Path where model was saved
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")

        if filepath is None:
            models_dir = Path(self.config.get("paths.models_dir", "models"))
            models_dir.mkdir(exist_ok=True)
            filepath = str(models_dir / f"{self.model_name}_{int(time.time())}.joblib")

        joblib.dump(self.model, filepath)
        logger.info(f"Model saved to {filepath}")

        return str(filepath)

    def load_model(self, filepath: str) -> None:
        """
        Load a saved model.

        Args:
            filepath: Path to the saved model
        """
        self.model = joblib.load(filepath)
        self.is_fitted = True
        logger.info(f"Model loaded from {filepath}")

    def _log_to_mlflow(
        self, train_metrics: dict, val_metrics: dict, training_time: float
    ) -> None:
        """Log metrics and model to MLflow."""
        if not MLFLOW_AVAILABLE:
            return

        # Log parameters
        if self.model is not None and hasattr(self.model, "get_params"):
            mlflow.log_params(self.model.get_params())

        # Log metrics
        for metric, value in train_metrics.items():
            mlflow.log_metric(f"train_{metric}", value)

        for metric, value in val_metrics.items():
            mlflow.log_metric(f"val_{metric}", value)

        mlflow.log_metric("training_time_seconds", training_time)

        # Log model
        if self.config.get("mlflow.log_model", True):
            mlflow_sklearn.log_model(self.model, "model")


class XGBoostTrainer(BaseTrainer):
    """XGBoost model trainer."""

    def create_model(self):
        """Create XGBoost classifier."""
        try:
            import xgboost as xgb
        except ImportError:
            raise ImportError(
                "XGBoost not installed. Run: pip install xgboost"
            ) from None

        model_config = self.config.get_model_config()
        return xgb.XGBClassifier(**{str(k): v for k, v in model_config.items()})


class LightGBMTrainer(BaseTrainer):
    """LightGBM model trainer."""

    def create_model(self):
        """Create LightGBM classifier."""
        try:
            import lightgbm as lgb
        except ImportError:
            raise ImportError(
                "LightGBM not installed. Run: pip install lightgbm"
            ) from None

        model_config = self.config.get_model_config()
        return lgb.LGBMClassifier(**{str(k): v for k, v in model_config.items()})


class SklearnTrainer(BaseTrainer):
    """Scikit-learn model trainer."""

    def create_model(self):
        """Create sklearn classifier."""
        from sklearn.ensemble import RandomForestClassifier

        model_config = self.config.get_model_config()
        return RandomForestClassifier(**{str(k): v for k, v in model_config.items()})


def get_trainer(model_type: str | None = None) -> BaseTrainer:
    """
    Factory function to get appropriate trainer.

    Args:
        model_type: Type of model ('xgboost', 'lightgbm', 'sklearn')

    Returns:
        Trainer instance
    """
    config = get_config()
    if model_type is None:
        model_type = config.get("model.type", "xgboost")

    trainers = {
        "xgboost": XGBoostTrainer,
        "lightgbm": LightGBMTrainer,
        "sklearn": SklearnTrainer,
    }

    if model_type not in trainers:
        raise ValueError(
            f"Unknown model type: {model_type}. Available: {list(trainers.keys())}"
        )

    return trainers[model_type](model_name=model_type)

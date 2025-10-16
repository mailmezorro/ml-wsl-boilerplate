#!/usr/bin/env python3
"""
Main training script for ML WSL Boilerplate.

This script provides a complete training pipeline with:
- Data loading and preprocessing
- Model training with cross-validation
- Model evaluation and metrics logging
- MLflow experiment tracking
- Model persistence

Usage:
    python scripts/train.py [--config CONFIG_PATH] [--model MODEL_TYPE]

Example:
    python scripts/train.py
    python scripts/train.py --config config/experiment1.yaml
    python scripts/train.py --model xgboost
"""

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from src.data.loader import DataLoader
from src.models.trainer import get_trainer
from src.utils.config import get_config, load_config
from src.utils.logging_config import get_logger, setup_logging

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))


# Setup logging
ml_logger = setup_logging()
logger = get_logger(component="training")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train ML model")

    parser.add_argument(
        "--config", type=str, default=None, help="Path to configuration file"
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        choices=["xgboost", "lightgbm", "sklearn"],
        help="Model type to train",
    )

    parser.add_argument(
        "--data", type=str, default=None, help="Path to training data CSV file"
    )

    parser.add_argument(
        "--experiment", type=str, default=None, help="Experiment name for MLflow"
    )

    parser.add_argument("--no-cv", action="store_true", help="Skip cross-validation")

    parser.add_argument(
        "--save-model", action="store_true", help="Save trained model to disk"
    )

    return parser.parse_args()


def load_data(config, data_path=None):
    """
    Load and prepare training data.

    Args:
        config: Configuration object
        data_path: Optional path to data file

    Returns:
        X_train, X_test, y_train, y_test: Training and test sets
    """
    if data_path and Path(data_path).exists():
        # Load real data
        logger.info(f"Loading data from {data_path}")
        data_loader = DataLoader()
        df = data_loader.load_csv(Path(data_path).name)

        # Assume last column is target
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]

        # Log data info
        ml_logger.log_data_info("Training Data", X.shape)

    else:
        # Generate synthetic data for demo
        logger.info("Generating synthetic classification data")
        X, y = make_classification(
            n_samples=config.get("data.n_samples", 10000),
            n_features=config.get("data.n_features", 20),
            n_informative=config.get("data.n_informative", 15),
            n_redundant=config.get("data.n_redundant", 5),
            n_clusters_per_class=config.get("data.n_clusters_per_class", 1),
            random_state=config.get("environment.seed", 42),
        )

        X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
        y = pd.Series(y, name="target")

        ml_logger.log_data_info("Synthetic Data", X.shape)

    # Validate data
    data_loader = DataLoader()
    validation_results = data_loader.validate_data(pd.concat([X, y], axis=1))
    logger.info(f"Data validation: {validation_results['duplicates']} duplicates found")

    # Split data
    test_size = config.get("data.train_test_split", 0.8)
    test_size = 1.0 - test_size  # Convert to test size

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=config.get("environment.seed", 42),
        stratify=y,
    )

    logger.info(f"Data split: {len(X_train)} train, {len(X_test)} test samples")

    return X_train, X_test, y_train, y_test


def train_model(
    config, X_train, y_train, X_test, y_test, model_type=None, skip_cv=False
):
    """
    Train and evaluate model.

    Args:
        config: Configuration object
        X_train, y_train: Training data
        X_test, y_test: Test data
        model_type: Model type override
        skip_cv: Skip cross-validation

    Returns:
        trainer: Trained model trainer
        results: Training results dictionary
    """
    # Get trainer
    if model_type:
        config.set("model.type", model_type)

    trainer = get_trainer()

    # Log model configuration
    model_config = config.get_model_config()
    ml_logger.log_model_info(config.get("model.type"), model_config)

    # Start experiment tracking
    experiment_name = f"training_{config.get('model.type')}_{int(time.time())}"
    ml_logger.log_experiment_start(experiment_name)

    results = {}

    try:
        # Cross-validation
        if not skip_cv:
            cv_folds = config.get("training.cv_folds", 5)
            logger.info(f"Running {cv_folds}-fold cross-validation")

            cv_results = trainer.cross_validate(
                X_train, y_train, cv_folds=cv_folds, scoring="accuracy"
            )
            results["cv_results"] = cv_results

            logger.info(
                f"CV Accuracy: {cv_results['cv_mean']:.4f} (±{cv_results['cv_std']:.4f})"
            )

        # Train model
        logger.info("Training final model")
        validation_split = config.get("data.validation_split", 0.1)

        if validation_split > 0:
            # Split training data for validation
            X_train_split, X_val, y_train_split, y_val = train_test_split(
                X_train,
                y_train,
                test_size=validation_split,
                random_state=config.get("environment.seed", 42),
                stratify=y_train,
            )

            train_metrics = trainer.fit(
                X_train_split, y_train_split, validation_data=(X_val, y_val)
            )
        else:
            train_metrics = trainer.fit(X_train, y_train)

        results["train_metrics"] = train_metrics

        # Evaluate on test set
        logger.info("Evaluating on test set")
        test_metrics = trainer.evaluate(X_test, y_test, stage="test")
        results["test_metrics"] = test_metrics

        # Log final metrics
        ml_logger.log_metrics(test_metrics, "FINAL TEST")

        ml_logger.log_experiment_end(success=True)

        return trainer, results

    except Exception as e:
        logger.error(f"Training failed: {e}")
        ml_logger.log_experiment_end(success=False)
        raise


def main():
    """Main training function."""
    args = parse_arguments()

    # Load configuration
    if args.config:
        config = load_config(args.config)
        logger.info(f"Loaded config from {args.config}")
    else:
        config = get_config()
        logger.info("Using default configuration")

    # Log configuration
    config_dict = {
        "model_type": config.get("model.type"),
        "data_split": config.get("data.train_test_split"),
        "cv_folds": config.get("training.cv_folds"),
        "random_seed": config.get("environment.seed"),
    }
    ml_logger.log_config(config_dict)

    # Validate configuration
    if not config.validate_config():
        logger.error("Configuration validation failed")
        sys.exit(1)

    # Load data
    try:
        X_train, X_test, y_train, y_test = load_data(config, args.data)
    except Exception as e:
        logger.error(f"Data loading failed: {e}")
        sys.exit(1)

    # Train model
    try:
        trainer, results = train_model(
            config,
            X_train,
            y_train,
            X_test,
            y_test,
            model_type=args.model,
            skip_cv=args.no_cv,
        )
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)

    # Save model if requested
    if args.save_model:
        try:
            model_path = trainer.save_model()
            logger.info(f"Model saved to {model_path}")
        except Exception as e:
            logger.error(f"Model saving failed: {e}")

    # Print summary
    logger.info("=" * 60)
    logger.info("TRAINING COMPLETED SUCCESSFULLY!")
    logger.info("=" * 60)

    if "test_metrics" in results:
        test_acc = results["test_metrics"].get("accuracy", 0)
        logger.info(f"Final Test Accuracy: {test_acc:.4f}")

    if "cv_results" in results:
        cv_acc = results["cv_results"]["cv_mean"]
        logger.info(f"Cross-validation Accuracy: {cv_acc:.4f}")

    logger.info("=" * 60)


if __name__ == "__main__":
    main()

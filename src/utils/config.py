"""
Configuration management using OmegaConf for flexible, hierarchical configs.
"""

import logging
import os
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


class ConfigManager:
    """
    Centralized configuration management for ML projects.

    Features:
    - YAML-based configuration with OmegaConf
    - Environment variable overrides
    - Config validation and merging
    - Structured access to nested configs
    """

    def __init__(self, config_path: str | None = None):
        """
        Initialize configuration manager.

        Args:
            config_path: Path to main config file. If None, uses default.
        """
        if config_path is None:
            config_path = str(
                Path(__file__).parent.parent.parent / "config" / "config.yaml"
            )

        self.config_path = Path(config_path)
        self.config = self._load_config()
        self._setup_directories()

    def _load_config(self) -> DictConfig:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        config = OmegaConf.load(self.config_path)

        # Apply environment variable overrides
        if isinstance(config, DictConfig):
            self._apply_env_overrides(config)
        else:
            raise TypeError(f"Expected DictConfig, got {type(config).__name__}")

        logger.info(f"Configuration loaded from {self.config_path}")
        return config

    def _apply_env_overrides(self, config: DictConfig) -> None:
        """Apply environment variable overrides to config."""
        # Example: ML_MODEL_TYPE=lightgbm overrides model.type
        env_mappings = {
            "ML_MODEL_TYPE": "model.type",
            "ML_DATA_DIR": "paths.data_dir",
            "ML_LOG_LEVEL": "logging.level",
            "ML_MLFLOW_URI": "mlflow.tracking_uri",
            "ML_RANDOM_SEED": "environment.seed",
        }

        for env_var, config_path in env_mappings.items():
            if env_var in os.environ:
                value = os.environ[env_var]
                # Try to convert to appropriate type
                try:
                    if value.lower() in ("true", "false"):
                        value = value.lower() == "true"
                    elif value.isdigit():
                        value = int(value)
                    elif value.replace(".", "").isdigit():
                        value = float(value)
                except Exception:
                    pass  # Keep as string

                OmegaConf.update(config, config_path, value, merge=True)
                logger.info(f"Applied env override: {env_var}={value} -> {config_path}")

    def _setup_directories(self) -> None:
        """Create necessary directories from config."""
        directories = [
            self.config.paths.data_dir,
            self.config.paths.models_dir,
            self.config.paths.logs_dir,
            self.config.paths.outputs_dir,
            self.config.paths.figures_dir,
            self.config.paths.reports_dir,
        ]

        for dir_path in directories:
            Path(dir_path).mkdir(parents=True, exist_ok=True)

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.

        Args:
            key: Configuration key (e.g., 'model.xgboost.n_estimators')
            default: Default value if key not found

        Returns:
            Configuration value
        """
        try:
            return OmegaConf.select(self.config, key, default=default)
        except Exception:
            return default

    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value using dot notation.

        Args:
            key: Configuration key
            value: Value to set
        """
        OmegaConf.update(self.config, key, value, merge=True)
        logger.debug(f"Config updated: {key}={value}")

    def get_model_config(self) -> DictConfig:
        """Get model-specific configuration."""
        model_type = self.config.model.type
        return self.config.model[model_type]

    def get_data_config(self) -> DictConfig:
        """Get data configuration."""
        return self.config.data

    def get_training_config(self) -> DictConfig:
        """Get training configuration."""
        return self.config.training

    def save_config(self, output_path: str | None = None) -> None:
        """
        Save current configuration to file.

        Args:
            output_path: Path to save config. If None, overwrites original.
        """
        if output_path is None:
            output_path = str(self.config_path)

        OmegaConf.save(self.config, output_path)
        logger.info(f"Configuration saved to {output_path}")

    def merge_config(self, other_config: dict[str, Any] | DictConfig) -> None:
        """
        Merge another configuration into current config.

        Args:
            other_config: Dictionary or DictConfig to merge
        """
        if isinstance(other_config, dict):
            other_config = OmegaConf.create(other_config)

        self.config = OmegaConf.merge(self.config, other_config)
        logger.info("Configuration merged successfully")

    def validate_config(self) -> bool:
        """
        Validate configuration for common issues.

        Returns:
            True if valid, False otherwise
        """
        try:
            # Check required fields
            required_fields = [
                "project.name",
                "model.type",
                "data.train_test_split",
                "paths.data_dir",
            ]

            for field in required_fields:
                if OmegaConf.select(self.config, field) is None:
                    logger.error(f"Missing required config field: {field}")
                    return False

            # Validate data split ratios
            train_split = self.config.data.train_test_split
            val_split = self.config.data.validation_split

            if not (0 < train_split < 1):
                logger.error(f"Invalid train_test_split: {train_split}")
                return False

            if not (0 <= val_split < 1):
                logger.error(f"Invalid validation_split: {val_split}")
                return False

            if train_split + val_split >= 1:
                logger.error("Sum of train and validation splits must be < 1")
                return False

            logger.info("Configuration validation passed")
            return True

        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            return False

    def print_config(self) -> None:
        """Print current configuration in a readable format."""
        print(OmegaConf.to_yaml(self.config))


# Global config instance (singleton pattern)
_config_manager = None


def get_config() -> ConfigManager:
    """Get global configuration manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


def load_config(config_path: str) -> ConfigManager:
    """Load configuration from specific path."""
    global _config_manager
    _config_manager = ConfigManager(config_path)
    return _config_manager

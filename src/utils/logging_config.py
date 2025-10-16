"""
Centralized logging setup with Rich formatting and file output.
"""

import logging
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler
from rich.traceback import install


class MLLogger:
    """
    Enhanced logging setup for ML projects with Rich formatting.

    Features:
    - Beautiful console output with Rich
    - File logging with rotation
    - Structured log formatting
    - Different log levels for different components
    """

    def __init__(self, name: str = "ml-project", log_dir: str = "logs"):
        """
        Initialize the ML Logger.

        Args:
            name: Logger name
            log_dir: Directory for log files
        """
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        # Install rich traceback handler for better error formatting
        install(show_locals=True)

        self.console = Console()
        self.logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        """Setup logger with Rich and file handlers."""
        logger = logging.getLogger(self.name)
        logger.setLevel(logging.DEBUG)

        # Clear existing handlers to avoid duplicates
        logger.handlers.clear()

        # Console handler with Rich formatting
        console_handler = RichHandler(
            console=self.console,
            show_time=True,
            show_path=False,
            markup=True,
            rich_tracebacks=True,
        )
        console_handler.setLevel(logging.INFO)

        # File handler for detailed logs
        log_file = self.log_dir / f"{self.name}_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)

        # Formatters
        console_format = "%(message)s"
        file_format = "%(asctime)s | %(name)s | %(levelname)s | %(filename)s:%(lineno)d | %(message)s"

        console_handler.setFormatter(logging.Formatter(console_format))
        file_handler.setFormatter(logging.Formatter(file_format))

        # Add handlers
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

        return logger

    def get_logger(self, component: str | None = None) -> logging.Logger:
        """
        Get a logger for a specific component.

        Args:
            component: Component name (e.g., 'data', 'model', 'training')

        Returns:
            Logger instance
        """
        if component:
            return logging.getLogger(f"{self.name}.{component}")
        return self.logger

    def log_config(self, config_dict: dict) -> None:
        """Log configuration in a structured way."""
        self.logger.info("=" * 50)
        self.logger.info("🔧 [bold blue]CONFIGURATION[/bold blue]")
        self.logger.info("=" * 50)

        for key, value in config_dict.items():
            if isinstance(value, dict):
                self.logger.info(f"[bold]{key}[/bold]:")
                for subkey, subvalue in value.items():
                    self.logger.info(f"  • {subkey}: {subvalue}")
            else:
                self.logger.info(f"🔹 {key}: {value}")

        self.logger.info("=" * 50)

    def log_data_info(
        self, name: str, shape: tuple, memory_mb: float | None = None
    ) -> None:
        """Log data information."""
        memory_str = f" ({memory_mb:.1f} MB)" if memory_mb else ""
        self.logger.info(
            f"[bold green]{name}[/bold green]: {shape[0]:,} rows x {shape[1]} cols{memory_str}"
        )

    def log_model_info(self, model_name: str, params: dict) -> None:
        """Log model information."""
        self.logger.info(f"[bold cyan]Model: {model_name}[/bold cyan]")
        for param, value in params.items():
            self.logger.info(f"  • {param}: {value}")

    def log_metrics(self, metrics: dict, stage: str = "training") -> None:
        """Log performance metrics."""
        self.logger.info(f"[bold yellow]{stage.upper()} METRICS[/bold yellow]")
        for metric, value in metrics.items():
            if isinstance(value, float):
                self.logger.info(f"  • {metric}: {value:.4f}")
            else:
                self.logger.info(f"  • {metric}: {value}")

    def log_timer(self, operation: str, duration: float) -> None:
        """Log operation timing."""
        if duration < 60:
            time_str = f"{duration:.2f}s"
        elif duration < 3600:
            time_str = f"{duration / 60:.1f}m"
        else:
            time_str = f"{duration / 3600:.1f}h"

        self.logger.info(f"[bold]{operation}[/bold] completed in {time_str}")

    def log_experiment_start(
        self, experiment_name: str, run_id: str | None = None
    ) -> None:
        """Log experiment start."""
        self.logger.info("=" * 50)
        self.logger.info(f"[bold magenta]EXPERIMENT: {experiment_name}[/bold magenta]")
        if run_id:
            self.logger.info(f"Run ID: {run_id}")
        self.logger.info("=" * 50)

    def log_experiment_end(self, success: bool = True) -> None:
        """Log experiment end."""
        status = (
            "[bold green]SUCCESS[/bold green]"
            if success
            else "[bold red]FAILED[/bold red]"
        )
        self.logger.info("=" * 50)
        self.logger.info(f"Experiment completed: {status}")
        self.logger.info("=" * 50)


# Global logger instance
_ml_logger = None


def get_logger(
    name: str = "ml-project", component: str | None = None
) -> logging.Logger:
    """
    Get a configured logger instance.

    Args:
        name: Logger name
        component: Component name

    Returns:
        Logger instance
    """
    global _ml_logger
    if _ml_logger is None:
        _ml_logger = MLLogger(name)

    return _ml_logger.get_logger(component)


def setup_logging(log_level: str = "INFO", log_dir: str = "logs") -> MLLogger:
    """
    Setup global logging configuration.

    Args:
        log_level: Logging level
        log_dir: Directory for log files

    Returns:
        MLLogger instance
    """
    global _ml_logger
    _ml_logger = MLLogger(log_dir=log_dir)

    # Set log level
    level = getattr(logging, log_level.upper(), logging.INFO)
    _ml_logger.logger.setLevel(level)

    return _ml_logger


class LoggerContext:
    """Context manager for temporary logger configuration."""

    def __init__(self, logger_name: str, log_level: str = "INFO"):
        self.logger_name = logger_name
        self.log_level = log_level
        self.original_level = None

    def __enter__(self):
        logger = logging.getLogger(self.logger_name)
        self.original_level = logger.level
        logger.setLevel(getattr(logging, self.log_level.upper()))
        return logger

    def __exit__(self, exc_type, exc_val, exc_tb):
        logger = logging.getLogger(self.logger_name)
        if self.original_level:
            logger.setLevel(self.original_level)

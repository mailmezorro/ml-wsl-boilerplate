"""
Data loading and preprocessing utilities.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DataLoader:
    """
    A flexible data loader for common ML datasets.

    Supports CSV, Excel, Parquet, and JSON formats.
    Includes basic preprocessing and validation.
    """

    def __init__(self, data_dir: str | Path = "data"):
        """
        Initialize the DataLoader.

        Args:
            data_dir: Directory containing data files
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)

    def load_csv(self, filename: str, **kwargs) -> pd.DataFrame:
        """
        Load data from CSV file.

        Args:
            filename: Name of the CSV file
            **kwargs: Additional arguments for pd.read_csv

        Returns:
            Loaded DataFrame
        """
        filepath = self.data_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        logger.info(f"Loading CSV data from {filepath}")
        df = pd.read_csv(filepath, **kwargs)
        logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")
        return df

    def validate_data(self, df: pd.DataFrame) -> dict:
        """
        Perform basic data validation.

        Args:
            df: DataFrame to validate

        Returns:
            Dictionary with validation results
        """
        validation = {
            "shape": df.shape,
            "missing_values": df.isnull().sum().to_dict(),
            "dtypes": df.dtypes.to_dict(),
            "duplicates": df.duplicated().sum(),
            "memory_usage": df.memory_usage(deep=True).sum(),
        }

        logger.info(
            f"Data validation complete: {validation['shape']} shape, "
            f"{validation['duplicates']} duplicates"
        )
        return validation

    def clean_data(
        self,
        df: pd.DataFrame,
        drop_duplicates: bool = True,
        fill_missing: str | None = None,
    ) -> pd.DataFrame:
        """
        Basic data cleaning operations.

        Args:
            df: DataFrame to clean
            drop_duplicates: Whether to remove duplicate rows
            fill_missing: Strategy for missing values ('mean', 'median', 'mode', or None)

        Returns:
            Cleaned DataFrame
        """
        df_clean = df.copy()

        if drop_duplicates:
            initial_rows = len(df_clean)
            df_clean = df_clean.drop_duplicates()
            removed = initial_rows - len(df_clean)
            if removed > 0:
                logger.info(f"Removed {removed} duplicate rows")

        if fill_missing:
            numeric_cols = df_clean.select_dtypes(include=[np.number]).columns

            if fill_missing == "mean":
                df_clean[numeric_cols] = df_clean[numeric_cols].fillna(
                    df_clean[numeric_cols].mean()
                )
            elif fill_missing == "median":
                df_clean[numeric_cols] = df_clean[numeric_cols].fillna(
                    df_clean[numeric_cols].median()
                )
            elif fill_missing == "mode":
                for col in df_clean.columns:
                    mode_val = df_clean[col].mode()
                    if not mode_val.empty:
                        df_clean[col] = df_clean[col].fillna(mode_val[0])

            logger.info(f"Applied {fill_missing} imputation for missing values")

        return df_clean


def train_test_split_temporal(
    df: pd.DataFrame, date_column: str, test_size: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data based on temporal order (important for time series).

    Args:
        df: DataFrame with temporal data
        date_column: Name of the date/datetime column
        test_size: Fraction of data for testing

    Returns:
        Training and testing DataFrames
    """
    df_sorted = df.sort_values(date_column)
    split_idx = int(len(df_sorted) * (1 - test_size))

    train_df = df_sorted.iloc[:split_idx].copy()
    test_df = df_sorted.iloc[split_idx:].copy()

    logger.info(f"Temporal split: {len(train_df)} train, {len(test_df)} test samples")
    return train_df, test_df

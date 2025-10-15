"""
Tests for data loading and processing functionality.
"""

import numpy as np
import pandas as pd
import pytest

from src.data.loader import DataLoader, train_test_split_temporal


class TestDataLoader:
    """Test cases for DataLoader class."""

    def test_init(self, temp_data_dir):
        """Test DataLoader initialization."""
        loader = DataLoader(data_dir=temp_data_dir)
        assert loader.data_dir == temp_data_dir
        assert loader.data_dir.exists()

    def test_load_csv(self, sample_csv_file):
        """Test CSV loading functionality."""
        loader = DataLoader(data_dir=sample_csv_file.parent)
        df = loader.load_csv(sample_csv_file.name)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1000
        assert "target" in df.columns
        assert len(df.columns) == 11  # 10 features + target

    def test_load_csv_file_not_found(self, temp_data_dir):
        """Test CSV loading with non-existent file."""
        loader = DataLoader(data_dir=temp_data_dir)

        with pytest.raises(FileNotFoundError):
            loader.load_csv("nonexistent.csv")

    def test_validate_data(self, sample_classification_data):
        """Test data validation functionality."""
        X, y = sample_classification_data
        df = X.copy()
        df["target"] = y

        loader = DataLoader()
        validation = loader.validate_data(df)

        assert "shape" in validation
        assert "missing_values" in validation
        assert "dtypes" in validation
        assert "duplicates" in validation
        assert "memory_usage" in validation

        assert validation["shape"] == (1000, 11)
        assert validation["duplicates"] == 0

    def test_validate_data_with_missing_values(self):
        """Test validation with missing values."""
        df = pd.DataFrame(
            {"A": [1, 2, np.nan, 4], "B": [1, np.nan, 3, 4], "C": [1, 2, 3, np.nan]}
        )

        loader = DataLoader()
        validation = loader.validate_data(df)

        assert validation["missing_values"]["A"] == 1
        assert validation["missing_values"]["B"] == 1
        assert validation["missing_values"]["C"] == 1

    def test_clean_data_drop_duplicates(self):
        """Test data cleaning with duplicate removal."""
        df = pd.DataFrame({"A": [1, 2, 2, 3], "B": [1, 2, 2, 3], "C": [1, 2, 2, 3]})

        loader = DataLoader()
        cleaned = loader.clean_data(df, drop_duplicates=True)

        assert len(cleaned) == 3  # One duplicate removed
        assert not cleaned.duplicated().any()

    def test_clean_data_fill_missing_mean(self):
        """Test data cleaning with mean imputation."""
        df = pd.DataFrame(
            {
                "A": [1.0, 2.0, np.nan, 4.0],
                "B": [10, 20, 30, 40],
                "C": ["a", "b", "c", "d"],  # Non-numeric column
            }
        )

        loader = DataLoader()
        cleaned = loader.clean_data(df, fill_missing="mean")

        assert not cleaned["A"].isna().any()
        assert cleaned["A"].iloc[2] == 2.333333333333333  # Mean of 1, 2, 4
        assert cleaned["B"].isna().sum() == 0

    def test_clean_data_fill_missing_median(self):
        """Test data cleaning with median imputation."""
        df = pd.DataFrame({"A": [1.0, 2.0, np.nan, 4.0], "B": [10, 20, 30, 40]})

        loader = DataLoader()
        cleaned = loader.clean_data(df, fill_missing="median")

        assert not cleaned["A"].isna().any()
        assert cleaned["A"].iloc[2] == 2.0  # Median of 1, 2, 4


class TestTemporalSplit:
    """Test cases for temporal train-test split."""

    def test_temporal_split_basic(self):
        """Test basic temporal splitting functionality."""
        # Create sample data with dates
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        df = pd.DataFrame(
            {
                "date": dates,
                "feature": np.random.randn(100),
                "target": np.random.randint(0, 2, 100),
            }
        )

        train_df, test_df = train_test_split_temporal(df, "date", test_size=0.2)

        assert len(train_df) == 80
        assert len(test_df) == 20

        # Check temporal order is maintained
        assert train_df["date"].max() <= test_df["date"].min()

    def test_temporal_split_custom_test_size(self):
        """Test temporal split with custom test size."""
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        df = pd.DataFrame({"date": dates, "feature": np.random.randn(100)})

        train_df, test_df = train_test_split_temporal(df, "date", test_size=0.3)

        assert len(train_df) == 70
        assert len(test_df) == 30

    def test_temporal_split_unsorted_data(self):
        """Test temporal split with unsorted data."""
        dates = pd.date_range("2023-01-01", periods=50, freq="D")
        # Shuffle the dates
        shuffled_dates = dates.to_series().sample(frac=1, random_state=42)

        df = pd.DataFrame({"date": shuffled_dates, "feature": np.random.randn(50)})

        train_df, test_df = train_test_split_temporal(df, "date", test_size=0.2)

        # Should still maintain temporal order after sorting
        assert train_df["date"].max() <= test_df["date"].min()
        assert len(train_df) == 40
        assert len(test_df) == 10

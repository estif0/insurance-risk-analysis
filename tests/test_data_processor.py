"""
Unit tests for DataProcessor module.

Tests cover:
- Data initialization and setting
- Missing value handling
- Outlier detection and treatment
- Feature engineering
- Data filtering
- Data saving
- Processing summaries
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil
from core.data_processor import DataProcessor


@pytest.fixture
def sample_data():
    """Create sample insurance data for testing."""
    np.random.seed(42)
    n = 1000

    data = pd.DataFrame(
        {
            "PolicyID": range(1, n + 1),
            "TransactionMonth": pd.date_range("2014-02-01", periods=n, freq="D"),
            "TotalPremium": np.random.uniform(100, 5000, n),
            "TotalClaims": np.random.uniform(0, 3000, n),
            "SumInsured": np.random.uniform(50000, 500000, n),
            "RegistrationYear": np.random.randint(2000, 2024, n),
            "Province": np.random.choice(
                ["Gauteng", "Western Cape", "KwaZulu-Natal"], n
            ),
            "VehicleType": np.random.choice(["Sedan", "SUV", "Truck"], n),
            "Gender": np.random.choice(["Male", "Female", "Not specified"], n),
            "Age": np.random.randint(18, 80, n),
        }
    )

    # Add some missing values
    data.loc[np.random.choice(n, 50, replace=False), "TotalPremium"] = np.nan
    data.loc[np.random.choice(n, 30, replace=False), "Province"] = np.nan

    # Add some outliers
    data.loc[np.random.choice(n, 10, replace=False), "TotalPremium"] = 50000

    # Set some claims to zero
    zero_claims_idx = np.random.choice(n, 300, replace=False)
    data.loc[zero_claims_idx, "TotalClaims"] = 0

    return data


@pytest.fixture
def processor():
    """Create DataProcessor instance."""
    return DataProcessor()


@pytest.fixture
def temp_dir():
    """Create temporary directory for file operations."""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path)


class TestInitialization:
    """Test DataProcessor initialization."""

    def test_init_without_data(self):
        """Test initialization without data."""
        processor = DataProcessor()
        assert processor.data is None
        assert processor.numeric_columns == []
        assert processor.categorical_columns == []

    def test_init_with_data(self, sample_data):
        """Test initialization with data."""
        processor = DataProcessor(sample_data)
        assert processor.data is not None
        assert len(processor.data) == len(sample_data)
        assert len(processor.numeric_columns) > 0
        assert len(processor.categorical_columns) > 0

    def test_set_data(self, processor, sample_data):
        """Test setting data after initialization."""
        processor.set_data(sample_data)
        assert processor.data is not None
        assert len(processor.data) == len(sample_data)

    def test_set_data_invalid_type(self, processor):
        """Test setting invalid data type."""
        with pytest.raises(TypeError):
            processor.set_data([1, 2, 3])

    def test_column_type_identification(self, sample_data):
        """Test automatic column type identification."""
        processor = DataProcessor(sample_data)
        assert "TotalPremium" in processor.numeric_columns
        assert "Province" in processor.categorical_columns


class TestMissingValueHandling:
    """Test missing value handling methods."""

    def test_handle_missing_auto_median(self, sample_data):
        """Test auto strategy with median for numeric."""
        processor = DataProcessor(sample_data)
        missing_before = processor.data["TotalPremium"].isnull().sum()

        processor.handle_missing_values(strategy="auto", numeric_strategy="median")
        missing_after = processor.data["TotalPremium"].isnull().sum()

        assert missing_before > 0
        assert missing_after == 0

    def test_handle_missing_auto_mean(self, sample_data):
        """Test auto strategy with mean for numeric."""
        processor = DataProcessor(sample_data)
        processor.handle_missing_values(strategy="auto", numeric_strategy="mean")

        assert processor.data["TotalPremium"].isnull().sum() == 0

    def test_handle_missing_auto_mode(self, sample_data):
        """Test auto strategy with mode for categorical."""
        processor = DataProcessor(sample_data)
        processor.handle_missing_values(strategy="auto", categorical_strategy="mode")

        assert processor.data["Province"].isnull().sum() == 0

    def test_handle_missing_zero(self, sample_data):
        """Test filling numeric with zero."""
        processor = DataProcessor(sample_data)
        processor.handle_missing_values(strategy="auto", numeric_strategy="zero")

        assert processor.data["TotalPremium"].isnull().sum() == 0

    def test_handle_missing_drop(self, sample_data):
        """Test dropping rows with missing values."""
        processor = DataProcessor(sample_data)
        rows_before = len(processor.data)

        processor.handle_missing_values(strategy="drop")
        rows_after = len(processor.data)

        assert rows_after < rows_before
        assert processor.data.isnull().sum().sum() == 0

    def test_handle_missing_fill_custom(self, sample_data):
        """Test filling with custom value."""
        processor = DataProcessor(sample_data)
        processor.handle_missing_values(strategy="fill", fill_value=-999)

        assert -999 in processor.data["TotalPremium"].values

    def test_handle_missing_specific_columns(self, sample_data):
        """Test handling missing values for specific columns."""
        processor = DataProcessor(sample_data)
        processor.handle_missing_values(
            strategy="auto", numeric_strategy="median", columns=["TotalPremium"]
        )

        assert processor.data["TotalPremium"].isnull().sum() == 0

    def test_handle_missing_no_data(self, processor):
        """Test error when no data set."""
        with pytest.raises(ValueError):
            processor.handle_missing_values()

    def test_handle_missing_invalid_strategy(self, sample_data):
        """Test invalid strategy raises error."""
        processor = DataProcessor(sample_data)
        with pytest.raises(ValueError):
            processor.handle_missing_values(strategy="invalid")


class TestOutlierHandling:
    """Test outlier detection and handling methods."""

    def test_handle_outliers_iqr_cap(self, sample_data):
        """Test IQR method with capping."""
        processor = DataProcessor(sample_data)
        max_before = processor.data["TotalPremium"].max()

        processor.handle_outliers(method="iqr", action="cap", columns=["TotalPremium"])
        max_after = processor.data["TotalPremium"].max()

        assert max_after < max_before

    def test_handle_outliers_zscore_cap(self, sample_data):
        """Test Z-score method with capping."""
        processor = DataProcessor(sample_data)
        max_before = processor.data["TotalPremium"].max()

        processor.handle_outliers(
            method="zscore", action="cap", columns=["TotalPremium"]
        )
        max_after = processor.data["TotalPremium"].max()

        # Check that extreme values were capped (max should be reduced)
        assert max_after < max_before

    def test_handle_outliers_modified_zscore(self, sample_data):
        """Test modified Z-score method."""
        processor = DataProcessor(sample_data)
        processor.handle_outliers(
            method="modified_zscore", action="cap", columns=["TotalPremium"]
        )

        assert processor.data is not None

    def test_handle_outliers_percentile(self, sample_data):
        """Test percentile method."""
        processor = DataProcessor(sample_data)
        processor.handle_outliers(
            method="percentile", action="cap", columns=["TotalPremium"]
        )

        assert processor.data is not None

    def test_handle_outliers_remove(self, sample_data):
        """Test removing outliers."""
        processor = DataProcessor(sample_data)
        rows_before = len(processor.data)

        processor.handle_outliers(
            method="iqr", action="remove", columns=["TotalPremium"]
        )
        rows_after = len(processor.data)

        assert rows_after < rows_before

    def test_handle_outliers_log_transform(self, sample_data):
        """Test log transformation."""
        processor = DataProcessor(sample_data)
        original_mean = processor.data["TotalPremium"].mean()

        processor.handle_outliers(
            method="iqr", action="log_transform", columns=["TotalPremium"]
        )
        transformed_mean = processor.data["TotalPremium"].mean()

        # After log transform, mean should be different
        assert transformed_mean != original_mean

    def test_handle_outliers_all_numeric(self, sample_data):
        """Test handling outliers for all numeric columns."""
        processor = DataProcessor(sample_data)
        processor.handle_outliers(method="iqr", action="cap")

        assert processor.data is not None

    def test_handle_outliers_no_data(self, processor):
        """Test error when no data set."""
        with pytest.raises(ValueError):
            processor.handle_outliers()

    def test_handle_outliers_invalid_method(self, sample_data):
        """Test invalid method raises error."""
        processor = DataProcessor(sample_data)
        with pytest.raises(ValueError):
            processor.handle_outliers(method="invalid")

    def test_handle_outliers_invalid_action(self, sample_data):
        """Test invalid action raises error."""
        processor = DataProcessor(sample_data)
        with pytest.raises(ValueError):
            processor.handle_outliers(action="invalid")


class TestFeatureEngineering:
    """Test feature engineering methods."""

    def test_create_vehicle_age(self, sample_data):
        """Test VehicleAge feature creation."""
        processor = DataProcessor(sample_data)
        processor.create_derived_features(features=["VehicleAge"])

        assert "VehicleAge" in processor.data.columns
        assert processor.data["VehicleAge"].min() >= 0

    def test_create_has_claim(self, sample_data):
        """Test HasClaim feature creation."""
        processor = DataProcessor(sample_data)
        processor.create_derived_features(features=["HasClaim"])

        assert "HasClaim" in processor.data.columns
        assert set(processor.data["HasClaim"].unique()).issubset({0, 1})

    def test_create_loss_ratio(self, sample_data):
        """Test LossRatio feature creation."""
        processor = DataProcessor(sample_data)
        processor.create_derived_features(features=["LossRatio"])

        assert "LossRatio" in processor.data.columns
        assert not np.isinf(processor.data["LossRatio"]).any()

    def test_create_premium_per_sum_insured(self, sample_data):
        """Test PremiumPerSumInsured feature creation."""
        processor = DataProcessor(sample_data)
        processor.create_derived_features(features=["PremiumPerSumInsured"])

        assert "PremiumPerSumInsured" in processor.data.columns

    def test_create_is_high_value(self, sample_data):
        """Test IsHighValue feature creation."""
        processor = DataProcessor(sample_data)
        processor.create_derived_features(features=["IsHighValue"])

        assert "IsHighValue" in processor.data.columns
        assert set(processor.data["IsHighValue"].unique()).issubset({0, 1})

    def test_create_transaction_year(self, sample_data):
        """Test TransactionYear feature creation."""
        processor = DataProcessor(sample_data)
        processor.create_derived_features(features=["TransactionYear"])

        assert "TransactionYear" in processor.data.columns
        assert processor.data["TransactionYear"].dtype in [np.int64, np.int32]

    def test_create_transaction_quarter(self, sample_data):
        """Test TransactionQuarter feature creation."""
        processor = DataProcessor(sample_data)
        processor.create_derived_features(features=["TransactionQuarter"])

        assert "TransactionQuarter" in processor.data.columns
        assert set(processor.data["TransactionQuarter"].unique()).issubset({1, 2, 3, 4})

    def test_create_vehicle_age_group(self, sample_data):
        """Test VehicleAgeGroup feature creation."""
        processor = DataProcessor(sample_data)
        processor.create_derived_features(features=["VehicleAge", "VehicleAgeGroup"])

        assert "VehicleAgeGroup" in processor.data.columns

    def test_create_all_features(self, sample_data):
        """Test creating all available features."""
        processor = DataProcessor(sample_data)
        cols_before = len(processor.data.columns)

        processor.create_derived_features()
        cols_after = len(processor.data.columns)

        assert cols_after > cols_before

    def test_create_features_no_data(self, processor):
        """Test error when no data set."""
        with pytest.raises(ValueError):
            processor.create_derived_features()


class TestDataManipulation:
    """Test data manipulation methods."""

    def test_remove_duplicates(self, sample_data):
        """Test removing duplicate rows."""
        # Add duplicates
        sample_data = pd.concat([sample_data, sample_data.head(10)], ignore_index=True)

        processor = DataProcessor(sample_data)
        rows_before = len(processor.data)

        processor.remove_duplicates()
        rows_after = len(processor.data)

        assert rows_after < rows_before

    def test_remove_duplicates_subset(self, sample_data):
        """Test removing duplicates based on subset of columns."""
        processor = DataProcessor(sample_data)
        processor.remove_duplicates(subset=["PolicyID"])

        assert processor.data["PolicyID"].is_unique

    def test_remove_duplicates_keep_last(self, sample_data):
        """Test keeping last duplicate."""
        sample_data = pd.concat([sample_data, sample_data.head(10)], ignore_index=True)

        processor = DataProcessor(sample_data)
        processor.remove_duplicates(keep="last")

        assert processor.data is not None

    def test_filter_data_greater_than(self, sample_data):
        """Test filtering with greater than condition."""
        processor = DataProcessor(sample_data)
        processor.filter_data({"TotalPremium": (">", 1000)})

        assert (processor.data["TotalPremium"] > 1000).all()

    def test_filter_data_in_list(self, sample_data):
        """Test filtering with 'in' condition."""
        processor = DataProcessor(sample_data)
        processor.filter_data({"Province": ("in", ["Gauteng", "Western Cape"])})

        assert processor.data["Province"].isin(["Gauteng", "Western Cape"]).all()

    def test_filter_data_between(self, sample_data):
        """Test filtering with between condition."""
        processor = DataProcessor(sample_data)
        processor.filter_data({"Age": ("between", (25, 65))})

        assert processor.data["Age"].between(25, 65).all()

    def test_filter_data_not_null(self, sample_data):
        """Test filtering not null values."""
        processor = DataProcessor(sample_data)
        processor.filter_data({"TotalPremium": ("not_null", None)})

        assert processor.data["TotalPremium"].notnull().all()

    def test_filter_data_multiple_conditions(self, sample_data):
        """Test filtering with multiple conditions."""
        processor = DataProcessor(sample_data)
        processor.filter_data(
            {"TotalPremium": (">", 500), "Province": ("in", ["Gauteng"])}
        )

        assert (processor.data["TotalPremium"] > 500).all()
        assert (processor.data["Province"] == "Gauteng").all()

    def test_filter_data_no_data(self, processor):
        """Test error when no data set."""
        with pytest.raises(ValueError):
            processor.filter_data({"col": (">", 0)})


class TestDataSaving:
    """Test data saving methods."""

    def test_save_csv(self, sample_data, temp_dir):
        """Test saving to CSV format."""
        processor = DataProcessor(sample_data)
        filepath = Path(temp_dir) / "test.csv"

        processor.save_processed_data(filepath, format="csv")

        assert filepath.exists()
        loaded = pd.read_csv(filepath)
        assert len(loaded) == len(sample_data)

    def test_save_txt(self, sample_data, temp_dir):
        """Test saving to TXT format."""
        processor = DataProcessor(sample_data)
        filepath = Path(temp_dir) / "test.txt"

        processor.save_processed_data(filepath, format="txt", sep="|")

        assert filepath.exists()
        loaded = pd.read_csv(filepath, sep="|")
        assert len(loaded) == len(sample_data)

    @pytest.mark.skip(
        reason="Parquet requires optional pyarrow/fastparquet dependencies"
    )
    def test_save_parquet(self, sample_data, temp_dir):
        """Test saving to Parquet format."""
        processor = DataProcessor(sample_data)
        filepath = Path(temp_dir) / "test.parquet"

        processor.save_processed_data(filepath, format="parquet")

        assert filepath.exists()

    def test_save_creates_directory(self, sample_data, temp_dir):
        """Test that save creates directory if not exists."""
        processor = DataProcessor(sample_data)
        filepath = Path(temp_dir) / "subdir" / "test.csv"

        processor.save_processed_data(filepath, format="csv")

        assert filepath.exists()

    def test_save_no_data(self, processor, temp_dir):
        """Test error when no data set."""
        filepath = Path(temp_dir) / "test.csv"
        with pytest.raises(ValueError):
            processor.save_processed_data(filepath)

    def test_save_invalid_format(self, sample_data, temp_dir):
        """Test invalid format raises error."""
        processor = DataProcessor(sample_data)
        filepath = Path(temp_dir) / "test.xyz"

        with pytest.raises(ValueError):
            processor.save_processed_data(filepath, format="invalid")


class TestProcessingSummary:
    """Test processing summary methods."""

    def test_get_processing_summary(self, sample_data):
        """Test getting processing summary."""
        processor = DataProcessor(sample_data)
        summary = processor.get_processing_summary()

        assert "shape" in summary
        assert "columns" in summary
        assert "numeric_columns" in summary
        assert "categorical_columns" in summary
        assert "total_missing" in summary
        assert "missing_percentage" in summary
        assert "duplicate_rows" in summary
        assert "memory_usage_mb" in summary

    def test_summary_shape(self, sample_data):
        """Test summary shape matches data."""
        processor = DataProcessor(sample_data)
        summary = processor.get_processing_summary()

        assert summary["shape"] == sample_data.shape

    def test_summary_after_processing(self, sample_data):
        """Test summary after data processing."""
        processor = DataProcessor(sample_data)

        # Process data
        processor.handle_missing_values(strategy="auto")
        processor.create_derived_features(features=["VehicleAge", "HasClaim"])

        summary = processor.get_processing_summary()

        assert summary["total_missing"] == 0
        assert summary["columns"] > len(sample_data.columns)

    def test_summary_no_data(self, processor):
        """Test error when no data set."""
        with pytest.raises(ValueError):
            processor.get_processing_summary()


class TestEndToEndProcessing:
    """Test complete processing workflows."""

    def test_complete_processing_workflow(self, sample_data, temp_dir):
        """Test complete data processing pipeline."""
        processor = DataProcessor(sample_data)

        # 1. Handle missing values
        processor.handle_missing_values(strategy="auto", numeric_strategy="median")

        # 2. Remove duplicates
        processor.remove_duplicates()

        # 3. Handle outliers
        processor.handle_outliers(method="iqr", action="cap", columns=["TotalPremium"])

        # 4. Create features
        processor.create_derived_features(
            features=["VehicleAge", "HasClaim", "LossRatio"]
        )

        # 5. Filter data
        processor.filter_data({"TotalPremium": (">", 0)})

        # 6. Get summary
        summary = processor.get_processing_summary()

        # 7. Save data
        filepath = Path(temp_dir) / "processed.csv"
        processor.save_processed_data(filepath)

        # Verify results
        assert summary["total_missing"] == 0
        assert "VehicleAge" in processor.data.columns
        assert "HasClaim" in processor.data.columns
        assert filepath.exists()

    def test_insurance_specific_workflow(self, sample_data):
        """Test insurance-specific processing workflow."""
        processor = DataProcessor(sample_data)

        # Process for insurance analysis
        processor.handle_missing_values(strategy="auto", numeric_strategy="median")
        processor.create_derived_features(
            features=[
                "VehicleAge",
                "HasClaim",
                "LossRatio",
                "PremiumPerSumInsured",
                "IsHighValue",
            ]
        )
        processor.filter_data({"TotalPremium": (">", 0), "TotalClaims": (">=", 0)})

        # Verify insurance-specific features
        assert "LossRatio" in processor.data.columns
        assert "HasClaim" in processor.data.columns
        assert (processor.data["TotalPremium"] > 0).all()

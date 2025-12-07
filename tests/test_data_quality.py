"""
Unit tests for DataQualityChecker module.

Tests cover:
- Missing value detection
- Duplicate detection
- Data type validation
- Value range checking
- Categorical distributions
- Outlier detection
- Quality report generation
"""

import pytest
import pandas as pd
import numpy as np
from core.data_quality import DataQualityChecker


@pytest.fixture
def sample_clean_data():
    """Create clean sample data without quality issues."""
    data = {
        "PolicyID": [1, 2, 3, 4, 5],
        "Province": [
            "Gauteng",
            "Western Cape",
            "KwaZulu-Natal",
            "Gauteng",
            "Western Cape",
        ],
        "TotalPremium": [1000.0, 1500.0, 2000.0, 1200.0, 1800.0],
        "TotalClaims": [500.0, 0.0, 1000.0, 300.0, 0.0],
        "Gender": ["Male", "Female", "Male", "Female", "Male"],
        "VehicleType": ["Car", "SUV", "Car", "Truck", "SUV"],
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_data_with_issues():
    """Create sample data with various quality issues."""
    data = {
        "PolicyID": [1, 2, 3, 4, 5, 5, 7],  # Duplicate row (5)
        "Province": [
            "Gauteng",
            None,
            "KZN",
            "Gauteng",
            None,
            None,
            "WC",
        ],  # Missing values
        "TotalPremium": [
            1000.0,
            1500.0,
            2000.0,
            1200.0,
            1800.0,
            1800.0,
            -100.0,
        ],  # Negative value
        "TotalClaims": [500.0, 0.0, 10000.0, 300.0, 0.0, 0.0, 50.0],  # Outlier (10000)
        "Gender": [
            "Male",
            "Female",
            None,
            "Female",
            "Male",
            "Male",
            "Other",
        ],  # Missing
        "Age": ["25", "30", "35", "40", "45", "45", "50"],  # Should be numeric
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_numeric_data():
    """Create sample data with numeric columns for range/outlier testing."""
    np.random.seed(42)
    data = {
        "Value1": np.concatenate(
            [np.random.normal(100, 10, 95), [200, 250, 10, 5, 300]]
        ),  # Outliers
        "Value2": np.random.uniform(0, 100, 100),
        "Value3": [0] * 50 + list(range(1, 51)),  # 50% zeros
        "Value4": list(range(-10, 90)),  # Some negative values
    }
    return pd.DataFrame(data)


class TestDataQualityChecker:
    """Test suite for DataQualityChecker class."""

    def test_initialization_valid(self, sample_clean_data):
        """Test successful initialization with valid data."""
        checker = DataQualityChecker(sample_clean_data)
        assert checker.data is not None
        assert len(checker.data) == 5
        assert checker.quality_report is None

    def test_initialization_none_data(self):
        """Test initialization with None data raises error."""
        with pytest.raises(ValueError, match="cannot be None"):
            DataQualityChecker(None)

    def test_initialization_empty_data(self):
        """Test initialization with empty DataFrame raises error."""
        with pytest.raises(ValueError, match="cannot be empty"):
            DataQualityChecker(pd.DataFrame())

    def test_check_missing_values_clean_data(self, sample_clean_data):
        """Test missing value check on clean data."""
        checker = DataQualityChecker(sample_clean_data)
        missing_info = checker.check_missing_values()

        assert missing_info["total_missing"] == 0
        assert missing_info["missing_percentage"] == 0.0
        assert missing_info["columns_with_missing"] == 0
        assert missing_info["complete_rows"] == 5

    def test_check_missing_values_with_nulls(self, sample_data_with_issues):
        """Test missing value detection."""
        checker = DataQualityChecker(sample_data_with_issues)
        missing_info = checker.check_missing_values()

        assert missing_info["total_missing"] > 0
        assert missing_info["columns_with_missing"] > 0
        assert "Province" in missing_info["missing_by_column"]
        assert "Gender" in missing_info["missing_by_column"]
        assert missing_info["rows_with_missing"] > 0

    def test_check_missing_values_with_threshold(self, sample_data_with_issues):
        """Test missing value check with threshold."""
        checker = DataQualityChecker(sample_data_with_issues)
        missing_info = checker.check_missing_values(threshold=20.0)

        # Only columns with >= 20% missing should be in columns_above_threshold
        assert "columns_above_threshold" in missing_info
        assert isinstance(missing_info["columns_above_threshold"], dict)

    def test_check_duplicates_no_duplicates(self, sample_clean_data):
        """Test duplicate check on data without duplicates."""
        checker = DataQualityChecker(sample_clean_data)
        dup_info = checker.check_duplicates()

        assert dup_info["duplicate_count"] == 0
        assert dup_info["duplicate_percentage"] == 0.0
        assert dup_info["unique_count"] == 5

    def test_check_duplicates_with_duplicates(self, sample_data_with_issues):
        """Test duplicate detection."""
        checker = DataQualityChecker(sample_data_with_issues)
        dup_info = checker.check_duplicates()

        # Row 5 is duplicated (PolicyID=5)
        assert dup_info["duplicate_count"] > 0
        assert dup_info["duplicate_percentage"] > 0
        assert isinstance(dup_info["duplicate_rows"], pd.DataFrame)

    def test_check_duplicates_with_subset(self, sample_data_with_issues):
        """Test duplicate check on specific columns."""
        checker = DataQualityChecker(sample_data_with_issues)
        dup_info = checker.check_duplicates(subset=["PolicyID"])

        assert "duplicate_count" in dup_info
        assert dup_info["columns_checked"] == ["PolicyID"]

    def test_check_data_types(self, sample_clean_data):
        """Test data type validation."""
        checker = DataQualityChecker(sample_clean_data)
        type_info = checker.check_data_types()

        assert "dtypes" in type_info
        assert "numeric_columns" in type_info
        assert "categorical_columns" in type_info
        assert "TotalPremium" in type_info["numeric_columns"]
        assert "Province" in type_info["categorical_columns"]

    def test_check_data_types_potential_issues(self, sample_data_with_issues):
        """Test detection of potential type issues."""
        checker = DataQualityChecker(sample_data_with_issues)
        type_info = checker.check_data_types()

        # 'Age' column is stored as string but should be numeric
        assert "potential_type_issues" in type_info
        assert isinstance(type_info["potential_type_issues"], list)

    def test_check_value_ranges(self, sample_numeric_data):
        """Test value range checking."""
        checker = DataQualityChecker(sample_numeric_data)
        ranges = checker.check_value_ranges()

        assert "Value1" in ranges
        assert "min" in ranges["Value1"]
        assert "max" in ranges["Value1"]
        assert "mean" in ranges["Value1"]
        assert "std" in ranges["Value1"]

    def test_check_value_ranges_zeros_negatives(self, sample_numeric_data):
        """Test detection of zeros and negative values."""
        checker = DataQualityChecker(sample_numeric_data)
        ranges = checker.check_value_ranges()

        # Value3 has 50% zeros
        assert ranges["Value3"]["zeros"] == 50
        assert ranges["Value3"]["zeros_percentage"] == 50.0

        # Value4 has negative values
        assert ranges["Value4"]["negatives"] > 0

    def test_check_value_ranges_specific_columns(self, sample_clean_data):
        """Test value range check on specific columns."""
        checker = DataQualityChecker(sample_clean_data)
        ranges = checker.check_value_ranges(["TotalPremium", "TotalClaims"])

        assert "TotalPremium" in ranges
        assert "TotalClaims" in ranges
        assert "Province" not in ranges  # Non-numeric column

    def test_check_categorical_distributions(self, sample_clean_data):
        """Test categorical distribution analysis."""
        checker = DataQualityChecker(sample_clean_data)
        cat_info = checker.check_categorical_distributions()

        assert "Province" in cat_info
        assert "unique_count" in cat_info["Province"]
        assert "top_values" in cat_info["Province"]
        assert "most_common" in cat_info["Province"]

    def test_check_categorical_distributions_specific_columns(self, sample_clean_data):
        """Test categorical analysis on specific columns."""
        checker = DataQualityChecker(sample_clean_data)
        cat_info = checker.check_categorical_distributions(["Province", "Gender"])

        assert "Province" in cat_info
        assert "Gender" in cat_info
        assert "TotalPremium" not in cat_info  # Numeric column

    def test_detect_outliers_iqr(self, sample_numeric_data):
        """Test IQR outlier detection."""
        checker = DataQualityChecker(sample_numeric_data)
        outliers = checker.detect_outliers_iqr()

        # Value1 has outliers (200, 250, 300, 5, 10)
        assert "Value1" in outliers
        assert outliers["Value1"]["outlier_count"] > 0
        assert "lower_bound" in outliers["Value1"]
        assert "upper_bound" in outliers["Value1"]
        assert "IQR" in outliers["Value1"]

    def test_detect_outliers_custom_multiplier(self, sample_numeric_data):
        """Test outlier detection with custom multiplier."""
        checker = DataQualityChecker(sample_numeric_data)

        # More strict (multiplier=1.0) should find more outliers
        outliers_strict = checker.detect_outliers_iqr(multiplier=1.0)
        # Less strict (multiplier=3.0) should find fewer outliers
        outliers_loose = checker.detect_outliers_iqr(multiplier=3.0)

        assert (
            outliers_strict["Value1"]["outlier_count"]
            >= outliers_loose["Value1"]["outlier_count"]
        )

    def test_detect_outliers_specific_columns(self, sample_clean_data):
        """Test outlier detection on specific columns."""
        checker = DataQualityChecker(sample_clean_data)
        outliers = checker.detect_outliers_iqr(["TotalPremium"])

        assert "TotalPremium" in outliers
        assert "TotalClaims" not in outliers

    def test_generate_quality_report(self, sample_clean_data):
        """Test comprehensive quality report generation."""
        checker = DataQualityChecker(sample_clean_data)
        report = checker.generate_quality_report()

        assert "overview" in report
        assert "missing_values" in report
        assert "duplicates" in report
        assert "data_types" in report
        assert "value_ranges" in report
        assert "categorical" in report
        assert "quality_score" in report
        assert "recommendations" in report

        # Clean data should have high quality score
        assert report["quality_score"] >= 80

    def test_generate_quality_report_with_issues(self, sample_data_with_issues):
        """Test quality report with data containing issues."""
        checker = DataQualityChecker(sample_data_with_issues)
        report = checker.generate_quality_report()

        # Data with issues should have lower quality score
        assert report["quality_score"] < 100

        # Should have recommendations
        assert len(report["recommendations"]) > 0

    def test_generate_quality_report_without_outliers(self, sample_clean_data):
        """Test quality report without outlier detection."""
        checker = DataQualityChecker(sample_clean_data)
        report = checker.generate_quality_report(include_outliers=False)

        assert "outliers" in report
        assert len(report["outliers"]) == 0

    def test_quality_score_calculation(self, sample_data_with_issues):
        """Test quality score is calculated correctly."""
        checker = DataQualityChecker(sample_data_with_issues)
        report = checker.generate_quality_report()

        # Score should be between 0 and 100
        assert 0 <= report["quality_score"] <= 100

        # Score should be deducted for issues
        missing_pct = report["missing_values"]["missing_percentage"]
        dup_pct = report["duplicates"]["duplicate_percentage"]

        if missing_pct > 0 or dup_pct > 0:
            assert report["quality_score"] < 100

    def test_print_quality_summary_without_report(self, sample_clean_data):
        """Test print summary raises error without report."""
        checker = DataQualityChecker(sample_clean_data)

        with pytest.raises(ValueError, match="Quality report not generated"):
            checker.print_quality_summary()

    def test_print_quality_summary_with_report(self, sample_clean_data, capsys):
        """Test print summary after generating report."""
        checker = DataQualityChecker(sample_clean_data)
        checker.generate_quality_report()

        # Should not raise error
        checker.print_quality_summary()

        # Check output contains expected sections
        captured = capsys.readouterr()
        assert "DATA QUALITY REPORT SUMMARY" in captured.out
        assert "OVERVIEW" in captured.out
        assert "MISSING VALUES" in captured.out
        assert "Quality Score" in captured.out

    def test_report_persistence(self, sample_clean_data):
        """Test that generated report is stored in instance."""
        checker = DataQualityChecker(sample_clean_data)

        assert checker.quality_report is None

        report = checker.generate_quality_report()

        assert checker.quality_report is not None
        assert checker.quality_report == report


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

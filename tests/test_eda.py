"""
Unit tests for the EDA Engine module.
"""

import pytest
import pandas as pd
import numpy as np
from core.eda import EDAEngine


class TestEDAEngine:
    """Test suite for EDAEngine class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        return pd.DataFrame(
            {
                "Age": np.random.randint(18, 80, 100),
                "Premium": np.random.uniform(1000, 5000, 100),
                "Claims": np.random.uniform(0, 3000, 100),
                "Province": np.random.choice(["GP", "WC", "KZN", "EC"], 100),
                "VehicleType": np.random.choice(["Sedan", "SUV", "Truck"], 100),
                "Gender": np.random.choice(["Male", "Female"], 100),
            }
        )

    @pytest.fixture
    def sample_with_outliers(self):
        """Create sample data with outliers."""
        np.random.seed(42)
        data = pd.DataFrame(
            {
                "Value1": list(range(90))
                + [100, 200, 300, 400, 500, 1000, 2000, 3000, 4000, 5000],
                "Value2": [10] * 95 + [100, 200, 300, 400, 500],
            }
        )
        return data

    @pytest.fixture
    def sample_insurance_data(self):
        """Create sample insurance data with loss ratios."""
        return pd.DataFrame(
            {
                "TotalPremium": [1000, 1500, 2000, 1200, 1800],
                "TotalClaims": [800, 1200, 2500, 600, 1500],
                "Province": ["GP", "WC", "GP", "KZN", "WC"],
                "VehicleType": ["Sedan", "SUV", "Sedan", "Truck", "SUV"],
            }
        )

    def test_initialization_valid(self, sample_data):
        """Test successful initialization."""
        engine = EDAEngine(sample_data)
        assert engine.data is not None
        assert len(engine.numeric_columns) == 3
        assert len(engine.categorical_columns) == 3

    def test_initialization_none_data(self):
        """Test initialization with None data."""
        with pytest.raises(ValueError, match="Data cannot be None"):
            EDAEngine(None)

    def test_initialization_empty_data(self):
        """Test initialization with empty DataFrame."""
        with pytest.raises(ValueError, match="Data cannot be empty"):
            EDAEngine(pd.DataFrame())

    def test_descriptive_statistics_all(self, sample_data):
        """Test descriptive statistics for all columns."""
        engine = EDAEngine(sample_data)
        stats = engine.descriptive_statistics()

        assert "numeric" in stats
        assert "categorical" in stats
        assert len(stats["numeric"]) == 3
        assert len(stats["categorical"]) == 3

        # Check numeric stats include additional metrics
        assert "median" in stats["numeric"].columns
        assert "variance" in stats["numeric"].columns
        assert "skewness" in stats["numeric"].columns
        assert "kurtosis" in stats["numeric"].columns

    def test_descriptive_statistics_specific_columns(self, sample_data):
        """Test descriptive statistics for specific columns."""
        engine = EDAEngine(sample_data)
        stats = engine.descriptive_statistics(columns=["Age", "Province"])

        assert "Age" in stats["numeric"].index
        assert "Province" in stats["categorical"].index
        assert len(stats["numeric"]) == 1
        assert len(stats["categorical"]) == 1

    def test_descriptive_statistics_invalid_columns(self, sample_data):
        """Test descriptive statistics with invalid columns."""
        engine = EDAEngine(sample_data)
        with pytest.raises(ValueError, match="Columns not found"):
            engine.descriptive_statistics(columns=["InvalidColumn"])

    def test_univariate_analysis_numeric(self, sample_data):
        """Test univariate analysis on numeric column."""
        engine = EDAEngine(sample_data)
        analysis = engine.univariate_analysis("Age")

        assert analysis["column"] == "Age"
        assert "statistics" in analysis
        assert "percentiles" in analysis
        assert "normality_test" in analysis

        # Check statistics
        stats = analysis["statistics"]
        assert "mean" in stats
        assert "median" in stats
        assert "std" in stats
        assert "skewness" in stats

        # Check percentiles
        assert "25%" in analysis["percentiles"]
        assert "75%" in analysis["percentiles"]
        assert "95%" in analysis["percentiles"]

    def test_univariate_analysis_categorical(self, sample_data):
        """Test univariate analysis on categorical column."""
        engine = EDAEngine(sample_data)
        analysis = engine.univariate_analysis("Province")

        assert analysis["column"] == "Province"
        assert "statistics" in analysis
        assert "value_counts" in analysis

        # Check statistics
        stats = analysis["statistics"]
        assert "count" in stats
        assert "unique" in stats
        assert "top" in stats

        # Check value counts
        assert "count" in analysis["value_counts"].columns
        assert "percentage" in analysis["value_counts"].columns

    def test_univariate_analysis_invalid_column(self, sample_data):
        """Test univariate analysis with invalid column."""
        engine = EDAEngine(sample_data)
        with pytest.raises(ValueError, match="Column .* not found"):
            engine.univariate_analysis("InvalidColumn")

    def test_bivariate_analysis_numeric_numeric(self, sample_data):
        """Test bivariate analysis between two numeric columns."""
        engine = EDAEngine(sample_data)
        analysis = engine.bivariate_analysis("Premium", "Claims")

        assert analysis["type1"] == "numeric"
        assert analysis["type2"] == "numeric"
        assert "correlation" in analysis

        # Check correlation metrics
        corr = analysis["correlation"]
        assert "pearson" in corr
        assert "spearman" in corr
        assert "pearson_p_value" in corr
        assert "relationship_strength" in corr

    def test_bivariate_analysis_numeric_categorical(self, sample_data):
        """Test bivariate analysis between numeric and categorical columns."""
        engine = EDAEngine(sample_data)
        analysis = engine.bivariate_analysis("Premium", "Province")

        assert "group_statistics" in analysis
        assert "anova" in analysis

        # Check group statistics
        stats = analysis["group_statistics"]
        assert "mean" in stats.columns
        assert "median" in stats.columns
        assert "std" in stats.columns

        # Check ANOVA
        anova = analysis["anova"]
        assert "f_statistic" in anova
        assert "p_value" in anova
        assert "significant" in anova

    def test_bivariate_analysis_categorical_categorical(self, sample_data):
        """Test bivariate analysis between two categorical columns."""
        engine = EDAEngine(sample_data)
        analysis = engine.bivariate_analysis("Province", "VehicleType")

        assert analysis["type1"] == "categorical"
        assert analysis["type2"] == "categorical"
        assert "contingency_table" in analysis
        assert "chi_square_test" in analysis

        # Check chi-square test
        chi2 = analysis["chi_square_test"]
        assert "chi2_statistic" in chi2
        assert "p_value" in chi2
        assert "degrees_of_freedom" in chi2
        assert "significant" in chi2

    def test_bivariate_analysis_invalid_columns(self, sample_data):
        """Test bivariate analysis with invalid columns."""
        engine = EDAEngine(sample_data)
        with pytest.raises(ValueError, match="not found"):
            engine.bivariate_analysis("Invalid1", "Invalid2")

    def test_multivariate_analysis(self, sample_data):
        """Test multivariate analysis on multiple numeric columns."""
        engine = EDAEngine(sample_data)
        analysis = engine.multivariate_analysis()

        assert "correlation_matrix" in analysis
        assert "covariance_matrix" in analysis
        assert "highly_correlated_pairs" in analysis

        # Check correlation matrix
        corr_matrix = analysis["correlation_matrix"]
        assert corr_matrix.shape[0] == len(engine.numeric_columns)
        assert corr_matrix.shape[1] == len(engine.numeric_columns)

    def test_multivariate_analysis_specific_columns(self, sample_data):
        """Test multivariate analysis on specific columns."""
        engine = EDAEngine(sample_data)
        analysis = engine.multivariate_analysis(columns=["Age", "Premium"])

        assert analysis["correlation_matrix"].shape == (2, 2)
        assert analysis["covariance_matrix"].shape == (2, 2)

    def test_multivariate_analysis_invalid_columns(self, sample_data):
        """Test multivariate analysis with non-numeric columns."""
        engine = EDAEngine(sample_data)
        with pytest.raises(ValueError, match="must be numeric"):
            engine.multivariate_analysis(columns=["Province", "Gender"])

    def test_multivariate_analysis_insufficient_columns(self, sample_data):
        """Test multivariate analysis with less than 2 columns."""
        engine = EDAEngine(sample_data)
        with pytest.raises(ValueError, match="at least 2 columns"):
            engine.multivariate_analysis(columns=["Age"])

    def test_detect_outliers_iqr(self, sample_with_outliers):
        """Test outlier detection using IQR method."""
        engine = EDAEngine(sample_with_outliers)
        outliers = engine.detect_outliers(method="iqr", threshold=1.5)

        assert "Value1" in outliers
        assert len(outliers["Value1"]) > 0

    def test_detect_outliers_zscore(self, sample_with_outliers):
        """Test outlier detection using Z-score method."""
        engine = EDAEngine(sample_with_outliers)
        outliers = engine.detect_outliers(method="zscore", threshold=3)

        assert "Value1" in outliers
        assert len(outliers["Value1"]) > 0

    def test_detect_outliers_modified_zscore(self, sample_with_outliers):
        """Test outlier detection using modified Z-score method."""
        engine = EDAEngine(sample_with_outliers)
        outliers = engine.detect_outliers(method="modified_zscore", threshold=3.5)

        assert "Value1" in outliers
        assert len(outliers["Value1"]) > 0

    def test_detect_outliers_invalid_method(self, sample_data):
        """Test outlier detection with invalid method."""
        engine = EDAEngine(sample_data)
        with pytest.raises(ValueError, match="Method must be one of"):
            engine.detect_outliers(method="invalid_method")

    def test_detect_outliers_specific_columns(self, sample_with_outliers):
        """Test outlier detection on specific columns."""
        engine = EDAEngine(sample_with_outliers)
        outliers = engine.detect_outliers(columns=["Value1"], method="iqr")

        assert "Value1" in outliers
        assert "Value2" not in outliers

    def test_calculate_loss_ratio_overall(self, sample_insurance_data):
        """Test overall loss ratio calculation."""
        engine = EDAEngine(sample_insurance_data)
        loss_ratio = engine.calculate_loss_ratio()

        assert isinstance(loss_ratio, float)
        assert loss_ratio > 0

        # Manual calculation
        expected_ratio = (
            sample_insurance_data["TotalClaims"].sum()
            / sample_insurance_data["TotalPremium"].sum()
        )
        assert abs(loss_ratio - expected_ratio) < 0.001

    def test_calculate_loss_ratio_by_group(self, sample_insurance_data):
        """Test loss ratio calculation by group."""
        engine = EDAEngine(sample_insurance_data)
        loss_ratios = engine.calculate_loss_ratio(group_by=["Province"])

        assert isinstance(loss_ratios, pd.DataFrame)
        assert "loss_ratio" in loss_ratios.columns
        assert "margin" in loss_ratios.columns
        assert "policy_count" in loss_ratios.columns

    def test_calculate_loss_ratio_multiple_groups(self, sample_insurance_data):
        """Test loss ratio calculation by multiple groups."""
        engine = EDAEngine(sample_insurance_data)
        loss_ratios = engine.calculate_loss_ratio(group_by=["Province", "VehicleType"])

        assert isinstance(loss_ratios, pd.DataFrame)
        assert "loss_ratio" in loss_ratios.columns

    def test_calculate_loss_ratio_invalid_claims_column(self, sample_insurance_data):
        """Test loss ratio with invalid claims column."""
        engine = EDAEngine(sample_insurance_data)
        with pytest.raises(ValueError, match="Claims column .* not found"):
            engine.calculate_loss_ratio(claims_column="InvalidColumn")

    def test_calculate_loss_ratio_invalid_premium_column(self, sample_insurance_data):
        """Test loss ratio with invalid premium column."""
        engine = EDAEngine(sample_insurance_data)
        with pytest.raises(ValueError, match="Premium column .* not found"):
            engine.calculate_loss_ratio(premium_column="InvalidColumn")

    def test_calculate_loss_ratio_zero_premiums(self):
        """Test loss ratio with zero premiums."""
        data = pd.DataFrame(
            {
                "TotalPremium": [0, 0, 0],
                "TotalClaims": [100, 200, 300],
            }
        )
        engine = EDAEngine(data)
        loss_ratio = engine.calculate_loss_ratio()

        assert np.isnan(loss_ratio)

    def test_interpret_correlation(self, sample_data):
        """Test correlation interpretation."""
        engine = EDAEngine(sample_data)

        # Test various correlation strengths
        assert "Very Strong" in engine._interpret_correlation(0.95)
        assert "Strong" in engine._interpret_correlation(0.75)
        assert "Moderate" in engine._interpret_correlation(0.55)
        assert "Weak" in engine._interpret_correlation(0.35)
        assert "Very Weak" in engine._interpret_correlation(0.15)

        # Test positive/negative
        assert "Positive" in engine._interpret_correlation(0.8)
        assert "Negative" in engine._interpret_correlation(-0.8)

    def test_get_summary(self, sample_data):
        """Test get_summary method."""
        engine = EDAEngine(sample_data)
        summary = engine.get_summary()

        assert summary["rows"] == 100
        assert summary["columns"] == 6
        assert len(summary["numeric_columns"]) == 3
        assert len(summary["categorical_columns"]) == 3
        assert "memory_usage_mb" in summary

    def test_data_immutability(self, sample_data):
        """Test that original data is not modified."""
        original_data = sample_data.copy()
        engine = EDAEngine(sample_data)

        # Perform various operations
        engine.descriptive_statistics()
        engine.univariate_analysis("Age")
        engine.bivariate_analysis("Premium", "Claims")

        # Check original data unchanged
        pd.testing.assert_frame_equal(sample_data, original_data)

"""
Unit tests for the hypothesis_testing module.

Tests the HypothesisTester class for calculating KPIs and performing
statistical tests on insurance data.
"""

import pytest
import pandas as pd
import numpy as np
from core.hypothesis_testing import HypothesisTester


@pytest.fixture
def sample_data():
    """Create sample insurance data for testing."""
    np.random.seed(42)
    n = 1000

    data = pd.DataFrame(
        {
            "Province": np.random.choice(
                ["Gauteng", "Western Cape", "KwaZulu-Natal"], n
            ),
            "PostalCode": np.random.choice(["1000", "2000", "3000", "4000"], n),
            "Gender": np.random.choice(["Male", "Female"], n),
            "TotalPremium": np.random.uniform(500, 5000, n),
            "TotalClaims": np.random.uniform(0, 3000, n),
        }
    )

    # Make some claims zero (no claim)
    data.loc[data["TotalClaims"] < 1000, "TotalClaims"] = 0

    return data


@pytest.fixture
def tester(sample_data):
    """Create HypothesisTester instance with sample data."""
    return HypothesisTester(sample_data)


class TestHypothesisTesterInit:
    """Test HypothesisTester initialization."""

    def test_init_valid_data(self, sample_data):
        """Test initialization with valid data."""
        tester = HypothesisTester(sample_data)
        assert tester.alpha == 0.05
        assert len(tester.data) == len(sample_data)

    def test_init_custom_alpha(self, sample_data):
        """Test initialization with custom alpha."""
        tester = HypothesisTester(sample_data, alpha=0.01)
        assert tester.alpha == 0.01

    def test_init_empty_data(self):
        """Test initialization with empty DataFrame."""
        with pytest.raises(ValueError, match="Data cannot be None or empty"):
            HypothesisTester(pd.DataFrame())

    def test_init_none_data(self):
        """Test initialization with None."""
        with pytest.raises(ValueError, match="Data cannot be None or empty"):
            HypothesisTester(None)

    def test_init_invalid_alpha(self, sample_data):
        """Test initialization with invalid alpha values."""
        with pytest.raises(ValueError, match="Alpha must be between 0 and 1"):
            HypothesisTester(sample_data, alpha=1.5)

        with pytest.raises(ValueError, match="Alpha must be between 0 and 1"):
            HypothesisTester(sample_data, alpha=0)


class TestCalculateKPIs:
    """Test KPI calculation methods."""

    def test_calculate_kpis_basic(self, tester):
        """Test basic KPI calculation."""
        result = tester.calculate_kpis("Province")

        assert isinstance(result, pd.DataFrame)
        assert "Group" in result.columns
        assert "PolicyCount" in result.columns
        assert "ClaimFrequency" in result.columns
        assert "ClaimSeverity" in result.columns
        assert "Margin" in result.columns
        assert len(result) > 0

    def test_calculate_kpis_values(self, tester):
        """Test KPI calculation produces valid values."""
        result = tester.calculate_kpis("Province")

        # Check value ranges
        assert all(result["PolicyCount"] > 0)
        assert all(result["ClaimFrequency"] >= 0)
        assert all(result["ClaimFrequency"] <= 100)
        assert all(result["ClaimSeverity"] >= 0)

    def test_calculate_kpis_selected_groups(self, tester):
        """Test KPI calculation with selected groups."""
        result = tester.calculate_kpis(
            "Province", selected_groups=["Gauteng", "Western Cape"]
        )

        assert len(result) == 2
        assert set(result["Group"]) == {"Gauteng", "Western Cape"}

    def test_calculate_kpis_invalid_column(self, tester):
        """Test KPI calculation with invalid column."""
        with pytest.raises(ValueError, match="Column 'InvalidColumn' not found"):
            tester.calculate_kpis("InvalidColumn")

    def test_calculate_kpis_missing_required_columns(self):
        """Test KPI calculation with missing required columns."""
        data = pd.DataFrame({"Province": ["A", "B"]})
        tester = HypothesisTester(data)

        with pytest.raises(ValueError, match="Required columns missing"):
            tester.calculate_kpis("Province")


class TestChiSquaredTest:
    """Test chi-squared testing methods."""

    def test_chi_squared_basic(self, tester):
        """Test basic chi-squared test."""
        result = tester.chi_squared_test("Province")

        assert "statistic" in result
        assert "p_value" in result
        assert "reject_null" in result
        assert "interpretation" in result
        assert "effect_size" in result
        assert isinstance(result["reject_null"], bool)

    def test_chi_squared_selected_groups(self, tester):
        """Test chi-squared with selected groups."""
        result = tester.chi_squared_test(
            "Province", selected_groups=["Gauteng", "Western Cape"]
        )

        assert "p_value" in result
        assert 0 <= result["p_value"] <= 1

    def test_chi_squared_invalid_column(self, tester):
        """Test chi-squared with invalid column."""
        with pytest.raises(ValueError, match="Column 'InvalidColumn' not found"):
            tester.chi_squared_test("InvalidColumn")

    def test_chi_squared_insufficient_groups(self):
        """Test chi-squared with only one group."""
        data = pd.DataFrame(
            {
                "Province": ["Gauteng"] * 100,
                "TotalPremium": np.random.uniform(500, 5000, 100),
                "TotalClaims": np.random.uniform(0, 3000, 100),
            }
        )
        tester = HypothesisTester(data)

        with pytest.raises(ValueError, match="Need at least 2 groups"):
            tester.chi_squared_test("Province")


class TestTTest:
    """Test t-test methods."""

    def test_t_test_basic(self, tester):
        """Test basic t-test."""
        result = tester.t_test("Gender", "Male", "Female", metric="Margin")

        assert "statistic" in result
        assert "p_value" in result
        assert "reject_null" in result
        assert "mean_group1" in result
        assert "mean_group2" in result
        assert "cohens_d" in result
        assert "interpretation" in result

    def test_t_test_claim_severity(self, tester):
        """Test t-test with ClaimSeverity metric."""
        result = tester.t_test("Gender", "Male", "Female", metric="ClaimSeverity")

        assert "p_value" in result
        assert 0 <= result["p_value"] <= 1

    def test_t_test_invalid_column(self, tester):
        """Test t-test with invalid column."""
        with pytest.raises(ValueError, match="Column 'InvalidColumn' not found"):
            tester.t_test("InvalidColumn", "A", "B")

    def test_t_test_invalid_metric(self, tester):
        """Test t-test with invalid metric."""
        with pytest.raises(ValueError, match="Metric must be"):
            tester.t_test("Gender", "Male", "Female", metric="InvalidMetric")

    def test_t_test_invalid_groups(self, tester):
        """Test t-test with non-existent groups."""
        with pytest.raises(ValueError, match="One or both groups have no data"):
            tester.t_test("Gender", "InvalidGroup1", "InvalidGroup2")


class TestANOVATest:
    """Test ANOVA testing methods."""

    def test_anova_basic(self, tester):
        """Test basic ANOVA test."""
        result = tester.anova_test("Province", metric="Margin")

        assert "statistic" in result
        assert "p_value" in result
        assert "reject_null" in result
        assert "group_means" in result
        assert "num_groups" in result
        assert "interpretation" in result

    def test_anova_claim_severity(self, tester):
        """Test ANOVA with ClaimSeverity metric."""
        result = tester.anova_test("Province", metric="ClaimSeverity")

        assert "p_value" in result
        assert 0 <= result["p_value"] <= 1
        assert result["num_groups"] >= 2

    def test_anova_selected_groups(self, tester):
        """Test ANOVA with selected groups."""
        result = tester.anova_test(
            "Province",
            metric="Margin",
            selected_groups=["Gauteng", "Western Cape", "KwaZulu-Natal"],
        )

        assert result["num_groups"] == 3

    def test_anova_invalid_column(self, tester):
        """Test ANOVA with invalid column."""
        with pytest.raises(ValueError, match="Column 'InvalidColumn' not found"):
            tester.anova_test("InvalidColumn")

    def test_anova_invalid_metric(self, tester):
        """Test ANOVA with invalid metric."""
        with pytest.raises(ValueError, match="Metric must be"):
            tester.anova_test("Province", metric="InvalidMetric")


class TestInterpretResults:
    """Test result interpretation methods."""

    def test_interpret_results_reject(self, tester):
        """Test interpretation when null is rejected."""
        result = tester.interpret_results(
            "Test Hypothesis", p_value=0.01, reject_null=True
        )

        assert "REJECT" in result
        assert "0.01" in result
        assert "Business Implications" in result

    def test_interpret_results_fail_to_reject(self, tester):
        """Test interpretation when null is not rejected."""
        result = tester.interpret_results(
            "Test Hypothesis", p_value=0.10, reject_null=False
        )

        assert "NOT REJECTED" in result
        assert "0.10" in result
        assert "Business Implications" in result

    def test_interpret_results_with_additional_info(self, tester):
        """Test interpretation with additional context."""
        result = tester.interpret_results(
            "Test Hypothesis",
            p_value=0.03,
            reject_null=True,
            additional_info={"Effect Size": 0.25, "Sample Size": 1000},
        )

        assert "Effect Size" in result
        assert "Sample Size" in result


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_all_zero_claims(self):
        """Test with data where all claims are zero."""
        data = pd.DataFrame(
            {
                "Province": ["A", "B"] * 50,
                "TotalPremium": np.random.uniform(500, 5000, 100),
                "TotalClaims": [0] * 100,
            }
        )
        tester = HypothesisTester(data)

        result = tester.calculate_kpis("Province")
        assert all(result["ClaimFrequency"] == 0)
        assert all(result["ClaimSeverity"] == 0)

    def test_all_claims_equal(self):
        """Test with identical claim values."""
        data = pd.DataFrame(
            {
                "Province": ["A", "B"] * 50,
                "TotalPremium": [1000] * 100,
                "TotalClaims": [500] * 100,
            }
        )
        tester = HypothesisTester(data)

        result = tester.calculate_kpis("Province")
        assert result["ClaimFrequency"].nunique() == 1

    def test_single_claim_per_group(self):
        """Test with minimal data per group."""
        data = pd.DataFrame(
            {
                "Province": ["A", "A", "B", "B"],
                "TotalPremium": [1000, 2000, 1500, 2500],
                "TotalClaims": [500, 600, 700, 800],
            }
        )
        tester = HypothesisTester(data)

        # Should work for KPIs
        result = tester.calculate_kpis("Province")
        assert len(result) == 2

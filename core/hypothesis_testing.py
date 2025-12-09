"""
Hypothesis Testing Module for Insurance Risk Analysis.

This module provides the HypothesisTester class for conducting A/B hypothesis
testing on insurance data. It calculates key performance indicators (KPIs) and
performs statistical tests to evaluate risk differences across various segments.

"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from scipy import stats
from scipy.stats import chi2_contingency, f_oneway, ttest_ind


class HypothesisTester:
    """
    A class for conducting hypothesis testing on insurance data.

    This class provides methods to calculate insurance KPIs (claim frequency,
    claim severity, margin) and perform statistical tests (chi-squared, t-test,
    ANOVA) to test hypotheses about risk differences across segments.

    Attributes:
        data (pd.DataFrame): The insurance dataset to analyze.
        alpha (float): Significance level for hypothesis tests (default 0.05).
    """

    def __init__(self, data: pd.DataFrame, alpha: float = 0.05):
        """
        Initialize the HypothesisTester.

        Args:
            data: DataFrame containing insurance policy data.
            alpha: Significance level for statistical tests (default 0.05).

        Raises:
            ValueError: If data is empty or alpha is not between 0 and 1.
        """
        if data is None or data.empty:
            raise ValueError("Data cannot be None or empty")
        if not 0 < alpha < 1:
            raise ValueError("Alpha must be between 0 and 1")

        self.data = data.copy()
        self.alpha = alpha

    def calculate_kpis(
        self, group_column: str, selected_groups: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Calculate key performance indicators by group.

        Calculates three main KPIs for insurance risk assessment:
        - Claim Frequency: Proportion of policies with claims (%)
        - Claim Severity: Average claim amount for policies with claims
        - Margin: Average profit per policy (TotalPremium - TotalClaims)

        Args:
            group_column: Column name to group by (e.g., 'Province', 'Gender').
            selected_groups: Optional list of specific groups to include.

        Returns:
            DataFrame with columns: Group, PolicyCount, ClaimFrequency,
            ClaimSeverity, Margin.

        Raises:
            ValueError: If group_column not in data or required columns missing.
        """
        if group_column not in self.data.columns:
            raise ValueError(f"Column '{group_column}' not found in data")

        required_cols = ["TotalPremium", "TotalClaims"]
        missing_cols = [col for col in required_cols if col not in self.data.columns]
        if missing_cols:
            raise ValueError(f"Required columns missing: {missing_cols}")

        # Filter to selected groups if specified
        df = self.data.copy()
        if selected_groups:
            df = df[df[group_column].isin(selected_groups)]

        # Calculate KPIs by group
        results = []
        for group in df[group_column].dropna().unique():
            group_data = df[df[group_column] == group]

            # Policy count
            policy_count = len(group_data)

            # Claim frequency: proportion with claims > 0
            has_claim = (group_data["TotalClaims"] > 0).sum()
            claim_frequency = (
                (has_claim / policy_count * 100) if policy_count > 0 else 0
            )

            # Claim severity: average claim amount (for policies with claims)
            claims_data = group_data[group_data["TotalClaims"] > 0]["TotalClaims"]
            claim_severity = claims_data.mean() if len(claims_data) > 0 else 0

            # Margin: average profit per policy
            margin = (group_data["TotalPremium"] - group_data["TotalClaims"]).mean()

            results.append(
                {
                    "Group": group,
                    "PolicyCount": policy_count,
                    "ClaimFrequency": claim_frequency,
                    "ClaimSeverity": claim_severity,
                    "Margin": margin,
                }
            )

        return pd.DataFrame(results)

    def chi_squared_test(
        self, group_column: str, selected_groups: Optional[List[str]] = None
    ) -> Dict[str, Union[float, bool, str]]:
        """
        Perform chi-squared test for claim frequency differences.

        Tests whether claim frequency (has claim vs no claim) differs
        significantly across groups. Used for categorical comparisons.

        Args:
            group_column: Column name to group by.
            selected_groups: Optional list of specific groups to test.

        Returns:
            Dictionary containing:
                - statistic: Chi-squared test statistic
                - p_value: P-value from the test
                - reject_null: Whether to reject null hypothesis
                - interpretation: Business interpretation of result
                - effect_size: Cramér's V effect size measure

        Raises:
            ValueError: If insufficient data or invalid inputs.
        """
        if group_column not in self.data.columns:
            raise ValueError(f"Column '{group_column}' not found in data")

        # Filter to selected groups if specified
        df = self.data.copy()
        if selected_groups:
            df = df[df[group_column].isin(selected_groups)]

        # Create contingency table: groups vs has_claim
        df["has_claim"] = (df["TotalClaims"] > 0).astype(int)
        contingency_table = pd.crosstab(df[group_column], df["has_claim"])

        if contingency_table.shape[0] < 2:
            raise ValueError("Need at least 2 groups for chi-squared test")

        # Perform chi-squared test
        chi2, p_value, dof, expected = chi2_contingency(contingency_table)
        reject_null = bool(p_value < self.alpha)

        # Calculate Cramér's V for effect size
        n = contingency_table.sum().sum()
        min_dim = min(contingency_table.shape[0] - 1, contingency_table.shape[1] - 1)
        cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0

        # Interpretation
        if reject_null:
            interpretation = (
                f"REJECT null hypothesis (p={p_value:.4f} < {self.alpha}). "
                f"Claim frequency differs significantly across {group_column} groups. "
                f"Effect size (Cramér's V): {cramers_v:.3f}"
            )
        else:
            interpretation = (
                f"FAIL TO REJECT null hypothesis (p={p_value:.4f} >= {self.alpha}). "
                f"No significant difference in claim frequency across {group_column} groups."
            )

        return {
            "statistic": float(chi2),
            "p_value": float(p_value),
            "degrees_of_freedom": int(dof),
            "reject_null": reject_null,
            "effect_size": float(cramers_v),
            "interpretation": interpretation,
        }

    def t_test(
        self, group_column: str, group1: str, group2: str, metric: str = "ClaimSeverity"
    ) -> Dict[str, Union[float, bool, str]]:
        """
        Perform independent samples t-test for two groups.

        Tests whether a numerical metric (claim severity or margin) differs
        significantly between two groups.

        Args:
            group_column: Column name containing groups.
            group1: First group identifier.
            group2: Second group identifier.
            metric: Metric to test ('ClaimSeverity' or 'Margin').

        Returns:
            Dictionary containing:
                - statistic: T-test statistic
                - p_value: P-value from the test
                - reject_null: Whether to reject null hypothesis
                - interpretation: Business interpretation of result
                - cohens_d: Cohen's d effect size measure

        Raises:
            ValueError: If groups not found or invalid metric.
        """
        if group_column not in self.data.columns:
            raise ValueError(f"Column '{group_column}' not found in data")

        if metric not in ["ClaimSeverity", "Margin"]:
            raise ValueError("Metric must be 'ClaimSeverity' or 'Margin'")

        # Get data for each group
        data1 = self.data[self.data[group_column] == group1]
        data2 = self.data[self.data[group_column] == group2]

        if len(data1) == 0 or len(data2) == 0:
            raise ValueError(f"One or both groups have no data: {group1}, {group2}")

        # Calculate metric values
        if metric == "ClaimSeverity":
            # Only policies with claims
            values1 = data1[data1["TotalClaims"] > 0]["TotalClaims"]
            values2 = data2[data2["TotalClaims"] > 0]["TotalClaims"]
        else:  # Margin
            values1 = data1["TotalPremium"] - data1["TotalClaims"]
            values2 = data2["TotalPremium"] - data2["TotalClaims"]

        if len(values1) < 2 or len(values2) < 2:
            raise ValueError(
                "Insufficient data for t-test (need at least 2 samples per group)"
            )

        # Perform independent samples t-test
        statistic, p_value = ttest_ind(values1, values2, equal_var=False)
        reject_null = bool(p_value < self.alpha)

        # Calculate Cohen's d for effect size
        mean1, mean2 = values1.mean(), values2.mean()
        pooled_std = np.sqrt((values1.std() ** 2 + values2.std() ** 2) / 2)
        cohens_d = (mean1 - mean2) / pooled_std if pooled_std > 0 else 0

        # Interpretation
        if reject_null:
            interpretation = (
                f"REJECT null hypothesis (p={p_value:.4f} < {self.alpha}). "
                f"{metric} differs significantly between {group1} (mean={mean1:.2f}) "
                f"and {group2} (mean={mean2:.2f}). "
                f"Effect size (Cohen's d): {cohens_d:.3f}"
            )
        else:
            interpretation = (
                f"FAIL TO REJECT null hypothesis (p={p_value:.4f} >= {self.alpha}). "
                f"No significant difference in {metric} between {group1} and {group2}."
            )

        return {
            "statistic": float(statistic),
            "p_value": float(p_value),
            "reject_null": reject_null,
            "mean_group1": float(mean1),
            "mean_group2": float(mean2),
            "cohens_d": float(cohens_d),
            "interpretation": interpretation,
        }

    def anova_test(
        self,
        group_column: str,
        metric: str = "ClaimSeverity",
        selected_groups: Optional[List[str]] = None,
    ) -> Dict[str, Union[float, bool, str]]:
        """
        Perform one-way ANOVA test for multiple groups.

        Tests whether a numerical metric differs significantly across
        multiple groups (3+). Used for comparing provinces, zip codes, etc.

        Args:
            group_column: Column name to group by.
            metric: Metric to test ('ClaimSeverity' or 'Margin').
            selected_groups: Optional list of specific groups to include.

        Returns:
            Dictionary containing:
                - statistic: F-statistic from ANOVA
                - p_value: P-value from the test
                - reject_null: Whether to reject null hypothesis
                - interpretation: Business interpretation of result
                - group_means: Dictionary of mean values by group

        Raises:
            ValueError: If insufficient groups or invalid metric.
        """
        if group_column not in self.data.columns:
            raise ValueError(f"Column '{group_column}' not found in data")

        if metric not in ["ClaimSeverity", "Margin"]:
            raise ValueError("Metric must be 'ClaimSeverity' or 'Margin'")

        # Filter to selected groups if specified
        df = self.data.copy()
        if selected_groups:
            df = df[df[group_column].isin(selected_groups)]

        # Collect data for each group
        groups_data = []
        group_means = {}

        for group in df[group_column].dropna().unique():
            group_df = df[df[group_column] == group]

            if metric == "ClaimSeverity":
                values = group_df[group_df["TotalClaims"] > 0]["TotalClaims"]
            else:  # Margin
                values = group_df["TotalPremium"] - group_df["TotalClaims"]

            if len(values) >= 2:  # Need at least 2 samples
                groups_data.append(values)
                group_means[str(group)] = float(values.mean())

        if len(groups_data) < 2:
            raise ValueError("Need at least 2 groups with sufficient data for ANOVA")

        # Perform one-way ANOVA
        statistic, p_value = f_oneway(*groups_data)
        reject_null = bool(p_value < self.alpha)

        # Interpretation
        if reject_null:
            interpretation = (
                f"REJECT null hypothesis (p={p_value:.4f} < {self.alpha}). "
                f"{metric} differs significantly across {group_column} groups. "
                f"At least one group has a significantly different mean."
            )
        else:
            interpretation = (
                f"FAIL TO REJECT null hypothesis (p={p_value:.4f} >= {self.alpha}). "
                f"No significant difference in {metric} across {group_column} groups."
            )

        return {
            "statistic": float(statistic),
            "p_value": float(p_value),
            "reject_null": reject_null,
            "group_means": group_means,
            "num_groups": len(groups_data),
            "interpretation": interpretation,
        }

    def interpret_results(
        self,
        test_name: str,
        p_value: float,
        reject_null: bool,
        additional_info: Optional[Dict] = None,
    ) -> str:
        """
        Generate business interpretation of statistical test results.

        Provides actionable business insights based on statistical test outcomes,
        explaining implications for insurance pricing and risk management.

        Args:
            test_name: Name of the hypothesis test performed.
            p_value: P-value from the statistical test.
            reject_null: Whether null hypothesis was rejected.
            additional_info: Optional dictionary with extra context (e.g., effect size).

        Returns:
            Formatted string with business interpretation and recommendations.
        """
        interpretation = f"\n{'='*70}\n"
        interpretation += f"HYPOTHESIS TEST: {test_name}\n"
        interpretation += f"{'='*70}\n\n"

        # Statistical decision
        if reject_null:
            interpretation += f"✗ NULL HYPOTHESIS REJECTED (p-value = {p_value:.4f} < {self.alpha})\n\n"
            interpretation += "Statistical Conclusion:\n"
            interpretation += (
                "  There IS statistically significant evidence of differences.\n\n"
            )
        else:
            interpretation += f"✓ NULL HYPOTHESIS NOT REJECTED (p-value = {p_value:.4f} >= {self.alpha})\n\n"
            interpretation += "Statistical Conclusion:\n"
            interpretation += (
                "  There is NO statistically significant evidence of differences.\n\n"
            )

        # Business implications
        interpretation += "Business Implications:\n"

        if reject_null:
            interpretation += "  • Risk profiles differ significantly across segments\n"
            interpretation += (
                "  • Premium adjustments may be warranted for different groups\n"
            )
            interpretation += "  • Consider differentiated pricing strategies\n"
            interpretation += "  • Target low-risk segments for acquisition\n"
        else:
            interpretation += (
                "  • Risk profiles are statistically similar across segments\n"
            )
            interpretation += "  • Current pricing strategy may be appropriate\n"
            interpretation += (
                "  • No strong evidence for segment-specific adjustments\n"
            )

        # Effect size interpretation if available
        if additional_info:
            interpretation += "\nAdditional Context:\n"
            for key, value in additional_info.items():
                interpretation += f"  • {key}: {value}\n"

        interpretation += f"\n{'='*70}\n"

        return interpretation

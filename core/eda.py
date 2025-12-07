"""
Exploratory Data Analysis (EDA) Engine Module.

This module provides a comprehensive EDA framework for insurance risk analysis,
including descriptive statistics, univariate/bivariate/multivariate analysis,
and insurance-specific metrics like loss ratios.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from scipy import stats


class EDAEngine:
    """
    Engine for performing comprehensive exploratory data analysis.

    This class provides methods for descriptive statistics, distribution analysis,
    correlation analysis, and insurance-specific metrics calculations.

    Attributes:
        data: DataFrame containing the data to analyze.
        numeric_columns: List of numeric column names.
        categorical_columns: List of categorical column names.
        logger: Logger instance for tracking operations.
    """

    def __init__(self, data: pd.DataFrame):
        """
        Initialize the EDA Engine.

        Args:
            data: DataFrame to analyze.

        Raises:
            ValueError: If data is None or empty.
        """
        if data is None:
            raise ValueError("Data cannot be None")
        if len(data) == 0:
            raise ValueError("Data cannot be empty")

        self.data = data.copy()
        self.numeric_columns = self.data.select_dtypes(
            include=[np.number]
        ).columns.tolist()
        self.categorical_columns = self.data.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()

        self.logger = logging.getLogger(__name__)
        self.logger.info(
            f"EDAEngine initialized with {len(self.data)} rows, "
            f"{len(self.numeric_columns)} numeric, "
            f"{len(self.categorical_columns)} categorical columns"
        )

    def descriptive_statistics(
        self, columns: Optional[List[str]] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Calculate comprehensive descriptive statistics.

        Generates detailed statistics for numeric and categorical columns including
        measures of central tendency, dispersion, and shape.

        Args:
            columns: Specific columns to analyze. If None, analyzes all columns.

        Returns:
            Dictionary with 'numeric' and 'categorical' DataFrames containing
            descriptive statistics.

        Raises:
            ValueError: If specified columns don't exist in the data.
        """
        if columns:
            invalid_cols = [col for col in columns if col not in self.data.columns]
            if invalid_cols:
                raise ValueError(f"Columns not found: {invalid_cols}")

        self.logger.info("Calculating descriptive statistics...")

        result = {}

        # Numeric statistics
        numeric_cols = (
            [col for col in columns if col in self.numeric_columns]
            if columns
            else self.numeric_columns
        )

        if numeric_cols:
            numeric_stats = self.data[numeric_cols].describe().T
            # Add additional statistics
            numeric_stats["median"] = self.data[numeric_cols].median()
            numeric_stats["variance"] = self.data[numeric_cols].var()
            numeric_stats["skewness"] = self.data[numeric_cols].skew()
            numeric_stats["kurtosis"] = self.data[numeric_cols].kurtosis()
            numeric_stats["missing"] = self.data[numeric_cols].isnull().sum()
            numeric_stats["missing_pct"] = (
                self.data[numeric_cols].isnull().sum() / len(self.data) * 100
            )

            result["numeric"] = numeric_stats

        # Categorical statistics
        categorical_cols = (
            [col for col in columns if col in self.categorical_columns]
            if columns
            else self.categorical_columns
        )

        if categorical_cols:
            cat_stats = []
            for col in categorical_cols:
                stats_dict = {
                    "column": col,
                    "count": self.data[col].count(),
                    "unique": self.data[col].nunique(),
                    "top": (
                        self.data[col].mode().iloc[0]
                        if len(self.data[col].mode()) > 0
                        else None
                    ),
                    "freq": (
                        self.data[col].value_counts().iloc[0]
                        if len(self.data[col]) > 0
                        else 0
                    ),
                    "missing": self.data[col].isnull().sum(),
                    "missing_pct": self.data[col].isnull().sum() / len(self.data) * 100,
                }
                cat_stats.append(stats_dict)

            result["categorical"] = pd.DataFrame(cat_stats).set_index("column")

        self.logger.info(
            f"Calculated statistics for {len(result.get('numeric', []))} numeric "
            f"and {len(result.get('categorical', []))} categorical columns"
        )

        return result

    def univariate_analysis(self, column: str) -> Dict[str, Union[pd.DataFrame, Dict]]:
        """
        Perform detailed univariate analysis on a single column.

        Args:
            column: Column name to analyze.

        Returns:
            Dictionary containing distribution statistics, value counts,
            percentiles, and normality test results.

        Raises:
            ValueError: If column doesn't exist in the data.
        """
        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' not found in data")

        self.logger.info(f"Performing univariate analysis on: {column}")

        result = {"column": column, "dtype": str(self.data[column].dtype)}

        if column in self.numeric_columns:
            # Numeric column analysis
            col_data = self.data[column].dropna()

            result["statistics"] = {
                "count": len(col_data),
                "mean": col_data.mean(),
                "median": col_data.median(),
                "mode": col_data.mode().iloc[0] if len(col_data.mode()) > 0 else None,
                "std": col_data.std(),
                "variance": col_data.var(),
                "min": col_data.min(),
                "max": col_data.max(),
                "range": col_data.max() - col_data.min(),
                "skewness": col_data.skew(),
                "kurtosis": col_data.kurtosis(),
                "missing": self.data[column].isnull().sum(),
                "missing_pct": self.data[column].isnull().sum() / len(self.data) * 100,
            }

            # Percentiles
            result["percentiles"] = {
                "1%": col_data.quantile(0.01),
                "5%": col_data.quantile(0.05),
                "10%": col_data.quantile(0.10),
                "25%": col_data.quantile(0.25),
                "50%": col_data.quantile(0.50),
                "75%": col_data.quantile(0.75),
                "90%": col_data.quantile(0.90),
                "95%": col_data.quantile(0.95),
                "99%": col_data.quantile(0.99),
            }

            # Normality test (Shapiro-Wilk for smaller samples, Anderson-Darling for larger)
            if len(col_data) < 5000:
                stat, p_value = stats.shapiro(col_data)
                result["normality_test"] = {
                    "test": "Shapiro-Wilk",
                    "statistic": stat,
                    "p_value": p_value,
                    "is_normal": p_value > 0.05,
                }
            else:
                # Use sample for Shapiro-Wilk with large datasets
                sample = col_data.sample(n=5000, random_state=42)
                stat, p_value = stats.shapiro(sample)
                result["normality_test"] = {
                    "test": "Shapiro-Wilk (on 5000 sample)",
                    "statistic": stat,
                    "p_value": p_value,
                    "is_normal": p_value > 0.05,
                }

        else:
            # Categorical column analysis
            value_counts = self.data[column].value_counts()
            value_percentages = self.data[column].value_counts(normalize=True) * 100

            result["statistics"] = {
                "count": self.data[column].count(),
                "unique": self.data[column].nunique(),
                "top": value_counts.index[0] if len(value_counts) > 0 else None,
                "top_freq": value_counts.iloc[0] if len(value_counts) > 0 else 0,
                "top_pct": (
                    value_percentages.iloc[0] if len(value_percentages) > 0 else 0
                ),
                "missing": self.data[column].isnull().sum(),
                "missing_pct": self.data[column].isnull().sum() / len(self.data) * 100,
            }

            result["value_counts"] = pd.DataFrame(
                {"count": value_counts, "percentage": value_percentages}
            )

        return result

    def bivariate_analysis(
        self, col1: str, col2: str
    ) -> Dict[str, Union[pd.DataFrame, Dict, float]]:
        """
        Analyze relationship between two columns.

        Performs appropriate analysis based on column types:
        - Numeric vs Numeric: Correlation, scatter plot data
        - Numeric vs Categorical: Group statistics
        - Categorical vs Categorical: Contingency table, chi-square test

        Args:
            col1: First column name.
            col2: Second column name.

        Returns:
            Dictionary containing relationship metrics and analysis results.

        Raises:
            ValueError: If columns don't exist in the data.
        """
        if col1 not in self.data.columns or col2 not in self.data.columns:
            raise ValueError(f"One or both columns not found: {col1}, {col2}")

        self.logger.info(f"Performing bivariate analysis: {col1} vs {col2}")

        result = {
            "column1": col1,
            "column2": col2,
            "type1": "numeric" if col1 in self.numeric_columns else "categorical",
            "type2": "numeric" if col2 in self.numeric_columns else "categorical",
        }

        # Numeric vs Numeric
        if col1 in self.numeric_columns and col2 in self.numeric_columns:
            clean_data = self.data[[col1, col2]].dropna()

            # Correlation metrics
            pearson_corr, pearson_p = stats.pearsonr(clean_data[col1], clean_data[col2])
            spearman_corr, spearman_p = stats.spearmanr(
                clean_data[col1], clean_data[col2]
            )

            result["correlation"] = {
                "pearson": pearson_corr,
                "pearson_p_value": pearson_p,
                "spearman": spearman_corr,
                "spearman_p_value": spearman_p,
                "relationship_strength": self._interpret_correlation(pearson_corr),
            }

        # Numeric vs Categorical
        elif (col1 in self.numeric_columns) != (col2 in self.numeric_columns):
            numeric_col = col1 if col1 in self.numeric_columns else col2
            categorical_col = col2 if col1 in self.numeric_columns else col1

            # Group statistics
            grouped = self.data.groupby(categorical_col)[numeric_col].agg(
                ["count", "mean", "median", "std", "min", "max"]
            )

            result["group_statistics"] = grouped

            # ANOVA test
            groups = [
                group[numeric_col].dropna()
                for name, group in self.data.groupby(categorical_col)
                if len(group[numeric_col].dropna()) > 0
            ]

            if len(groups) > 1:
                f_stat, p_value = stats.f_oneway(*groups)
                result["anova"] = {
                    "f_statistic": f_stat,
                    "p_value": p_value,
                    "significant": p_value < 0.05,
                }

        # Categorical vs Categorical
        else:
            # Contingency table
            contingency = pd.crosstab(self.data[col1], self.data[col2])
            result["contingency_table"] = contingency

            # Chi-square test
            chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
            result["chi_square_test"] = {
                "chi2_statistic": chi2,
                "p_value": p_value,
                "degrees_of_freedom": dof,
                "significant": p_value < 0.05,
            }

        return result

    def multivariate_analysis(
        self, columns: Optional[List[str]] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Perform multivariate analysis on multiple columns.

        Calculates correlation matrix, covariance matrix, and provides
        insights into multi-dimensional relationships.

        Args:
            columns: List of columns to analyze. If None, uses all numeric columns.

        Returns:
            Dictionary containing correlation matrix, covariance matrix,
            and highly correlated pairs.

        Raises:
            ValueError: If specified columns don't exist or are not numeric.
        """
        if columns is None:
            columns = self.numeric_columns
        else:
            invalid_cols = [col for col in columns if col not in self.numeric_columns]
            if invalid_cols:
                raise ValueError(f"Columns must be numeric. Invalid: {invalid_cols}")

        if len(columns) < 2:
            raise ValueError("Need at least 2 columns for multivariate analysis")

        self.logger.info(f"Performing multivariate analysis on {len(columns)} columns")

        result = {}

        # Correlation matrix
        result["correlation_matrix"] = self.data[columns].corr()

        # Covariance matrix
        result["covariance_matrix"] = self.data[columns].cov()

        # Find highly correlated pairs (|r| > 0.7)
        corr_matrix = result["correlation_matrix"]
        high_corr = []

        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                if abs(corr_value) > 0.7:
                    high_corr.append(
                        {
                            "column1": corr_matrix.columns[i],
                            "column2": corr_matrix.columns[j],
                            "correlation": corr_value,
                            "strength": self._interpret_correlation(corr_value),
                        }
                    )

        result["highly_correlated_pairs"] = pd.DataFrame(high_corr)

        return result

    def detect_outliers(
        self,
        columns: Optional[List[str]] = None,
        method: str = "iqr",
        threshold: float = 1.5,
    ) -> Dict[str, pd.DataFrame]:
        """
        Detect outliers in numeric columns using specified method.

        Args:
            columns: Columns to check. If None, checks all numeric columns.
            method: Outlier detection method ('iqr', 'zscore', or 'modified_zscore').
            threshold: Threshold for outlier detection:
                       - IQR: multiplier for IQR (default 1.5)
                       - Z-score: number of standard deviations (default 1.5, typically 3)
                       - Modified Z-score: threshold (default 1.5, typically 3.5)

        Returns:
            Dictionary with outlier DataFrames for each column containing
            outlier indices, values, and statistics.

        Raises:
            ValueError: If invalid method or columns specified.
        """
        valid_methods = ["iqr", "zscore", "modified_zscore"]
        if method not in valid_methods:
            raise ValueError(f"Method must be one of {valid_methods}")

        if columns is None:
            columns = self.numeric_columns
        else:
            invalid_cols = [col for col in columns if col not in self.numeric_columns]
            if invalid_cols:
                raise ValueError(f"Columns must be numeric. Invalid: {invalid_cols}")

        self.logger.info(
            f"Detecting outliers using {method} method for {len(columns)} columns"
        )

        result = {}

        for col in columns:
            col_data = self.data[col].dropna()

            if method == "iqr":
                Q1 = col_data.quantile(0.25)
                Q3 = col_data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                outlier_mask = (col_data < lower_bound) | (col_data > upper_bound)

            elif method == "zscore":
                z_scores = np.abs(stats.zscore(col_data))
                outlier_mask = z_scores > threshold

            else:  # modified_zscore
                median = col_data.median()
                mad = np.median(np.abs(col_data - median))
                if mad != 0:
                    modified_z_scores = 0.6745 * (col_data - median) / mad
                    outlier_mask = np.abs(modified_z_scores) > threshold
                else:
                    # If MAD is 0, no outliers can be detected
                    outlier_mask = pd.Series(
                        [False] * len(col_data), index=col_data.index
                    )

            outliers = col_data[outlier_mask]

            if len(outliers) > 0:
                result[col] = pd.DataFrame(
                    {
                        "value": outliers,
                        "index": outliers.index,
                        "count": len(outliers),
                        "percentage": len(outliers) / len(col_data) * 100,
                    }
                )

        self.logger.info(f"Found outliers in {len(result)} columns")

        return result

    def calculate_loss_ratio(
        self,
        claims_column: str = "TotalClaims",
        premium_column: str = "TotalPremium",
        group_by: Optional[List[str]] = None,
    ) -> Union[float, pd.DataFrame]:
        """
        Calculate insurance loss ratio (TotalClaims / TotalPremium).

        Loss ratio is a key insurance metric indicating profitability.
        A ratio > 1.0 means claims exceed premiums (loss).
        A ratio < 1.0 means premiums exceed claims (profit).

        Args:
            claims_column: Name of the total claims column.
            premium_column: Name of the total premium column.
            group_by: Optional columns to group by (e.g., Province, VehicleType).

        Returns:
            Overall loss ratio as float, or DataFrame with loss ratios by group.

        Raises:
            ValueError: If specified columns don't exist or are not numeric.
        """
        if claims_column not in self.data.columns:
            raise ValueError(f"Claims column '{claims_column}' not found")
        if premium_column not in self.data.columns:
            raise ValueError(f"Premium column '{premium_column}' not found")

        if claims_column not in self.numeric_columns:
            raise ValueError(f"Claims column '{claims_column}' must be numeric")
        if premium_column not in self.numeric_columns:
            raise ValueError(f"Premium column '{premium_column}' must be numeric")

        self.logger.info("Calculating loss ratios...")

        if group_by is None:
            # Overall loss ratio
            total_claims = self.data[claims_column].sum()
            total_premiums = self.data[premium_column].sum()

            if total_premiums == 0:
                self.logger.warning(
                    "Total premiums are zero, cannot calculate loss ratio"
                )
                return np.nan

            loss_ratio = total_claims / total_premiums
            self.logger.info(f"Overall loss ratio: {loss_ratio:.4f}")

            return loss_ratio

        else:
            # Group-wise loss ratios
            invalid_cols = [col for col in group_by if col not in self.data.columns]
            if invalid_cols:
                raise ValueError(f"Group by columns not found: {invalid_cols}")

            grouped = self.data.groupby(group_by).agg(
                {
                    claims_column: "sum",
                    premium_column: "sum",
                }
            )

            grouped["loss_ratio"] = grouped[claims_column] / grouped[premium_column]
            grouped["policy_count"] = self.data.groupby(group_by).size()

            # Add margin
            grouped["margin"] = grouped[premium_column] - grouped[claims_column]

            # Sort by loss ratio descending
            grouped = grouped.sort_values("loss_ratio", ascending=False)

            self.logger.info(f"Calculated loss ratios for {len(grouped)} groups")

            return grouped

    def _interpret_correlation(self, correlation: float) -> str:
        """
        Interpret the strength of correlation coefficient.

        Args:
            correlation: Correlation coefficient (-1 to 1).

        Returns:
            String description of correlation strength.
        """
        abs_corr = abs(correlation)

        if abs_corr >= 0.9:
            strength = "Very Strong"
        elif abs_corr >= 0.7:
            strength = "Strong"
        elif abs_corr >= 0.5:
            strength = "Moderate"
        elif abs_corr >= 0.3:
            strength = "Weak"
        else:
            strength = "Very Weak"

        direction = "Positive" if correlation >= 0 else "Negative"

        return f"{strength} {direction}"

    def get_summary(self) -> Dict[str, Union[int, List[str]]]:
        """
        Get a summary of the data structure for EDA.

        Returns:
            Dictionary containing data dimensions and column information.
        """
        return {
            "rows": len(self.data),
            "columns": len(self.data.columns),
            "numeric_columns": self.numeric_columns,
            "categorical_columns": self.categorical_columns,
            "memory_usage_mb": self.data.memory_usage(deep=True).sum() / 1024 / 1024,
        }

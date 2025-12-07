"""
Data Quality Checker Module for Insurance Risk Analysis.

This module provides comprehensive data quality assessment functionality
including missing value analysis, duplicate detection, data type validation,
and quality report generation.
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import pandas as pd
import numpy as np
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataQualityChecker:
    """
    A class for assessing and reporting data quality issues.

    This class provides comprehensive data quality checks including missing values,
    duplicates, data types, outliers, and generates detailed quality reports for
    insurance policy data.

    Attributes:
        data (pd.DataFrame): The data to check for quality issues.
        quality_report (Dict): Comprehensive quality assessment report.

    Example:
        >>> checker = DataQualityChecker(df)
        >>> missing_info = checker.check_missing_values()
        >>> duplicates = checker.check_duplicates()
        >>> report = checker.generate_quality_report()
    """

    def __init__(self, data: pd.DataFrame):
        """
        Initialize the DataQualityChecker with data.

        Args:
            data: DataFrame to check for quality issues.

        Raises:
            ValueError: If data is None or empty.
        """
        if data is None:
            raise ValueError("Data cannot be None")

        if data.empty:
            raise ValueError("Data cannot be empty")

        self.data = data
        self.quality_report: Optional[Dict] = None
        logger.info(
            f"DataQualityChecker initialized with {len(data)} rows and {len(data.columns)} columns"
        )

    def check_missing_values(
        self, threshold: float = 0.0
    ) -> Dict[str, Union[int, float, Dict]]:
        """
        Analyze missing values in the dataset.

        Args:
            threshold: Minimum percentage of missing values to report (0-100).
                      Only columns with missing percentage >= threshold are included.

        Returns:
            Dict containing:
                - 'total_missing': Total count of missing values
                - 'missing_percentage': Overall percentage of missing values
                - 'columns_with_missing': Number of columns with missing values
                - 'missing_by_column': Dict of {column: count} for columns above threshold
                - 'missing_percentage_by_column': Dict of {column: percentage}
                - 'rows_with_missing': Number of rows with at least one missing value
                - 'complete_rows': Number of rows with no missing values

        Example:
            >>> checker = DataQualityChecker(df)
            >>> missing_info = checker.check_missing_values(threshold=5.0)
            >>> print(f"Total missing: {missing_info['total_missing']}")
        """
        # Calculate missing values per column
        missing_counts = self.data.isnull().sum()
        total_rows = len(self.data)
        missing_percentages = (missing_counts / total_rows * 100).round(2)

        # Filter by threshold
        columns_above_threshold = missing_percentages[missing_percentages >= threshold]

        # Count rows with any missing values
        rows_with_missing = self.data.isnull().any(axis=1).sum()

        # Create detailed report
        missing_info = {
            "total_missing": int(missing_counts.sum()),
            "missing_percentage": round(
                (missing_counts.sum() / (total_rows * len(self.data.columns)) * 100), 2
            ),
            "columns_with_missing": int((missing_counts > 0).sum()),
            "missing_by_column": missing_counts[missing_counts > 0].to_dict(),
            "missing_percentage_by_column": missing_percentages[
                missing_percentages > 0
            ].to_dict(),
            "columns_above_threshold": columns_above_threshold.to_dict(),
            "rows_with_missing": int(rows_with_missing),
            "rows_with_missing_percentage": round(
                rows_with_missing / total_rows * 100, 2
            ),
            "complete_rows": int(total_rows - rows_with_missing),
            "complete_rows_percentage": round(
                (total_rows - rows_with_missing) / total_rows * 100, 2
            ),
        }

        logger.info(
            f"Missing values check: {missing_info['total_missing']} "
            f"({missing_info['missing_percentage']}%) missing values found"
        )

        return missing_info

    def check_duplicates(
        self, subset: Optional[List[str]] = None, keep: str = "first"
    ) -> Dict[str, Union[int, float, pd.DataFrame]]:
        """
        Check for duplicate rows in the dataset.

        Args:
            subset: List of column names to consider for identifying duplicates.
                   If None, all columns are used.
            keep: Which duplicates to mark. Options: 'first', 'last', False.
                 - 'first': Mark duplicates as True except for the first occurrence.
                 - 'last': Mark duplicates as True except for the last occurrence.
                 - False: Mark all duplicates as True.

        Returns:
            Dict containing:
                - 'duplicate_count': Number of duplicate rows
                - 'duplicate_percentage': Percentage of rows that are duplicates
                - 'unique_count': Number of unique rows
                - 'duplicate_rows': DataFrame of duplicate rows (if any)

        Example:
            >>> checker = DataQualityChecker(df)
            >>> dup_info = checker.check_duplicates()
            >>> print(f"Found {dup_info['duplicate_count']} duplicates")
        """
        # Find duplicates
        duplicates_mask = self.data.duplicated(subset=subset, keep=keep)
        duplicate_count = duplicates_mask.sum()

        # Get duplicate rows
        duplicate_rows = (
            self.data[duplicates_mask] if duplicate_count > 0 else pd.DataFrame()
        )

        duplicate_info = {
            "duplicate_count": int(duplicate_count),
            "duplicate_percentage": round(duplicate_count / len(self.data) * 100, 2),
            "unique_count": int(len(self.data) - duplicate_count),
            "unique_percentage": round(
                (len(self.data) - duplicate_count) / len(self.data) * 100, 2
            ),
            "duplicate_rows": duplicate_rows,
            "columns_checked": subset if subset else "all columns",
        }

        logger.info(
            f"Duplicate check: {duplicate_count} duplicate rows found "
            f"({duplicate_info['duplicate_percentage']}%)"
        )

        return duplicate_info

    def check_data_types(self) -> Dict[str, Union[Dict, List]]:
        """
        Validate and report data types for all columns.

        Returns:
            Dict containing:
                - 'dtypes': Dictionary of {column: dtype}
                - 'type_distribution': Count of columns by data type
                - 'numeric_columns': List of numeric column names
                - 'categorical_columns': List of categorical/object column names
                - 'datetime_columns': List of datetime column names
                - 'boolean_columns': List of boolean column names
                - 'potential_type_issues': Columns that might have incorrect types

        Example:
            >>> checker = DataQualityChecker(df)
            >>> type_info = checker.check_data_types()
            >>> print(f"Numeric columns: {len(type_info['numeric_columns'])}")
        """
        dtypes_dict = {col: str(dtype) for col, dtype in self.data.dtypes.items()}

        # Categorize columns by type
        numeric_columns = self.data.select_dtypes(include=[np.number]).columns.tolist()
        categorical_columns = self.data.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()
        datetime_columns = self.data.select_dtypes(
            include=["datetime64"]
        ).columns.tolist()
        boolean_columns = self.data.select_dtypes(include=["bool"]).columns.tolist()

        # Check for potential type issues
        potential_issues = []

        # Check if numeric columns stored as objects
        for col in categorical_columns:
            sample = self.data[col].dropna().head(100)
            if len(sample) > 0:
                try:
                    pd.to_numeric(sample, errors="raise")
                    potential_issues.append(
                        {
                            "column": col,
                            "current_type": "object",
                            "suggested_type": "numeric",
                            "reason": "All non-null values appear to be numeric",
                        }
                    )
                except (ValueError, TypeError):
                    pass

        # Get type distribution
        type_distribution = self.data.dtypes.value_counts().to_dict()
        type_distribution = {str(k): int(v) for k, v in type_distribution.items()}

        type_info = {
            "dtypes": dtypes_dict,
            "type_distribution": type_distribution,
            "numeric_columns": numeric_columns,
            "categorical_columns": categorical_columns,
            "datetime_columns": datetime_columns,
            "boolean_columns": boolean_columns,
            "potential_type_issues": potential_issues,
        }

        logger.info(
            f"Data types: {len(numeric_columns)} numeric, "
            f"{len(categorical_columns)} categorical, "
            f"{len(datetime_columns)} datetime"
        )

        return type_info

    def check_value_ranges(
        self, columns: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, Union[float, int]]]:
        """
        Check value ranges for numeric columns.

        Args:
            columns: List of numeric columns to check. If None, checks all numeric columns.

        Returns:
            Dict of {column: {'min', 'max', 'mean', 'median', 'std', 'zeros', 'negatives'}}

        Example:
            >>> checker = DataQualityChecker(df)
            >>> ranges = checker.check_value_ranges(['TotalPremium', 'TotalClaims'])
            >>> print(ranges['TotalPremium'])
        """
        if columns is None:
            columns = self.data.select_dtypes(include=[np.number]).columns.tolist()

        range_info = {}

        for col in columns:
            if col not in self.data.columns:
                logger.warning(f"Column '{col}' not found in data")
                continue

            if not pd.api.types.is_numeric_dtype(self.data[col]):
                logger.warning(f"Column '{col}' is not numeric")
                continue

            col_data = self.data[col].dropna()

            if len(col_data) == 0:
                range_info[col] = {"status": "all_null"}
                continue

            range_info[col] = {
                "min": float(col_data.min()),
                "max": float(col_data.max()),
                "mean": float(col_data.mean()),
                "median": float(col_data.median()),
                "std": float(col_data.std()),
                "zeros": int((col_data == 0).sum()),
                "zeros_percentage": round(
                    (col_data == 0).sum() / len(col_data) * 100, 2
                ),
                "negatives": int((col_data < 0).sum()),
                "negatives_percentage": round(
                    (col_data < 0).sum() / len(col_data) * 100, 2
                ),
                "non_null_count": int(len(col_data)),
            }

        return range_info

    def check_categorical_distributions(
        self, columns: Optional[List[str]] = None, top_n: int = 10
    ) -> Dict[str, Dict[str, Union[int, Dict]]]:
        """
        Analyze categorical column distributions.

        Args:
            columns: List of categorical columns to analyze. If None, checks all object columns.
            top_n: Number of top categories to include in the report.

        Returns:
            Dict of {column: {'unique_count', 'top_values', 'value_counts'}}

        Example:
            >>> checker = DataQualityChecker(df)
            >>> cat_info = checker.check_categorical_distributions(['Province', 'Gender'])
            >>> print(cat_info['Province']['unique_count'])
        """
        if columns is None:
            columns = self.data.select_dtypes(
                include=["object", "category"]
            ).columns.tolist()

        categorical_info = {}

        for col in columns:
            if col not in self.data.columns:
                logger.warning(f"Column '{col}' not found in data")
                continue

            value_counts = self.data[col].value_counts()
            unique_count = self.data[col].nunique()

            # Get top N values
            top_values = value_counts.head(top_n).to_dict()

            categorical_info[col] = {
                "unique_count": int(unique_count),
                "top_values": top_values,
                "most_common": (
                    str(value_counts.index[0]) if len(value_counts) > 0 else None
                ),
                "most_common_count": (
                    int(value_counts.iloc[0]) if len(value_counts) > 0 else 0
                ),
                "least_common": (
                    str(value_counts.index[-1]) if len(value_counts) > 0 else None
                ),
                "least_common_count": (
                    int(value_counts.iloc[-1]) if len(value_counts) > 0 else 0
                ),
                "cardinality_ratio": round(unique_count / len(self.data), 4),
            }

        return categorical_info

    def detect_outliers_iqr(
        self, columns: Optional[List[str]] = None, multiplier: float = 1.5
    ) -> Dict[str, Dict[str, Union[int, float, List]]]:
        """
        Detect outliers using the IQR (Interquartile Range) method.

        Args:
            columns: List of numeric columns to check. If None, checks all numeric columns.
            multiplier: IQR multiplier for outlier detection (default 1.5).
                       Values beyond Q1 - multiplier*IQR or Q3 + multiplier*IQR are outliers.

        Returns:
            Dict of {column: {'outlier_count', 'outlier_percentage', 'lower_bound',
                              'upper_bound', 'Q1', 'Q3', 'IQR'}}

        Example:
            >>> checker = DataQualityChecker(df)
            >>> outliers = checker.detect_outliers_iqr(['TotalPremium', 'TotalClaims'])
            >>> print(f"TotalPremium outliers: {outliers['TotalPremium']['outlier_count']}")
        """
        if columns is None:
            columns = self.data.select_dtypes(include=[np.number]).columns.tolist()

        outlier_info = {}

        for col in columns:
            if col not in self.data.columns:
                continue

            if not pd.api.types.is_numeric_dtype(self.data[col]):
                continue

            col_data = self.data[col].dropna()

            if len(col_data) == 0:
                continue

            # Calculate IQR
            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1

            # Define bounds
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR

            # Find outliers
            outliers_mask = (col_data < lower_bound) | (col_data > upper_bound)
            outlier_count = outliers_mask.sum()

            outlier_info[col] = {
                "outlier_count": int(outlier_count),
                "outlier_percentage": round(outlier_count / len(col_data) * 100, 2),
                "lower_bound": float(lower_bound),
                "upper_bound": float(upper_bound),
                "Q1": float(Q1),
                "Q3": float(Q3),
                "IQR": float(IQR),
                "min_outlier": (
                    float(col_data[outliers_mask].min()) if outlier_count > 0 else None
                ),
                "max_outlier": (
                    float(col_data[outliers_mask].max()) if outlier_count > 0 else None
                ),
            }

        return outlier_info

    def generate_quality_report(self, include_outliers: bool = True) -> Dict[str, Any]:
        """
        Generate a comprehensive data quality report.

        Args:
            include_outliers: Whether to include outlier detection (can be slow for large datasets).

        Returns:
            Dict containing comprehensive quality assessment with sections:
                - 'overview': Basic dataset information
                - 'missing_values': Missing value analysis
                - 'duplicates': Duplicate row detection
                - 'data_types': Data type information
                - 'value_ranges': Numeric value ranges
                - 'categorical': Categorical distributions
                - 'outliers': Outlier detection (if enabled)
                - 'quality_score': Overall quality score (0-100)
                - 'recommendations': List of recommended actions

        Example:
            >>> checker = DataQualityChecker(df)
            >>> report = checker.generate_quality_report()
            >>> print(f"Quality score: {report['quality_score']}")
            >>> for rec in report['recommendations']:
            ...     print(f"- {rec}")
        """
        logger.info("Generating comprehensive quality report...")

        # Overview
        overview = {
            "total_rows": len(self.data),
            "total_columns": len(self.data.columns),
            "memory_usage_mb": round(
                self.data.memory_usage(deep=True).sum() / (1024 * 1024), 2
            ),
            "report_timestamp": datetime.now().isoformat(),
        }

        # Run all checks
        missing_values = self.check_missing_values()
        duplicates = self.check_duplicates()
        data_types = self.check_data_types()
        value_ranges = self.check_value_ranges()
        categorical = self.check_categorical_distributions()

        outliers = {}
        if include_outliers:
            outliers = self.detect_outliers_iqr()

        # Calculate quality score (0-100)
        score = 100.0

        # Deduct points for missing values
        if missing_values["missing_percentage"] > 0:
            score -= min(missing_values["missing_percentage"], 20)

        # Deduct points for duplicates
        if duplicates["duplicate_percentage"] > 0:
            score -= min(duplicates["duplicate_percentage"] * 2, 15)

        # Deduct points for potential type issues
        if len(data_types["potential_type_issues"]) > 0:
            score -= min(len(data_types["potential_type_issues"]) * 2, 10)

        quality_score = max(round(score, 2), 0)

        # Generate recommendations
        recommendations = []

        if missing_values["missing_percentage"] > 5:
            recommendations.append(
                f"High percentage of missing values ({missing_values['missing_percentage']}%). "
                f"Consider imputation or removal strategies."
            )

        if duplicates["duplicate_count"] > 0:
            recommendations.append(
                f"Found {duplicates['duplicate_count']} duplicate rows. "
                f"Review and remove if appropriate."
            )

        if len(data_types["potential_type_issues"]) > 0:
            recommendations.append(
                f"Found {len(data_types['potential_type_issues'])} potential data type issues. "
                f"Review and convert types if needed."
            )

        if missing_values["columns_with_missing"] > len(self.data.columns) * 0.5:
            recommendations.append(
                "More than 50% of columns have missing values. "
                "Consider data source quality review."
            )

        # Compile report
        quality_report = {
            "overview": overview,
            "missing_values": missing_values,
            "duplicates": {
                k: v for k, v in duplicates.items() if k != "duplicate_rows"
            },
            "data_types": data_types,
            "value_ranges": value_ranges,
            "categorical": categorical,
            "outliers": outliers,
            "quality_score": quality_score,
            "recommendations": recommendations,
        }

        self.quality_report = quality_report

        logger.info(f"Quality report generated. Overall score: {quality_score}/100")

        return quality_report

    def print_quality_summary(self) -> None:
        """
        Print a human-readable summary of data quality.

        Raises:
            ValueError: If quality report hasn't been generated yet.

        Example:
            >>> checker = DataQualityChecker(df)
            >>> checker.generate_quality_report()
            >>> checker.print_quality_summary()
        """
        if self.quality_report is None:
            raise ValueError(
                "Quality report not generated. Call generate_quality_report() first."
            )

        report = self.quality_report

        print("\n" + "=" * 70)
        print("DATA QUALITY REPORT SUMMARY")
        print("=" * 70)

        print(f"\n📊 OVERVIEW:")
        print(f"  Rows: {report['overview']['total_rows']:,}")
        print(f"  Columns: {report['overview']['total_columns']}")
        print(f"  Memory: {report['overview']['memory_usage_mb']} MB")
        print(f"  Quality Score: {report['quality_score']}/100")

        print(f"\n❌ MISSING VALUES:")
        print(
            f"  Total: {report['missing_values']['total_missing']:,} "
            f"({report['missing_values']['missing_percentage']}%)"
        )
        print(f"  Columns affected: {report['missing_values']['columns_with_missing']}")
        print(
            f"  Complete rows: {report['missing_values']['complete_rows']:,} "
            f"({report['missing_values']['complete_rows_percentage']}%)"
        )

        print(f"\n🔄 DUPLICATES:")
        print(
            f"  Duplicate rows: {report['duplicates']['duplicate_count']:,} "
            f"({report['duplicates']['duplicate_percentage']}%)"
        )

        print(f"\n📝 DATA TYPES:")
        print(f"  Numeric: {len(report['data_types']['numeric_columns'])}")
        print(f"  Categorical: {len(report['data_types']['categorical_columns'])}")
        print(f"  DateTime: {len(report['data_types']['datetime_columns'])}")
        print(
            f"  Potential issues: {len(report['data_types']['potential_type_issues'])}"
        )

        if report["outliers"]:
            print(f"\n⚠️  OUTLIERS (IQR method):")
            for col, info in list(report["outliers"].items())[:5]:
                print(
                    f"  {col}: {info['outlier_count']:,} ({info['outlier_percentage']}%)"
                )

        if report["recommendations"]:
            print(f"\n💡 RECOMMENDATIONS:")
            for i, rec in enumerate(report["recommendations"], 1):
                print(f"  {i}. {rec}")

        print("\n" + "=" * 70 + "\n")

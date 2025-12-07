"""
Data Processing Module for Insurance Risk Analysis.

This module provides comprehensive data processing capabilities including:
- Missing value handling with multiple imputation strategies
- Outlier detection and treatment
- Feature engineering for insurance domain
- Data transformation and scaling
- Data quality validation

Classes:
    DataProcessor: Main class for data processing operations

Author: AlphaCare Insurance Solutions
Date: December 2025
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
import logging


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataProcessor:
    """
    Comprehensive data processor for insurance risk analysis.

    This class handles data cleaning, outlier treatment, feature engineering,
    and data transformation operations specific to insurance data.

    Attributes:
        data (pd.DataFrame): The dataset being processed.
        numeric_columns (List[str]): List of numeric column names.
        categorical_columns (List[str]): List of categorical column names.
    """

    def __init__(self, data: Optional[pd.DataFrame] = None):
        """
        Initialize DataProcessor.

        Args:
            data: Optional DataFrame to process. Can be set later using set_data().
        """
        self.data = data.copy() if data is not None else None
        self.numeric_columns = []
        self.categorical_columns = []

        if self.data is not None:
            self._identify_column_types()

    def set_data(self, data: pd.DataFrame) -> None:
        """
        Set or update the data to be processed.

        Args:
            data: DataFrame to process.

        Raises:
            TypeError: If data is not a pandas DataFrame.
        """
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Data must be a pandas DataFrame")

        self.data = data.copy()
        self._identify_column_types()
        logger.info(f"Data set with shape: {self.data.shape}")

    def _identify_column_types(self) -> None:
        """Identify numeric and categorical columns in the dataset."""
        if self.data is None:
            return

        self.numeric_columns = self.data.select_dtypes(
            include=[np.number]
        ).columns.tolist()
        self.categorical_columns = self.data.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()

        logger.info(f"Identified {len(self.numeric_columns)} numeric columns")
        logger.info(f"Identified {len(self.categorical_columns)} categorical columns")

    def handle_missing_values(
        self,
        strategy: str = "auto",
        numeric_strategy: str = "median",
        categorical_strategy: str = "mode",
        fill_value: Optional[Any] = None,
        columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Handle missing values in the dataset.

        Args:
            strategy: Overall strategy ('auto', 'drop', 'fill', 'custom').
                - 'auto': Use numeric_strategy for numeric, categorical_strategy for categorical
                - 'drop': Drop rows with missing values
                - 'fill': Fill with fill_value
                - 'custom': Use custom logic
            numeric_strategy: Strategy for numeric columns ('mean', 'median', 'mode', 'zero', 'forward_fill', 'backward_fill').
            categorical_strategy: Strategy for categorical columns ('mode', 'constant', 'forward_fill', 'backward_fill').
            fill_value: Value to use when strategy is 'fill' or categorical_strategy is 'constant'.
            columns: Specific columns to process. If None, process all columns.

        Returns:
            DataFrame with missing values handled.

        Raises:
            ValueError: If data is not set or invalid strategy provided.
        """
        if self.data is None:
            raise ValueError("No data set. Use set_data() first.")

        df = self.data.copy()
        cols_to_process = columns if columns else df.columns.tolist()

        # Count missing values before
        missing_before = df[cols_to_process].isnull().sum().sum()

        if strategy == "drop":
            df = df.dropna(subset=cols_to_process)
            logger.info(f"Dropped rows with missing values. New shape: {df.shape}")

        elif strategy == "fill":
            if fill_value is None:
                raise ValueError("fill_value must be provided when strategy is 'fill'")
            df[cols_to_process] = df[cols_to_process].fillna(fill_value)
            logger.info(f"Filled missing values with {fill_value}")

        elif strategy == "auto":
            # Handle numeric columns
            numeric_cols = [
                col for col in cols_to_process if col in self.numeric_columns
            ]
            for col in numeric_cols:
                if df[col].isnull().sum() > 0:
                    if numeric_strategy == "mean":
                        df[col].fillna(df[col].mean(), inplace=True)
                    elif numeric_strategy == "median":
                        df[col].fillna(df[col].median(), inplace=True)
                    elif numeric_strategy == "mode":
                        df[col].fillna(
                            df[col].mode()[0] if len(df[col].mode()) > 0 else 0,
                            inplace=True,
                        )
                    elif numeric_strategy == "zero":
                        df[col].fillna(0, inplace=True)
                    elif numeric_strategy == "forward_fill":
                        df[col].fillna(method="ffill", inplace=True)
                    elif numeric_strategy == "backward_fill":
                        df[col].fillna(method="bfill", inplace=True)
                    else:
                        raise ValueError(
                            f"Invalid numeric_strategy: {numeric_strategy}"
                        )

            # Handle categorical columns
            categorical_cols = [
                col for col in cols_to_process if col in self.categorical_columns
            ]
            for col in categorical_cols:
                if df[col].isnull().sum() > 0:
                    if categorical_strategy == "mode":
                        df[col].fillna(
                            df[col].mode()[0] if len(df[col].mode()) > 0 else "Unknown",
                            inplace=True,
                        )
                    elif categorical_strategy == "constant":
                        df[col].fillna(
                            fill_value if fill_value else "Unknown", inplace=True
                        )
                    elif categorical_strategy == "forward_fill":
                        df[col].fillna(method="ffill", inplace=True)
                    elif categorical_strategy == "backward_fill":
                        df[col].fillna(method="bfill", inplace=True)
                    else:
                        raise ValueError(
                            f"Invalid categorical_strategy: {categorical_strategy}"
                        )

            logger.info(f"Handled missing values using auto strategy")

        else:
            raise ValueError(f"Invalid strategy: {strategy}")

        # Count missing values after
        missing_after = df[cols_to_process].isnull().sum().sum()
        logger.info(f"Missing values: {missing_before} -> {missing_after}")

        self.data = df
        return df

    def handle_outliers(
        self,
        method: str = "iqr",
        action: str = "cap",
        columns: Optional[List[str]] = None,
        threshold: float = 1.5,
        z_threshold: float = 3.0,
    ) -> pd.DataFrame:
        """
        Detect and handle outliers in numeric columns.

        Args:
            method: Detection method ('iqr', 'zscore', 'modified_zscore', 'percentile').
            action: Action to take ('cap', 'remove', 'log_transform').
                - 'cap': Cap outliers at boundaries
                - 'remove': Remove outlier rows
                - 'log_transform': Apply log transformation
            columns: Specific columns to process. If None, process all numeric columns.
            threshold: Multiplier for IQR method (default: 1.5).
            z_threshold: Threshold for z-score methods (default: 3.0).

        Returns:
            DataFrame with outliers handled.

        Raises:
            ValueError: If data is not set or invalid parameters provided.
        """
        if self.data is None:
            raise ValueError("No data set. Use set_data() first.")

        df = self.data.copy()
        cols_to_process = columns if columns else self.numeric_columns

        if not cols_to_process:
            logger.warning("No numeric columns to process for outliers")
            return df

        outliers_count = 0

        for col in cols_to_process:
            if col not in df.columns:
                logger.warning(f"Column {col} not found in data")
                continue

            if col not in self.numeric_columns:
                logger.warning(f"Column {col} is not numeric, skipping")
                continue

            # Detect outliers based on method
            if method == "iqr":
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)

            elif method == "zscore":
                mean = df[col].mean()
                std = df[col].std()
                z_scores = np.abs((df[col] - mean) / std)
                outlier_mask = z_scores > z_threshold
                lower_bound = mean - z_threshold * std
                upper_bound = mean + z_threshold * std

            elif method == "modified_zscore":
                median = df[col].median()
                mad = np.median(np.abs(df[col] - median))
                modified_z_scores = 0.6745 * (df[col] - median) / mad if mad != 0 else 0
                outlier_mask = np.abs(modified_z_scores) > z_threshold
                lower_bound = median - z_threshold * mad / 0.6745
                upper_bound = median + z_threshold * mad / 0.6745

            elif method == "percentile":
                lower_bound = df[col].quantile(0.01)
                upper_bound = df[col].quantile(0.99)
                outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)

            else:
                raise ValueError(f"Invalid method: {method}")

            outliers_in_col = outlier_mask.sum()
            outliers_count += outliers_in_col

            if outliers_in_col > 0:
                # Handle outliers based on action
                if action == "cap":
                    df.loc[df[col] < lower_bound, col] = lower_bound
                    df.loc[df[col] > upper_bound, col] = upper_bound
                    logger.info(f"Capped {outliers_in_col} outliers in {col}")

                elif action == "remove":
                    df = df[~outlier_mask]
                    logger.info(f"Removed {outliers_in_col} outlier rows for {col}")

                elif action == "log_transform":
                    # Add 1 to handle zeros
                    min_val = df[col].min()
                    if min_val <= 0:
                        df[col] = df[col] - min_val + 1
                    df[col] = np.log(df[col])
                    logger.info(f"Applied log transform to {col}")

                else:
                    raise ValueError(f"Invalid action: {action}")

        logger.info(f"Total outliers detected: {outliers_count}")
        self.data = df
        return df

    def create_derived_features(
        self, features: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Create derived features specific to insurance domain.

        Args:
            features: List of features to create. If None, create all available features.
                Available features:
                - 'VehicleAge': Age of vehicle from RegistrationYear
                - 'HasClaim': Binary indicator if TotalClaims > 0
                - 'LossRatio': TotalClaims / TotalPremium
                - 'PremiumPerSumInsured': TotalPremium / SumInsured
                - 'ClaimFrequency': Proportion of policies with claims
                - 'AvgClaimAmount': Average claim amount per policy
                - 'IsHighValue': Binary indicator for high-value vehicles
                - 'TransactionYear': Year from TransactionMonth
                - 'TransactionQuarter': Quarter from TransactionMonth
                - 'VehicleAgeGroup': Categorical age groups

        Returns:
            DataFrame with derived features added.

        Raises:
            ValueError: If data is not set.
        """
        if self.data is None:
            raise ValueError("No data set. Use set_data() first.")

        df = self.data.copy()
        available_features = {
            "VehicleAge",
            "HasClaim",
            "LossRatio",
            "PremiumPerSumInsured",
            "ClaimFrequency",
            "AvgClaimAmount",
            "IsHighValue",
            "TransactionYear",
            "TransactionQuarter",
            "VehicleAgeGroup",
        }

        features_to_create = features if features else list(available_features)
        created_count = 0

        # Current year for age calculation
        current_year = pd.Timestamp.now().year

        # VehicleAge
        if "VehicleAge" in features_to_create and "RegistrationYear" in df.columns:
            df["VehicleAge"] = current_year - df["RegistrationYear"]
            df["VehicleAge"] = df["VehicleAge"].clip(lower=0)
            created_count += 1
            logger.info("Created VehicleAge feature")

        # HasClaim
        if "HasClaim" in features_to_create and "TotalClaims" in df.columns:
            df["HasClaim"] = (df["TotalClaims"] > 0).astype(int)
            created_count += 1
            logger.info("Created HasClaim feature")

        # LossRatio
        if "LossRatio" in features_to_create and all(
            col in df.columns for col in ["TotalClaims", "TotalPremium"]
        ):
            df["LossRatio"] = df["TotalClaims"] / df["TotalPremium"].replace(0, np.nan)
            df["LossRatio"] = df["LossRatio"].replace([np.inf, -np.inf], np.nan)
            created_count += 1
            logger.info("Created LossRatio feature")

        # PremiumPerSumInsured
        if "PremiumPerSumInsured" in features_to_create and all(
            col in df.columns for col in ["TotalPremium", "SumInsured"]
        ):
            df["PremiumPerSumInsured"] = df["TotalPremium"] / df["SumInsured"].replace(
                0, np.nan
            )
            df["PremiumPerSumInsured"] = df["PremiumPerSumInsured"].replace(
                [np.inf, -np.inf], np.nan
            )
            created_count += 1
            logger.info("Created PremiumPerSumInsured feature")

        # ClaimFrequency (requires grouping - calculate per policy)
        if "ClaimFrequency" in features_to_create and "HasClaim" in df.columns:
            if "PolicyID" in df.columns:
                claim_freq = df.groupby("PolicyID")["HasClaim"].mean().reset_index()
                claim_freq.columns = ["PolicyID", "ClaimFrequency"]
                df = df.merge(claim_freq, on="PolicyID", how="left")
                created_count += 1
                logger.info("Created ClaimFrequency feature")

        # AvgClaimAmount
        if "AvgClaimAmount" in features_to_create and "TotalClaims" in df.columns:
            if "PolicyID" in df.columns:
                avg_claim = df.groupby("PolicyID")["TotalClaims"].mean().reset_index()
                avg_claim.columns = ["PolicyID", "AvgClaimAmount"]
                df = df.merge(avg_claim, on="PolicyID", how="left")
                created_count += 1
                logger.info("Created AvgClaimAmount feature")

        # IsHighValue
        if "IsHighValue" in features_to_create and "SumInsured" in df.columns:
            # Define high value as top 25%
            threshold = df["SumInsured"].quantile(0.75)
            df["IsHighValue"] = (df["SumInsured"] > threshold).astype(int)
            created_count += 1
            logger.info("Created IsHighValue feature")

        # TransactionYear
        if "TransactionYear" in features_to_create and "TransactionMonth" in df.columns:
            df["TransactionYear"] = pd.to_datetime(df["TransactionMonth"]).dt.year
            created_count += 1
            logger.info("Created TransactionYear feature")

        # TransactionQuarter
        if (
            "TransactionQuarter" in features_to_create
            and "TransactionMonth" in df.columns
        ):
            df["TransactionQuarter"] = pd.to_datetime(df["TransactionMonth"]).dt.quarter
            created_count += 1
            logger.info("Created TransactionQuarter feature")

        # VehicleAgeGroup
        if "VehicleAgeGroup" in features_to_create and "VehicleAge" in df.columns:
            df["VehicleAgeGroup"] = pd.cut(
                df["VehicleAge"],
                bins=[-1, 3, 7, 12, 100],
                labels=["New (0-3)", "Mid (4-7)", "Old (8-12)", "Very Old (13+)"],
            )
            created_count += 1
            logger.info("Created VehicleAgeGroup feature")

        logger.info(f"Created {created_count} derived features")
        self.data = df
        return df

    def remove_duplicates(
        self, subset: Optional[List[str]] = None, keep: str = "first"
    ) -> pd.DataFrame:
        """
        Remove duplicate rows from the dataset.

        Args:
            subset: Columns to consider for identifying duplicates. If None, use all columns.
            keep: Which duplicates to keep ('first', 'last', False for remove all).

        Returns:
            DataFrame with duplicates removed.

        Raises:
            ValueError: If data is not set.
        """
        if self.data is None:
            raise ValueError("No data set. Use set_data() first.")

        df = self.data.copy()
        rows_before = len(df)

        df = df.drop_duplicates(subset=subset, keep=keep)
        rows_after = len(df)
        duplicates_removed = rows_before - rows_after

        logger.info(f"Removed {duplicates_removed} duplicate rows")
        self.data = df
        return df

    def filter_data(self, conditions: Dict[str, Any]) -> pd.DataFrame:
        """
        Filter data based on specified conditions.

        Args:
            conditions: Dictionary of column names and filter conditions.
                Examples:
                - {'TotalPremium': ('>', 0)} - Greater than 0
                - {'Province': ('in', ['Gauteng', 'Western Cape'])} - In list
                - {'VehicleAge': ('between', (0, 10))} - Between values

        Returns:
            Filtered DataFrame.

        Raises:
            ValueError: If data is not set or invalid conditions.
        """
        if self.data is None:
            raise ValueError("No data set. Use set_data() first.")

        df = self.data.copy()
        rows_before = len(df)

        for col, condition in conditions.items():
            if col not in df.columns:
                logger.warning(f"Column {col} not found, skipping filter")
                continue

            operator, value = condition

            if operator == ">":
                df = df[df[col] > value]
            elif operator == ">=":
                df = df[df[col] >= value]
            elif operator == "<":
                df = df[df[col] < value]
            elif operator == "<=":
                df = df[df[col] <= value]
            elif operator == "==":
                df = df[df[col] == value]
            elif operator == "!=":
                df = df[df[col] != value]
            elif operator == "in":
                df = df[df[col].isin(value)]
            elif operator == "not_in":
                df = df[~df[col].isin(value)]
            elif operator == "between":
                df = df[df[col].between(value[0], value[1])]
            elif operator == "is_null":
                df = df[df[col].isnull()]
            elif operator == "not_null":
                df = df[df[col].notnull()]
            else:
                raise ValueError(f"Invalid operator: {operator}")

        rows_after = len(df)
        rows_filtered = rows_before - rows_after

        logger.info(f"Filtered {rows_filtered} rows based on conditions")
        self.data = df
        return df

    def save_processed_data(
        self, filepath: Union[str, Path], format: str = "csv", **kwargs
    ) -> None:
        """
        Save processed data to file.

        Args:
            filepath: Path where to save the file.
            format: File format ('csv', 'parquet', 'excel', 'txt').
            **kwargs: Additional arguments to pass to pandas save methods.

        Raises:
            ValueError: If data is not set or invalid format.
        """
        if self.data is None:
            raise ValueError("No data set. Use set_data() first.")

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        if format == "csv":
            self.data.to_csv(filepath, index=False, **kwargs)
        elif format == "parquet":
            self.data.to_parquet(filepath, index=False, **kwargs)
        elif format == "excel":
            self.data.to_excel(filepath, index=False, **kwargs)
        elif format == "txt":
            # Default to pipe delimiter for .txt files
            sep = kwargs.pop("sep", "|")
            self.data.to_csv(filepath, sep=sep, index=False, **kwargs)
        else:
            raise ValueError(
                f"Invalid format: {format}. Supported: csv, parquet, excel, txt"
            )

        logger.info(f"Saved processed data to {filepath} ({format} format)")

    def get_processing_summary(self) -> Dict[str, Any]:
        """
        Get summary of current data state after processing.

        Returns:
            Dictionary containing data summary statistics.

        Raises:
            ValueError: If data is not set.
        """
        if self.data is None:
            raise ValueError("No data set. Use set_data() first.")

        summary = {
            "shape": self.data.shape,
            "columns": len(self.data.columns),
            "numeric_columns": len(self.numeric_columns),
            "categorical_columns": len(self.categorical_columns),
            "total_missing": self.data.isnull().sum().sum(),
            "missing_percentage": (
                self.data.isnull().sum().sum()
                / (self.data.shape[0] * self.data.shape[1])
            )
            * 100,
            "duplicate_rows": self.data.duplicated().sum(),
            "memory_usage_mb": self.data.memory_usage(deep=True).sum() / (1024 * 1024),
        }

        return summary

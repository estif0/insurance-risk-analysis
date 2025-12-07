"""
Data Loader Module for Insurance Risk Analysis.

This module provides functionality to load insurance data from .txt files
with various delimiters (pipe, tab, comma) and validate the data schema.
"""

import os
from typing import Dict, List, Optional, Union, Tuple
import pandas as pd
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """
    A class for loading and validating insurance data from text files.
    
    This class handles loading of insurance policy data from .txt files with
    various delimiters, validates the schema, and provides data information.
    
    Attributes:
        data (pd.DataFrame): The loaded insurance data.
        file_path (str): Path to the loaded data file.
        delimiter (str): Delimiter used in the file.
        expected_columns (List[str]): Expected column names for validation.
    
    Example:
        >>> loader = DataLoader()
        >>> df = loader.load_data('data/clean/MachineLearningRating_v3.txt', delimiter='|')
        >>> info = loader.get_data_info()
        >>> print(info['shape'])
    """
    
    def __init__(self):
        """Initialize the DataLoader with default attributes."""
        self.data: Optional[pd.DataFrame] = None
        self.file_path: Optional[str] = None
        self.delimiter: Optional[str] = None
        
        # Expected columns based on insurance data structure
        self.expected_columns = [
            # Policy information
            'UnderwrittenCoverID', 'PolicyID', 'TransactionMonth',
            
            # Client demographics
            'IsVATRegistered', 'Citizenship', 'LegalType', 'Title', 
            'Language', 'Bank', 'AccountType', 'MaritalStatus', 'Gender',
            
            # Location data
            'Country', 'Province', 'PostalCode', 'MainCrestaZone', 'SubCrestaZone',
            
            # Vehicle details
            'ItemType', 'mmcode', 'VehicleType', 'RegistrationYear', 'make', 
            'Model', 'Cylinders', 'cubiccapacity', 'kilowatts', 'bodytype', 
            'NumberOfDoors', 'VehicleIntroDate', 'CustomValueEstimate',
            'AlarmImmobiliser', 'TrackingDevice', 'CapitalOutstanding',
            'NewVehicle', 'WrittenOff', 'Rebuilt', 'Converted', 'CrossBorder',
            'NumberOfVehiclesInFleet',
            
            # Plan details
            'SumInsured', 'TermFrequency', 'CalculatedPremiumPerTerm',
            'ExcessSelected', 'CoverCategory', 'CoverType', 'CoverGroup',
            'Section', 'Product', 'StatutoryClass', 'StatutoryRiskType',
            
            # Financial metrics
            'TotalPremium', 'TotalClaims'
        ]
    
    def load_data(
        self, 
        file_path: str, 
        delimiter: str = '|',
        encoding: str = 'utf-8',
        low_memory: bool = False
    ) -> pd.DataFrame:
        """
        Load insurance data from a text file.
        
        Args:
            file_path: Path to the .txt data file.
            delimiter: Delimiter used in the file. Common values: '|', '\t', ','.
            encoding: File encoding. Default is 'utf-8'.
            low_memory: If False, read entire file at once for better type inference.
        
        Returns:
            pd.DataFrame: Loaded insurance data.
        
        Raises:
            FileNotFoundError: If the specified file does not exist.
            ValueError: If the file is empty or cannot be parsed.
            Exception: For other file reading errors.
        
        Example:
            >>> loader = DataLoader()
            >>> df = loader.load_data('data/clean/insurance_data.txt', delimiter='|')
            >>> print(f"Loaded {len(df)} records")
        """
        # Validate file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"Data file not found at: {file_path}\n"
                f"Please ensure the file exists and the path is correct."
            )
        
        # Validate file is not empty
        if os.path.getsize(file_path) == 0:
            raise ValueError(f"Data file is empty: {file_path}")
        
        logger.info(f"Loading data from: {file_path}")
        logger.info(f"Using delimiter: '{delimiter}'")
        
        try:
            # Load data with specified delimiter
            self.data = pd.read_csv(
                file_path,
                delimiter=delimiter,
                encoding=encoding,
                low_memory=low_memory
            )
            
            self.file_path = file_path
            self.delimiter = delimiter
            
            # Log success
            logger.info(f"Successfully loaded {len(self.data)} records with {len(self.data.columns)} columns")
            
            # Basic validation
            if self.data.empty:
                raise ValueError("Loaded data is empty")
            
            return self.data
            
        except pd.errors.EmptyDataError:
            raise ValueError(f"No data found in file: {file_path}")
        
        except pd.errors.ParserError as e:
            raise ValueError(
                f"Error parsing file with delimiter '{delimiter}': {str(e)}\n"
                f"Try a different delimiter (e.g., '\\t' for tab, ',' for comma)"
            )
        
        except Exception as e:
            raise Exception(f"Unexpected error loading data: {str(e)}")
    
    def validate_schema(self, strict: bool = False) -> Dict[str, Union[bool, List[str]]]:
        """
        Validate the schema of loaded data against expected columns.
        
        Args:
            strict: If True, require all expected columns to be present.
                   If False, only warn about missing columns.
        
        Returns:
            Dict containing:
                - 'is_valid': Boolean indicating if schema is valid
                - 'missing_columns': List of expected columns not found
                - 'extra_columns': List of columns not in expected schema
                - 'total_columns': Total number of columns in data
        
        Raises:
            ValueError: If no data has been loaded yet.
            ValueError: If strict=True and required columns are missing.
        
        Example:
            >>> loader = DataLoader()
            >>> loader.load_data('data/clean/insurance_data.txt')
            >>> validation = loader.validate_schema()
            >>> if not validation['is_valid']:
            ...     print(f"Missing: {validation['missing_columns']}")
        """
        if self.data is None:
            raise ValueError("No data loaded. Please call load_data() first.")
        
        # Get actual columns (case-insensitive comparison)
        actual_columns = set(self.data.columns)
        expected_columns_set = set(self.expected_columns)
        
        # Find missing and extra columns
        missing_columns = list(expected_columns_set - actual_columns)
        extra_columns = list(actual_columns - expected_columns_set)
        
        # Determine if valid
        is_valid = len(missing_columns) == 0
        
        # Create validation report
        validation_result = {
            'is_valid': is_valid,
            'missing_columns': missing_columns,
            'extra_columns': extra_columns,
            'total_columns': len(actual_columns),
            'expected_columns': len(expected_columns_set),
            'match_percentage': (len(actual_columns & expected_columns_set) / len(expected_columns_set)) * 100
        }
        
        # Log results
        if is_valid:
            logger.info("Schema validation passed: All expected columns present")
        else:
            logger.warning(f"Schema validation: {len(missing_columns)} columns missing")
            logger.warning(f"Missing columns: {missing_columns[:5]}...")  # Show first 5
        
        if extra_columns:
            logger.info(f"Found {len(extra_columns)} additional columns not in expected schema")
        
        # Strict mode: raise error if columns are missing
        if strict and not is_valid:
            raise ValueError(
                f"Schema validation failed in strict mode.\n"
                f"Missing columns: {missing_columns}"
            )
        
        return validation_result
    
    def get_data_info(self) -> Dict[str, Union[Tuple, List, int, float]]:
        """
        Get comprehensive information about the loaded data.
        
        Returns:
            Dict containing:
                - 'shape': Tuple of (rows, columns)
                - 'columns': List of column names
                - 'dtypes': Dictionary of column data types
                - 'memory_usage_mb': Memory usage in megabytes
                - 'missing_values': Total count of missing values
                - 'missing_percentage': Percentage of missing values
                - 'duplicate_rows': Count of duplicate rows
        
        Raises:
            ValueError: If no data has been loaded yet.
        
        Example:
            >>> loader = DataLoader()
            >>> loader.load_data('data/clean/insurance_data.txt')
            >>> info = loader.get_data_info()
            >>> print(f"Shape: {info['shape']}")
            >>> print(f"Missing: {info['missing_percentage']:.2f}%")
        """
        if self.data is None:
            raise ValueError("No data loaded. Please call load_data() first.")
        
        # Calculate memory usage
        memory_usage = self.data.memory_usage(deep=True).sum()
        memory_mb = memory_usage / (1024 * 1024)
        
        # Calculate missing values
        total_cells = self.data.shape[0] * self.data.shape[1]
        missing_values = self.data.isnull().sum().sum()
        missing_percentage = (missing_values / total_cells * 100) if total_cells > 0 else 0
        
        # Get data types
        dtypes_dict = {col: str(dtype) for col, dtype in self.data.dtypes.items()}
        
        info = {
            'shape': self.data.shape,
            'columns': list(self.data.columns),
            'dtypes': dtypes_dict,
            'memory_usage_mb': round(memory_mb, 2),
            'missing_values': int(missing_values),
            'missing_percentage': round(missing_percentage, 2),
            'duplicate_rows': int(self.data.duplicated().sum()),
            'file_path': self.file_path,
            'delimiter': self.delimiter
        }
        
        return info
    
    def get_column_types(self) -> Dict[str, List[str]]:
        """
        Categorize columns by their data types.
        
        Returns:
            Dict with keys: 'numerical', 'categorical', 'datetime', 'other'
            Each containing a list of column names.
        
        Raises:
            ValueError: If no data has been loaded yet.
        
        Example:
            >>> loader = DataLoader()
            >>> loader.load_data('data/clean/insurance_data.txt')
            >>> types = loader.get_column_types()
            >>> print(f"Numerical columns: {len(types['numerical'])}")
        """
        if self.data is None:
            raise ValueError("No data loaded. Please call load_data() first.")
        
        column_types = {
            'numerical': [],
            'categorical': [],
            'datetime': [],
            'other': []
        }
        
        for col in self.data.columns:
            dtype = self.data[col].dtype
            
            if pd.api.types.is_numeric_dtype(dtype):
                column_types['numerical'].append(col)
            elif pd.api.types.is_datetime64_any_dtype(dtype):
                column_types['datetime'].append(col)
            elif pd.api.types.is_object_dtype(dtype) or pd.api.types.is_categorical_dtype(dtype):
                column_types['categorical'].append(col)
            else:
                column_types['other'].append(col)
        
        return column_types
    
    def detect_delimiter(self, file_path: str, n_lines: int = 5) -> str:
        """
        Automatically detect the delimiter used in a text file.
        
        Args:
            file_path: Path to the text file.
            n_lines: Number of lines to sample for detection.
        
        Returns:
            str: Most likely delimiter ('|', '\t', ',', ';', or ' ').
        
        Raises:
            FileNotFoundError: If file doesn't exist.
        
        Example:
            >>> loader = DataLoader()
            >>> delimiter = loader.detect_delimiter('data/raw/data.txt')
            >>> df = loader.load_data('data/raw/data.txt', delimiter=delimiter)
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Common delimiters to test
        delimiters = ['|', '\t', ',', ';', ' ']
        delimiter_counts = {d: 0 for d in delimiters}
        
        # Read first n_lines
        with open(file_path, 'r', encoding='utf-8') as f:
            for _ in range(n_lines):
                line = f.readline()
                if not line:
                    break
                
                for delimiter in delimiters:
                    delimiter_counts[delimiter] += line.count(delimiter)
        
        # Return delimiter with highest count
        detected_delimiter = max(delimiter_counts, key=delimiter_counts.get)
        logger.info(f"Detected delimiter: '{detected_delimiter}'")
        
        return detected_delimiter
    
    def preview_data(self, n_rows: int = 5) -> pd.DataFrame:
        """
        Preview the first n rows of loaded data.
        
        Args:
            n_rows: Number of rows to display.
        
        Returns:
            pd.DataFrame: First n rows of data.
        
        Raises:
            ValueError: If no data has been loaded yet.
        
        Example:
            >>> loader = DataLoader()
            >>> loader.load_data('data/clean/insurance_data.txt')
            >>> preview = loader.preview_data(10)
            >>> print(preview)
        """
        if self.data is None:
            raise ValueError("No data loaded. Please call load_data() first.")
        
        return self.data.head(n_rows)
    
    def get_sample(self, n: int = 1000, random_state: int = 42) -> pd.DataFrame:
        """
        Get a random sample of the loaded data.
        
        Args:
            n: Number of rows to sample.
            random_state: Random seed for reproducibility.
        
        Returns:
            pd.DataFrame: Random sample of data.
        
        Raises:
            ValueError: If no data has been loaded yet.
        
        Example:
            >>> loader = DataLoader()
            >>> loader.load_data('data/clean/insurance_data.txt')
            >>> sample = loader.get_sample(500)
            >>> print(f"Sample size: {len(sample)}")
        """
        if self.data is None:
            raise ValueError("No data loaded. Please call load_data() first.")
        
        # If n is greater than data size, return all data
        sample_size = min(n, len(self.data))
        
        return self.data.sample(n=sample_size, random_state=random_state)

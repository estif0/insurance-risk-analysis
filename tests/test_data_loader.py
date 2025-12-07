"""
Unit tests for DataLoader module.

Tests cover:
- File loading with different delimiters
- Schema validation
- Data info retrieval
- Error handling
- Edge cases
"""

import pytest
import pandas as pd
import os
import tempfile
from core.data_loader import DataLoader


@pytest.fixture
def sample_data_pipe():
    """Create a temporary pipe-delimited test file."""
    data = """PolicyID|Province|TotalPremium|TotalClaims|Gender
1|Gauteng|1000.0|500.0|Male
2|Western Cape|1500.0|0.0|Female
3|KwaZulu-Natal|2000.0|1000.0|Male
4|Gauteng|1200.0|300.0|Female
5|Western Cape|1800.0|0.0|Male"""
    
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        f.write(data)
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    if os.path.exists(temp_path):
        os.remove(temp_path)


@pytest.fixture
def sample_data_tab():
    """Create a temporary tab-delimited test file."""
    data = """PolicyID\tProvince\tTotalPremium\tTotalClaims\tGender
1\tGauteng\t1000.0\t500.0\tMale
2\tWestern Cape\t1500.0\t0.0\tFemale
3\tKwaZulu-Natal\t2000.0\t1000.0\tMale"""
    
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        f.write(data)
        temp_path = f.name
    
    yield temp_path
    
    if os.path.exists(temp_path):
        os.remove(temp_path)


@pytest.fixture
def empty_file():
    """Create an empty temporary file."""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.txt') as f:
        temp_path = f.name
    
    yield temp_path
    
    if os.path.exists(temp_path):
        os.remove(temp_path)


class TestDataLoader:
    """Test suite for DataLoader class."""
    
    def test_initialization(self):
        """Test DataLoader initialization."""
        loader = DataLoader()
        assert loader.data is None
        assert loader.file_path is None
        assert loader.delimiter is None
        assert len(loader.expected_columns) > 0
    
    def test_load_data_pipe_delimiter(self, sample_data_pipe):
        """Test loading data with pipe delimiter."""
        loader = DataLoader()
        df = loader.load_data(sample_data_pipe, delimiter='|')
        
        assert df is not None
        assert len(df) == 5
        assert len(df.columns) == 5
        assert 'PolicyID' in df.columns
        assert 'TotalPremium' in df.columns
    
    def test_load_data_tab_delimiter(self, sample_data_tab):
        """Test loading data with tab delimiter."""
        loader = DataLoader()
        df = loader.load_data(sample_data_tab, delimiter='\t')
        
        assert df is not None
        assert len(df) == 3
        assert 'Province' in df.columns
    
    def test_load_data_file_not_found(self):
        """Test error handling for missing file."""
        loader = DataLoader()
        
        with pytest.raises(FileNotFoundError):
            loader.load_data('nonexistent_file.txt')
    
    def test_load_data_empty_file(self, empty_file):
        """Test error handling for empty file."""
        loader = DataLoader()
        
        with pytest.raises(ValueError, match="empty"):
            loader.load_data(empty_file)
    
    def test_get_data_info(self, sample_data_pipe):
        """Test get_data_info method."""
        loader = DataLoader()
        loader.load_data(sample_data_pipe, delimiter='|')
        
        info = loader.get_data_info()
        
        assert 'shape' in info
        assert 'columns' in info
        assert 'dtypes' in info
        assert 'memory_usage_mb' in info
        assert 'missing_values' in info
        assert info['shape'] == (5, 5)
        assert len(info['columns']) == 5
    
    def test_get_data_info_no_data(self):
        """Test get_data_info raises error when no data loaded."""
        loader = DataLoader()
        
        with pytest.raises(ValueError, match="No data loaded"):
            loader.get_data_info()
    
    def test_validate_schema(self, sample_data_pipe):
        """Test schema validation."""
        loader = DataLoader()
        loader.load_data(sample_data_pipe, delimiter='|')
        
        validation = loader.validate_schema(strict=False)
        
        assert 'is_valid' in validation
        assert 'missing_columns' in validation
        assert 'extra_columns' in validation
        assert isinstance(validation['is_valid'], bool)
    
    def test_validate_schema_no_data(self):
        """Test validate_schema raises error when no data loaded."""
        loader = DataLoader()
        
        with pytest.raises(ValueError, match="No data loaded"):
            loader.validate_schema()
    
    def test_get_column_types(self, sample_data_pipe):
        """Test column type categorization."""
        loader = DataLoader()
        loader.load_data(sample_data_pipe, delimiter='|')
        
        column_types = loader.get_column_types()
        
        assert 'numerical' in column_types
        assert 'categorical' in column_types
        assert 'datetime' in column_types
        assert 'other' in column_types
        
        # TotalPremium and TotalClaims should be numerical
        assert 'TotalPremium' in column_types['numerical']
        assert 'TotalClaims' in column_types['numerical']
    
    def test_preview_data(self, sample_data_pipe):
        """Test data preview functionality."""
        loader = DataLoader()
        loader.load_data(sample_data_pipe, delimiter='|')
        
        preview = loader.preview_data(n_rows=3)
        
        assert len(preview) == 3
        assert isinstance(preview, pd.DataFrame)
    
    def test_preview_data_no_data(self):
        """Test preview_data raises error when no data loaded."""
        loader = DataLoader()
        
        with pytest.raises(ValueError, match="No data loaded"):
            loader.preview_data()
    
    def test_get_sample(self, sample_data_pipe):
        """Test random sampling."""
        loader = DataLoader()
        loader.load_data(sample_data_pipe, delimiter='|')
        
        sample = loader.get_sample(n=3, random_state=42)
        
        assert len(sample) == 3
        assert isinstance(sample, pd.DataFrame)
    
    def test_get_sample_larger_than_data(self, sample_data_pipe):
        """Test sampling when n > data size."""
        loader = DataLoader()
        loader.load_data(sample_data_pipe, delimiter='|')
        
        sample = loader.get_sample(n=100, random_state=42)
        
        assert len(sample) == 5  # Should return all 5 rows
    
    def test_detect_delimiter(self, sample_data_pipe, sample_data_tab):
        """Test automatic delimiter detection."""
        loader = DataLoader()
        
        # Test pipe delimiter
        detected = loader.detect_delimiter(sample_data_pipe)
        assert detected == '|'
        
        # Test tab delimiter
        detected = loader.detect_delimiter(sample_data_tab)
        assert detected == '\t'
    
    def test_detect_delimiter_file_not_found(self):
        """Test delimiter detection with missing file."""
        loader = DataLoader()
        
        with pytest.raises(FileNotFoundError):
            loader.detect_delimiter('nonexistent.txt')
    
    def test_data_persistence(self, sample_data_pipe):
        """Test that data persists in the loader instance."""
        loader = DataLoader()
        loader.load_data(sample_data_pipe, delimiter='|')
        
        # Access data multiple times
        assert loader.data is not None
        assert len(loader.data) == 5
        
        # Get info should work without reloading
        info = loader.get_data_info()
        assert info['shape'] == (5, 5)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

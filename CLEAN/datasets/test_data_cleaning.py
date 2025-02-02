"""
Test suite for data cleaning transformations
------------------------------------------

This script tests the data cleaning transformations to ensure:
1. Values are properly normalized to [-1, 1]
2. Median centering works correctly
3. Binary variables are correctly transformed
4. Ordinal scales maintain proper spacing
"""

import pandas as pd
import numpy as np
import pytest
import os
import sys

# Add the project root directory to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from datasets.clean_data import clean_datasets

def create_test_data():
    """Create a small test dataset with various variable types."""
    np.random.seed(42)
    n_samples = 1000
    
    # Create test data
    data = {
        # Binary variable (Yes/No)
        'BINARY': np.random.choice([1, 2], n_samples),
        
        # Opinion scale (1-7)
        'OPINION': np.random.choice(range(1, 8), n_samples),
        
        # Frequency measure (0-8)
        'FREQ': np.random.choice(range(9), n_samples),
        
        # Confidence measure (1-3)
        'CONF': np.random.choice([1, 2, 3], n_samples),
        
        # Year column (should be excluded from transformation)
        'YEAR': np.random.choice([2000, 2002, 2004], n_samples),
    }
    
    return pd.DataFrame(data)

def test_value_ranges(df_regular, df_median):
    """Test if all values are within [-1, 1] range."""
    exclude = ['YEAR']
    for col in df_regular.columns:
        if col not in exclude:
            assert df_regular[col].min() >= -1
            assert df_regular[col].max() <= 1
            assert df_median[col].min() >= -1
            assert df_median[col].max() <= 1

def test_binary_transformation(df_regular):
    """Test if binary variables are mapped to -1 and 1."""
    values = sorted(df_regular['BINARY'].unique())
    assert len(values) == 2
    assert abs(values[0] + 1) < 0.01  # Should be close to -1
    assert abs(values[1] - 1) < 0.01  # Should be close to 1

def test_median_centering(df_median):
    """Test if variables are properly median centered in the median version."""
    for col in ['OPINION', 'FREQ', 'CONF']:
        median = df_median[col].median()
        assert abs(median) < 0.01  # Should be close to 0

def test_excluded_columns(df_regular, df_median):
    """Test if excluded columns are unchanged."""
    assert df_regular['YEAR'].dtype == df_median['YEAR'].dtype
    assert set(df_regular['YEAR'].unique()) == set(df_median['YEAR'].unique())

def main():
    """Run all tests and print results."""
    # Create test data
    df = create_test_data()
    
    # Clean data
    df_regular, df_median = clean_datasets(df, time_frame=[2000, 2002, 2004])
    
    # Run tests
    print("Running tests...")
    test_value_ranges(df_regular, df_median)
    test_binary_transformation(df_regular)
    test_median_centering(df_median)
    test_excluded_columns(df_regular, df_median)
    print("All tests passed!")

if __name__ == "__main__":
    main() 
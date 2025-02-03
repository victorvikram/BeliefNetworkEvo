"""
Validation script for GSS data cleaning functionality.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add the parent directory to Python path
sys.path.append(str(Path(__file__).parent))

from clean_data_copy import clean_datasets, DataConfig
from import_gss import import_dataset

def test_raw_values(year=None):
    """
    Show total frequency counts for all values in both raw and cleaned datasets.
    
    Args:
        year: Optional year to filter by
    """
    # Get raw data
    df_raw, _ = import_dataset()
    
    if year is not None:
        df_raw = df_raw[df_raw['YEAR'] == year]
        print(f"\nValues for year {year}:")
    
    # Get cleaned data
    df_clean = clean_datasets()
    if year is not None:
        df_clean = df_clean[df_clean['YEAR'] == year]
    
    # Count frequencies across all columns
    print("\nRAW DATA VALUE COUNTS:")
    print("-" * 50)
    raw_counts = pd.Series(df_raw.values.ravel()).value_counts()
    print(raw_counts)
    
    print("\nCLEANED DATA VALUE COUNTS:")
    print("-" * 50)
    clean_counts = pd.Series(df_clean.values.ravel()).value_counts()
    print(clean_counts)


if __name__ == "__main__":
    test_raw_values(2000)
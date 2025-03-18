"""
This module handles loading and filtering of General Social Survey (GSS) data from pickle files.
It provides functionality to load pre-cleaned GSS data and filter it by year range.

Example:
    >>> df, metadata = load_and_filter_data('gss_cleaned.pkl', 2000, 2020)
    >>> print(df.shape)  # View dimensions of filtered dataset
    >>> print(metadata['description'])  # Access dataset metadata
"""

import pandas as pd

def load_and_filter_data(file_path: str, start_year: int, end_year: int) -> tuple:
    """
    Load and filter GSS data for a specific time period.
    
    Args:
        file_path: Path to the pickle file containing GSS data
        start_year: Start year for filtering
        end_year: End year for filtering
        
    Returns:
        tuple: (filtered_dataframe, metadata)
    """
    # Load the cleaned dataset
    df_cleaned, meta = pd.read_pickle(file_path)
    
    # Filter the data between the specified time period
    df_filtered = df_cleaned[
        (df_cleaned['YEAR'] >= start_year) & 
        (df_cleaned['YEAR'] <= end_year)
    ]
    
    return df_filtered, meta 
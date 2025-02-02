"""
This module analyzes and visualizes the overlap of non-null values between variables in a dataset.
It's particularly useful for understanding data completeness and identifying potential biases
in missing data patterns across variables.

The module calculates the percentage of rows where pairs of variables both have valid (non-null)
values and visualizes these relationships in a heatmap.

Example:
    >>> df = pd.DataFrame({'A': [1,2,None,4], 'B': [1,None,3,4]})
    >>> overlap_mat = calculate_overlap_matrix(df)
    >>> plot_overlap_matrix(overlap_mat)  # Shows 75% overlap for A-B
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_overlap_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the percentage of overlapping non-null values between variables.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame: Matrix of overlap percentages
    """
    total_rows = len(df)
    overlap_matrix = pd.DataFrame(index=df.columns, columns=df.columns)
    
    # Only iterate through upper triangle
    for i, col1 in enumerate(df.columns):
        for j, col2 in enumerate(df.columns[i:], i):
            overlap = df[[col1, col2]].dropna().shape[0]
            percentage = (overlap / total_rows) * 100
            overlap_matrix.loc[col1, col2] = percentage
            if col1 != col2:  # Don't copy diagonal values
                overlap_matrix.loc[col2, col1] = percentage
    
    return overlap_matrix.astype(float)

def plot_overlap_matrix(overlap_matrix: pd.DataFrame, show: bool = True) -> None:
    """
    Create a heatmap visualization of the overlap matrix.
    
    Args:
        overlap_matrix: Matrix of overlap percentages
        show: Whether to display the plot immediately
    """
    plt.figure(figsize=(12, 10))
    sns.heatmap(overlap_matrix, 
                cmap='YlOrRd',
                xticklabels=True,
                yticklabels=True,
                fmt='.1f')
    plt.title('Percentage of Overlapping Values Between Variables (%)')
    plt.xticks(rotation=45, ha='right', fontsize=6)
    plt.yticks(rotation=0, fontsize=6)
    plt.tight_layout()
    
    if show:
        plt.show() 
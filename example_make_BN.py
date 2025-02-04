# CLEAN/source_code/data_loader.py
# CLEAN/source_code/generators/corr_make_network.py

import pickle
import os

"""
Example script demonstrating how to create a correlation matrix (belief network)
from GSS data using the CLEAN framework.

This script:
1. Loads filtered GSS data
2. Creates correlation matrices using different methods
3. Demonstrates both regular and partial correlations
4. Shows how to apply edge suppression
"""

from CLEAN.source_code.generators.data_loader import load_and_filter_data
from CLEAN.source_code.generators.corr_make_network import (
    calculate_correlation_matrix,
    CorrelationMethod,
    EdgeSuppressionMethod
)
from CLEAN.source_code.visualizers.network_visualizer import generate_html_visualization, create_network_data


import matplotlib.pyplot as plt
import seaborn as sns

def main():
    
    with open(os.path.join('CLEAN', 'datasets', 'cached_data', 'cleaned_data_1.pkl'), 'rb') as f:
            df = pickle.load(f)

    start_year = 2000
    end_year = 2004

    print(df)
    # Filter to time window
    df = df[
        (df['YEAR'] >= start_year) & 
        (df['YEAR'] <= end_year)
    ]
    
    
    # 2. Calculate regular Spearman correlation matrix
    corr_matrix = calculate_correlation_matrix(
        df,
        method=CorrelationMethod.SPEARMAN,
        partial=True,
        edge_suppression=EdgeSuppressionMethod.SQUARE
    )
    

    # Create and save network visualization for the final time period        # Create network data
    nodes, edges = create_network_data(corr_matrix)
    
    generate_html_visualization(nodes, edges, "Spearman Correlations")

    # Generate HTML visualization
    output_file = f'correlation_network_{start_year}-{end_year}.html'
    generate_html_visualization(nodes, edges, output_file)
    print(f"Network visualization saved as {output_file}")


if __name__ == "__main__":
    main()
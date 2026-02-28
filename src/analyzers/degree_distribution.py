import numpy as np
import pandas as pd
import plotly.express as px

def plot_degree_distribution(correlation_matrix, threshold=0, log_scale=False):
    """
    Plot degree distribution of a correlation matrix.
    
    Parameters:
    -----------
    correlation_matrix : pd.DataFrame or np.ndarray
        Correlation matrix to analyze
    threshold : float, optional (default=0)
        Absolute correlation threshold for significant connections
    log_scale : bool, optional (default=False)
        If True, use logarithmic scale for both axes
    
    Returns:
    --------
    plotly.graph_objs._figure.Figure
        Interactive degree distribution plot
    """
    # Ensure input is numpy array
    if isinstance(correlation_matrix, pd.DataFrame):
        corr_array = correlation_matrix.to_numpy()
    else:
        corr_array = np.array(correlation_matrix)
    
    # Ensure diagonal is zero
    np.fill_diagonal(corr_array, 0)

    # Calculate degrees (number of significant connections)
    degrees = np.sum(np.abs(corr_array) > threshold, axis=1)    
    # Calculate degree distribution
    unique, counts = np.unique(degrees, return_counts=True)
    
    # Create DataFrame for Plotly Express
    degree_df = pd.DataFrame({
        'Degree': unique,
        'Frequency': counts
    })
    
    # Create plot
    fig = px.scatter(
        degree_df, 
        x='Degree', 
        y='Frequency',
        title='Correlation Matrix Degree Distribution',
        labels={
            'Degree': 'Degree', 
            'Frequency': 'Count of Nodes'
        }
    )
    
    # Apply log scale if requested
    if log_scale:
        fig.update_layout(
            xaxis_type="log",
            yaxis_type="log"
        )
    
    return fig
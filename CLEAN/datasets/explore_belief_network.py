"""
Belief Network Exploration Module
-------------------------------

This module provides functions for exploring belief networks in the GSS data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional, Tuple
from CLEAN.datasets.create_clean_dataset import load_cleaned_datasets
from datasets.explore_gss import (
    analyze_variable_relationships,
    display_variable_info,
    plot_variable_distribution
)

# Define belief-related variables
BELIEF_VARS = [
    'GOD',      # Belief in God
    'BIBLE',    # View of Bible
    'RELPERSN', # Consider self religious person
    'SPRTPRSN', # Consider self spiritual person
    'POSTLIFE', # Belief in afterlife
    'PRAYER',   # Frequency of prayer
    'RELITEN',  # Strength of religious affiliation
    'ATTEND',   # Religious service attendance
    'PRAY',     # How often pray
    'RELACTIV', # Religious activities
    'RELIG',    # Religious preference
    'FUND',     # Fundamentalist/Liberal
    'RELDENOM', # Specific denomination
]

# Define moral/social attitude variables
ATTITUDE_VARS = [
    'ABORTION', # Abortion views
    'HOMOSEX',  # Views on homosexuality
    'PREMARSX', # Views on premarital sex
    'XMARSEX',  # Views on extramarital sex
    'SUICIDE',  # Views on suicide
    'PORNLAW',  # Views on pornography laws
    'GRASS',    # Marijuana legalization
    'CAPPUN',   # Capital punishment
    'GUNLAW',   # Gun control
    'EUTHANAS', # Euthanasia
]

# Define political variables
POLITICAL_VARS = [
    'POLVIEWS', # Political ideology
    'PARTYID',  # Party identification
    'VOTE',     # Voting behavior
    'TRUST',    # Trust in people
    'CONFED',   # Confidence in education
    'CONFINAN', # Confidence in financial institutions
    'CONPRESS', # Confidence in press
    'CONJUDGE', # Confidence in courts
    'CONLEGIS', # Confidence in congress
    'CONARMY',  # Confidence in military
]

def analyze_belief_network(df: pd.DataFrame, 
                         meta: Dict,
                         target_var: str,
                         var_groups: Optional[List[List[str]]] = None,
                         year_range: Optional[Tuple[int, int]] = None,
                         min_overlap: float = 0.01,
                         plot_distributions: bool = True) -> Dict:
    """
    Analyze relationships between belief variables and other variable groups.
    
    Args:
        df: Input dataframe
        meta: GSS metadata
        target_var: Target variable to analyze
        var_groups: List of variable groups to analyze (default: [BELIEF_VARS, ATTITUDE_VARS, POLITICAL_VARS])
        year_range: Optional tuple of (start_year, end_year)
        min_overlap: Minimum required overlap between variables
        plot_distributions: Whether to plot distributions of variables
    
    Returns:
        Dictionary containing analysis results
    """
    if var_groups is None:
        var_groups = [BELIEF_VARS, ATTITUDE_VARS, POLITICAL_VARS]
    
    results = {
        'target_var': target_var,
        'correlations': [],
        'year_range': year_range,
        'group_summaries': []
    }
    
    # First, analyze the target variable distribution
    if plot_distributions:
        print(f"\nAnalyzing distribution of {target_var}...")
        fig = plot_variable_distribution(df, target_var)
        if fig:
            plt.show()
    
    # Analyze relationships with each group of variables
    for group in var_groups:
        # Filter to variables that exist in the dataset
        existing_vars = [var for var in group if var in df.columns]
        if not existing_vars:
            continue
            
        # Get correlations for this group
        print(f"\nAnalyzing relationships with {len(existing_vars)} variables...")
        correlations = analyze_variable_relationships(
            df=df,
            meta=meta,
            target_var=target_var,
            specific_vars=existing_vars,
            year_range=year_range,
            min_overlap=min_overlap,
            plot_trends=True
        )
        
        # Store results
        results['correlations'].extend(correlations)
        
        # Compute summary statistics for this group
        group_summary = {
            'n_vars': len(existing_vars),
            'n_correlated': len([c for c in correlations if abs(c['correlation']) > 0.1]),
            'mean_correlation': np.mean([abs(c['correlation']) for c in correlations]),
            'strongest_correlation': max([abs(c['correlation']) for c in correlations]),
            'most_correlated_var': None
        }
        
        # Find most strongly correlated variable
        if correlations:
            strongest = max(correlations, key=lambda x: abs(x['correlation']))
            group_summary['most_correlated_var'] = {
                'variable': strongest['variable'],
                'correlation': strongest['correlation'],
                'label': strongest['label']
            }
        
        results['group_summaries'].append(group_summary)
    
    return results

def plot_belief_network(df: pd.DataFrame,
                       vars_to_plot: Optional[List[str]] = None,
                       min_correlation: float = 0.1,
                       year_range: Optional[Tuple[int, int]] = None) -> None:
    """
    Create a network visualization of relationships between belief variables.
    
    Args:
        df: Input dataframe
        vars_to_plot: List of variables to include (default: BELIEF_VARS)
        min_correlation: Minimum absolute correlation to show
        year_range: Optional tuple of (start_year, end_year)
    """
    if vars_to_plot is None:
        vars_to_plot = BELIEF_VARS
    
    # Filter to existing variables
    existing_vars = [var for var in vars_to_plot if var in df.columns]
    
    if len(existing_vars) < 2:
        print("Not enough variables to create network plot")
        return
    
    # Filter by year if specified
    if year_range:
        start_year, end_year = year_range
        df = df[df['YEAR'].between(start_year, end_year)].copy()
    
    # Create correlation matrix
    corr_matrix = pd.DataFrame(index=existing_vars, columns=existing_vars)
    
    for var1 in existing_vars:
        for var2 in existing_vars:
            if var1 != var2:
                # Get complete cases for this pair
                valid_data = df[[var1, var2]].dropna()
                if len(valid_data) > 100:  # Require sufficient observations
                    corr = valid_data[var1].corr(valid_data[var2], method='spearman')
                    corr_matrix.loc[var1, var2] = corr
    
    # Create network plot
    plt.figure(figsize=(12, 8))
    
    # Calculate node positions using spring layout
    import networkx as nx
    G = nx.Graph()
    
    # Add edges with absolute correlation above threshold
    for var1 in existing_vars:
        for var2 in existing_vars:
            if var1 < var2:  # Avoid duplicate edges
                corr = corr_matrix.loc[var1, var2]
                if pd.notna(corr) and abs(corr) >= min_correlation:
                    G.add_edge(var1, var2, weight=abs(corr))
    
    if len(G.edges()) == 0:
        print(f"No correlations above threshold {min_correlation}")
        return
    
    # Calculate layout
    pos = nx.spring_layout(G, k=1, iterations=50)
    
    # Draw edges
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]
    nx.draw_networkx_edges(G, pos, alpha=0.2,
                          width=[w*5 for w in weights],
                          edge_color='gray')
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=1000,
                          node_color='lightblue',
                          alpha=0.6)
    
    # Add labels
    nx.draw_networkx_labels(G, pos, font_size=8)
    
    plt.title("Belief Variable Network\n"
             f"(showing correlations >= {min_correlation})")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def analyze_belief_trends(df: pd.DataFrame,
                         vars_to_analyze: Optional[List[str]] = None,
                         start_year: Optional[int] = None,
                         end_year: Optional[int] = None) -> None:
    """
    Analyze trends in belief variables over time.
    
    Args:
        df: Input dataframe
        vars_to_analyze: List of variables to analyze (default: BELIEF_VARS)
        start_year: Optional start year
        end_year: Optional end year
    """
    if vars_to_analyze is None:
        vars_to_analyze = BELIEF_VARS
    
    # Filter to existing variables
    existing_vars = [var for var in vars_to_analyze if var in df.columns]
    
    if not existing_vars:
        print("No variables to analyze")
        return
    
    # Filter by year if specified
    if start_year and end_year:
        df = df[df['YEAR'].between(start_year, end_year)].copy()
    
    # Create trend plots
    n_vars = len(existing_vars)
    n_cols = min(3, n_vars)
    n_rows = (n_vars + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    if n_vars == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for i, var in enumerate(existing_vars):
        # Calculate mean and confidence interval for each year
        yearly_stats = df.groupby('YEAR')[var].agg(['mean', 'std', 'count']).reset_index()
        yearly_stats['ci'] = 1.96 * yearly_stats['std'] / np.sqrt(yearly_stats['count'])
        
        # Plot trend
        ax = axes[i]
        ax.plot(yearly_stats['YEAR'], yearly_stats['mean'], 'b-', label='Mean')
        ax.fill_between(yearly_stats['YEAR'],
                       yearly_stats['mean'] - yearly_stats['ci'],
                       yearly_stats['mean'] + yearly_stats['ci'],
                       alpha=0.2)
        
        # Add labels
        ax.set_title(var)
        ax.set_xlabel('Year')
        ax.set_ylabel('Mean Value')
        ax.grid(True, alpha=0.3)
        
        # Rotate x-axis labels
        ax.tick_params(axis='x', rotation=45)
    
    # Remove empty subplots
    for i in range(n_vars, len(axes)):
        fig.delaxes(axes[i])
    
    plt.suptitle("Trends in Belief Variables Over Time", y=1.02)
    plt.tight_layout()
    plt.show()

def main():
    """Main function to run belief network analysis."""
    # Load data
    print("Loading GSS data...")
    df_cleaned_1, df_cleaned_2, meta = load_cleaned_datasets()
    
    # Display available variables
    print("\nAvailable variables in the dataset:")
    display_variable_info(meta)
    
    # Analyze belief network using the first cleaned dataset
    print("\nAnalyzing belief network using regular cleaned dataset...")
    results_1 = analyze_belief_network(
        df=df_cleaned_1,
        meta=meta,
        target_var='GOD',  # Example: using belief in God as target
        year_range=(2000, 2020)  # Focus on recent years
    )
    
    # Create network visualization
    print("\nCreating belief network visualization...")
    plot_belief_network(df_cleaned_1, year_range=(2000, 2020))
    
    # Analyze trends
    print("\nAnalyzing trends in belief variables...")
    analyze_belief_trends(df_cleaned_1, start_year=2000, end_year=2020)
    
    # Optionally analyze using the median-centered dataset
    print("\nAnalyzing belief network using median-centered dataset...")
    results_2 = analyze_belief_network(
        df=df_cleaned_2,
        meta=meta,
        target_var='GOD',
        year_range=(2000, 2020)
    )
    
    # Compare results between the two datasets
    print("\nComparing results between regular and median-centered datasets:")
    for group_summary_1, group_summary_2 in zip(results_1['group_summaries'], results_2['group_summaries']):
        if group_summary_1['most_correlated_var'] and group_summary_2['most_correlated_var']:
            print(f"\nStrongest correlation in regular dataset:")
            print(f"Variable: {group_summary_1['most_correlated_var']['variable']}")
            print(f"Correlation: {group_summary_1['most_correlated_var']['correlation']:.3f}")
            print(f"Label: {group_summary_1['most_correlated_var']['label']}")
            
            print(f"\nStrongest correlation in median-centered dataset:")
            print(f"Variable: {group_summary_2['most_correlated_var']['variable']}")
            print(f"Correlation: {group_summary_2['most_correlated_var']['correlation']:.3f}")
            print(f"Label: {group_summary_2['most_correlated_var']['label']}")

if __name__ == "__main__":
    main() 
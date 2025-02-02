"""
GSS Data Exploration Module
-------------------------

This module provides functions for loading and exploring GSS data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
from typing import Tuple, Dict, List, Optional
import random
from datasets.raw_data.import_gss import import_dataset

def load_or_cache_data() -> Tuple[pd.DataFrame, Dict]:
    """
    Load GSS data using the import_dataset function.
    
    Returns:
        Tuple of (dataframe, metadata)
    """
    return import_dataset()

def get_complete_columns(df: pd.DataFrame, min_complete: float = 0.5) -> pd.Series:
    """
    Get columns that have at least min_complete non-missing values.
    
    Args:
        df: Input dataframe
        min_complete: Minimum fraction of non-missing values required
    
    Returns:
        Series with column names and their completion rates
    """
    completion_rates = 1 - df.isna().mean()
    return completion_rates[completion_rates >= min_complete]

def get_numeric_columns(df: pd.DataFrame) -> List[str]:
    """Get list of numeric columns."""
    return df.select_dtypes(include=[np.number]).columns.tolist()

def make_scatter_plot(df: pd.DataFrame, 
                     complete_cols: pd.Series,
                     var1: Optional[str] = None,
                     var2: Optional[str] = None,
                     min_overlap: float = 0.3) -> Tuple[Optional[plt.Figure], dict]:
    """
    Create a scatter plot of two variables.
    
    Args:
        df: Input dataframe
        complete_cols: Series of column completion rates
        var1: First variable (optional)
        var2: Second variable (optional)
        min_overlap: Minimum required overlap between variables
    
    Returns:
        Tuple of (figure, info_dict)
    """
    info = {
        'variables_tried': [],
        'overlap_percentage': None,
        'message': None
    }
    
    # Get numeric columns with sufficient completion
    complete_numeric = [col for col in get_numeric_columns(df) 
                       if col in complete_cols.index]
    
    # If variables not specified, randomly select them
    if var1 is None or var2 is None:
        if len(complete_numeric) < 2:
            info['message'] = "Not enough numeric columns to select from"
            return None, info
        
        # Try up to 5 random pairs to find ones with sufficient overlap
        max_attempts = 5
        for attempt in range(max_attempts):
            selected_vars = random.sample(complete_numeric, 2)
            v1, v2 = selected_vars
            
            # Check overlap
            valid_rows = df[[v1, v2]].dropna()
            overlap_pct = len(valid_rows) / len(df)
            
            info['variables_tried'].extend([v1, v2])
            
            if overlap_pct >= min_overlap:
                var1, var2 = v1, v2
                info['overlap_percentage'] = overlap_pct
                break
            
            if attempt == max_attempts - 1:
                info['message'] = f"Could not find variable pair with >{min_overlap*100}% overlap after {max_attempts} attempts"
                return None, info
    
    # Verify the specified variables exist and are numeric
    for var in [var1, var2]:
        if var not in df.columns:
            info['message'] = f"Variable {var} not found in dataset"
            return None, info
        if not pd.api.types.is_numeric_dtype(df[var]):
            info['message'] = f"Variable {var} is not numeric"
            return None, info
    
    # Create scatter plot
    valid_data = df[[var1, var2]].dropna()
    overlap_pct = len(valid_data) / len(df)
    
    if overlap_pct < min_overlap:
        info['message'] = f"Insufficient overlap ({overlap_pct*100:.1f}%) between {var1} and {var2}"
        return None, info
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(valid_data[var1], valid_data[var2], alpha=0.5)
    ax.set_xlabel(var1)
    ax.set_ylabel(var2)
    ax.set_title(f"{var1} vs {var2}\n(n={len(valid_data):,}, {overlap_pct*100:.1f}% overlap)")
    
    # Add trend line
    z = np.polyfit(valid_data[var1], valid_data[var2], 1)
    p = np.poly1d(z)
    ax.plot(valid_data[var1], p(valid_data[var1]), "r--", alpha=0.8)
    
    info['overlap_percentage'] = overlap_pct
    return fig, info

def plot_variable_distribution(df: pd.DataFrame, var: str, 
                             by_year: bool = True) -> Optional[plt.Figure]:
    """
    Plot distribution of a variable, optionally showing changes over time.
    
    Args:
        df: Input dataframe
        var: Variable to plot
        by_year: Whether to show distribution by year
    
    Returns:
        matplotlib Figure or None if error
    """
    if var not in df.columns:
        print(f"Variable {var} not found in dataset")
        return None
    
    if not pd.api.types.is_numeric_dtype(df[var]):
        print(f"Variable {var} is not numeric")
        return None
    
    if by_year and 'YEAR' in df.columns:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Overall distribution
        sns.histplot(data=df, x=var, ax=ax1)
        ax1.set_title(f"Overall Distribution of {var}")
        
        # Distribution by year
        sns.boxplot(data=df, x='YEAR', y=var, ax=ax2)
        ax2.set_title(f"Distribution of {var} by Year")
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
    else:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(data=df, x=var, ax=ax)
        ax.set_title(f"Distribution of {var}")
    
    return fig

def calculate_correlation(x: pd.Series, y: pd.Series) -> Optional[float]:
    """Calculate Spearman correlation between two series."""
    try:
        return x.corr(y, method='spearman')
    except:
        return None

def standardize_array(x: np.ndarray) -> Optional[np.ndarray]:
    """
    Standardize array to zero mean and unit variance, handling edge cases.
    
    Args:
        x: Input array
        
    Returns:
        Standardized array or None if standardization is not possible
    """
    if len(x) == 0 or np.all(np.isnan(x)):
        return None
        
    # Remove NaN values for calculations
    valid_x = x[~np.isnan(x)]
    if len(valid_x) == 0:
        return None
        
    # Check if all values are the same
    if np.all(valid_x == valid_x[0]):
        return np.zeros_like(x)
        
    mean = np.mean(valid_x)
    std = np.std(valid_x)
    
    # Handle zero standard deviation
    if std == 0:
        return np.zeros_like(x)
        
    # Standardize while preserving NaN values
    result = np.copy(x)
    result[~np.isnan(x)] = (valid_x - mean) / std
    return result

def calculate_partial_correlation(data: pd.DataFrame, x: str, y: str, 
                                control_vars: Optional[List[str]] = None) -> Optional[float]:
    """
    Calculate partial correlation between x and y controlling for other variables.
    
    Args:
        data: Input dataframe
        x: First variable
        y: Second variable
        control_vars: List of variables to control for
    
    Returns:
        Partial correlation coefficient or None if error
    """
    if control_vars is None or len(control_vars) == 0:
        return calculate_correlation(data[x], data[y])
    
    try:
        # Get complete cases
        variables = [x, y] + control_vars
        complete_data = data[variables].dropna()
        
        if len(complete_data) < 30:  # Need sufficient observations
            return None
        
        # Standardize all variables
        X = standardize_array(complete_data[x].values)
        Y = standardize_array(complete_data[y].values)
        Z = complete_data[control_vars].values
        
        if X is None or Y is None:
            return None
        
        # Standardize control variables
        Z_std = np.column_stack([standardize_array(Z[:, i]) for i in range(Z.shape[1])])
        
        # Add constant term
        Z_with_const = np.c_[np.ones(len(Z)), Z_std]
        
        # Check matrix condition number for stability
        if np.linalg.cond(Z_with_const.T @ Z_with_const) > 1e10:
            return None
        
        # Compute residuals using QR decomposition for stability
        Q, R = np.linalg.qr(Z_with_const)
        
        # Project out the effect of control variables
        X_resid = X - Q @ (Q.T @ X)
        Y_resid = Y - Q @ (Q.T @ Y)
        
        # Calculate correlation between residuals
        return np.corrcoef(X_resid, Y_resid)[0, 1]
        
    except Exception as e:
        print(f"Warning: Failed to calculate partial correlation between {x} and {y}: {str(e)}")
        return None

def calculate_full_partial_correlation(data, x, y, exclude_vars=None):
    """
    Calculate partial correlation controlling for all other numeric variables.
    
    This is a more sophisticated version that:
    1. Uses pairwise complete observations
    2. Handles collinearity
    3. Controls for all numeric variables except those in exclude_vars
    """
    try:
        # Get pair data first
        pair_data = data[[x, y]].dropna()
        if len(pair_data) < 30:  # Need sufficient observations
            return None
        
        # Get potential control variables
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        control_vars = [col for col in numeric_cols 
                       if col not in [x, y] 
                       and (exclude_vars is None or col not in exclude_vars)]
        
        # Standardize main variables
        X_std = standardize_array(pair_data[x].values)
        Y_std = standardize_array(pair_data[y].values)
        
        if X_std is None or Y_std is None:
            return None
            
        X_resid = X_std.copy()
        Y_resid = Y_std.copy()
        
        n_controls_used = 0
        
        for control in control_vars:
            try:
                # Get complete cases for this control variable
                control_data = data[[x, y, control]].dropna()
                if len(control_data) < 30:  # Skip if too few observations
                    continue
                
                # Standardize all variables
                X = standardize_array(control_data[x].values)
                Y = standardize_array(control_data[y].values)
                Z = standardize_array(control_data[control].values)
                
                if X is None or Y is None or Z is None:
                    continue
                
                # Add constant term
                Z_with_const = np.c_[np.ones(len(Z)), Z]
                
                # Check matrix condition number for stability
                if np.linalg.cond(Z_with_const.T @ Z_with_const) > 1e10:
                    continue
                
                # Compute residuals using QR decomposition for stability
                Q, R = np.linalg.qr(Z_with_const)
                
                # Project out the effect of Z
                X_resid = X - Q @ (Q.T @ X)
                Y_resid = Y - Q @ (Q.T @ Y)
                
                n_controls_used += 1
                
            except Exception as e:
                continue
        
        if n_controls_used == 0:
            return calculate_correlation(pair_data[x], pair_data[y])
            
        # Calculate final correlation between residuals
        valid_mask = ~np.isnan(X_resid) & ~np.isnan(Y_resid)
        if np.sum(valid_mask) < 30:
            return None
            
        partial_corr = np.corrcoef(X_resid[valid_mask], Y_resid[valid_mask])[0, 1]
        
        # Check if result is valid
        if not np.isfinite(partial_corr):
            return None
            
        return partial_corr
        
    except Exception as e:
        print(f"Warning: Failed to calculate partial correlation between {x} and {y}: {str(e)}")
        return None

def plot_correlation_comparison(correlations, target_var):
    """Plot comparison of raw vs partial correlations."""
    # Extract correlations
    vars_to_plot = []
    raw_corrs = []
    partial_corrs = []
    
    for corr in correlations:
        if corr['partial_correlation'] is not None:
            vars_to_plot.append(corr['variable'])
            raw_corrs.append(corr['correlation'])
            partial_corrs.append(corr['partial_correlation'])
    
    if len(vars_to_plot) == 0:
        print("No valid correlations to plot")
        return
    
    # Create plot
    plt.figure(figsize=(12, 6))
    x = range(len(vars_to_plot))
    width = 0.35
    
    plt.bar([i - width/2 for i in x], raw_corrs, width, 
            label='Raw Correlation', alpha=0.7)
    plt.bar([i + width/2 for i in x], partial_corrs, width,
            label='Partial Correlation', alpha=0.7)
    
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.2)
    plt.xticks(x, vars_to_plot, rotation=45, ha='right')
    plt.ylabel("Correlation Coefficient")
    plt.title(f"Raw vs Partial Correlations with {target_var}")
    plt.legend()
    plt.tight_layout()
    plt.show()

def analyze_variable_relationships(df, meta, target_var, min_overlap=0.01, top_n=100, specific_vars=None, 
                               year_range=None, specific_years=None, plot_trends=True,
                               control_vars=None, full_partial=True):
    """
    Analyze relationships between a target variable and other variables in the dataset.
    
    Args:
        df: Input dataframe
        meta: Metadata from GSS
        target_var: Target variable to analyze
        min_overlap: Minimum required overlap between variables
        top_n: Number of top correlations to show
        specific_vars: List of specific variables to analyze (optional)
        year_range: Tuple of (start_year, end_year) to analyze (optional)
        specific_years: List of specific years to analyze (optional)
        plot_trends: Whether to plot correlation trends over time
        control_vars: List of variables to control for in partial correlations
        full_partial: Whether to use full partial correlations
    
    Returns:
        List of correlation results
    """
    # Verify target variable exists
    if target_var not in df.columns:
        raise ValueError(f"Target variable {target_var} not found in dataset")
    
    # Filter to specified years if provided
    if year_range:
        start_year, end_year = year_range
        df_subset = df[df['YEAR'].between(start_year, end_year)].copy()
    elif specific_years:
        df_subset = df[df['YEAR'].isin(specific_years)].copy()
    else:
        df_subset = df.copy()
    
    # Filter to years where target variable has data
    # This ensures we only analyze years where the target variable was measured
    target_years = df[df[target_var].notna()]['YEAR'].unique()
    if len(target_years) == 0:
        raise ValueError(f"No data for {target_var} in the specified years")
    
    df_subset = df[df['YEAR'].isin(target_years)].copy()
    print(f"\nAnalyzing years with {target_var} data: {sorted(target_years)}")
    
    # Display descriptive statistics for the target variable
    print(f"\n{target_var} statistics:")
    print(df_subset[target_var].describe())
    print(f"\nMissing values in selected years: {df_subset[target_var].isna().sum()} "
          f"({df_subset[target_var].isna().mean()*100:.1f}%)")
    
    # Show data availability by year
    print(f"\n{target_var} measurements by year:")
    year_counts = df_subset[df_subset[target_var].notna()].groupby('YEAR').size()
    print(year_counts)
    
    # Filter numeric columns based on data quality first
    quality_numeric_cols = filter_numeric_columns(df_subset)
    print(f"\nIdentified {len(quality_numeric_cols)} quality numeric variables")
    print("Quality criteria:")
    print("- Must be numeric")
    print("- Not an ID or year variable")
    print(f"- Less than 95% missing values")
    print(f"- At least {min_overlap*100:.1f}% unique values")
    print("- More than 1 unique value")
    
    # Determine which variables to analyze
    if specific_vars:
        # If specific variables are requested, verify they exist in the dataset
        missing_vars = [var for var in specific_vars if var not in df_subset.columns]
        if missing_vars:
            raise ValueError(f"Variables not found in dataset: {missing_vars}")
        analyze_cols = specific_vars
        print("\nUsing specified variables instead of quality-filtered variables")
    else:
        # Only analyze quality numeric columns
        analyze_cols = quality_numeric_cols
    
    print(f"\nWill analyze relationships with {len(analyze_cols)} variables")
    
    # Calculate correlations with target variable
    correlations = []
    n_total = len(df_subset)
    
    # Get list of variables to exclude from controls
    non_numeric_cols = df_subset.select_dtypes(exclude=[np.number]).columns.tolist()
    exclude_from_controls = ['YEAR'] + non_numeric_cols
    
    # Pre-calculate valid data once for full partial analysis
    if full_partial:
        n_controls = len([col for col in quality_numeric_cols 
                         if col not in [target_var, 'YEAR']])
        print(f"Will control for up to {n_controls} variables using pairwise complete observations")
        print("\nCalculating partial correlations...")
    
    # Analyze relationship with each variable
    for i, col in enumerate(analyze_cols):
        if col != target_var and col != 'YEAR':
            if i % 10 == 0:  # Show progress every 10 variables
                print(f"Processing variable {i+1}/{len(analyze_cols)}...")
            
            try:
                # Get data for this pair of variables
                valid_data = df_subset[[target_var, col]].dropna()
                overlap_pct = len(valid_data) / n_total
                
                if overlap_pct >= min_overlap:
                    raw_corr = calculate_correlation(valid_data[target_var], valid_data[col])
                    
                    if full_partial:
                        partial_corr = calculate_full_partial_correlation(
                            df_subset, target_var, col,
                            exclude_vars=exclude_from_controls
                        )
                    else:
                        # For regular partial, we need all variables simultaneously
                        valid_vars = [target_var, col] + (control_vars or [])
                        valid_data = df_subset[valid_vars].dropna()
                        partial_corr = calculate_partial_correlation(
                            valid_data, target_var, col, control_vars
                        )
                    
                    if raw_corr is not None:
                        label = meta.column_labels[meta.column_names.index(col)]
                        correlations.append({
                            'variable': col,
                            'label': label,
                            'correlation': raw_corr,
                            'partial_correlation': partial_corr,
                            'overlap': overlap_pct,
                            'n_valid': len(valid_data),
                            'control_type': 'full partial' if full_partial else 'specified controls'
                        })
            except Exception as e:
                print(f"Warning: Failed to analyze relationship between {target_var} and {col}: {str(e)}")
                continue
    
    # Sort results by strength of relationship
    correlations.sort(key=lambda x: abs(x['partial_correlation'] if x['partial_correlation'] is not None else x['correlation']), 
                     reverse=True)
    
    # Display correlation results with updated header
    if specific_vars:
        print(f"\nCorrelations with specified variables:")
    else:
        print(f"\nTop {top_n} correlations with {target_var}:")
    
    print("-" * 120)
    print(f"{'Variable':15} | {'Raw Corr':12} | {'Partial Corr':12} | {'Overlap %':10} | {'N':8} | Description")
    print("-" * 120)
    
    # Display correlation details
    display_correlations = correlations if specific_vars else correlations[:top_n]
    for corr in display_correlations:
        partial_corr_str = f"{corr['partial_correlation']:+.3f}" if corr['partial_correlation'] is not None else "N/A"
        print(f"{corr['variable']:15} | {corr['correlation']:+.3f}      | {partial_corr_str:12} | "
              f"{corr['overlap']*100:8.1f}% | {corr['n_valid']:8d} | {corr['label']}")
    
    # Plot correlation comparison
    if len(correlations) > 0:
        print("\nGenerating correlation comparison plots...")
        plot_correlation_comparison(correlations, target_var)
    
    # Create scatter plots for top relationships
    n_plots = len(correlations) if specific_vars else min(6, len(correlations))
    if n_plots > 0:
        # Determine plot layout based on number of relationships
        if n_plots <= 3:
            fig, axes = plt.subplots(1, n_plots, figsize=(5*n_plots, 5))
            if n_plots == 1:
                axes = [axes]
        else:
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes = axes.flatten()
        
        # Create scatter plot for each relationship
        for i, corr in enumerate(correlations[:n_plots]):
            var = corr['variable']
            valid_data = df_subset[[target_var, var]].dropna()
            
            # Add jitter to better visualize discrete variables
            x_jitter = np.random.normal(0, 0.1, size=len(valid_data))
            y_jitter = np.random.normal(0, 0.1, size=len(valid_data))
            
            # Plot points with transparency for density visualization
            axes[i].scatter(valid_data[target_var] + x_jitter, valid_data[var] + y_jitter, 
                          alpha=0.5, s=20)
            
            # Add trend line if possible
            try:
                z = np.polyfit(valid_data[target_var], valid_data[var], 1)
                p = np.poly1d(z)
                axes[i].plot(valid_data[target_var], p(valid_data[target_var]), "r--", alpha=0.8)
            except:
                pass
            
            # Add labels and title
            axes[i].set_xlabel(target_var)
            axes[i].set_ylabel(var)
            axes[i].set_title(f'{var}\nρ={corr["correlation"]:.3f}, n={corr["n_valid"]}')
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    # Analyze and plot time trends if requested
    if plot_trends and len(df_subset['YEAR'].unique()) > 1:
        print("\nAnalyzing time trends for top correlations...")
        plot_time_trends(df_subset, target_var, correlations)
    
    return correlations

def display_variable_info(meta):
    """Display information about variables from the metadata."""
    print("\nVariable Information:")
    print("-" * 80)
    for i, (name, label) in enumerate(zip(meta.column_names, meta.column_labels)):
        print(f"{name:20} | {label}")
    print("-" * 80)

def filter_numeric_columns(df: pd.DataFrame, min_unique_pct: float = 0.01,
                         max_missing_pct: float = 0.95) -> List[str]:
    """
    Filter numeric columns based on data quality criteria.
    
    Args:
        df: Input dataframe
        min_unique_pct: Minimum percentage of unique values required
        max_missing_pct: Maximum percentage of missing values allowed
    
    Returns:
        List of column names meeting criteria
    """
    numeric_cols = []
    
    for col in df.columns:
        # Skip ID and year columns
        if col in ['ID', 'YEAR']:
            continue
            
        # Must be numeric
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
            
        # Get non-missing values
        non_missing = df[col].dropna()
        
        # Skip if too many missing values
        missing_pct = 1 - len(non_missing) / len(df)
        if missing_pct > max_missing_pct:
            continue
            
        # Must have more than 1 unique value
        n_unique = non_missing.nunique()
        if n_unique <= 1:
            continue
            
        # Must have sufficient unique values
        unique_pct = n_unique / len(non_missing)
        if unique_pct < min_unique_pct:
            continue
            
        numeric_cols.append(col)
    
    return numeric_cols
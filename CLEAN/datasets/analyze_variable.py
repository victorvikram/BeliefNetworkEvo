import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from explore_gss import load_or_cache_data, calculate_correlation
from scipy.stats import spearmanr
from pingouin import partial_corr

def plot_time_trends(df_subset, target_var, correlations, min_year_overlap=100):
    """
    Plot how relationships between variables change over time.
    
    This function creates two subplots:
    1. Top panel: Shows how correlations with the target variable change across years
    2. Bottom panel: Shows the sample size available for each correlation by year
    
    Args:
        df_subset (pd.DataFrame): Filtered DataFrame containing the variables to analyze
        target_var (str): Name of the target variable being analyzed
        correlations (list): List of dictionaries containing correlation information
        min_year_overlap (int): Minimum number of observations required in a year to calculate correlation
    
    The function plots trends for the top 5 most strongly correlated variables.
    Each trend line shows how the relationship strength changes over time.
    The sample size plot helps identify potential data quality issues or collection patterns.
    """
    # Get years with sufficient data
    years = sorted(df_subset['YEAR'].unique())
    
    # Take top 5 correlated variables for time trends
    top_vars = [c['variable'] for c in correlations[:5]]
    
    # Initialize dictionaries to store yearly correlations and sample sizes
    yearly_correlations = {var: [] for var in top_vars}
    yearly_counts = {var: [] for var in top_vars}
    
    # Calculate correlations and sample sizes for each year
    for year in years:
        year_data = df_subset[df_subset['YEAR'] == year]
        for var in top_vars:
            valid_data = year_data[[target_var, var]].dropna()
            if len(valid_data) >= min_year_overlap:
                corr = calculate_correlation(valid_data[target_var], valid_data[var])
                if corr is not None:
                    yearly_correlations[var].append((year, corr))
                    yearly_counts[var].append((year, len(valid_data)))
    
    # Create the visualization
    plt.figure(figsize=(15, 10))
    
    # Top subplot: Correlation trends
    plt.subplot(2, 1, 1)
    for var in top_vars:
        if yearly_correlations[var]:  # Only plot if we have data
            years, corrs = zip(*yearly_correlations[var])
            plt.plot(years, corrs, 'o-', label=var, alpha=0.7)
    
    plt.title(f'Correlation Trends with {target_var} Over Time')
    plt.xlabel('Year')
    plt.ylabel("Spearman's ρ")
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Bottom subplot: Sample size trends
    plt.subplot(2, 1, 2)
    for var in top_vars:
        if yearly_counts[var]:  # Only plot if we have data
            years, counts = zip(*yearly_counts[var])
            plt.plot(years, counts, 'o-', label=var, alpha=0.7)
    
    plt.title('Sample Size Over Time')
    plt.xlabel('Year')
    plt.ylabel('Number of Observations')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.show()

def calculate_partial_correlation(data, x, y, covariates):
    """
    Calculate the partial correlation between two variables while controlling for covariates.
    
    Partial correlation measures the relationship between two variables while removing
    the effect of one or more control variables. This helps identify direct relationships
    by accounting for confounding factors.
    
    Args:
        data (pd.DataFrame): DataFrame containing all variables (x, y, and covariates)
        x (str): Name of the first variable
        y (str): Name of the second variable
        covariates (list): List of variable names to control for
    
    Returns:
        float or None: Partial correlation coefficient if calculation succeeds,
                      None if there's insufficient variation or calculation fails
    
    Example:
        >>> df = pd.DataFrame({'x': [1,2,3], 'y': [2,4,6], 'z': [1,2,3]})
        >>> calculate_partial_correlation(df, 'x', 'y', ['z'])
    """
    try:
        # Check for sufficient variation in all variables
        # Partial correlation requires at least 2 unique values for each variable
        if (len(data[x].unique()) <= 1 or 
            len(data[y].unique()) <= 1 or 
            any(len(data[cov].unique()) <= 1 for cov in covariates)):
            return None
        
        # Calculate partial correlation using pingouin library
        # Using Spearman method for consistency with regular correlations
        result = partial_corr(data=data, x=x, y=y, covar=covariates, method='spearman')
        return result['r'].iloc[0]
    except:
        return None

def standardize_array(x):
    """
    Safely standardize an array with checks for numerical stability.
    
    Args:
        x: numpy array to standardize
    
    Returns:
        Standardized array or None if standardization is not possible
    """
    if len(x) < 2:
        return None
        
    std = np.std(x, ddof=1)
    if std < 1e-10:  # Check for near-zero standard deviation
        return None
        
    mean = np.mean(x)
    return (x - mean) / std

def calculate_full_partial_correlation(data, x, y, exclude_vars=None):
    """
    Calculate partial correlation controlling for all numeric variables except those specified.
    Uses pairwise complete observations to handle missing data gracefully.
    
    Args:
        data (pd.DataFrame): DataFrame containing all variables
        x (str): Name of the first variable
        y (str): Name of the second variable
        exclude_vars (list): Variables to exclude from controls
    
    Returns:
        float or None: Partial correlation coefficient if calculation succeeds
    """
    try:
        # Get all numeric columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        # Determine control variables (all numeric except x, y, and excluded)
        if exclude_vars is None:
            exclude_vars = []
        control_vars = [col for col in numeric_cols 
                       if col not in [x, y] + exclude_vars]
        
        if not control_vars:
            return calculate_correlation(data[x], data[y])
        
        # Get data for the pair we're analyzing
        pair_data = data[[x, y]].dropna()
        if len(pair_data) < 30:  # Minimum sample size for reliable correlation
            return None
            
        # Initialize with standardized target variables
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

def plot_correlation_comparison(correlations, target_var, top_n=20):
    """
    Create visualizations comparing raw and partial correlations to understand
    how controlling for other variables affects relationships.
    """
    # Convert correlations to DataFrame for easier analysis
    df_corr = pd.DataFrame(correlations)
    
    # Calculate absolute changes and percentages
    df_corr['abs_raw'] = df_corr['correlation'].abs()
    df_corr['abs_partial'] = df_corr['partial_correlation'].abs()
    df_corr['abs_change'] = df_corr['abs_partial'] - df_corr['abs_raw']
    df_corr['pct_change'] = (df_corr['abs_change'] / df_corr['abs_raw']) * 100
    
    # Get top correlations by partial correlation strength
    df_top = df_corr.nlargest(top_n, 'abs_partial')
    
    # Create figure with three subplots
    fig = plt.figure(figsize=(15, 15))
    
    # Subplot 1: Parallel coordinates plot
    plt.subplot(3, 1, 1)
    for _, row in df_top.iterrows():
        color = 'red' if abs(row['partial_correlation']) < abs(row['correlation']) else 'green'
        plt.plot(['Raw', 'Partial'], 
                [row['correlation'], row['partial_correlation']], 
                '-o', alpha=0.5, color=color, label=row['variable'])
    
    plt.grid(True, alpha=0.3)
    plt.ylabel("Correlation Coefficient")
    plt.title(f"How Correlations Change After Controlling for Other Variables\n(Top {top_n} relationships with {target_var})")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    
    # Subplot 2: Bar plot of absolute changes
    plt.subplot(3, 1, 2)
    df_changes = df_top.sort_values('abs_change')
    bars = plt.bar(range(len(df_changes)), df_changes['abs_change'], alpha=0.6)
    
    # Color bars based on direction of change
    for i, bar in enumerate(bars):
        if df_changes['abs_change'].iloc[i] > 0:
            bar.set_color('green')
        else:
            bar.set_color('red')
    
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.grid(True, alpha=0.3)
    plt.ylabel("Change in |Correlation|\n(|Partial| - |Raw|)")
    plt.title("Absolute Change in Correlation Strength")
    plt.xticks(range(len(df_changes)), df_changes['variable'], rotation=45, ha='right')
    
    # Subplot 3: Scatter plot of raw vs partial correlations
    plt.subplot(3, 1, 3)
    plt.scatter(df_top['correlation'], df_top['partial_correlation'], alpha=0.6)
    
    # Add variable labels to points
    for _, row in df_top.iterrows():
        plt.annotate(row['variable'], 
                    (row['correlation'], row['partial_correlation']),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=8)
    
    # Add diagonal line
    lims = [
        min(plt.xlim()[0], plt.ylim()[0]),
        max(plt.xlim()[1], plt.ylim()[1]),
    ]
    plt.plot(lims, lims, 'k--', alpha=0.5, label='No Change Line')
    plt.grid(True, alpha=0.3)
    plt.xlabel("Raw Correlation")
    plt.ylabel("Partial Correlation")
    plt.title("Raw vs Partial Correlations\n(Points below line indicate weakened relationships)")
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print("\nSummary of correlation changes:")
    print("-" * 80)
    print("Overall changes:")
    print(f"Mean absolute change: {df_corr['abs_change'].mean():.3f}")
    print(f"Median absolute change: {df_corr['abs_change'].median():.3f}")
    print(f"Relationships weakened: {(df_corr['abs_change'] < 0).sum()} ({(df_corr['abs_change'] < 0).mean()*100:.1f}%)")
    print(f"Relationships strengthened: {(df_corr['abs_change'] > 0).sum()} ({(df_corr['abs_change'] > 0).mean()*100:.1f}%)")
    
    print("\nMost affected relationships (absolute change):")
    print("-" * 80)
    print("Largest decreases in correlation strength:")
    decreases = df_corr.nsmallest(5, 'abs_change')[['variable', 'label', 'correlation', 'partial_correlation', 'abs_change']]
    for _, row in decreases.iterrows():
        print(f"{row['variable']}: {row['correlation']:.3f} → {row['partial_correlation']:.3f} (Δ={row['abs_change']:.3f})")
        print(f"  {row['label']}")
    
    print("\nLargest increases in correlation strength:")
    increases = df_corr.nlargest(5, 'abs_change')[['variable', 'label', 'correlation', 'partial_correlation', 'abs_change']]
    for _, row in increases.iterrows():
        print(f"{row['variable']}: {row['correlation']:.3f} → {row['partial_correlation']:.3f} (Δ={row['abs_change']:.3f})")
        print(f"  {row['label']}")
    
    print("\nInterpretation guide:")
    print("-" * 80)
    print("- Weakened relationships suggest the original correlation was partly explained by other variables")
    print("- Strengthened relationships suggest the true relationship was masked by other variables")
    print("- Stable relationships (small changes) suggest direct, unmediated relationships")
    print("- Large changes suggest complex relationships involving multiple variables")

def filter_numeric_columns(df, min_unique_ratio=0.01, max_missing_ratio=0.95):
    """
    Filter numeric columns based on data quality criteria.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        min_unique_ratio (float): Minimum ratio of unique values to total values
        max_missing_ratio (float): Maximum allowed ratio of missing values
    
    Returns:
        list: List of column names meeting the criteria
    """
    numeric_cols = []
    n_rows = len(df)
    
    for col in df.select_dtypes(include=[np.number]).columns:
        # Skip columns that are just IDs or years
        if col.upper() in ['ID', 'YEAR', 'CASEID']:
            continue
            
        series = df[col]
        n_unique = series.nunique()
        missing_ratio = series.isna().mean()
        unique_ratio = n_unique / n_rows
        
        # Include column if it meets all criteria
        if (missing_ratio < max_missing_ratio and 
            unique_ratio >= min_unique_ratio and
            n_unique > 1):
            numeric_cols.append(col)
    
    return numeric_cols

def analyze_variable_relationships(df, meta, target_var, min_overlap=0.01, top_n=100, specific_vars=None, 
                               year_range=None, specific_years=None, plot_trends=True,
                               control_vars=None, full_partial=True):
    """
    Comprehensive analysis of relationships between a target variable and other variables.
    
    This function performs several analyses:
    1. Basic statistics of the target variable
    2. Raw correlations with other variables
    3. Partial correlations controlling for demographic factors
    4. Visualization of relationships
    5. Time trend analysis (if multiple years present)
    
    Args:
        df (pd.DataFrame): DataFrame containing the GSS survey data
        meta: Metadata object containing variable descriptions
        target_var (str): Name of the variable to analyze relationships with
        min_overlap (float): Minimum required data overlap between variables (0-1)
        top_n (int): Number of top correlations to display
        specific_vars (list): Optional list of specific variables to analyze
        year_range (tuple): Optional (start_year, end_year) to filter data
        specific_years (list): Optional list of specific years to analyze
        plot_trends (bool): Whether to plot correlation trends over time
        control_vars (list): Variables to control for in partial correlations
        full_partial (bool): If True, control for all numeric variables except the pair
                           being compared. If False, use specified control_vars.
    
    Returns:
        list: List of dictionaries containing correlation information
    
    Example:
        >>> correlations = analyze_variable_relationships(
        ...     df, meta, 'POLVIEWS',
        ...     year_range=(2018, 2022),
        ...     control_vars=['AGE', 'SEX', 'EDUC']
        ... )
    """
    # Set default control variables if none specified and not using full partial
    if not full_partial and control_vars is None:
        potential_controls = ['AGE', 'SEX', 'YEAR', 'EDUC']
        control_vars = [var for var in potential_controls if var in df.columns and var != target_var]
    
    # Validate target variable exists
    if target_var not in df.columns:
        raise ValueError(f"Target variable '{target_var}' not found in dataset")
    
    # Get and display target variable description
    target_label = meta.column_labels[meta.column_names.index(target_var)]
    print(f"\nAnalyzing {target_var}: {target_label}")
    
    # Validate year filtering options
    if year_range and specific_years:
        raise ValueError("Cannot specify both year_range and specific_years")
    
    # Apply year filtering based on specified range or specific years
    if year_range:
        start_year, end_year = year_range
        df = df[df['YEAR'].between(start_year, end_year)].copy()
        print(f"\nFiltering data to years {start_year}-{end_year}")
    elif specific_years:
        df = df[df['YEAR'].isin(specific_years)].copy()
        print(f"\nFiltering data to specific years: {sorted(specific_years)}")
    
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
    
    # Analyze and plot time trends if requested and we have multiple years
    if plot_trends and len(df_subset['YEAR'].unique()) > 1:
        print("\nAnalyzing time trends for top correlations...")
        plot_time_trends(df_subset, target_var, correlations)
    
    return correlations

if __name__ == "__main__":
    # Load the GSS dataset
    df, meta = load_or_cache_data()
    
    # Display available years in the dataset
    print("\nYears in dataset:", sorted(df['YEAR'].unique()))
    
    # Example analysis with optimized full partial correlations
    target_var = 'POLVIEWS'
    time_range = (2022, 2022)
    
    print(f"\nAnalyzing {target_var} with full partial correlations")
    correlations = analyze_variable_relationships(
        df, meta, target_var,
        year_range=time_range,
        plot_trends=True,  # Enable plotting
        full_partial=True
    )
    
    print("\nAnalysis complete. Results available in 'correlations' variable.") 
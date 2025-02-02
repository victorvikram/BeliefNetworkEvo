"""
Data Quality Assessment Script for Cleaned GSS Datasets
----------------------------------------------------
This script performs comprehensive validation of the cleaned datasets to ensure:
1. Data ranges are correct (-1 to 1 for normalized variables)
2. Missing values are handled appropriately
3. Transformations were applied correctly
4. Data consistency between versions
5. Variable relationships make sense
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, List, Optional
import os
import sys
import warnings
from contextlib import contextmanager

# Add the project root directory to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from datasets.prepare_cleaned_datasets import load_cleaned_datasets, VARS_TO_KEEP
from datasets.clean_data import DataConfig

@contextmanager
def figure_context():
    """Context manager to ensure figures are closed even if an error occurs."""
    try:
        yield
    finally:
        plt.close()

def load_datasets() -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[Dict]]:
    """Load both cleaned versions and the metadata."""
    try:
        print("Loading cleaned datasets...")
        df_cleaned_1, df_cleaned_2, meta = load_cleaned_datasets()
        return df_cleaned_1, df_cleaned_2, meta
    except FileNotFoundError as e:
        print(f"Error loading datasets: {e}")
        return None, None, None
    except Exception as e:
        print(f"Unexpected error loading datasets: {e}")
        return None, None, None

def validate_data_types(df: pd.DataFrame, config: DataConfig) -> List[str]:
    """Validate data types before transformation."""
    issues = []
    for col in df.columns:
        if col not in config.EXCLUDE_COLS:
            if not pd.api.types.is_numeric_dtype(df[col]):
                issues.append(f"{col}: Non-numeric type {df[col].dtype}")
            elif df[col].isna().all():
                issues.append(f"{col}: All values are NaN")
    return issues

def check_value_ranges(df: pd.DataFrame, exclude_cols: List[str]) -> Dict:
    """
    Check if normalized variables are within expected ranges (-1 to 1).
    
    Returns:
        Dictionary with statistics about value ranges
    """
    results = {}
    for col in df.columns:
        if col not in exclude_cols:
            non_null = df[col].dropna()
            if len(non_null) == 0:
                results[col] = {
                    'error': 'All values are NaN'
                }
                continue
                
            col_stats = {
                'min': non_null.min(),
                'max': non_null.max(),
                'median': non_null.median(),
                'within_bounds': (non_null.min() >= -1.1) and (non_null.max() <= 1.1),
                'unique_values': non_null.nunique(),
                'missing_pct': (df[col].isna().sum() / len(df)) * 100
            }
            results[col] = col_stats
    return results

def check_excluded_columns(df: pd.DataFrame, config: DataConfig) -> List[str]:
    """Verify that excluded columns weren't transformed."""
    exclude_cols = set(config.EXCLUDE_COLS)
    issues = []
    
    for col in exclude_cols:
        if col in df.columns:
            # Check if values match original data types and ranges
            if df[col].dtype != pd.Int64Dtype() and df[col].dtype != pd.Float64Dtype():
                issues.append(f"{col}: Unexpected data type {df[col].dtype}")
            # Check for unexpected modifications
            if col == 'YEAR':
                if not df[col].between(1900, 2100).all():
                    issues.append(f"{col}: Values outside expected range")
    return issues

def compare_datasets(df1: pd.DataFrame, df2: pd.DataFrame, config: DataConfig) -> Dict:
    """
    Compare the two cleaned versions to identify significant differences.
    """
    comparison = {
        'common_cols': list(set(df1.columns) & set(df2.columns)),
        'unique_to_df1': list(set(df1.columns) - set(df2.columns)),
        'unique_to_df2': list(set(df2.columns) - set(df1.columns)),
        'correlations': {},
        'value_diffs': {}
    }
    
    # Compare distributions of common columns
    for col in comparison['common_cols']:
        if col not in config.EXCLUDE_COLS:
            # Handle NaN values safely
            valid_mask = df1[col].notna() & df2[col].notna()
            if valid_mask.any():
                # Calculate correlation between versions
                try:
                    correlation = df1.loc[valid_mask, col].corr(
                        df2.loc[valid_mask, col], 
                        method='spearman'
                    )
                    comparison['correlations'][col] = correlation
                except Exception as e:
                    comparison['correlations'][col] = f"Error: {str(e)}"
                
                # Calculate distribution differences
                try:
                    value_diff = (
                        df1.loc[valid_mask, col].mean() - 
                        df2.loc[valid_mask, col].mean()
                    )
                    comparison['value_diffs'][col] = value_diff
                except Exception as e:
                    comparison['value_diffs'][col] = f"Error: {str(e)}"
            else:
                comparison['correlations'][col] = "No valid pairs"
                comparison['value_diffs'][col] = "No valid pairs"
    
    return comparison

def plot_distribution_comparisons(df1: pd.DataFrame, df2: pd.DataFrame, 
                                cols_to_plot: List[str], output_dir: str,
                                config: DataConfig):
    """
    Create comparison plots of variable distributions between the two versions.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Filter out excluded columns and verify existence
    cols_to_plot = [col for col in cols_to_plot 
                    if col in df1.columns 
                    and col in df2.columns 
                    and col not in config.EXCLUDE_COLS]
    
    for col in cols_to_plot:
        with figure_context():
            try:
                plt.figure(figsize=(12, 6))
                
                # Plot distributions
                plt.subplot(1, 2, 1)
                sns.histplot(data=df1, x=col, bins=30, alpha=0.5, label='Version 1')
                sns.histplot(data=df2, x=col, bins=30, alpha=0.5, label='Version 2')
                plt.title(f'Distribution Comparison: {col}')
                plt.legend()
                
                # Plot Q-Q plot
                plt.subplot(1, 2, 2)
                sorted1 = np.sort(df1[col].dropna())
                sorted2 = np.sort(df2[col].dropna())
                min_len = min(len(sorted1), len(sorted2))
                if min_len > 0:  # Only plot if we have valid data
                    plt.scatter(sorted1[:min_len], sorted2[:min_len], alpha=0.5)
                    plt.plot([-1, 1], [-1, 1], 'r--')
                plt.title(f'Q-Q Plot: {col}')
                plt.xlabel('Version 1')
                plt.ylabel('Version 2')
                
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'{col}_comparison.png'))
            except Exception as e:
                warnings.warn(f"Error plotting {col}: {str(e)}")

def validate_transformations(df1: pd.DataFrame, df2: pd.DataFrame, 
                           meta: Dict, config: DataConfig) -> Dict:
    """
    Validate that transformations were applied correctly according to our rules.
    """
    validation = {
        'binary_vars': {},
        'ordinal_vars': {},
        'median_centered_vars': {}
    }
    
    # Check binary transformations (should be close to -1 or 1)
    for var in config.BINARY_VARS:
        if var in df1.columns:
            values = df1[var].dropna().value_counts()
            unexpected = [v for v in values.index 
                         if not any(abs(abs(v) - 1) < 0.1 for v in values.index)]
            validation['binary_vars'][var] = {
                'unique_values': values.to_dict(),
                'properly_binary': len(unexpected) == 0,
                'unexpected_values': unexpected
            }
    
    # Check ordinal transformations
    for category, variables in config.MEDIAN_VARS.items():
        for var in variables:
            if var in df1.columns:
                values = df1[var].dropna().value_counts()
                validation['ordinal_vars'][var] = {
                    'unique_values': values.to_dict(),
                    'proper_range': (-1 <= df1[var].min() <= 1) and (-1 <= df1[var].max() <= 1),
                    'missing_pct': (df1[var].isna().sum() / len(df1)) * 100
                }
    
    # Check median-centered variables in version 2
    for category, variables in config.MEDIAN_VARS.items():
        for var in variables:
            if var in df2.columns:
                non_null = df2[var].dropna()
                if len(non_null) > 0:
                    median = non_null.median()
                    validation['median_centered_vars'][var] = {
                        'median': median,
                        'properly_centered': abs(median) < 0.1,  # Should be close to 0
                        'range': [-1 <= non_null.min() <= 1, -1 <= non_null.max() <= 1],
                        'missing_pct': (df2[var].isna().sum() / len(df2)) * 100
                    }
                else:
                    validation['median_centered_vars'][var] = {
                        'error': 'No valid data'
                    }
    
    return validation

def analyze_year_spans(df1: pd.DataFrame, df2: pd.DataFrame) -> Dict:
    """Analyze the year spans in both datasets."""
    spans = {}
    for version, df in [("Version 1", df1), ("Version 2", df2)]:
        if 'YEAR' in df.columns:
            spans[version] = {
                'min_year': int(df['YEAR'].min()),
                'max_year': int(df['YEAR'].max()),
                'n_years': len(df['YEAR'].unique()),
                'unique_years': sorted(df['YEAR'].unique().tolist())
            }
        else:
            spans[version] = "No YEAR column found"
    return spans

def plot_data_completeness(df1: pd.DataFrame, df2: pd.DataFrame, output_dir: str):
    """Create visualization of data completeness for both datasets."""
    with figure_context():
        plt.figure(figsize=(15, 10))
        
        # Calculate completeness percentages
        completeness1 = (1 - df1.isna().mean()) * 100
        completeness2 = (1 - df2.isna().mean()) * 100
        
        # Combine data for plotting
        completeness_df = pd.DataFrame({
            'Version 1': completeness1,
            'Version 2': completeness2
        })
        
        # Sort by Version 1 completeness for better visualization
        completeness_df = completeness_df.sort_values('Version 1', ascending=True)
        
        # Create heatmap
        sns.heatmap(completeness_df.T, annot=True, fmt='.1f', 
                    cmap='YlOrRd', cbar_kws={'label': 'Completeness %'})
        plt.title('Data Completeness by Variable (%)')
        plt.xlabel('Variables')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Save plot
        plt.savefig(os.path.join(output_dir, 'data_completeness.png'))

def compare_dataset_headers(df1: pd.DataFrame, df2: pd.DataFrame) -> Dict:
    """Compare headers and their overlap between datasets."""
    vars1 = set(df1.columns)
    vars2 = set(df2.columns)
    
    comparison = {
        'version1_vars': sorted(list(vars1)),
        'version2_vars': sorted(list(vars2)),
        'common_vars': sorted(list(vars1 & vars2)),
        'only_in_version1': sorted(list(vars1 - vars2)),
        'only_in_version2': sorted(list(vars2 - vars1)),
        'overlap_percentage': len(vars1 & vars2) / max(len(vars1), len(vars2)) * 100
    }
    return comparison

def main():
    """Run all validation checks and generate report."""
    # Load datasets
    df_cleaned_1, df_cleaned_2, meta = load_datasets()
    if df_cleaned_1 is None or df_cleaned_2 is None:
        print("Failed to load datasets. Exiting.")
        return
    
    print("\nRunning validation checks...")
    
    config = DataConfig()
    
    # New checks
    print("\nAnalyzing year spans...")
    year_spans = analyze_year_spans(df_cleaned_1, df_cleaned_2)
    
    print("\nAnalyzing dataset headers and overlap...")
    header_comparison = compare_dataset_headers(df_cleaned_1, df_cleaned_2)
    
    print("\nGenerating data completeness visualization...")
    plot_dir = 'datasets/validation_plots'
    os.makedirs(plot_dir, exist_ok=True)
    plot_data_completeness(df_cleaned_1, df_cleaned_2, plot_dir)
    
    # Original checks
    # 0. Validate data types
    print("\nValidating data types...")
    type_issues_1 = validate_data_types(df_cleaned_1, config)
    type_issues_2 = validate_data_types(df_cleaned_2, config)
    
    # 1. Check value ranges
    print("\nChecking value ranges...")
    ranges_1 = check_value_ranges(df_cleaned_1, config.EXCLUDE_COLS)
    ranges_2 = check_value_ranges(df_cleaned_2, config.EXCLUDE_COLS)
    
    # 2. Check excluded columns
    print("\nChecking excluded columns...")
    exclude_issues_1 = check_excluded_columns(df_cleaned_1, config)
    exclude_issues_2 = check_excluded_columns(df_cleaned_2, config)
    
    # 3. Compare datasets
    print("\nComparing datasets...")
    comparison = compare_datasets(df_cleaned_1, df_cleaned_2, config)
    
    # 4. Validate transformations
    print("\nValidating transformations...")
    transformation_validation = validate_transformations(df_cleaned_1, df_cleaned_2, meta, config)
    
    # 5. Generate plots
    print("\nGenerating comparison plots...")
    # Select variables to plot based on actual data
    vars_to_plot = [var for var in VARS_TO_KEEP 
                    if var in df_cleaned_1.columns 
                    and var in df_cleaned_2.columns 
                    and var not in config.EXCLUDE_COLS][:10]  # Limit to first 10
    plot_distribution_comparisons(df_cleaned_1, df_cleaned_2, vars_to_plot, plot_dir, config)
    
    # Print new analysis results
    print("\n" + "="*80)
    print("ADDITIONAL VALIDATION RESULTS")
    print("="*80)
    
    print("\nYear Spans:")
    for version, span in year_spans.items():
        if isinstance(span, dict):
            print(f"\n{version}:")
            print(f"Range: {span['min_year']} - {span['max_year']} ({span['n_years']} unique years)")
            print(f"Unique years: {span['unique_years']}")
        else:
            print(f"\n{version}: {span}")
    
    print("\nDataset Headers:")
    print(f"\nVersion 1 variables ({len(header_comparison['version1_vars'])}):")
    print(header_comparison['version1_vars'])
    print(f"\nVersion 2 variables ({len(header_comparison['version2_vars'])}):")
    print(header_comparison['version2_vars'])
    
    print("\nVariable Overlap Analysis:")
    print(f"Overlap percentage: {header_comparison['overlap_percentage']:.1f}%")
    if header_comparison['only_in_version1']:
        print("\nVariables only in Version 1:")
        print(header_comparison['only_in_version1'])
    if header_comparison['only_in_version2']:
        print("\nVariables only in Version 2:")
        print(header_comparison['only_in_version2'])
    
    print("\nData completeness visualization has been saved to:", 
          os.path.join(plot_dir, 'data_completeness.png'))
    
    # Print summary report
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)
    
    print("\n0. Data Type Issues:")
    if type_issues_1 or type_issues_2:
        print("\nVersion 1:")
        for issue in type_issues_1:
            print(f"- {issue}")
            # Print example values for the problematic variable
            var_name = issue.split(':')[0].strip()
            if var_name in df_cleaned_1.columns:
                print("  Example values:")
                print(f"  Data type: {df_cleaned_1[var_name].dtype}")
                print(f"  First 5 unique values: {df_cleaned_1[var_name].unique()[:5].tolist()}")
        
        print("\nVersion 2:")
        for issue in type_issues_2:
            print(f"- {issue}")
            # Print example values for the problematic variable
            var_name = issue.split(':')[0].strip()
            if var_name in df_cleaned_2.columns:
                print("  Example values:")
                print(f"  Data type: {df_cleaned_2[var_name].dtype}")
                print(f"  First 5 unique values: {df_cleaned_2[var_name].unique()[:5].tolist()}")
    else:
        print("No data type issues found")
    
    print("\n1. Dataset Shapes:")
    print(f"Version 1: {df_cleaned_1.shape}")
    print(f"Version 2: {df_cleaned_2.shape}")
    
    print("\n2. Value Range Issues:")
    for version, ranges, df in [("Version 1", ranges_1, df_cleaned_1), 
                              ("Version 2", ranges_2, df_cleaned_2)]:
        print(f"\n{version}:")
        issues = [col for col, stats in ranges.items() 
                 if 'error' in stats or not stats.get('within_bounds', False)]
        if issues:
            print(f"Variables outside [-1, 1] range or with errors:")
            for col in issues:
                stats = ranges[col]
                print(f"\n- {col}:")
                if 'error' in stats:
                    print(f"  Error: {stats['error']}")
                else:
                    print(f"  Range: [{df[col].min():.3f}, {df[col].max():.3f}]")
                    print(f"  Example values: {df[col].dropna().sample(min(5, len(df[col].dropna()))).tolist()}")
                    print(f"  Missing values: {(df[col].isna().sum() / len(df)) * 100:.1f}%")
        else:
            print("All variables within expected range")
    
    print("\n3. Excluded Column Issues:")
    if exclude_issues_1 or exclude_issues_2:
        print("Issues found with excluded columns:")
        for issue in exclude_issues_1 + exclude_issues_2:
            print(f"- {issue}")
    else:
        print("No issues with excluded columns")
    
    print("\n4. Dataset Differences:")
    print(f"Common columns: {len(comparison['common_cols'])}")
    print(f"Unique to Version 1: {len(comparison['unique_to_df1'])}")
    print(f"Unique to Version 2: {len(comparison['unique_to_df2'])}")
    
    print("\n5. Transformation Issues:")
    print("\nBinary Variables:")
    binary_issues = [var for var, stats in transformation_validation['binary_vars'].items() 
                    if not stats['properly_binary']]
    if binary_issues:
        print(f"Issues found in: {binary_issues}")
        for var in binary_issues:
            stats = transformation_validation['binary_vars'][var]
            print(f"\n{var}:")
            print(f"Unexpected values: {stats['unexpected_values']}")
    else:
        print("No issues found")
        
    print("\nMedian-Centered Variables:")
    median_issues = [var for var, stats in transformation_validation['median_centered_vars'].items() 
                    if 'error' in stats or not stats.get('properly_centered', False)]
    if median_issues:
        print(f"Issues found in: {median_issues}")
        for var in median_issues:
            stats = transformation_validation['median_centered_vars'][var]
            if 'error' in stats:
                print(f"\n{var}: {stats['error']}")
            else:
                print(f"\n{var}:")
                print(f"Median: {stats['median']:.3f}")
                print(f"Missing: {stats['missing_pct']:.1f}%")
    else:
        print("All variables properly centered")
    
    print("\nValidation plots have been saved to:", plot_dir)
    print("\nValidation complete!")

if __name__ == "__main__":
    main() 
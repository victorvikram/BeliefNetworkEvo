"""
GSS Dataset Preparation Script
----------------------------

This script prepares the GSS dataset by:
1. Loading the raw data
2. Creating two cleaned versions:
   - Regular version with natural zero points
   - Median-centered version
3. Caching the results for future use
"""

import pandas as pd
import numpy as np
import os
import pickle
import sys
import warnings
import logging
from typing import Dict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Add the project root directory to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)
    logger.info(f"Added {project_root} to Python path")

try:
    from datasets.explore_gss import load_or_cache_data
    from datasets.clean_data import clean_datasets, DataConfig
    logger.info("Successfully imported required modules")
except Exception as e:
    logger.error(f"Error importing modules: {str(e)}")
    raise

# Define our desired variables
BELIEF_VARS = [
    "YEAR", "PARTYID", "POLVIEWS", "NATSPAC", "NATENVIR", "NATHEAL", "NATCITY",
    "NATCRIME", "NATDRUG", "NATEDUC", "NATRACE", "NATARMS", "NATAID", "NATFARE",
    "NATROAD", "NATSOC", "NATMASS", "NATPARK", "NATCHLD", "NATSCI", "NATENRGY",
    "EQWLTH", "SPKATH", "COLATH", "LIBATH", "SPKRAC", "COLRAC", "LIBRAC",
    "SPKCOM", "COLCOM", "LIBCOM", "SPKMIL", "COLMIL", "LIBMIL", "SPKHOMO",
    "COLHOMO", "LIBHOMO", "SPKMSLM", "COLMSLM", "LIBMSLM", "CAPPUN", "GUNLAW",
    "COURTS", "GRASS", "ATTEND", "RELITEN", "POSTLIFE", "PRAYER", "AFFRMACT",
    "WRKWAYUP", "HELPFUL", "FAIR", "TRUST", "CONFINAN", "CONBUS", "CONCLERG",
    "CONEDUC", "CONFED", "CONLABOR", "CONPRESS", "CONMEDIC", "CONTV", "CONJUDGE",
    "CONSCI", "CONLEGIS", "CONARMY", "GETAHEAD", "FEPOL", "ABDEFECT", "ABNOMORE",
    "ABHLTH", "ABPOOR", "ABRAPE", "ABSINGLE", "ABANY", "SEXEDUC", "DIVLAW",
    "PREMARSX", "TEENSEX", "XMARSEX", "HOMOSEX", "PORNLAW", "SPANKING", "LETDIE1",
    "SUICIDE1", "SUICIDE2", "POLHITOK", "POLABUSE", "POLMURDR", "POLESCAP",
    "POLATTAK", "NEWS", "TVHOURS", "FECHLD", "FEPRESCH", "FEFAM", "RACDIF1",
    "RACDIF2", "RACDIF3", "RACDIF4", "HELPPOOR", "MARHOMO", "RACOPEN", "HELPNOT",
    "HELPBLK"
]

# Combine all variables we want to keep
VARS_TO_KEEP = BELIEF_VARS

def perform_quality_checks(df_raw: pd.DataFrame, df_cleaned_1: pd.DataFrame, df_cleaned_2: pd.DataFrame) -> None:
    """
    Perform quality checks on the cleaned datasets.
    
    Args:
        df_raw: Original raw dataframe
        df_cleaned_1: First cleaned version (regular normalization)
        df_cleaned_2: Second cleaned version (median-centered)
    """
    issues_found = False
    
    # Check 1: All-NaN columns
    logger.info("1. Checking for all-NaN columns...")
    for df, version in [(df_cleaned_1, "Regular"), (df_cleaned_2, "Median-centered")]:
        all_nan_cols = df.columns[df.isna().all()].tolist()
        if all_nan_cols:
            issues_found = True
            logger.warning(f"\n{version} version has columns with all NaN values:")
            for col in all_nan_cols:
                raw_nan_pct = df_raw[col].isna().mean() * 100
                logger.warning(f"  • {col:<15} (Raw data NaN: {raw_nan_pct:.1f}%)")
    
    # Check 2: NaN correspondence
    logger.info("\n2. Checking NaN correspondence with raw data...")
    for df, version in [(df_cleaned_1, "Regular"), (df_cleaned_2, "Median-centered")]:
        nan_mismatches = []
        for col in df.columns:
            if col in df_raw.columns and col != 'YEAR':
                raw_non_nan = ~df_raw[col].isna()
                cleaned_non_nan = ~df[col].isna()
                if not (raw_non_nan == cleaned_non_nan).all():
                    issues_found = True
                    raw_nan_count = df_raw[col].isna().sum()
                    cleaned_nan_count = df[col].isna().sum()
                    nan_mismatches.append({
                        'col': col,
                        'raw_nans': raw_nan_count,
                        'cleaned_nans': cleaned_nan_count
                    })
        if nan_mismatches:
            logger.warning(f"\n{version} version has NaN mismatches:")
            for mismatch in nan_mismatches:
                logger.warning(f"  • {mismatch['col']:<15} Raw NaNs: {mismatch['raw_nans']:,} | "
                             f"Cleaned NaNs: {mismatch['cleaned_nans']:,}")
    
    # Check 3: Value ranges
    logger.info("\n3. Checking value ranges (-1 to 1)...")
    for df, version in [(df_cleaned_1, "Regular"), (df_cleaned_2, "Median-centered")]:
        range_issues = []
        for col in df.columns:
            if col != 'YEAR':
                non_nan_vals = df[col].dropna()
                if len(non_nan_vals) > 0:
                    col_min = non_nan_vals.min()
                    col_max = non_nan_vals.max()
                    if col_min < -1.001 or col_max > 1.001:
                        issues_found = True
                        range_issues.append({
                            'col': col,
                            'min': col_min,
                            'max': col_max,
                            'examples': non_nan_vals.sample(min(5, len(non_nan_vals))).tolist()
                        })
        if range_issues:
            logger.warning(f"\n{version} version has values outside [-1, 1] range:")
            for issue in range_issues:
                logger.warning(f"\n  • {issue['col']}")
                logger.warning(f"    Range: [{issue['min']:.3f}, {issue['max']:.3f}]")
                logger.warning(f"    Examples: {issue['examples']}")
    
    if not issues_found:
        logger.info("\n✓ All quality checks passed successfully!")
    else:
        logger.warning("\n⚠ Issues were found in quality checks. Please review the output above.")

# Add a separator function for cleaner output
def log_section(title: str) -> None:
    """Print a section separator with title."""
    separator = "=" * 80
    logger.info(f"\n{separator}\n{title}\n{separator}")

def prepare_and_cache_datasets():
    """
    Load the raw GSS data and create two cleaned versions using different approaches.
    Cache both versions for future use.
    """
    try:
        log_section("Loading Data")
        logger.info("Loading raw GSS data...")
        df, meta = load_or_cache_data()
        logger.info(f"Raw data loaded successfully")
        logger.info(f"Shape: {df.shape[0]:,} rows × {df.shape[1]:,} columns")
        
        # Filter variables before transformation
        df_filtered = df[VARS_TO_KEEP].copy()
        years = sorted(df['YEAR'].unique())
        logger.info(f"Filtered to {len(VARS_TO_KEEP):,} variables")
        logger.info(f"Years available: {min(years):.0f}-{max(years):.0f}")
        
        # Create both cleaned versions
        log_section("Creating Cleaned Datasets")
        df_cleaned_1, df_cleaned_2 = clean_datasets(df_filtered, time_frame=years)
        logger.info(f"Regular version shape: {df_cleaned_1.shape[0]:,} rows × {df_cleaned_1.shape[1]:,} columns")
        logger.info(f"Median-centered version shape: {df_cleaned_2.shape[0]:,} rows × {df_cleaned_2.shape[1]:,} columns")
        
        # Run quality checks
        log_section("Quality Checks")
        perform_quality_checks(df_filtered, df_cleaned_1, df_cleaned_2)
        
        # Cache the cleaned datasets
        log_section("Caching Results")
        cache_dir = 'datasets/cached_data'
        os.makedirs(cache_dir, exist_ok=True)
        logger.info("Created cache directory")
        
        # Cache version 1
        cache_file_1 = os.path.join(cache_dir, 'gss_cleaned_1.pkl')
        with open(cache_file_1, 'wb') as f:
            pickle.dump((df_cleaned_1, meta), f)
        logger.info(f"Cached regular version to: {cache_file_1}")
        
        # Cache version 2
        cache_file_2 = os.path.join(cache_dir, 'gss_cleaned_2.pkl')
        with open(cache_file_2, 'wb') as f:
            pickle.dump((df_cleaned_2, meta), f)
        logger.info(f"Cached median-centered version to: {cache_file_2}")
        
        log_section("Complete")
        logger.info("Dataset preparation completed successfully!")
        
    except Exception as e:
        logger.error(f"Error preparing datasets: {str(e)}")
        raise

def load_cleaned_datasets():
    """
    Load the cached cleaned datasets if available.
    """
    cache_dir = 'datasets/cached_data'  # Changed from cleaned_data to cached_data
    cache_file_1 = os.path.join(cache_dir, 'gss_cleaned_1.pkl')
    cache_file_2 = os.path.join(cache_dir, 'gss_cleaned_2.pkl')
    
    # Check if cached files exist
    if not os.path.exists(cache_file_1) or not os.path.exists(cache_file_2):
        print("Cached cleaned datasets not found. Creating new ones...")
        return prepare_and_cache_datasets()
    
    print("Loading cached cleaned datasets...")
    
    # Load version 1
    with open(cache_file_1, 'rb') as f:
        df_cleaned_1, meta = pickle.load(f)
    print(f"Loaded cleaned dataset 1: {df_cleaned_1.shape}")
    
    # Load version 2
    with open(cache_file_2, 'rb') as f:
        df_cleaned_2, _ = pickle.load(f)  # We already have meta from version 1
    print(f"Loaded cleaned dataset 2: {df_cleaned_2.shape}")
    
    return df_cleaned_1, df_cleaned_2, meta

if __name__ == "__main__":
    try:
        prepare_and_cache_datasets()
    except Exception as e:
        logger.error(f"Script failed: {str(e)}")
        sys.exit(1) 
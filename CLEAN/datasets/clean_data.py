"""
GSS Data Cleaning Script
-----------------------

This script provides functions to clean and transform GSS data in two ways:
1. Regular version: Variables normalized to [-1, 1] with natural zero points preserved
2. Median-centered version: Variables normalized to [-1, 1] and centered around their median

The script handles:
- Binary variables (Yes/No responses)
- Opinion scales (e.g., 1-7 agreement scales)
- Frequency measures (e.g., 0-8 attendance scales)
- Confidence measures (1-3 scales)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Union
from dataclasses import dataclass
import warnings

@dataclass
class DataConfig:
    """Configuration for data cleaning and transformation."""
    
    # Columns that should not be transformed
    EXCLUDE_COLS = ["YEAR", "BALLOT", "ID"]
    
    # Binary variables (Yes/No, Agree/Disagree)
    BINARY_VARS = [
        'GRASS',  # Should marijuana be made legal
        'CAPPUN',  # Favor or oppose death penalty
        'GUNLAW',  # Favor or oppose gun permits
        'SPKATH',   # Allow anti-religionist to speak
        'LIBATH',   # Allow anti-religionist books in library
        'SPKRAC',   # Allow racist to speak
        'LIBRAC',   # Allow racist books in library
        'SPKCOM',   # Allow communist to speak
        'LIBCOM',   # Allow communist books in library
        'SPKMIL',   # Allow militarist to speak
        'LIBMIL',   # Allow militarist books in library
        'SPKHOMO',  # Allow homosexual to speak
        'LIBHOMO',  # Allow homosexual books in library
        'SPKMSLM',  # Allow anti-American Muslim clergymen to speak
        'LIBMSLM'   # Allow anti-American Muslim books in library
    ]
    BINARY_MAP = {1: 1, 2: -1}  # Yes/Favor=1, No/Oppose=2 mapping
    
    # Variables with 5-point scales
    SCALE_5PT_VARS = [
        'HELPNOT',  # People should help themselves vs govt aid (1-5)
        'HELPBLK',  # Govt help for blacks/minorities (1-5)
        'COLATH',   # Allow anti-religionist to teach (1-5)
        'COLRAC',   # Allow racist to teach (1-5)
        'COLCOM',   # Allow communist to teach (1-5)
        'COLMIL',   # Allow militarist to teach (1-5)
        'COLHOMO',  # Allow homosexual to teach (1-5)
        'COLMSLM',  # Allow anti-American Muslim clergymen to teach (1-5)
    ]
    SCALE_5PT_MAP = {1: 1, 2: 0.5, 3: 0, 4: -0.5, 5: -1}  # 5-point scale mapping
    
    # Variables requiring median centering in the second version
    MEDIAN_VARS = {
        'opinion_scales': [
            'EQWLTH',     # Should govt reduce income differences (1-7)
            'POLVIEWS',   # Political views (1-7)
            'TRUST',      # Can people be trusted (1-3)
            'HELPFUL',    # People are helpful (1-3)
            'FAIR',       # People are fair (1-3)
            'PARTYID'     # Party identification (0-7)
        ],
        'frequency_measures': [
            'ATTEND',    # How often attend religious services (0-8)
            'TVHOURS',   # Hours per day watching TV (0-24)
            'NEWS'       # How often follow news (1-5)
        ],
        'confidence_measures': [
            'CONFINAN',  # Confidence in financial institutions
            'CONBUS',    # Confidence in business
            'CONCLERG',  # Confidence in organized religion
            'CONEDUC',   # Confidence in education
            'CONFED',    # Confidence in executive branch
            'CONLABOR',  # Confidence in organized labor
            'CONPRESS',  # Confidence in press
            'CONMEDIC',  # Confidence in medicine
            'CONTV',     # Confidence in television
            'CONJUDGE',  # Confidence in supreme court
            'CONSCI',    # Confidence in scientific community
            'CONLEGIS',  # Confidence in congress
            'CONARMY'    # Confidence in military
        ],
        'support_measures': [
            'NATSPAC',   # Space exploration program
            'NATENVIR',  # Environmental protection
            'NATHEAL',   # Health care
            'NATCITY',   # Solving problems of big cities
            'NATCRIME',  # Halting rising crime rate
            'NATDRUG',   # Dealing with drug addiction
            'NATEDUC',   # Improving education
            'NATRACE',   # Improving conditions for blacks
            'NATARMS',   # Military/armaments/defense
            'NATAID',    # Foreign aid
            'NATFARE',   # Welfare
            'NATROAD',   # Highways and bridges
            'NATSOC',    # Social security
            'NATMASS',   # Mass transportation
            'NATPARK',   # Parks and recreation
            'NATCHLD',   # Assistance for childcare
            'NATSCI',    # Supporting scientific research
            'NATENRGY'   # Developing alternative energy
        ],
        'binary_measures': [
            'POSTLIFE', 'PRAYER', 'FEPOL', 'ABDEFECT', 'ABNOMORE', 'ABHLTH',
            'ABPOOR', 'ABRAPE', 'ABSINGLE', 'ABANY', 'LETDIE1', 'SUICIDE1',
            'SUICIDE2', 'POLHITOK', 'POLABUSE', 'POLMURDR', 'POLESCAP',
            'POLATTAK', 'RACDIF1', 'RACDIF2', 'RACDIF3', 'RACDIF4'
        ],
        'ordinal_3pt': ['COURTS', 'SEXEDUC', 'DIVLAW', 'PORNLAW', 'RACOPEN'],
        'ordinal_4pt': [
            'AFFRMACT', 'GETAHEAD', 'PREMARSX', 'TEENSEX', 
            'XMARSEX', 'SPANKING', 'FECHLD', 'FEPRESCH', 'FEFAM',
            'RELITEN'  # Strength of religious affiliation (1-4)
        ],
        'ordinal_5pt': ['WRKWAYUP', 'HOMOSEX', 'HELPPOOR', 'MARHOMO']
    }
    
    # Scale mappings for different variable types
    SCALE_MAPS = {
        'binary_2pt': {1: 1, 2: -1},  # Yes/No, Agree/Disagree
        'support_3pt': {1: 1, 2: 0, 3: -1},  # Too much/About right/Too little
        'ordinal_3pt': {1: 1, 2: 0, 3: -1},  # High/Medium/Low
        'ordinal_4pt': {1: 1, 2: 0.333, 3: -0.333, 4: -1},  # Strongly agree to Strongly disagree
        'ordinal_5pt': {1: 1, 2: 0.5, 3: 0, 4: -0.5, 5: -1},  # Strongly agree to Strongly disagree
        'opinion_7pt': {1: 1, 2: 0.667, 3: 0.333, 4: 0, 5: -0.333, 6: -0.667, 7: -1}  # 7-point scales
    }
    
    def __post_init__(self):
        """Initialize additional attributes after instance creation."""
        # Add scale mappings for variables with more values
        self.SCALE_MAPS['party_id'] = {i: 2 * (i/7 - 0.5) for i in range(8)}  # 0-7 scale
        self.SCALE_MAPS['attend'] = {i: 2 * (i/8 - 0.5) for i in range(9)}  # 0-8 scale
        self.SCALE_MAPS['news'] = {i: 2 * ((i-1)/4 - 0.5) for i in range(1, 6)}  # 1-5 scale
        self.SCALE_MAPS['tvhours'] = {i: 2 * (i/24 - 0.5) for i in range(25)}  # 0-24 scale

def normalize_to_range(series: pd.Series, target_min: float = -1, 
                      target_max: float = 1) -> pd.Series:
    """Normalize a series to a target range while preserving NaN values."""
    if not series.notna().any():
        return series
    
    current_min = series[series.notna()].min()
    current_max = series[series.notna()].max()
    
    if current_min == current_max:
        return pd.Series(target_max, index=series.index)
    
    normalized = (series - current_min) / (current_max - current_min)
    return normalized * (target_max - target_min) + target_min

def safe_map(series: pd.Series, mapping: Dict) -> pd.Series:
    """Map values while preserving NaN and warning about unmapped values."""
    unique_vals = series.unique()
    unmapped = [v for v in unique_vals if pd.notna(v) and v not in mapping]
    if unmapped:
        warnings.warn(f"Found unmapped values in {series.name}: {unmapped}")
    return series.map(mapping)

def transform_regular(df: pd.DataFrame, config: DataConfig) -> pd.DataFrame:
    """Transform data using regular normalization (preserving natural zero points)."""
    df_clean = df.copy()
    
    # Ensure YEAR stays as integer
    if 'YEAR' in df_clean.columns:
        df_clean['YEAR'] = df_clean['YEAR'].astype('Int64')
    
    # Transform binary variables
    for var in config.BINARY_VARS:
        if var in df.columns:
            df_clean[var] = safe_map(df[var], config.BINARY_MAP)
    
    # Transform 5-point scale variables
    for var in config.SCALE_5PT_VARS:
        if var in df.columns:
            df_clean[var] = safe_map(df[var], config.SCALE_5PT_MAP)
    
    # Transform variables by category
    for category, vars_list in config.MEDIAN_VARS.items():
        for var in vars_list:
            if var in df.columns:
                if category == 'opinion_scales':
                    if var == 'PARTYID':
                        df_clean[var] = safe_map(df[var], config.SCALE_MAPS['party_id'])
                    elif var in ['TRUST', 'HELPFUL', 'FAIR']:
                        df_clean[var] = safe_map(df[var], config.SCALE_MAPS['ordinal_3pt'])
                    else:
                        df_clean[var] = safe_map(df[var], config.SCALE_MAPS['opinion_7pt'])
                elif category == 'frequency_measures':
                    if var == 'TVHOURS':
                        df_clean[var] = safe_map(df[var], config.SCALE_MAPS['tvhours'])
                    elif var == 'NEWS':
                        df_clean[var] = safe_map(df[var], config.SCALE_MAPS['news'])
                    else:
                        df_clean[var] = safe_map(df[var], config.SCALE_MAPS['attend'])
                elif category == 'confidence_measures':
                    df_clean[var] = safe_map(df[var], config.SCALE_MAPS['ordinal_3pt'])
                elif category == 'support_measures':
                    df_clean[var] = safe_map(df[var], config.SCALE_MAPS['support_3pt'])
                elif category == 'binary_measures':
                    df_clean[var] = safe_map(df[var], config.SCALE_MAPS['binary_2pt'])
                elif category == 'ordinal_3pt':
                    df_clean[var] = safe_map(df[var], config.SCALE_MAPS['ordinal_3pt'])
                elif category == 'ordinal_4pt':
                    df_clean[var] = safe_map(df[var], config.SCALE_MAPS['ordinal_4pt'])
                elif category == 'ordinal_5pt':
                    df_clean[var] = safe_map(df[var], config.SCALE_MAPS['ordinal_5pt'])
    
    return df_clean

def transform_median_centered(df: pd.DataFrame, config: DataConfig) -> pd.DataFrame:
    """Transform data using median centering."""
    df_clean = df.copy()
    
    # Ensure YEAR stays as integer
    if 'YEAR' in df_clean.columns:
        df_clean['YEAR'] = df_clean['YEAR'].astype('Int64')
    
    # Transform binary variables
    for var in config.BINARY_VARS:
        if var in df.columns:
            df_clean[var] = safe_map(df[var], config.BINARY_MAP)
    
    # Transform 5-point scale variables
    for var in config.SCALE_5PT_VARS:
        if var in df.columns:
            df_clean[var] = safe_map(df[var], config.SCALE_5PT_MAP)
    
    # Transform variables by category
    for category, vars_list in config.MEDIAN_VARS.items():
        for var in vars_list:
            if var in df.columns:
                if category == 'opinion_scales':
                    if var == 'PARTYID':
                        df_clean[var] = safe_map(df[var], config.SCALE_MAPS['party_id'])
                    elif var in ['TRUST', 'HELPFUL', 'FAIR']:
                        df_clean[var] = safe_map(df[var], config.SCALE_MAPS['ordinal_3pt'])
                    else:
                        df_clean[var] = safe_map(df[var], config.SCALE_MAPS['opinion_7pt'])
                elif category == 'frequency_measures':
                    if var == 'TVHOURS':
                        df_clean[var] = safe_map(df[var], config.SCALE_MAPS['tvhours'])
                    elif var == 'NEWS':
                        df_clean[var] = safe_map(df[var], config.SCALE_MAPS['news'])
                    else:
                        df_clean[var] = safe_map(df[var], config.SCALE_MAPS['attend'])
                elif category == 'confidence_measures':
                    df_clean[var] = safe_map(df[var], config.SCALE_MAPS['ordinal_3pt'])
                elif category == 'support_measures':
                    df_clean[var] = safe_map(df[var], config.SCALE_MAPS['support_3pt'])
                elif category == 'binary_measures':
                    df_clean[var] = safe_map(df[var], config.SCALE_MAPS['binary_2pt'])
                elif category == 'ordinal_3pt':
                    df_clean[var] = safe_map(df[var], config.SCALE_MAPS['ordinal_3pt'])
                elif category == 'ordinal_4pt':
                    df_clean[var] = safe_map(df[var], config.SCALE_MAPS['ordinal_4pt'])
                elif category == 'ordinal_5pt':
                    df_clean[var] = safe_map(df[var], config.SCALE_MAPS['ordinal_5pt'])
    
    # Center all variables (except YEAR) around their median and normalize to [-1, 1]
    for col in df_clean.columns:
        if col != 'YEAR':
            non_null = df_clean[col].dropna()
            if len(non_null) > 0:
                # Center around median
                median = non_null.median()
                centered = df_clean[col] - median
                
                # Normalize to [-1, 1] range
                max_abs = max(abs(centered[centered.notna()].min()), 
                            abs(centered[centered.notna()].max()))
                if max_abs > 0:
                    df_clean[col] = centered / max_abs
    
    return df_clean

def clean_datasets(df: pd.DataFrame, time_frame: List[int]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Clean and transform GSS data, producing two versions:
    1. Regular version with natural zero points preserved
    2. Median-centered version
    
    Args:
        df: Raw GSS data
        time_frame: List of years to include
    
    Returns:
        Tuple of (regular_version, median_centered_version)
    """
    # Filter to time frame
    df = df[df['YEAR'].isin(time_frame)].copy()
    
    # Initialize config
    config = DataConfig()
    
    # Create both versions
    df_regular = transform_regular(df, config)
    df_median = transform_median_centered(df, config)
    
    return df_regular, df_median 
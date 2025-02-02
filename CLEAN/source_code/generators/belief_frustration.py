import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add the root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.explore_gss import load_or_cache_data
from typing import List, Tuple, Dict

def identify_political_groups(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify Democrats and Republicans in the dataset.
    Uses PARTYID variable where:
    0: Strong Democrat
    1: Not Strong Democrat
    2: Independent, Near Democrat
    4: Independent, Near Republican
    5: Not Strong Republican
    6: Strong Republican
    """
    # Create binary party affiliation
    df = df.copy()
    df['is_democrat'] = df['PARTYID'].isin([0, 1, 2])  # Democrat and Democrat-leaning
    df['is_republican'] = df['PARTYID'].isin([4, 5, 6])  # Republican and Republican-leaning
    return df

def get_belief_variables(df: pd.DataFrame, meta, min_responses: int = 1000) -> List[str]:
    """
    Get list of belief-related variables that have sufficient data.
    
    Args:
        df: DataFrame containing GSS data
        meta: Metadata object
        min_responses: Minimum number of non-null responses required
    
    Returns:
        List of variable names related to beliefs/attitudes
    """
    belief_vars = []
    
    for col in df.columns:
        # Skip demographic and administrative variables
        if col in ['YEAR', 'PARTYID', 'is_democrat', 'is_republican']:
            continue
            
        # Check if variable has sufficient responses
        if df[col].count() < min_responses:
            continue
        
        # Get variable label
        try:
            label = meta.column_labels[meta.column_names.index(col)].lower()
            
            # Include variables that might represent beliefs/attitudes
            belief_keywords = ['think', 'believe', 'opinion', 'view', 'feel', 'agree', 
                             'should', 'favor', 'oppose', 'attitude', 'support']
            
            if any(keyword in label for keyword in belief_keywords):
                belief_vars.append(col)
        except:
            continue
    
    return belief_vars

def calculate_average_beliefs(df: pd.DataFrame, 
                            belief_vars: List[str], 
                            group_col: str,
                            year: int) -> pd.Series:
    """
    Calculate average belief vector for a specific group in a given year.
    """
    group_data = df[df['YEAR'] == year & df[group_col]]
    return group_data[belief_vars].mean()

def calculate_frustration_delta(dem_beliefs: pd.Series, 
                              rep_beliefs: pd.Series) -> float:
    """
    Calculate frustration delta between two belief vectors.
    Higher values indicate more disagreement/frustration.
    """
    # Normalize beliefs to [0,1] range if needed
    # Then calculate Euclidean distance as frustration measure
    return np.sqrt(((dem_beliefs - rep_beliefs) ** 2).sum())

def analyze_belief_frustration(df: pd.DataFrame, 
                             meta,
                             min_year: int = 1972,
                             min_responses: int = 1000) -> Dict:
    """
    Analyze frustration deltas between Democrats and Republicans over time.
    
    Returns:
        Dictionary containing:
        - Time series of frustration deltas for each belief
        - Beliefs with growing frustration
        - Beliefs with stable frustration
    """
    # Prepare data
    df = identify_political_groups(df)
    belief_vars = get_belief_variables(df, meta, min_responses)
    years = sorted(df['YEAR'].unique())
    years = [y for y in years if y >= min_year]
    
    # Calculate frustration deltas over time
    frustration_deltas = {}
    for var in belief_vars:
        deltas = []
        for year in years:
            year_data = df[df['YEAR'] == year]
            
            # Skip if insufficient data for either group
            if (year_data['is_democrat'].sum() < 30 or 
                year_data['is_republican'].sum() < 30):
                continue
            
            # Calculate average beliefs
            dem_belief = year_data[year_data['is_democrat']][var].mean()
            rep_belief = year_data[year_data['is_republican']][var].mean()
            
            # Skip if missing data
            if pd.isna(dem_belief) or pd.isna(rep_belief):
                continue
                
            deltas.append({
                'year': year,
                'delta': abs(dem_belief - rep_belief),
                'dem_belief': dem_belief,
                'rep_belief': rep_belief
            })
        
        if deltas:  # Only include variables with data
            frustration_deltas[var] = deltas
    
    # Analyze trends
    trend_analysis = {}
    for var, deltas in frustration_deltas.items():
        if len(deltas) < 2:
            continue
            
        # Calculate trend
        years = [d['year'] for d in deltas]
        values = [d['delta'] for d in deltas]
        
        if len(years) > 1:
            slope = np.polyfit(years, values, 1)[0]
            variance = np.var(values)
            
            trend_analysis[var] = {
                'slope': slope,
                'variance': variance,
                'deltas': deltas,
                'label': meta.column_labels[meta.column_names.index(var)]
            }
    
    # Identify growing vs stable frustration
    sorted_trends = sorted(trend_analysis.items(), key=lambda x: x[1]['slope'], reverse=True)
    
    n_top = min(10, len(sorted_trends))
    growing_frustration = dict(sorted_trends[:n_top])
    stable_frustration = dict(sorted(
        [(k, v) for k, v in trend_analysis.items() if abs(v['slope']) < 0.01],
        key=lambda x: x[1]['variance']
    )[:n_top])
    
    return {
        'all_trends': trend_analysis,
        'growing_frustration': growing_frustration,
        'stable_frustration': stable_frustration
    }

def plot_frustration_trends(results: Dict):
    """
    Create visualizations of frustration trends.
    """
    # Plot growing frustration trends
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 1, 1)
    for var, data in results['growing_frustration'].items():
        years = [d['year'] for d in data['deltas']]
        values = [d['delta'] for d in data['deltas']]
        plt.plot(years, values, 'o-', label=f"{var}: {data['label'][:50]}...")
    
    plt.title("Beliefs with Growing Democrat-Republican Frustration")
    plt.xlabel("Year")
    plt.ylabel("Frustration Delta")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Plot stable frustration trends
    plt.subplot(2, 1, 2)
    for var, data in results['stable_frustration'].items():
        years = [d['year'] for d in data['deltas']]
        values = [d['delta'] for d in data['deltas']]
        plt.plot(years, values, 'o-', label=f"{var}: {data['label'][:50]}...")
    
    plt.title("Beliefs with Stable Democrat-Republican Frustration")
    plt.xlabel("Year")
    plt.ylabel("Frustration Delta")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Load data
    print("Loading GSS data...")
    df, meta = load_or_cache_data()
    
    # Analyze frustration
    print("\nAnalyzing belief frustration between Democrats and Republicans...")
    results = analyze_belief_frustration(df, meta)
    
    # Display results
    print("\nBeliefs with growing frustration:")
    print("-" * 80)
    for var, data in results['growing_frustration'].items():
        print(f"\n{var}: {data['label']}")
        print(f"Trend slope: {data['slope']:.4f}")
        print(f"Recent frustration delta: {data['deltas'][-1]['delta']:.3f}")
    
    print("\nBeliefs with stable frustration:")
    print("-" * 80)
    for var, data in results['stable_frustration'].items():
        print(f"\n{var}: {data['label']}")
        print(f"Trend slope: {data['slope']:.4f}")
        print(f"Average frustration delta: {np.mean([d['delta'] for d in data['deltas']]):.3f}")
    
    # Plot trends
    plot_frustration_trends(results) 
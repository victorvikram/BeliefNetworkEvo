import pandas as pd
from data_loader import load_and_filter_data
from analyzers.overlap_analyzer import calculate_overlap_matrix, plot_overlap_matrix
from source_code.corr_make_network import calculate_correlation_matrix
from visualizers.network_visualizer import create_network_data, generate_html_visualization
from analyzers.centrality_analyzer import calculate_centrality_measures, create_centrality_dataframes
import matplotlib.pyplot as plt
from typing import Tuple, List
from multiprocessing import Pool
from functools import partial

def process_time_period(data_file: str, time_window: int, year: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Process a single time period and return centrality DataFrames."""
    print(f"\nProcessing time period {year}-{year+time_window}")
    
    # Load and filter data
    df, meta = load_and_filter_data(data_file, year, year+time_window)

    # Calculate correlation matrix
    correlation_matrix = calculate_correlation_matrix(df, method='spearman', edge_suppression='square', partial=True)

    # Hard filter the matrix, setting all correlations below 0.05 to 0
    #correlation_matrix = correlation_matrix.applymap(lambda x: 0 if x < 0.05 else x)

    # Calculate centrality measures
    bc, degrees, edge_weight_sums, filtered_vars = calculate_centrality_measures(correlation_matrix)
    
    # Create formatted DataFrames for centrality measures
    betweenness_df, degree_df, strength_df = create_centrality_dataframes(
        correlation_matrix, bc, degrees, edge_weight_sums
    )
 

    # Create and save network visualization for the final time period        # Create network data
    nodes, edges = create_network_data(correlation_matrix)
    
    # Generate HTML visualization
    output_file = f'correlation_network_{year}-{year + time_window - 1}.html'
    generate_html_visualization(nodes, edges, output_file)
    print(f"Network visualization saved as {output_file}")

    return year, betweenness_df, degree_df, strength_df

def get_centrality_data(df_list: pd.DataFrame, year: int) -> pd.DataFrame:
    """Add year column and prepare DataFrame for plotting."""
    df = df_list.reset_index()
    df['Year'] = year
    return df

def print_centrality_rankings(all_data: pd.DataFrame, centrality_type: str, start_year: int, end_year: int, time_step: int):
    """Print top 6 centrality measures for each year."""
    # Group by year and get top 6 for each year
    for year in range(start_year, end_year, time_step):
        year_data = all_data[all_data['Year'] == year]
        top_6 = year_data.nlargest(6, centrality_type)
        
        print(f"\nTop 6 {centrality_type} for year {year}:")
        for idx, row in top_6.iterrows():
            print(f"{row['Variable']}: {row[centrality_type]:.4f}")

def main():
    # Configuration
    data_file = 'datasets/cached_data/gss_cleaned_1.pkl'
    start_year = 2000
    time_step = 2
    time_window = 4
    end_year = 2012
    
    # Choose centrality type: 'Betweenness Centrality', 'Degree', or 'Total Correlation'
    centrality_type = 'Betweenness Centrality'
    
    print("Starting analysis...")
    print(f"Data file: {data_file}")
    print(f"Time range: {start_year}-{end_year}")
    print(f"Analyzing {centrality_type}")
    
    try:
        # Create a partial function with fixed arguments
        process_func = partial(process_time_period, data_file, time_window)
        
        # Create list of years to process
        years = list(range(start_year, end_year, time_step))
        
        # Process time periods in parallel
        with Pool(processes=4) as pool:  # Limit to 4 cores
            results = pool.map(process_func, years)
        
        # Organize results
        processed_results = []
        for year, betweenness_df, degree_df, strength_df in results:
            # Select appropriate DataFrame based on centrality type
            if centrality_type == 'Betweenness Centrality':
                df = betweenness_df
            elif centrality_type == 'Degree':
                df = degree_df
            else:  # Total Correlation
                df = strength_df
            
            # Prepare data for printing
            df = get_centrality_data(df, year)
            processed_results.append(df)

        print(f"\nPrinting {centrality_type} rankings over time...")
        
        # Combine all results
        all_data = pd.concat(processed_results, ignore_index=True)
        
        # Print the rankings
        print_centrality_rankings(all_data, centrality_type, start_year, end_year, time_step)
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        print(traceback.format_exc())


if __name__ == "__main__":
    main()
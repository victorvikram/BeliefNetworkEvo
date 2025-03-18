import sys
import os
from pathlib import Path
# Get the project root directory
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from CLEAN.source_code.loaders.clean_raw_data import clean_datasets
from source_code.generators.corr_make_network import CorrelationMethod, EdgeSuppressionMethod
from source_code.visualizers.temporal_network_visualizer import (
    generate_temporal_html_visualization,
    test_temporal_correlation_matrix
)

def test_both_approaches():
    """
    Test both direct function call and the test example from the original file.
    This will help identify any differences in behavior.
    """
    cleaned_df = clean_datasets()
    
    print("\nTest 1: Running original test function...")
    try:
        test_temporal_correlation_matrix()
        print("✓ Original test function succeeded")
    except Exception as e:
        print(f"✗ Original test function failed: {str(e)}")
    
    print("\nTest 2: Running with identical parameters via direct call...")
    try:
        generate_temporal_html_visualization(
            cleaned_df,
            nodes_to_highlight=['POLVIEWS'],
            time_window_length=4,
            start_year=2000,
            end_year=2010,
            step_size=2,
            method=CorrelationMethod.PEARSON,
            partial=True,
            edge_suppression=EdgeSuppressionMethod.REGULARIZATION,
            suppression_params={'regularization': 0.2},
            output_path='direct_call_network.html'
        )
        print("✓ Direct function call succeeded")
    except Exception as e:
        print(f"✗ Direct function call failed: {str(e)}")

    # Print the cleaned DataFrame info to check for any differences
    print("\nDataFrame Info:")
    print(cleaned_df.info())

if __name__ == "__main__":
    test_both_approaches() 
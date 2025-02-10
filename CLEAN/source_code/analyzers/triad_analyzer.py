import numpy as np  # Ensure numpy is imported
import pandas as pd  # Ensure pandas is imported
from typing import Tuple, Union, Dict, Any

def count_triads(
    correlation_matrix: pd.DataFrame, 
    return_names: bool = False,
    return_sums: bool = False
) -> Union[Tuple[int, int, int], Dict[str, Any]]:
    """
    Efficiently count and analyze triads in the correlation matrix.
    
    Args:
        correlation_matrix: A square pandas DataFrame representing the correlation matrix.
        return_names: If True, return triplet names for positive/negative triads.
        return_sums: If True, return absolute edge sums for triads.
        
    Returns:
        Triad count analysis with optional additional details
    """
    # Convert to numpy array for faster computations
    corr_array = correlation_matrix.to_numpy()
    n = corr_array.shape[0]
    
    # Preallocate containers
    total_triads = 0
    positive_triads = 0
    negative_triads = 0
    
    # Detailed tracking if return_names or return_sums is True
    positive_triad_details = [] if return_names or return_sums else None
    negative_triad_details = [] if return_names or return_sums else None
    positive_triad_sums = [] if return_sums else None
    negative_triad_sums = [] if return_sums else None
    
    # Use numpy advanced indexing to reduce nested loops
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                # Extract triad correlations
                triad_edges = [
                    corr_array[i, j],
                    corr_array[i, k],
                    corr_array[j, k]
                ]
                
                total_triads += 1
                
                # Calculate the product of the triad edges
                product = triad_edges[0] * triad_edges[1] * triad_edges[2]
                
                # Count and track triads
                if product > 0:
                    positive_triads += 1
                    if return_names or return_sums:
                        triad_nodes = (
                            correlation_matrix.index[i],
                            correlation_matrix.index[j],
                            correlation_matrix.index[k]
                        )
                        if return_names:
                            positive_triad_details.append(triad_nodes)
                        if return_sums:
                            positive_triad_sums.append(sum(abs(edge) for edge in triad_edges))
                elif product < 0:
                    negative_triads += 1
                    if return_names or return_sums:
                        triad_nodes = (
                            correlation_matrix.index[i],
                            correlation_matrix.index[j],
                            correlation_matrix.index[k]
                        )
                        if return_names:
                            negative_triad_details.append(triad_nodes)
                        if return_sums:
                            negative_triad_sums.append(sum(abs(edge) for edge in triad_edges))
    
    # Construct return value based on flags
    if return_names and return_sums:
        return {
            'total_triads': total_triads,
            'positive_triads': positive_triads,
            'negative_triads': negative_triads,
            'positive_triad_nodes': positive_triad_details,
            'negative_triad_nodes': negative_triad_details,
            'positive_triad_sums': positive_triad_sums,
            'negative_triad_sums': negative_triad_sums
        }
    elif return_names:
        return {
            'total_triads': total_triads,
            'positive_triads': positive_triads,
            'negative_triads': negative_triads,
            'positive_triad_nodes': positive_triad_details,
            'negative_triad_nodes': negative_triad_details
        }
    elif return_sums:
        return {
            'total_triads': total_triads,
            'positive_triads': positive_triads,
            'negative_triads': negative_triads,
            'positive_triad_sums': positive_triad_sums,
            'negative_triad_sums': negative_triad_sums
        }
    
    return {'total_triads': total_triads, 'positive_triads': positive_triads, 'negative_triads': negative_triads}
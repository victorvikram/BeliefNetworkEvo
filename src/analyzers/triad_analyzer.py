import numpy as np  # Ensure numpy is imported
import pandas as pd  # Ensure pandas is imported
from typing import Tuple, Union, Dict, Any

def count_triads(
    correlation_matrix: pd.DataFrame, 
    return_names: bool = False,
    return_sums: bool = False
) -> Union[Tuple[int, int, int], Dict[str, Any]]:
    """
    Count and analyze triads in a correlation matrix, identifying positive and negative triads.
    
    A triad is a set of three nodes and their connecting edges in the correlation network.
    The sign of a triad is determined by the product of its three edge correlations:
    - Positive triad: product > 0 (even number of negative edges: 0 or 2)
    - Negative triad: product < 0 (odd number of negative edges: 1 or 3)
    
    Args:
        correlation_matrix: A square pandas DataFrame representing the correlation matrix.
            Must be symmetric with values between -1 and 1, where the index and columns
            represent the same nodes.
        return_names: If True, includes the node names for each triad in the results.
        return_sums: If True, includes the sum of absolute correlation values for each triad.
        
    Returns:
        If return_names=False and return_sums=False:
            Dict with keys:
            - 'total_triads': Total number of triads found
            - 'positive_triads': Number of triads with positive product
            - 'negative_triads': Number of triads with negative product
            
        If return_names=True:
            Also includes:
            - 'positive_triad_nodes': List of (node1, node2, node3) tuples for positive triads
            - 'negative_triad_nodes': List of (node1, node2, node3) tuples for negative triads
            
        If return_sums=True:
            Also includes:
            - 'positive_triad_sums': List of absolute correlation sums for positive triads
            - 'negative_triad_sums': List of absolute correlation sums for negative triads
            
    Examples:
        >>> # Create a sample correlation matrix
        >>> data = [[1.0, 0.5, -0.3],
        ...        [0.5, 1.0, 0.6],
        ...        [-0.3, 0.6, 1.0]]
        >>> corr_matrix = pd.DataFrame(data, 
        ...                           index=['A', 'B', 'C'],
        ...                           columns=['A', 'B', 'C'])
        >>> 
        >>> # Get basic triad counts
        >>> result = count_triads(corr_matrix)
        >>> print(f"Total triads: {result['total_triads']}")
        >>> print(f"Positive triads: {result['positive_triads']}")
        >>> print(f"Negative triads: {result['negative_triads']}")
        >>>
        >>> # Get detailed triad information
        >>> detailed = count_triads(corr_matrix, return_names=True, return_sums=True)
        >>> for nodes, sum_val in zip(detailed['negative_triad_nodes'], 
        ...                          detailed['negative_triad_sums']):
        ...     print(f"Negative triad: {nodes}, Sum of correlations: {sum_val:.2f}")
    
    Notes:
        - The function processes the upper triangle of the correlation matrix to avoid
          counting the same triad multiple times.
        - Edge correlations are considered in their absolute form when calculating sums.
        - The function assumes the correlation matrix is symmetric and contains valid
          correlation values (-1 to 1).
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
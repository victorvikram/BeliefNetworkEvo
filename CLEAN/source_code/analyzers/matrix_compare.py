import numpy as np
import pandas as pd
import networkx as nx
from scipy.spatial.distance import euclidean
from typing import Dict, List, Tuple, Set, Any, Optional, Union

def compare_matrices(
    matrix1: pd.DataFrame, 
    matrix2: pd.DataFrame, 
    threshold: float = 0.0, 
    triangle_threshold: float = 0.0
) -> Dict[str, Any]:
    """
    Calculate comparison metrics between two correlation matrices (belief networks).
    
    Parameters
    ----------
    matrix1, matrix2 : pd.DataFrame
        Correlation matrices to compare, with the same index and columns representing nodes
    threshold : float, default 0.0
        Threshold for considering an edge to exist
    weight_tolerance : float, default 0.0
        Tolerance for considering edges with slightly different weights as the same edge
    
    Returns
    -------
    Dict[str, Any]
        Dictionary containing comparison metrics:
        - num_edges: Number of edges in each network
        - delta_edges: Difference in number of edges
        - density: Network density for each network
        - delta_density: Difference in network density
        - avg_degree: Average node degree for each network
        - delta_avg_degree: Difference in average node degree
        - avg_weight_sum: Average sum of edge weights per node for each network
        - delta_avg_weight_sum: Difference in average weight sum
        - clustering_coefficient: Clustering coefficient for each network
        - delta_clustering_coefficient: Difference in clustering coefficient
        - calc_num_triangles: Number of triangles in each network
        - delta_num_triangles: Difference in number of triangles
        - taxi_cab_distance: 1-norm (taxi-cab) distance between matrices
        - euclidean_distance: 2-norm (Euclidean) distance between matrices
        - pearson_correlation: Pearson correlation coefficient between matrices (-1 to 1)
        - spearman_correlation: Spearman rank correlation between matrices (-1 to 1)
        - Spectral gap: Spectral gap of the networks (the difference between the first and second eigenvalues of the adjacency matrix. 
            A larger gap suggests more well-defined community structure. 
        - num_communities: Number of communities in the network
        """
    # Ensure both matrices have the same variables/nodes
    common_vars = matrix1.columns.intersection(matrix2.columns)
    if len(common_vars) < len(matrix1.columns) or len(common_vars) < len(matrix2.columns):
        print(f"Warning: Matrices have different variables. Using only {len(common_vars)} common variables.")
    
    matrix1 = matrix1.loc[common_vars, common_vars].copy()
    matrix2 = matrix2.loc[common_vars, common_vars].copy()
    
    # Create NetworkX graphs (undirected)
    G1 = nx.Graph()
    G2 = nx.Graph()
    G1.add_nodes_from(common_vars)
    G2.add_nodes_from(common_vars)
    
    # Add edges based on threshold
    edges1 = []
    edges2 = []
    edge_weights1 = {}
    edge_weights2 = {}
    
    # Only iterate through upper triangle to avoid double counting in undirected networks
    for i, node1 in enumerate(common_vars):
        for j, node2 in enumerate(common_vars):
            if i < j:  # Only consider upper triangle since the networks are undirected
                val1 = matrix1.loc[node1, node2]
                val2 = matrix2.loc[node1, node2]
                
                # Check if edge exists in network 1
                if abs(val1) > threshold:
                    G1.add_edge(node1, node2, weight=val1)
                    edges1.append((node1, node2))
                    # Store canonical edge representation (sorted tuple) to ensure consistent lookup
                    edge_weights1[(node1, node2)] = abs(val1)
                
                # Check if edge exists in network 2
                if abs(val2) > threshold:
                    G2.add_edge(node1, node2, weight=val2)
                    edges2.append((node1, node2))
                    edge_weights2[(node1, node2)] = abs(val2)
    
    # Calculate metrics
    num_edges1 = G1.number_of_edges()
    num_edges2 = G2.number_of_edges()
    
    density1 = nx.density(G1)
    density2 = nx.density(G2)
    
    avg_degree1 = sum(dict(G1.degree()).values()) / G1.number_of_nodes() if G1.number_of_nodes() > 0 else 0
    avg_degree2 = sum(dict(G2.degree()).values()) / G2.number_of_nodes() if G2.number_of_nodes() > 0 else 0
    
    # Calculate average summed edge weight per node
    node_weight_sums1 = {node: 0.0 for node in common_vars}
    node_weight_sums2 = {node: 0.0 for node in common_vars}
    
    for (node1, node2), weight in edge_weights1.items():
        node_weight_sums1[node1] += weight
        node_weight_sums1[node2] += weight
    
    for (node1, node2), weight in edge_weights2.items():
        node_weight_sums2[node1] += weight
        node_weight_sums2[node2] += weight
    
    avg_weight_sum1 = sum(node_weight_sums1.values()) / len(common_vars) if len(common_vars) > 0 else 0
    avg_weight_sum2 = sum(node_weight_sums2.values()) / len(common_vars) if len(common_vars) > 0 else 0
    
    # Calculate clustering coefficient
    clustering_coefficient1 = nx.average_clustering(G1)
    clustering_coefficient2 = nx.average_clustering(G2)

    # Calculate number of triangles (only consider triangles where all edges have weight greater than triangle_threshold)
    # Calculate number of triangles where all edges exceed triangle_threshold
    
    # Function to count triangles with all edges above threshold
    def count_strong_triangles(G, threshold):
        triangles_count = 0
        # Get all nodes in the graph
        nodes = list(G.nodes())
        
        # Check all possible triplets of nodes
        for i in range(len(nodes)):
            for j in range(i+1, len(nodes)):
                for k in range(j+1, len(nodes)):
                    node1, node2, node3 = nodes[i], nodes[j], nodes[k]
                    
                    # Check if all three edges exist and have weights above threshold
                    if (G.has_edge(node1, node2) and G.has_edge(node2, node3) and G.has_edge(node1, node3) and
                        abs(G[node1][node2]['weight']) > threshold and
                        abs(G[node2][node3]['weight']) > threshold and
                        abs(G[node1][node3]['weight']) > threshold):
                        triangles_count += 1
        
        return triangles_count
    
    # Count triangles in both networks
    triangles_count1 = count_strong_triangles(G1, triangle_threshold)
    triangles_count2 = count_strong_triangles(G2, triangle_threshold)


    # Flatten matrices for distance calculations - convert to numpy arrays first
    mat1_np = matrix1.values
    mat2_np = matrix2.values
    
    # Get upper triangular elements (excluding diagonal) to avoid double counting
    indices = np.triu_indices(len(common_vars), k=1)
    upper1 = mat1_np[indices]
    upper2 = mat2_np[indices]

    # Calculate the 1-norm (taxi-cab) distance
    taxi_dist = np.sum(np.abs(upper1 - upper2))

    # Calculate the 2-norm (Euclidean) distance
    euc_dist = np.linalg.norm(upper1 - upper2)

    # Calculate various correlation measures between matrices
    # Use only the upper triangular part for correlation calculations to avoid duplicates
    pearson_corr = np.corrcoef(upper1, upper2)[0, 1]  # Pearson correlation
    spearman_corr = pd.Series(upper1).corr(pd.Series(upper2), method='spearman')  # Spearman rank correlation

    # Calculate the spectral gap for both networks - with robust error handling
    spectral_gap1 = compute_spectral_gap(G1)
    spectral_gap2 = compute_spectral_gap(G2)
    
    # Calculate delta spectral gap, handling the case where one or both might be None
    if spectral_gap1 is not None and spectral_gap2 is not None:
        delta_spectral_gap = spectral_gap1 - spectral_gap2
    else:
        delta_spectral_gap = None

    # Calculate number of communities (louvain_communities)
    communities1 = list(nx.community.louvain_communities(G1))
    communities2 = list(nx.community.louvain_communities(G2))


    return {
        "num_edges": {"matrix1": num_edges1, "matrix2": num_edges2},
        "delta_edges": num_edges1 - num_edges2,
        "density": {"matrix1": density1, "matrix2": density2},
        "delta_density": density1 - density2,
        "avg_degree": {"matrix1": avg_degree1, "matrix2": avg_degree2},
        "delta_avg_degree": avg_degree1 - avg_degree2,
        "avg_weight_sum": {"matrix1": avg_weight_sum1, "matrix2": avg_weight_sum2},
        "delta_avg_weight_sum": avg_weight_sum1 - avg_weight_sum2,
        "clustering_coefficient": {"matrix1": clustering_coefficient1, "matrix2": clustering_coefficient2},
        "delta_clustering_coefficient": clustering_coefficient1 - clustering_coefficient2,
        "calc_num_triangles": {"matrix1": triangles_count1, "matrix2": triangles_count2},
        "delta_num_triangles": triangles_count1 - triangles_count2,
        "euclidean_distance": euc_dist,
        "taxi_cab_distance": taxi_dist,
        "pearson_correlation": pearson_corr,
        "spearman_correlation": spearman_corr,
        "spectral_gap": {"matrix1": spectral_gap1, "matrix2": spectral_gap2},
        "delta_spectral_gap": delta_spectral_gap,
        "num_communities": {"matrix1": len(communities1), "matrix2": len(communities2)},
        "delta_num_communities": len(communities1) - len(communities2)
    }

def compute_spectral_gap(G: nx.Graph) -> Optional[float]:
    """
    Only computes the spectral gap for the largest connected component if the graph is disconnected.
    
    Parameters
    ----------
    G : nx.Graph
        NetworkX graph
        
    Returns
    -------
    float
        The spectral gap of the largest connected component of the graph
    """
    
    # Get the largest connected component
    connected_components = list(nx.connected_components(G))
    largest_cc = max(connected_components, key=len)
    G = G.subgraph(largest_cc).copy()    
       
    # Compute the spectral gap
    eig_vals = nx.laplacian_spectrum(G)
    return float(eig_vals[1] - eig_vals[0])
    

def find_differential_edges(
    matrix1: pd.DataFrame, 
    matrix2: pd.DataFrame, 
    threshold: float = 0.0, 
    top_n: Optional[int] = None
) -> Tuple[List[Tuple[str, str, float]], List[Tuple[str, str, float]]]:
    """
    Find edges that differ the most between two correlation matrices.
    
    Parameters
    ----------
    matrix1, matrix2 : pd.DataFrame
        Correlation matrices to compare
    threshold : float, default 0.0
        Threshold for considering an edge to exist
    top_n : Optional[int], default None
        If specified, return only the top N most different edges
    
    Returns
    -------
    Tuple[List[Tuple[str, str, float]], List[Tuple[str, str, float]]]
        Two lists of tuples (node1, node2, difference):
        - First list: Edges stronger in matrix1 than matrix2
        - Second list: Edges stronger in matrix2 than matrix1
    """
    # Ensure both matrices have the same variables/nodes
    common_vars = matrix1.columns.intersection(matrix2.columns)
    matrix1 = matrix1.loc[common_vars, common_vars].copy()
    matrix2 = matrix2.loc[common_vars, common_vars].copy()
    
    # Calculate the difference matrix
    diff_matrix = matrix1 - matrix2
    
    # Lists to store differential edges
    stronger_in_matrix1 = []
    stronger_in_matrix2 = []
    
    # Find differential edges - only look at upper triangle since networks are undirected
    for i, node1 in enumerate(common_vars):
        for j, node2 in enumerate(common_vars):
            if i < j:  # Only consider upper triangle
                diff_val = diff_matrix.loc[node1, node2]
                
                val1 = matrix1.loc[node1, node2]
                val2 = matrix2.loc[node1, node2]
                
                # Check if either edge exists based on threshold
                edge1_exists = abs(val1) > threshold
                edge2_exists = abs(val2) > threshold
                
                if edge1_exists or edge2_exists:
                    # Compare absolute strengths
                    if abs(val1) > abs(val2):
                        stronger_in_matrix1.append((node1, node2, abs(val1) - abs(val2)))
                    elif abs(val2) > abs(val1):
                        stronger_in_matrix2.append((node1, node2, abs(val2) - abs(val1)))
    
    # Sort by difference magnitude
    stronger_in_matrix1.sort(key=lambda x: x[2], reverse=True)
    stronger_in_matrix2.sort(key=lambda x: x[2], reverse=True)
    
    # Return top N if specified
    if top_n is not None:
        stronger_in_matrix1 = stronger_in_matrix1[:top_n]
        stronger_in_matrix2 = stronger_in_matrix2[:top_n]
    
    return stronger_in_matrix1, stronger_in_matrix2

def print_comparison_results(
    results: Dict[str, Any],
    title1: str = "Matrix 1",
    title2: str = "Matrix 2",
    precision: int = 4
) -> None:
    """
    Print the results of a matrix comparison in a well-formatted table.
    
    Parameters
    ----------
    results : Dict[str, Any]
        Dictionary of comparison metrics as returned by compare_matrices
    title1, title2 : str, default "Matrix 1" and "Matrix 2"
        Names to use for the two matrices being compared
    precision : int, default 4
        Number of decimal places to display for floating point values
    """
    print(f"\n{'=' * 60}")
    print(f"NETWORK COMPARISON: {title1} vs. {title2}")
    print(f"{'=' * 60}")
    
    # Network size metrics
    print(f"\n{'NETWORK SIZE METRICS':-^60}")
    print(f"Number of edges:  {title1}: {results['num_edges']['matrix1']}, {title2}: {results['num_edges']['matrix2']}")
    print(f"Delta edges:      {results['delta_edges']} ({'+' if results['delta_edges'] > 0 else ''}{results['delta_edges'] / max(1, results['num_edges']['matrix2']) * 100:.2f}%)")
    print(f"Network density:  {title1}: {results['density']['matrix1']:.{precision}f}, {title2}: {results['density']['matrix2']:.{precision}f}")
    print(f"Delta density:    {results['delta_density']:.{precision}f}")
    print(f"Average degree:   {title1}: {results['avg_degree']['matrix1']:.{precision}f}, {title2}: {results['avg_degree']['matrix2']:.{precision}f}")
    print(f"Delta avg degree: {results['delta_avg_degree']:.{precision}f}")
    print(f"Avg weight sum:   {title1}: {results['avg_weight_sum']['matrix1']:.{precision}f}, {title2}: {results['avg_weight_sum']['matrix2']:.{precision}f}")
    print(f"Delta avg weight: {results['delta_avg_weight_sum']:.{precision}f}")
    
    # Similarity metrics
    print(f"\n{'SIMILARITY METRICS':-^60}")
    print(f"Euclidean distance:     {results['euclidean_distance']:.{precision}f}")
    print(f"Taxi-cab distance:      {results['taxi_cab_distance']:.{precision}f}")
    print(f"Pearson correlation:    {results['pearson_correlation']:.{precision}f} (-1 to +1, higher is more similar)")
    print(f"Spearman correlation:   {results['spearman_correlation']:.{precision}f} (-1 to +1, higher is more similar)")
    
    print(f"{'=' * 60}\n")
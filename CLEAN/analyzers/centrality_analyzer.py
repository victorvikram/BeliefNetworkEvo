"""
This module calculates and analyzes various centrality measures for weighted networks,
particularly useful for correlation or similarity networks. It implements three key
centrality measures:

1. Betweenness Centrality: Measures how often a node acts as a bridge between other nodes
2. Degree Centrality: Counts the number of connections each node has
3. Strength Centrality: Sums the weights of all connections for each node

The module is particularly useful for identifying influential variables in correlation networks
and understanding the network's structure.

Note: Metadata variables (YEAR, BALLOT, ID) are excluded from correlation calculations
upstream, but remain available in the original DataFrame for analysis purposes.
The correlation matrices passed to this module will not contain these metadata columns.

Example:
    >>> corr_matrix = pd.DataFrame([[1,0.5,0.3], [0.5,1,0.2], [0.3,0.2,1]])
    >>> bc, degrees, strengths = calculate_centrality_measures(corr_matrix)
    >>> bc_df, deg_df, str_df = create_centrality_dataframes(corr_matrix, bc, degrees, strengths)
"""

import numpy as np
import pandas as pd
import heapq

def betweenness_centrality_weighted(adj_matrix: np.ndarray) -> np.ndarray:
    """
    Calculate the betweenness centrality for each node in a weighted graph.
    
    Args:
        adj_matrix: Square numpy array representing the weighted adjacency matrix
        
    Returns:
        numpy.ndarray: Array of betweenness centrality values
    """
    n = adj_matrix.shape[0]
    bc = np.zeros(n, dtype=float)

    print(f"Processing {n} nodes...")

    for s in range(n):
        # Initialization
        S = []
        P = [[] for _ in range(n)]
        sigma = np.zeros(n, dtype=float)
        sigma[s] = 1.0
        dist = np.full(n, np.inf)
        dist[s] = 0.0
        queue = []
        heapq.heappush(queue, (0.0, s))

        while queue:
            d, v = heapq.heappop(queue)
            if d > dist[v]:
                continue
            S.append(v)
            for w in range(n):
                weight = adj_matrix[v, w]
                if weight != 0:
                    new_dist = dist[v] + weight
                    if new_dist < dist[w]:
                        dist[w] = new_dist
                        sigma[w] = sigma[v]
                        heapq.heappush(queue, (new_dist, w))
                        P[w] = [v]
                    elif np.isclose(new_dist, dist[w]):
                        sigma[w] += sigma[v]
                        P[w].append(v)

        delta = np.zeros(n, dtype=float)
        while S:
            w = S.pop()
            for v in P[w]:
                delta[v] += (sigma[v] / sigma[w]) * (1 + delta[w])
            if w != s:
                bc[w] += delta[w]
        #print("ite")        

    return bc / 2.0

def calculate_centrality_measures(correlation_matrix: pd.DataFrame) -> tuple:
    """
    Calculate various centrality measures for the network.
    
    Args:
        correlation_matrix: Correlation matrix DataFrame (metadata already filtered)
        
    Returns:
        tuple: (betweenness_centrality, degrees, edge_weight_sums, variables)
    """
    print(f"Processing network with {len(correlation_matrix)} nodes...")
    
    # Calculate betweenness centrality
    bc = betweenness_centrality_weighted(correlation_matrix.to_numpy())
    
    # Calculate degree (excluding self-loops)
    degrees = np.sum(correlation_matrix != 0, axis=0) - 1
    
    # Calculate edge weight sums (excluding self-loops)
    edge_weight_sums = np.sum(correlation_matrix, axis=0) - 1
    
    return bc, degrees, edge_weight_sums, correlation_matrix.index

def create_centrality_dataframes(correlation_matrix: pd.DataFrame, bc: np.ndarray, 
                               degrees: np.ndarray, edge_weight_sums: np.ndarray,
                               top_n: int = 10) -> tuple:
    """
    Create formatted DataFrames for centrality measures.
    
    Args:
        correlation_matrix: Correlation matrix DataFrame (metadata already filtered)
        bc: Betweenness centrality values
        degrees: Degree values
        edge_weight_sums: Edge weight sum values
        top_n: Number of top entries to include
        
    Returns:
        tuple: (betweenness_df, degree_df, strength_df)
    """
    # Create DataFrames for each measure
    betweenness_df = pd.DataFrame({
        'Variable': correlation_matrix.columns,
        'Betweenness Centrality': bc
    }).sort_values('Betweenness Centrality', ascending=False).head(top_n)

    degree_df = pd.DataFrame({
        'Variable': correlation_matrix.columns,
        'Degree': degrees
    }).sort_values('Degree', ascending=False).head(top_n)

    strength_df = pd.DataFrame({
        'Variable': correlation_matrix.columns,
        'Total Correlation': edge_weight_sums
    }).sort_values('Total Correlation', ascending=False).head(top_n)

    # Set Variable as index for plotting
    betweenness_df.set_index('Variable', inplace=True)
    degree_df.set_index('Variable', inplace=True)
    strength_df.set_index('Variable', inplace=True)

    return betweenness_df, degree_df, strength_df 
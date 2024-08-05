import networkx as nx
import numpy as np
import pandas as pd

def create_graph(adj_matrix, node_names=None):
    if isinstance(adj_matrix, pd.DataFrame):
        G = nx.from_pandas_adjacency(adj_matrix)
    else:
        G = nx.from_numpy_array(adj_matrix)
    
    if node_names:
        mapping = {i: name for i, name in enumerate(node_names)}
        G = nx.relabel_nodes(G, mapping)
    
    return G

def compute_centrality_measures(G):
    degree_centrality = nx.degree_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G)
    eigenvector_centrality = nx.eigenvector_centrality(G)
    return degree_centrality, betweenness_centrality, eigenvector_centrality

def find_top_central_nodes(centrality, top_n=5):
    sorted_nodes = sorted(centrality.items(), key=lambda item: item[1], reverse=True)
    return sorted_nodes[:top_n]

def count_components(G):
    return nx.number_connected_components(G)

def get_basic_info(G):
    size = G.number_of_nodes()
    avg_degree = sum(dict(G.degree()).values()) / size
    return size, avg_degree

def find_strongest_correlations(adj_matrix, node_names=None, top_n=5):
    adj_matrix_array = np.array(adj_matrix)
    np.fill_diagonal(adj_matrix_array, -np.inf)  # Exclude self-loops
    strongest_correlations_indices = np.unravel_index(np.argsort(-adj_matrix_array, axis=None), adj_matrix_array.shape)
    strongest_correlations = [(i, j, adj_matrix_array[i, j]) for i, j in zip(*strongest_correlations_indices) if i < j][:top_n]
    
    if node_names:
        strongest_correlations = [(node_names[i], node_names[j], value) for i, j, value in strongest_correlations]
    
    return strongest_correlations

def compute_global_properties(G):
    avg_path_length = nx.average_shortest_path_length(G) if nx.is_connected(G) else float('inf')
    clustering_coefficient = nx.average_clustering(G)
    network_diameter = nx.diameter(G) if nx.is_connected(G) else float('inf')
    return avg_path_length, clustering_coefficient, network_diameter

def get_network_info(adj_matrix, node_names=None):
    G = create_graph(adj_matrix, node_names)

    degree_centrality, betweenness_centrality, eigenvector_centrality = compute_centrality_measures(G)
    top_degree_centrality = find_top_central_nodes(degree_centrality)
    top_betweenness_centrality = find_top_central_nodes(betweenness_centrality)
    top_eigenvector_centrality = find_top_central_nodes(eigenvector_centrality)
    
    num_components = count_components(G)
    size, avg_degree = get_basic_info(G)
    strongest_correlations = find_strongest_correlations(adj_matrix, node_names)
    avg_path_length, clustering_coefficient, network_diameter = compute_global_properties(G)

    network_info = {
        'top_degree_centrality': top_degree_centrality,
        'top_betweenness_centrality': top_betweenness_centrality,
        'top_eigenvector_centrality': top_eigenvector_centrality,
        'number_of_components': num_components,
        'size': size,
        'average_degree': avg_degree,
        'strongest_correlations': strongest_correlations,
        'global_properties': {
            'average_path_length': avg_path_length,
            'clustering_coefficient': clustering_coefficient,
            'network_diameter': network_diameter
        }
    }

    return network_info

def print_network_info(info):
    print("\n" + "="*50)
    print("NETWORK INFORMATION")
    print("="*50)
    
    print("\nTop 5 Nodes by Degree Centrality:")
    for i, (node, centrality) in enumerate(info['top_degree_centrality'], 1):
        print("  {}. {}: {:.4f}".format(i, node, centrality))
    
    print("\nTop 5 Nodes by Betweenness Centrality:")
    for i, (node, centrality) in enumerate(info['top_betweenness_centrality'], 1):
        print("  {}. {}: {:.4f}".format(i, node, centrality))
    
    print("\nTop 5 Nodes by Eigenvector Centrality:")
    for i, (node, centrality) in enumerate(info['top_eigenvector_centrality'], 1):
        print("  {}. {}: {:.4f}".format(i, node, centrality))
    
    print("\nNumber of Components: {}".format(info['number_of_components']))
    
    print("\nBasic Info:")
    print("  - Size:           {}".format(info['size']))
    print("  - Average Degree: {:.2f}".format(info['average_degree']))
    
    print("\nStrongest Correlations:")
    for i, (node1, node2, value) in enumerate(info['strongest_correlations'], 1):
        print("  {}. {} <--> {} (Strength: {:.4f})".format(i, node1, node2, value))
    
    print("\nGlobal Network Properties:")
    print("  - Average Path Length:     {:.2f}".format(info['global_properties']['average_path_length']))
    print("  - Clustering Coefficient:  {:.2f}".format(info['global_properties']['clustering_coefficient']))
    print("  - Network Diameter:        {:.2f}".format(info['global_properties']['network_diameter']))
    print("="*50 + "\n")


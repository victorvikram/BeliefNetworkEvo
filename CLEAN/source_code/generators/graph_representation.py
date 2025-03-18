"""
This file has code to go from an adjacency matrix representation to a networkx graph representation
"""

import networkx as nx

def create_graph_from_adj_mat(correlation_matrix, variables_list=None, threshold=0):
    """
    converts a correlation matrix to a networkx graph
    names the nodes according to the variables_list
    only adds edges if the correlation is greater than the threshold
    """
    graph = nx.Graph()

    if variables_list is None:
        variables_list = list(range(correlation_matrix.shape[0]))
    
    for node in variables_list:
        graph.add_node(node)
    
    # Add edges between nodes (variables) with significant correlations as per the threshold.
    for i, var1 in enumerate(variables_list):
        for j, var2 in enumerate(variables_list[i+1:], start=i+1):
            # If a threshold is specified, only add an edge if the absolute correlation exceeds it.
            if abs(correlation_matrix[i, j]) > threshold:
                graph.add_edge(var1, var2, weight=correlation_matrix[i, j])

    return graph
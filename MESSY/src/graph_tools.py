"""
functions to handle networkx graphs 
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def create_graph_from_adj_mat(correlation_matrix, variables_list=None, threshold=0):
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

def make_consistent_matrix(shuffled_var_list, var_list, mat):
    indices = []
    for var in shuffled_var_list:
        indices.append(var_list.index(var))
    
    return mat[indices,:][:, indices]


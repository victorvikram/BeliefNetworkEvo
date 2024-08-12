"""
functions to handle networkx graphs 
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def create_graph_from_adjacency_matrix(adj_matrix):
    """
    makes a networkx graph from adjacency matrix
    TODO: combine with the other function that does this with a var list `make_networkx_graph...`
    """
    G = nx.Graph()
    n = adj_matrix.shape[0]
    for i in range(n):
        G.add_node(i)
        for j in range(i + 1, n):
            if adj_matrix[i, j] > 0:
                G.add_edge(i, j, weight=adj_matrix[i, j])
    return G

def make_consistent_matrix(shuffled_var_list, var_list, mat):
    indices = []
    for var in shuffled_var_list:
        indices.append(var_list.index(var))
    
    return mat[indices,:][:, indices]


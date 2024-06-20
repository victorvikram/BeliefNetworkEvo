
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def create_graph_from_adjacency_matrix(adj_matrix):
    G = nx.Graph()
    n = adj_matrix.shape[0]
    for i in range(n):
        G.add_node(i)
        for j in range(i + 1, n):
            if adj_matrix[i, j] > 0:
                G.add_edge(i, j, weight=adj_matrix[i, j])
    return G

def plot_graph(G, node_values, labels):
    # Position nodes using the spring layout
    pos = nx.spring_layout(G)
    
    # Draw nodes with color corresponding to node values
    node_color = node_values
    nx.draw_networkx_nodes(G, pos, node_color=node_color, cmap=plt.cm.viridis, node_size=30)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, width=0.05)
    
    # Draw node labels
    nx.draw_networkx_labels(G, pos, labels, font_size=7)

    
    # Add color bar
    # sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=min(node_values), vmax=max(node_values)))
    # sm.set_array([])
    # plt.colorbar(sm)
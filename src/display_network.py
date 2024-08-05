"""
functions for visualizing the graph
"""

from pyvis.network import Network
import networkx as nx
import numpy as np
from IPython.display import display, HTML

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def display_graph_pyvis(G, node_values=None, abs_val_edges=True, remove_zero_edges=True):
    """
    takes a networkx graph and makes interactive visualization, color coding based on node values
    """
    edges_to_remove = []
    
    if abs_val_edges:
        for u, v, d in G.edges(data=True):
            d['weight'] = abs(d['weight'])

            if d['weight'] == 0 and remove_zero_edges:
                edges_to_remove.append((u, v))
    
    G.remove_edges_from(edges_to_remove)
    
    if node_values is not None:
        for i, node in enumerate(G.nodes):
            G.nodes[node]['value'] = node_values[i]
    
    # pos = nx.spring_layout(G, scale=1000)
    net = Network('1000px', '1000px', notebook=True, cdn_resources='remote')
    net.from_nx(G)
    
    
    for e in net.edges:
        e['width'] = e['width'] * 20
    
    cmap = plt.cm.coolwarm  # You can choose other colormaps as well
    norm = mcolors.Normalize(vmin=-1, vmax=1)
    
    fixed_font_size = 20
    for node in net.nodes:
        value = node['value']
        node['color'] = mcolors.to_hex(cmap(norm(value))) 
        node['value'] = 1 # otherwise they end up being different sizes
        
        if 'font' not in node:
            node['font'] = {}
        node['font']['size'] = fixed_font_size
        
    return net

def display_graph_nx(G, node_values, labels):
    """
    displays the graph using networkx, non-interactive
    """
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
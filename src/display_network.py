"""
functions for visualizing the graph
"""

from pyvis.network import Network
import networkx as nx
import numpy as np
from scipy.stats import rankdata
from IPython.display import display, HTML

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def add_node_values_to_graph(G, node_values):
    
    for i, node in enumerate(G.nodes):
        G.nodes[node]['value'] = node_values[i]
    
    return G


def display_graph_pyvis(G, abs_val_edges=True, remove_zero_edges=True, include_physics_buttons=True, 
                        equal_edge_weights=False, rank_edge_coloring=False, size_highlight=[], border_highlight=[], shape_highlight=[], 
                        node_colormap=plt.cm.gist_earth, edge_colormap=plt.cm.PiYG, hi_fix_node=None, lo_fix_node=None):
    """
    takes a networkx graph and makes interactive visualization, color coding based on node values
    """
    edges_to_remove = []
    for u, v, d in G.edges(data=True):
        if d['weight'] == 0 and remove_zero_edges:
            edges_to_remove.append((u, v))
    
    G.remove_edges_from(edges_to_remove)
    
    # pos = nx.spring_layout(G, scale=1000)
    net = Network('1000px', '1000px', notebook=True, cdn_resources='remote')
    net.from_nx(G)
    
    edge_color_values = [e["type"] if "type" in e else e["width"] for e in net.edges]
    max_edge_color_value = max(abs(min(edge_color_values)), max(edge_color_values))
    
    if rank_edge_coloring:
        edge_color_values = rankdata(edge_color_values)
        edge_norm = mcolors.Normalize(vmin=0, vmax=len(edge_color_values))
    else:
        edge_norm = mcolors.Normalize(vmin=-max_edge_color_value, vmax=max_edge_color_value)

    node_norm = mcolors.Normalize(vmin=-1, vmax=1)


    for i, e in enumerate(net.edges):
        if equal_edge_weights:
            e['width'] = 2
        else:
            e['width'] = e['width'] * 30
           

        if abs_val_edges:
            e['width'] = abs(e["width"])
        
        e["color"] = mcolors.to_hex(edge_colormap(edge_norm(edge_color_values[i])))
        
        # if i % 100 == 0:
            # print(e)
    
    
    fixed_font_size = 20
    for node in net.nodes:
        node['size'] = 18
        if 'value' in node:
            value = node['value']
            node['color'] = mcolors.to_hex(node_colormap(node_norm(-value))) 
            node.pop('value') # otherwise they end up being different sizes
        
        if 'font' not in node:
            node['font'] = {}
        
        if node['id'] in size_highlight:
            node['size'] = 1.5 * node['size']
        
        if node['id'] in border_highlight:
            node["borderWidth"] = 6
        
        if node['id'] in shape_highlight:
            node['shape'] = 'box'
        
        if node['id'] == hi_fix_node:
            node['x'] = 500
            node['y'] = 1200
            node['fixed'] = True
            node['physics'] = False
        
        if node['id'] == lo_fix_node:
            node['x'] = 500
            node['y'] = -200
            node['fixed'] = True
            node['physics'] = False
        
        # print(node)
        node['font']['size'] = fixed_font_size
                            
    if include_physics_buttons:
        net.show_buttons(filter_=['physics'])
        
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

if __name__ == "__main__":
    G1 = nx.Graph()
    G1.add_edge("zero", "one", weight=2)
    G1.add_edge("one", "two", weight=4)
    G1.add_edge("two", "three", weight=6)
    
    add_node_values_to_graph(G1, [0, 1, 2, 3])

    print([G1.nodes[n]["value"] for n in G1.nodes])
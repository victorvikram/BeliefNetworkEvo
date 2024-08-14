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

def recenter_ranks(ranks, values):
    smallest_pos_ind = np.argmin(np.where(np.array(values) > 0, values, np.inf))
    actual_midpoint = ranks[smallest_pos_ind]
    naive_vmin = 1
    naive_vmax = len(ranks)
    naive_midpoint = np.ceil(len(ranks) / 2) + 1

    print("old", naive_vmin, actual_midpoint, naive_vmax)

    vmin = naive_vmin if actual_midpoint >= naive_midpoint else 2*(actual_midpoint - 1) - naive_vmax
    vmax = naive_vmax if actual_midpoint < naive_midpoint else 2*(actual_midpoint - 2) + 1

    print("adjusted", vmin, actual_midpoint, vmax)
    return vmin, vmax, actual_midpoint

def display_graph_pyvis(G, abs_val_edges=True, remove_zero_edges=True, include_physics_buttons=True, 
                        equal_edge_weights=False, rank_edge_coloring=False, make_zero_edge_midpoint=True, size_highlight=[], border_highlight=[], shape_highlight=[], 
                        node_colormap=plt.cm.gist_earth, edge_colormap=plt.cm.PiYG, hi_fix_node=None, lo_fix_node=None):
    """
    `abs_val_edges` - make the edge thicknesses and gravity based on the absolute value 
    `remove_zero_edges` - remove any edges that have zero weights
    `include_physics_buttons` - show physics buttons in the html to mess with the gravity and spring parameters 
    `equal_edge_weights` - display all edges as having the same weight
    `rank_edge_coloring` - color edges based on their rank rather than their value
    `make_zero_edge_midpoint` - make zero the midpoint of the color scale for edges
    `size_highlight` - a list of nodes to highlight by making them bigger
    `border_highlight` - a list of nodes to highlight by making their border heavier
    `node_colormap` - the colormap to use for coloring the nodes (which is done based on the node['value'])
    `edge_colormap` - the colormap to use for coloring the edges (which is done based on e['type'] if it is in the edge dictionary, and if not
    e['width'])
    `hi_fix_node` - the node fixed to the top part of the canvas
    `lo_fix_node` - the node fixed to the bottom part of the canvas 
    
    takes a networkx graph and makes interactive visualization, color coding based on node values

    # TODO make robust to zero edge weights
    # TODO check centering
    """
    edges_to_remove = []
    for u, v, d in G.edges(data=True):
        if d['weight'] == 0 and remove_zero_edges:
            edges_to_remove.append((u, v))
    
    G.remove_edges_from(edges_to_remove)
    
    print("number of edges", len(G.edges(data=True)))
    # pos = nx.spring_layout(G, scale=1000)
    net = Network('1000px', '1000px', notebook=True, cdn_resources='remote')
    net.from_nx(G)
    
    edge_color_values = [e["type"] if "type" in e else e["width"] for e in net.edges]
    max_edge_color_value = max(abs(min(edge_color_values)), max(edge_color_values)) if make_zero_edge_midpoint else max(edge_color_values)
    min_edge_color_value = max(abs(min(edge_color_values)), max(edge_color_values)) if make_zero_edge_midpoint else min(edge_color_values)
    
    if rank_edge_coloring:
        edge_rank_values = rankdata(edge_color_values)

        if make_zero_edge_midpoint:
            vmin, vmax, actual_midpoint = recenter_ranks(edge_rank_values, edge_color_values)
            edge_norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
            
            print("min", edge_norm(vmin))
            print("zero", edge_norm(0))
            print("first pos", edge_norm(actual_midpoint))
            print("max", edge_norm(vmax))
        else:
            edge_norm = mcolors.Normalize(vmin=0, vmax=len(edge_rank_values))
        
        edge_color_values = edge_rank_values
    else:
        edge_norm = mcolors.Normalize(vmin=-min_edge_color_value, vmax=max_edge_color_value)

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
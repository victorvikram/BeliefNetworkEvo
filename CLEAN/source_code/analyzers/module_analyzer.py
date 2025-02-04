"""
This file contains functions to find different sorts of modules in a network. These include:
- consistent components: that is, groups of nodes that can be understood as a single entity since they are all positively connected
and agree on connectinos to all their neighbors
- modules: parts of the network that are tightly connected by only sparsely connected to other parts of the network
"""


def find_consistent_clusters_greedy(G):
    """
    TODO write this function. It is supposed to greedily find consistent components by adding
    nodes to a component if they are consistent with the component
    """
    connected_components = list(nx.connected_components(G))

    for component in connected_components:
        consistent_sets = component.copy()

def calculate_interaction_strength(G, set_1, set_2):
    """
    finds the interaction strength between two sets of nodes in a network by adding up the 
    absolute values of the edge weights between both sets of nodes
    """
    total_strength = 0
    for node in set_1:
        for u, v, d in G.edges(node, data=True):
            if v in set_2:
                total_strength += np.abs(d["weight"])
    
    return total_strength
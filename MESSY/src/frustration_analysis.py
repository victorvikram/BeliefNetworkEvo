"""
contains functions for exploring where there is frustration in the graph
"""

import networkx as nx
import numpy as np

def merge_nodes_greedy(G):
    """
    Merges consistent nodes iteratively by picking the two consistent nodes with the highest edge strength
    between them and merging them. Continues to do this until no nodes are consistent.
    """
    return 

def merge_two_consistent_nodes(G, G_cons, node1, node2):
    """
    This function should first:
    1. reverse the graph for node2 if needed 
    2. make a new G with the supernode and all edges into it
    3. update G_cons with the supernode
    4. remove the old nodes from G
    """
    node_name = node1 + "-" + node2
    
    all_edges = (G.edges(node1) | G.edges(node2))

    for u, v, d in all_edges:
        if G.get_edge_data(node_name, v) is None:
            G.add_edge(node_name, v, d["weight"])
        else:
            G[node_name][v] += d["weight"]
    
    G.remove_node(node1)
    G.remove_node(node2)
    
    for node in G_cons.nodes():
        if node != node1 and node != node2:
            node1_cons = G.get_edge_data(node1, node)
            node2_cons = G.get_edge_data(node2, node)

            if node1_cons is not None and node2_cons is not None:
                G_cons.add_edge(node_name, node)
            
    G_cons.remove_node(node1)
    G_cons.remove_node(node2)

    return G, G_cons

def update_consistency_graph(G, G_cons, node1, node2, new_node):

    """
    this function updates the consistency graph after `node1` and `node2` have 
    been merged into new_node. It takes as input `G` and `G_cons` such that
    - they have been modified to make `node1` and `node2` positively consistent (by reversing sign of one of them if necessary)
    - the graphs contain `node1`, `node2`, AND `new_node` (`node1` and `node2` should be deleted later)
    - `G` contains all edges that the final graph will have. `G_cons` contains all the nodes but not yet the edges

    The function employs the following procedure:

    """

    for node in G.nodes():
        node1_edges = G_cons.get_edge_data(node1, node)
        node2_edges = G_cons.get_edge_data(node2, node)

        pos_con_n1 = {"weight": 1} in node1_edges.values()
        neg_con_n1 = {"weight": -1} in node1_edges.values()

        pos_con_n2 = {"weight": 1} in node2_edges.values()
        neg_con_n2 = {"weight": -1} in node2_edges.values()

        if pos_con_n1 and pos_con_n2:
            G_cons.add_edge(new_node, node, weight=1)
        
        if neg_con_n1 and neg_con_n2:
            G_cons.add_edge(new_node, node, weight=-1)
        
    node1_neighbors = list(G.neighbors(node1))
    node2_neighbors = list(G.neighbors(node2))

    for n1_neighbor in node1_neighbors:
        edge_weight_to_new_node_n1 = np.sign(G.get_edge_data(new_node, n1_neighbor))
        for n2_neighbor in node2_neighbors:
            existing_edges = G_cons.edges(n1_neighbor, n2_neighbor, keys=True)
            G.remove_edges_from(existing_edges)

            con_edges = G_cons.get_edge_data(n1_neighbor, n2_neighbor)
            edge_weight_to_new_node_n2 = np.sign(G.get_edge_data(new_node, n2_neighbor))

            pos_con = {"weight": 1} in con_edges.values()
            neg_con = {"weight": -1} in con_edges.values()

            if pos_con and edge_weight_to_new_node_n1 == edge_weight_to_new_node_n2:
                G_cons.add_edge(n1_neighbor, n2_neighbor, weight=1)

            if neg_con and edge_weight_to_new_node_n1 == -edge_weight_to_new_node_n2:
                G_cons.add_edge(n1_neighbor, n2_neighbor, weight=-1)
    
    G_cons.remove_node(node1)
    G_cons.remove_node(node2)

    return G_cons

def reverse_node(G, G_cons, node):

    """
    This function reverses the sign of a node, so that all positive correlations become
    negative and vice versa. Therefore all nodes it was positively consistent with become 
    negatively consistent and vice versa.

    **tested**

    MUTATING
    """

    # reverse all the edge signs in the main graph
    for u, v in G.edges(node):
        G[u][v]["weight"] = -G[u][v]["weight"]
    
    # reverse all edge signs in the consistency graph
    for u, v, key, data in G_cons.edges(node, keys=True, data=True):
        G_cons[u][v][key]["weight"] = -G_cons[u][v][key]["weight"]

def get_consistency_graph(G):
    """
    This iterates trhough all pairs of nodes in `G` and checks if they are consistent by the 
    definition in `check_for_consistency(...)`. It returns a multigraph, where if two nodes are
    negatively consistent, they have an edge between them with weight negative one, and if they
    are positively consistent, they have an edge between them with weight positive one.

    It is possible for two nodes to have two edges between each other if they are both negatively
    and positively consistent.

    **tested**
    """
    G_cons = nx.MultiGraph()

    nodes = list(G.nodes())
    num_nodes = len(nodes)

    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            p_con, n_con = check_for_consistency(G, nodes[i], nodes[j])
            
            if p_con:
                G_cons.add_edge(nodes[i], nodes[j], weight=1)
            
            if n_con:
                G_cons.add_edge(nodes[i], nodes[j], weight=-1)
    
    return G_cons

def check_for_consistency(G, node1, node2):

    """
    Two nodes are 
    1. positively consistent if they are not negatively connected, and on all neighbors they share, they agree on the sign of the edge
    2. negatively consistent if they are not positively connected, and on all neighbors they share, they disagree on the sign of the edge
    
    It is possible to be both if you share no neighbors.

    This function checks if `node1` and `node2` are positively and/or negatively consistent in 
    the graph `G`

    **tested**
    """

    opposite_count = 0
    same_count = 0

    edge = G.get_edge_data(node1, node2)
    not_neg_connect = (edge is None) or (np.sign(edge["weight"]) >= 0)
    not_pos_connect = (edge is None) or (np.sign(edge["weight"]) <= 0)
    for u, v, d1 in G.edges(node1, data=True):
        d2 = G.get_edge_data(node2, v)

        if d2 is not None:
            opposite_count += (np.sign(d1["weight"]) != np.sign(d2["weight"]))
            same_count += (np.sign(d1["weight"]) == np.sign(d2["weight"]))
        
        if opposite_count > 0 and same_count > 0:
            return False, False
    
    positive_consistent = (opposite_count == 0 and same_count >= 0) and not_neg_connect
    negative_consistent = (opposite_count >= 0 and same_count == 0) and not_pos_connect

    return positive_consistent, negative_consistent
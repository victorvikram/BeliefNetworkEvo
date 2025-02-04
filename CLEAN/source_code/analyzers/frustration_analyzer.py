"""
This file contains functions to find edges that are frustrated in the network
"""

import numpy as np

def get_satisfaction_mats(adj_mat, vectors):
    """ 
    this function uses the `adj_mat` to find the frustration_matrix of each row vector in `vectors`
    `satisfaction_mat[i, j, k]` is the satisfaction of the edge between j and k in the ith vector, and the satisfaction of 
    an edge is defined as the edge weight times the product of the two node values. So the more negative
    the satisfaction is, the more frustrated the edge is taken to be
    """
    satisfaction_mats = vectors[:, :, None] * (vectors[:, None, :] * adj_mat[None,:,:])
    return satisfaction_mats

def get_frust_percentage(satisfaction_mats):
    """
    this function finds the percentage of the time an edge is frustrated, which can be found
    by how often the value is negative in the satisfaction_matrix. `satisfaction_mats` is a stack of
    satisfaction matrices, wbere `satisfaction_mat[i, j, k]` is the satisfaction of the edge between 
    j and k in the ith vector, and the satisfaction of an edge is defined as the edge weight times 
    the product of the two node values
    """
    num_measurements = satisfaction_mats.shape[0]
    frust_percentage = np.sum(np.sign(satisfaction_mats) == -1, axis=0) / num_measurements
    return frust_percentage

def assign_nodes_search(G, breadth_first=True):
    """
    this function assigns nodes to +1 or -1 by performing a search on the network, and assigning
    a node according to the value of the node at the previous step, and the sign of the edge weight.

    The search algorithm can be breadth-first search or depth-first search. Breadth first seems to provide
    more consistent results and fewer frustrated edges
    """
    connected_components = list(nx.connected_components(G))

    start_sign = 1
    nodes_assigned = {}

    for component in connected_components:
        start_node = np.random.choice(list(component))    
        nodes_to_visit = [(start_node, start_sign, None)]

        while len(nodes_to_visit) > 0:
            pop_ind = 0 if breadth_first else -1
            curr_node, curr_val, parent = nodes_to_visit.pop(pop_ind)
            
            if curr_node not in nodes_assigned:
                nodes_assigned[curr_node] =  (curr_val, parent)

                for node, other_node, dat in G.edges(curr_node, data=True):
                    if other_node not in nodes_assigned:
                        nodes_to_visit.append((other_node, np.sign(dat["weight"]) * curr_val, curr_node))
    
    return nodes_assigned


def assign_nodes_random_walk(G, weighted=True, iterations_per_node=100):

    """
    this function assigns nodes to +1 or -1 by performing a random walk on the network, and assigning
    a node according to the value of the node at the previous step, and the sign of the edge weight
    """
    connected_components = list(nx.connected_components(G))

    start_sign = 1
    nodes_assigned = {node: 0 for node in G.nodes()}

    for component in connected_components:
        component_size = len(component)
        curr_node = np.random.choice(list(component))  
        curr_val = start_sign
        curr_edges = G.edges(curr_node, data=True)
        counter = 0

        while (len(component) > 0 or counter < iterations_per_node * component_size) and len(curr_edges) > 0:
            nodes_assigned[curr_node] += curr_val

            if curr_node in component:
                component.remove(curr_node)
            
            neighbor_list = [v for u, v, d in curr_edges]
            weight_arr = np.array([d['weight'] for u, v, d in curr_edges])
            prob_arr = np.abs(weight_arr) / np.abs(weight_arr).sum() if weighted else np.ones((len(neighbor_list))) / len(neighbor_list)

            curr_node_ind = np.random.choice(range(len(neighbor_list)), p=prob_arr)
            
            curr_val = np.sign(weight_arr[curr_node_ind]) * np.sign(nodes_assigned[curr_node])
            curr_node = neighbor_list[curr_node_ind]
            curr_edges = G.edges(curr_node, data=True)
            
            counter += 1
    
    for node in nodes_assigned:
        val = np.sign(nodes_assigned[node])
        nodes_assigned[node] = val

    return nodes_assigned
"""
This file contains functions to find edges that are frustrated in the network
"""

import numpy as np
import pandas as pd
import networkx as nx
import sys

sys.path.append("..")  # Adjust the path as necessary

from analyzers.optimization_analyzer import multi_pass_optimize, simulated_annealing, hamiltonian_objective_function, flip_step_function

def calculate_frustration(adj_mat, optimizer="multi_pass"):
    """
    `adj_mat` is the usual dataframe with column and row names

    calculates the frustration of edges in the network. the procedure is as follows
    1. first, optimize the belief vector, which is to say, find the assignments of edges that locally minimize the 
        the number of unsatisfied edges. Do this many times to get many optimal belief vectors
    2. second, calculate the percentage of the time that the edge is frustrated
    3. put that in a matrix `frust_percentage` and return it

    there are different optimization methods to use. "multi_pass" is fast and works well. "simulated annealing" is a more standard 
    method which takes more time. They produce similar results. 
    """
    var_list = adj_mat.index.tolist()
    np_adj_mat = adj_mat.values
    n_vars = adj_mat.shape[0]
    if optimizer == "multi_pass":
        initial_vectors = np.random.choice([-1, 0, 1], size=(1000, n_vars))
        minima, _ = multi_pass_optimize(initial_vectors, np_adj_mat, max_iterations=int(1e4))
    elif optimizer == "simulated_annealing":
        initial_vectors = np.random.choice([-1, 0, 1], size=(1000, n_vars))
        minima, _ = simulated_annealing(initial_vectors, 100, 0.99, int(1e5), 
                              lambda vecs: hamiltonian_objective_function(vecs, np_adj_mat),
                              lambda vecs: flip_step_function(vecs, num_flips=1))
    
    satisfaction_mats = get_satisfaction_mats(np_adj_mat, minima)
    frust_percentage = get_frust_percentage(satisfaction_mats)

    # convert the result back to a dataframe with the same index and columns as the original adj_mat
    frust_percentage = pd.DataFrame(frust_percentage, index=var_list, columns=var_list)

    return frust_percentage

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
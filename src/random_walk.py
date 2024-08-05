<<<<<<< HEAD
"""
functions for doing random walks
"""

import numpy as np

def non_markov_random_walk(adj_mat, start_node, num_steps, decay=1):
    """
    does a random walk that factors in transition probabilities from more than just the last node
    """
=======
import numpy as np

def non_markov_random_walk(adj_mat, start_node, num_steps, decay=1):
>>>>>>> e6bf9692b558f3d75297def83afb3e35d94878a7
    adj_mat = np.where(adj_mat > 0, adj_mat, 0)
    row_sums = adj_mat.sum(axis=1, keepdims=True)
    transition_matrix = adj_mat / row_sums

    current_node = start_node
    path = [current_node]
    
    for _ in range(num_steps):
        transition_probs = transition_matrix[path, :]
        exponent = np.arange(len(path))[::-1].reshape(len(path), 1)
        
        factor = (1 - decay)**exponent
        averaged_transition_probs = (transition_probs * factor).sum(axis=0) / factor.sum()

        # print(exponent)
        # print(factor)
        # print("path", path)
        # print("prob", averaged_transition_probs)
        # print("rowp", transition_matrix[current_node, :])

        next_node = np.random.choice(
            range(len(transition_matrix)),
            p=averaged_transition_probs
        )
        path.append(next_node)
        current_node = next_node
    

    visit_count = np.zeros((adj_mat.shape[0],))
    np.add.at(visit_count, path, 1)

    return path, visit_count


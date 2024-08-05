import numpy as np
# We calculate the "stress" of the belief vector i as
def stress_slow(belief_vector, interaction_matrix): # Element-wise construction of H (slow)
    H = 0
    num_nodes = len(belief_vector)
    for i in range(num_nodes):
        for j in range(num_nodes):
            H += - interaction_matrix[i, j] * belief_vector[i] * belief_vector[j] 
    return H

def stress(belief_vector, interaction_matrix): # Matrix multiplication H = - (b dot Jb) (fast)
    belief_vector = np.array(belief_vector)
    interaction_matrix = np.array(interaction_matrix)
    H = - np.dot(belief_vector, np.dot(interaction_matrix, belief_vector))
    return H
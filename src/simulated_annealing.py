import numpy as np

def symm_matrix_to_vec(matrix_stack):
    """
    **tested**
    """
    # Get the upper triangular indices, including the diagonal
    dim = matrix_stack.shape[-1]
    triu_indices = np.triu_indices(dim)

    # Extract the elements using the indices
    vector_stack = matrix_stack[...,triu_indices[0], triu_indices[1]]
    return vector_stack

"""
def vec_to_symm_matrix(vector_stack):
    length = vector_stack.shape[-1]
    dim = np.sqrt(2 * length + 0.25) + 0.5
    triu_indices = np.triu_indices(dim)
    matrix_stack = np.zeros(vector_stack.shape[:-1] + (dim, dim))

    
    matrix_stack[...,triu_indices] = vector_stack

    transposed_matrices = matrices.transpose(0, 2, 1)
"""


def flip_step_function(vector_arr, num_flips=1, seed=None):
    """
    **tested**
    """
    new_vector_arr = vector_arr.copy()
    num_vectors = new_vector_arr.shape[0]
    num_components = new_vector_arr.shape[1]
    
    if seed is not None:
        np.random.seed(seed)

    flips = np.random.randint(0, num_components, size=(num_vectors, num_flips))
    new_values = np.random.choice([-1, 0, 1], size=(num_vectors, num_flips))    
    new_vector_arr[np.arange(num_vectors).reshape(-1, 1), flips] = new_values

    return new_vector_arr

def sum_of_squares_objective_function(vectors): 
    return (vectors**2).sum(axis=1, keepdims=True)

def hamiltonian_objective_function(vectors, couplings):
    vector_outer = vectors[:,None,:] * vectors[:,:,None]
    cost = - (vector_outer * couplings).sum(axis=(1, 2))

    cost = cost.reshape(-1, 1)

    return cost

def accept_new_vector(old_cost_vector, new_cost_vector, temperature, seed=None):
    """
    **tested**
    """
    acceptance_prob_vector = np.where(new_cost_vector < old_cost_vector, 1, np.exp((old_cost_vector - new_cost_vector) / temperature))

    if seed is not None:
        np.random.seed(seed)
    
    rand_vector = np.random.rand(old_cost_vector.shape[0], 1)
    acceptance_vector =  rand_vector < acceptance_prob_vector
    
    return acceptance_vector

def simulated_annealing(initial_vectors, initial_temperature, cooling_rate, max_iterations, objective_function, step_function):
    current_vectors = initial_vectors.copy()
    current_cost = objective_function(current_vectors)
    temperature = initial_temperature

    for iteration in range(max_iterations):
        new_vectors = step_function(current_vectors)
        new_cost = objective_function(new_vectors)

        acceptance_vector = accept_new_vector(current_cost, new_cost, temperature)
        current_vectors = np.where(acceptance_vector, new_vectors, current_vectors)
        current_cost = np.where(acceptance_vector, new_cost, current_cost)

        temperature *= cooling_rate  # Cool down the system

        if temperature < 1e-10:
            break

        if iteration % (max_iterations // 10) == 0:
            print(f"Iteration {iteration}, Current Cost: {current_cost}, Temperature: {temperature}")

    return current_vectors, current_cost
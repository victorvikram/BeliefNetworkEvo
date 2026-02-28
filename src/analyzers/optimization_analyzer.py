"""
this file contains functions to find belief vectors that minimize energy
"""

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
    function that takes an array of vectors and changes `num_flips` of the bits by choosing randomly to reassign
    each component to -1, 0, 1
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
    """
    note: this will only work if there are no entries on the diagonal
    TODO: redo so that I only multiply the upper triangles together
    """
    vector_outer = vectors[:,None,:] * vectors[:,:,None]
    cost = - (vector_outer * couplings).sum(axis=(1, 2)) / 2

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

def multi_pass_optimize(initial_vectors, couplings, random_order=True, change_threshold=0, max_iterations=1000):
    iterations = 0
    num_changed = change_threshold + 1
    current_vectors = initial_vectors.copy()

    while num_changed > change_threshold and iterations < max_iterations:
        current_vectors, num_changed = single_pass_optimize(current_vectors, couplings, random_order)
        iterations += 1
    
    return current_vectors, num_changed

def single_pass_optimize(initial_vectors, couplings, random_order=True):
    """
    this goes through the vector one component at a time, and pushes that component to the extreme value that optimizes 
    for an ising model, you always do better by going to the extreme value
    """
    num_components = initial_vectors.shape[1]
    
    if random_order:
        component_queue = np.random.choice(range(num_components), size=(num_components,), replace=False)
    else:
        component_queue = np.arange(num_components)

    current_vectors = initial_vectors.copy()

    for component in component_queue:
        component_multiplier = (current_vectors * couplings[component:component+1,:]).sum(axis=1)
        new_component_value = np.where(component_multiplier > 0, 1, -1)
        current_vectors[:,component] = new_component_value

    num_changed = (initial_vectors != current_vectors).any(axis=1).sum()

    return current_vectors, num_changed

def simulated_annealing(initial_vectors, initial_temperature, cooling_rate, max_iterations, objective_function, step_function, print_stuff=False):
    """
    **tested**
    """
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

        if print_stuff and iteration % (max_iterations // 10) == 0:
            print(f"Iteration {iteration}, Current Cost: {current_cost}, Temperature: {temperature}")

    return current_vectors, current_cost

def gradient_descent(initial_vectors, step_size, couplings, diff_threshold=0.01, max_iterations=10000):
    current_vectors = initial_vectors.copy()

    iterations = 0
    still_changing = True
     
    while iterations < max_iterations and still_changing:
        # print("vecs", current_vectors)
        # print("cost", hamiltonian_objective_function(current_vectors, couplings))
        gradient = - np.matmul(couplings[None,:, :], current_vectors[:,:,None]).reshape(*current_vectors.shape)
        step = - gradient / np.abs(gradient).sum(axis=1, keepdims=True) * step_size
        # print("step", step)
        new_vectors = np.maximum(np.minimum(current_vectors + step, 1), -1)
        # print("new vecs", new_vectors)
        still_changing = (np.abs(current_vectors - new_vectors).sum(axis=1) > step_size * diff_threshold).any()
        current_vectors = new_vectors
        iterations += 1

    return current_vectors

if __name__ == "__main__":
    couplings = np.array([[-0.54530848, -0.61689409,  0.48662589,  0.        ,  0.        ],
                              [-0.61689409, -0.73337935, -0.21204274,  0.        ,  0.        ],
                              [ 0.48662589, -0.21204274,  0.87779761,  0.        ,  0.        ],
                              [ 0.        ,  0.        ,  0.        ,  0.06709374, -0.31310036],
                              [ 0.        ,  0.        ,  0.        , -0.31310036,  0.38020617]])
    
    vectors = np.array([[-0.1, 0.2, 0.5, -0.3, 1]])

    minima = gradient_descent(vectors, 0.1, couplings)
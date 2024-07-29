import unittest
import pandas as pd
import numpy as np
from numpy.testing import assert_almost_equal

import sys
sys.path.append('../src')

from simulated_annealing import *

class TestModuleFunctions(unittest.TestCase):

    def test_flip_step_function(self):

        vectors = np.array([[-1, -1, -1,  0,  1],
                            [ 1, -1, -1,  1, -1],
                            [ 0,  0,  0,  1,  1],
                            [ 1, -1,  1,  0, -1],
                            [ 0,  1, -1, -1,  1],
                            [-1, -1, -1, -1, -1],
                            [-1,  1, -1,  1,  0],
                            [ 0,  0, -1,  0,  0],
                            [ 0, -1,  0,  1, -1]])
        
        flipped_vector = flip_step_function(vectors, num_flips=2, seed=0)

        # expected indices
        # np.array([[4, 0],
        #           [3, 3],
        #           [3, 1],
        #           [3, 2],
        #           [4, 0],
        #           [0, 4],
        #           [2, 1],
        #           [0, 1],
        #           [1, 0]])

        # expected replacements
        # array([[ 0 -1]
        #        [-1  0]
        #        [ 1 -1]
        #        [ 1 -1]
        #        [ 0  0]
        #        [ 1 -1]
        #        [ 0  0]
        #        [ 0 -1]
        #        [ 1 -1]])

        expected_flipped_vector = np.array([[-1, -1, -1,  0,  0],
                                            [ 1, -1, -1,  0, -1],
                                            [ 0, -1,  0,  1,  1],
                                            [ 1, -1, -1,  1, -1],
                                            [ 0,  1, -1, -1,  0],
                                            [ 1, -1, -1, -1, -1],
                                            [-1,  0,  0,  1,  0],
                                            [ 0, -1, -1,  0,  0],
                                            [-1,  1,  0,  1, -1]])
        
        self.assertTrue((flipped_vector == expected_flipped_vector).all())
    
    def test_symm_matrix_to_vec(self):
        arr = np.array([[-0.54530848, -0.61689409,  0.48662589,  0.        ,  0.        ],
                        [-0.61689409, -0.73337935, -0.21204274,  0.        ,  0.        ],
                        [ 0.48662589, -0.21204274,  0.87779761,  0.        ,  0.        ],
                        [ 0.        ,  0.        ,  0.        ,  0.06709374, -0.31310036],
                        [ 0.        ,  0.        ,  0.        , -0.31310036,  0.38020617]])
        expected_vec = np.array([-0.54530848, -0.61689409,  0.48662589,  0.,  0.,
                                              -0.73337935, -0.21204274,  0.,  0.,
                                                            0.87779761,  0.,  0.,
                                                                         0.06709374, -0.31310036,
                                                                                      0.38020617])
        
        vec = symm_matrix_to_vec(arr)
        assert_almost_equal(vec, expected_vec)

        arr = np.arange(18).reshape(2, 3, 3) 
        arr = (arr * arr.transpose(0, 2, 1))
        expected_vec = np.array([[0, 3, 12, 16, 35, 64],[81, 120, 165, 169, 224, 289]])

        vec = symm_matrix_to_vec(arr)
        self.assertTrue((vec == expected_vec).all())
    
    def test_accept_new_vector(self):
        old_cost_vector = np.array([[1],[2],[3],[4],[2],[4],[8],[9],[3],[3]])
        new_cost_vector = np.array([[2],[1],[2],[5],[3],[6],[7],[9],[4],[5]])

        """
        [[0.5488135 ]
        [0.71518937]
        [0.60276338]

        [0.54488318]
        [0.4236548 ]
        [0.64589411]

        [0.43758721]
        [0.891773  ]
        [0.96366276]

        [0.38344152]]
        """

        temperature = 1
        acceptances = accept_new_vector(old_cost_vector, new_cost_vector, temperature, seed=0)
        exp_acceptances = np.array([False, True, True, False, False, False, True, True, False, False]).reshape(-1, 1)
        self.assertTrue((acceptances == exp_acceptances).all())

        temperature = 3
        acceptances = accept_new_vector(old_cost_vector, new_cost_vector, temperature, seed=0)
        exp_acceptances = np.array([True, True, True, True, True, False, True, True, False, True]).reshape(-1, 1)
        self.assertTrue((acceptances == exp_acceptances).all())

        temperature = 10
        acceptances = accept_new_vector(old_cost_vector, new_cost_vector, temperature, seed=0)
        exp_acceptances = np.array([True, True, True, True, True, True, True, True, False, True]).reshape(-1, 1)
        self.assertTrue((acceptances == exp_acceptances).all())
    
    def test_hamiltonian_objective_function(self):
        couplings = np.array([[ -1,     1, 0.5],
                              [  1, -0.25, 0.2],
                              [0.5,  0.35, 0.4]])
        
        vectors = np.array([[1, -1, 1], [-1, 1, 1]])

        cost = hamiltonian_objective_function(vectors, couplings)
        exp_cost = np.array([[2.4], [3.3]])
        assert_almost_equal(cost, exp_cost)

    def test_simulated_annealing(self):
        couplings = np.array([[-0.54530848, -0.61689409,  0.48662589,  0.        ,  0.        ],
                              [-0.61689409, -0.73337935, -0.21204274,  0.        ,  0.        ],
                              [ 0.48662589, -0.21204274,  0.87779761,  0.        ,  0.        ],
                              [ 0.        ,  0.        ,  0.        ,  0.06709374, -0.31310036],
                              [ 0.        ,  0.        ,  0.        , -0.31310036,  0.38020617]])
                
        initial_vectors = np.random.choice([-1, 0, 1], size=(100, 5))
        final_vectors, final_cost = simulated_annealing(initial_vectors, 
                                                        10, 0.95, 1000, 
                                                        lambda vec: hamiltonian_objective_function(vec, couplings), 
                                                        lambda vec: flip_step_function(vec, num_flips=1))
        
        unique_rows = np.unique(final_vectors, axis=0)
        self.assertEqual(unique_rows.shape[0], 2**2)

        couplings = [[ 0.00,  0.35,  0.44,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00],
                     [ 0.35,  0.00,  0.72,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00],
                     [ 0.44,  0.72,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00],
                     [ 0.00,  0.00,  0.00,  0.00,  0.50, -0.25,  0.80,  0.00,  0.00,  0.00],
                     [ 0.00,  0.00,  0.00,  0.50,  0.00, -0.70,  0.60,  0.00,  0.00,  0.00],
                     [ 0.00,  0.00,  0.00, -0.25, -0.70,  0.00, -0.55,  0.00,  0.00,  0.00],
                     [ 0.00,  0.00,  0.00,  0.80,  0.60, -0.55,  0.00,  0.00,  0.00,  0.00],
                     [ 0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00, -0.20,  0.75],
                     [ 0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00, -0.20,  0.00, -0.45],
                     [ 0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.75, -0.45,  0.00]]
        
        initial_vectors = np.random.choice([-1, 0, 1], size=(100, 10))
        final_vectors, final_cost = simulated_annealing(initial_vectors, 
                                                        20, 0.95, 1000, 
                                                        lambda vec: hamiltonian_objective_function(vec, couplings), 
                                                        lambda vec: flip_step_function(vec, num_flips=1))
        
        unique_rows = np.unique(final_vectors, axis=0)
        print(unique_rows)
        self.assertEqual(unique_rows.shape[0], 2**3)


if __name__ == "__main__":
    unittest.main()


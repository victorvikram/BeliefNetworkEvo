import unittest
import numpy as np
from numpy.testing import assert_almost_equal

import sys
sys.path.append('../src')

from corr_networks import filter_nans, precision_mat_to_partial_corr, corr_mat_to_partial_corr_mat, cov_mat_to_regularized_partial_corr

class TestModuleFunctions(unittest.TestCase):
    def test_my_pairwise_correlations(self):
        # Define mean vector and covariance matrix
        dim = 4
        random_mat = 2 * np.random.rand(dim, dim) - 1 # get a matrix with values from -1 to 1
        random_cov_mat = np.dot(random_mat, random_mat.T) # make it pos semi-definite

        std_deviations = np.sqrt(np.diag(random_cov_mat))
        random_cor_mat = random_cov_mat / np.outer(std_deviations, std_deviations)
            
        mean = np.zeros((dim,))  # Mean vector

        # Generate random samples from the multivariate normal distribution
        num_samples = 20000
        samples = np.random.multivariate_normal(mean, random_cov_mat, size=num_samples)

        sample_df = pd.DataFrame(samples)

        # Print the first few samples

        precision_matrix = np.linalg.inv(random_cov_mat)
        scale_factor = np.matmul(np.diag(precision_matrix).reshape(-1, 1), np.diag(precision_matrix).reshape(1, -1))
        partial_corr_mat = (- precision_matrix / scale_factor) + (2 * np.identity(dim))

        sample_df_ord = pd.DataFrame()

        for var in list(range(dim)):
            num_ordinal_values = np.random.randint(2, 11)
            # signed = np.random.rand() > 0.5
            var_std = np.sqrt(random_cov_mat[var, var])

            interval_spread = np.random.rand() * var_std
            leftmost_border = interval_spread * ((num_ordinal_values - 2)/ 2) + (np.random.rand() - 0.5)
            
            cutoffs = interval_spread * np.arange(num_ordinal_values - 1) - leftmost_border
            cutoffs = np.concatenate(([-np.inf], cutoffs, [np.inf]))
            sample_df_ord[var] = pd.cut(sample_df[var], bins=cutoffs, labels=np.arange(num_ordinal_values)).cat.codes
        
        vars, corr_mat = my_pairwise_correlations([0, 1, 2, 3], sample_df, method="pearson", partial=True, regularization=0)

        assert_almost_equal(corr_mat, )
        




    def test_filter_nans(self):
        input_mat = np.arange(36).reshape(6, 6).astype(float)
        input_mat = (input_mat + input_mat.T) / 2

        output, removed = filter_nans(input_mat)

        self.assertTrue((output == input_mat).all())
        self.assertEqual(removed, [])

        input_mat[np.triu_indices(6, k=4)] = np.nan
        input_mat[np.tril_indices(6, k=-4)] = np.nan
        expected_output = np.array([[14,  17.5, 21,  24.5],
                                    [17.5, 21, 24.5, 28],
                                    [21,  24.5, 28,  31.5],
                                    [24.5, 28,  31.5, 35]])

        output, removed = filter_nans(input_mat)
        self.assertTrue((output == expected_output).all())
        self.assertEqual(removed, [0, 1])

        input_mat = input_mat[[[3], [5], [1], [4], [0], [2]], [[3, 5, 1, 4, 0, 2]]]
        expected_output = [[21,  14,  10.5, 17.5],
                           [14,   7,  3.5, 10.5],
                           [10.5,  3.5,  0,   7],
                           [17.5, 10.5,  7,  14]]
        
        output, removed = filter_nans(input_mat)
        self.assertTrue((output == expected_output).all())
        self.assertEqual(removed, [1, 3])

        input_mat = input_mat[[[0], [4], [2], [3], [1], [5]], [[0, 4, 2, 3, 1, 5]]]
        input_mat[4, 4] = np.nan
        input_mat[5, 2] = np.nan
        input_mat[2, 5] = np.nan
        input_mat[0, 2] = np.nan
        input_mat[2, 0] = np.nan
        expected_output = [[21, 24.5, 17.5],
                           [24.5, 28, 21],
                           [17.5, 21, 14]]

        output, removed = filter_nans(input_mat)
        self.assertTrue((output == expected_output).all())
        self.assertEqual(removed, [2, 1, 4])

    def test_precision_mat_to_partial_corr(self):
        prec_mat = np.array([[ 1.0, -0.3,  0.2,  0.4],
                             [-0.3,  0.4,  0.1,  0.8],
                             [ 0.2,  0.1,  0.9, -0.4],
                             [ 0.4,  0.8, -0.4,  0.6]])
        scale_factor = np.sqrt(np.matmul([[1.0],[0.4],[0.9],[0.6]],[[1.0, 0.4, 0.9, 0.6]]))
        expected_partial_corr = - prec_mat / scale_factor
        np.fill_diagonal(expected_partial_corr, 1)
        partial_corr = precision_mat_to_partial_corr(prec_mat)

        self.assertTrue((expected_partial_corr == partial_corr).all())

        prec_mat = ((np.arange(16) + 1) / 16).reshape(4, 4)
        checkerboard = np.array([[ 1,-1, 1,-1],
                                 [-1, 1,-1, 1],
                                 [ 1,-1, 1,-1],
                                 [-1, 1,-1, 1]])
        prec_mat = prec_mat * checkerboard
        scale_factor = np.sqrt(np.matmul([[0.0625], [0.375], [0.6875], [1]], [[0.0625, 0.375, 0.6875, 1]]))
        expected_partial_corr = -prec_mat / scale_factor
        np.fill_diagonal(expected_partial_corr, 1)
        partial_corr = precision_mat_to_partial_corr(prec_mat)

        self.assertTrue((expected_partial_corr == partial_corr).all())
    
    def test_corr_mat_to_partial_corr_mat(self):
        corr_mat = np.array([[ 1.,    -0.582, -0.017,  0.439],
                             [-0.582,  1.,    -0.528,  0.415],
                             [-0.017, -0.528,  1.,    -0.607],
                             [ 0.439,  0.415, -0.607,  1.   ]])

        first_partial_corr = corr_mat_to_partial_corr_mat(corr_mat)
        second_partial_corr = cov_mat_to_regularized_partial_corr(corr_mat)

        assert_almost_equal(first_partial_corr, second_partial_corr)
    
    def test_cov_mat_to_regularized_partial_corr(self):
        corr_mat = np.array([[ 1.,    -0.425,  0.759, -0.187],
                             [-0.425,  1.,     0.166, -0.654],
                             [ 0.759,  0.166,  1.,    -0.615],
                             [-0.187, -0.654, -0.615,  1.   ]])
        first_partial_corr = corr_mat_to_partial_corr_mat(corr_mat)
        second_partial_corr = cov_mat_to_regularized_partial_corr(corr_mat)

        assert_almost_equal(first_partial_corr, second_partial_corr)

        reged_partial_corr = cov_mat_to_regularized_partial_corr(corr_mat, alpha=0.05)

        print(first_partial_corr)
        print(reged_partial_corr)
        self.assertGreater((np.abs(first_partial_corr) - np.abs(reged_partial_corr)).sum(), 0)

        
                                                    
if __name__ == '__main__':
    unittest.main()
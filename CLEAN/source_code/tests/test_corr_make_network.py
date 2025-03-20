import unittest
import pandas as pd
import numpy as np
from numpy.testing import assert_almost_equal

import sys
sys.path.append('../generators')

from corr_make_network import *

class TestModuleFunctions(unittest.TestCase):
    def test_calculate_correlation_matrix(self):
        # Define mean vector and covariance matrix
        dim = 6
        random_mat = 2 * np.random.rand(dim, dim) - 1 # get a matrix with values from -1 to 1
        random_cov_mat = np.dot(random_mat, random_mat.T) # make it pos semi-definite

        std_deviations = np.sqrt(np.diag(random_cov_mat))
        random_cor_mat = random_cov_mat / np.outer(std_deviations, std_deviations) - np.identity(dim)
            
        mean = np.zeros((dim,))  # Mean vector

        # Generate random samples from the multivariate normal distribution
        num_samples = 300000
        samples = np.random.multivariate_normal(mean, random_cov_mat, size=num_samples)

        sample_df = pd.DataFrame(samples)

        # Print the first few samples

        precision_matrix = np.linalg.inv(random_cov_mat)
        scale_factor = np.sqrt(np.outer(np.diag(precision_matrix), np.diag(precision_matrix)))
        partial_corr_mat = (- precision_matrix / scale_factor) + (2 * np.identity(dim)) - np.identity(6)
        
        corr_df = calculate_correlation_matrix(sample_df, 
                                                  variables_of_interest=[0, 1, 2, 3, 4, 5], 
                                                  method=CorrelationMethod.PEARSON, 
                                                  partial=False)
        corr_mat = corr_df.values
        vars = list(corr_df.columns)

        assert_almost_equal(corr_mat, random_cor_mat, decimal=2)
        self.assertEqual(vars, [0, 1, 2, 3, 4, 5])

        corr_df = calculate_correlation_matrix(sample_df,
                                                       variables_of_interest=[0, 1, 2, 3, 4, 5], 
                                                       method=CorrelationMethod.PEARSON, 
                                                       partial=True, 
                                                       edge_suppression=EdgeSuppressionMethod.REGULARIZATION,
                                                       suppression_params={"regularization": 0})
        
        corr_mat = corr_df.values
        vars = list(corr_df.columns)

        assert_almost_equal(corr_mat, partial_corr_mat, decimal=2)
        self.assertEqual(vars, [0, 1, 2, 3, 4, 5])

        corr_df = calculate_correlation_matrix(sample_df,
                                                        variables_of_interest=[0, 1, 2, 3, 4, 5],
                                                        method=CorrelationMethod.PEARSON, 
                                                        partial=True, 
                                                        edge_suppression=EdgeSuppressionMethod.REGULARIZATION,
                                                        suppression_params={"regularization": 0.1})
        
        corr_mat = corr_df.values
        vars = list(corr_df.columns)

        self.assertGreater((np.abs(random_cor_mat) - np.abs(corr_mat)).sum(), 0)

        non_overlapping_df = sample_df.copy()
        non_overlapping_df.loc[0:num_samples//2,[4, 5]] = np.nan
        non_overlapping_df.loc[num_samples//2:,[0, 1, 2, 3]] = np.nan

        corr_df = calculate_correlation_matrix(non_overlapping_df, 
                                           variables_of_interest=[0, 1, 2, 3, 4, 5],
                                           method=CorrelationMethod.PEARSON, 
                                           partial=True)
        corr_mat = corr_df.values
        vars = list(corr_df.columns)
        
        truncated_partial_corr = calculate_partial_correlations(random_cov_mat[0:4, 0:4]) - np.identity(4)
        assert_almost_equal(corr_mat, truncated_partial_corr, decimal=2)
        self.assertEqual(vars, [0, 1, 2, 3])

        corr_df = calculate_correlation_matrix(non_overlapping_df, 
                                                        variables_of_interest=[0, 1, 2, 3, 4, 5], 
                                                        method=CorrelationMethod.PEARSON, 
                                                        partial=False, 
                                                        edge_suppression=EdgeSuppressionMethod.REGULARIZATION,
                                                        suppression_params={"regularization": 0})
        corr_mat = corr_df.values
        vars = list(corr_df.columns)
        
        value_mask = np.logical_xor((np.arange(6) > 3).reshape(-1, 1), (np.arange(6) > 3).reshape(1, -1)) # both row and column indices either above 3 or below 3
        exp_corr_mat = np.where(value_mask, np.nan, random_cor_mat)
        assert_almost_equal(corr_mat, exp_corr_mat, decimal=2)
        self.assertEqual(vars, [0, 1, 2, 3, 4, 5])

        non_overlapping_df = sample_df.copy()
        non_overlapping_df.loc[0:num_samples//2,[2, 4]] = np.nan
        non_overlapping_df.loc[num_samples//2:,[1]] = np.nan

        corr_df = calculate_correlation_matrix(non_overlapping_df, 
                                                 variables_of_interest=[0, 1, 2, 3, 4, 5],
                                                 method=CorrelationMethod.PEARSON, 
                                                 partial=True)
        
        corr_mat = corr_df.values
        vars = list(corr_df.columns)

        truncated_partial_corr = calculate_partial_correlations(random_cov_mat[[[0],[2],[3],[4],[5]],[[0, 2, 3, 4, 5]]]) - np.identity(5)
        self.assertEqual(vars, [0, 2, 3, 4, 5])
        assert_almost_equal(corr_mat, truncated_partial_corr, decimal=2)
        
        fade_out_df = sample_df.copy()
        fade_out_df.loc[:num_samples//3,[2, 4]] = np.nan
        fade_out_df.loc[:2*(num_samples//3),[3, 5]] = np.nan 

        corr_df = calculate_correlation_matrix(fade_out_df,
                                                        variables_of_interest=[0, 1, 2, 3, 4, 5], 
                                                        method=CorrelationMethod.PEARSON, 
                                                        partial=True, 
                                                        sample_threshold=0.5)
        
        corr_mat = corr_df.values
        vars = list(corr_df.columns)
        
        truncated_partial_corr = calculate_partial_correlations(random_cov_mat[[[0], [1], [2], [4]], [[0, 1, 2, 4]]]) - np.identity(4)
        self.assertEqual(vars, [0, 1, 2, 4])
        assert_almost_equal(truncated_partial_corr, corr_mat, decimal=2)

        corr_df = calculate_correlation_matrix(sample_df,
                                            [0, 1, 2, 3, 4, 5],
                                            method=CorrelationMethod.SPEARMAN, 
                                            partial=True)
        corr_mat = corr_df.values
        vars = list(corr_df.columns)

        _, corr_mat_exp = alternative_calculate_pairwise_correlations([0, 1, 2, 3, 4, 5], sample_df, method="spearman", partial=True)
        corr_mat_exp = corr_mat_exp - np.identity(6)
        assert_almost_equal(corr_mat, corr_mat_exp, decimal=2)
        self.assertEqual(vars, [0, 1, 2, 3, 4, 5])

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
        
        """
        TODO check this
        vars, corr_mat = my_pairwise_correlations([0, 1, 2, 3, 4, 5], sample_df_ord, method="polychoric", partial=True, regularization=0, sample_threshold=0)
        assert_almost_equal(partial_corr_mat, corr_mat, decimal=2)
        """

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
    
    def test_calculate_partial_correlations(self):
        corr_mat = np.array([[ 1.,    -0.582, -0.017,  0.439],
                             [-0.582,  1.,    -0.528,  0.415],
                             [-0.017, -0.528,  1.,    -0.607],
                             [ 0.439,  0.415, -0.607,  1.   ]])

        first_partial_corr = calculate_partial_correlations(corr_mat)
        second_partial_corr = calculate_regularized_partial_correlations(corr_mat, alpha=0)

        assert_almost_equal(first_partial_corr, second_partial_corr)
    
    def test_cov_mat_to_regularized_partial_corr(self):
        corr_mat = np.array([[ 1.,    -0.425,  0.759, -0.187],
                             [-0.425,  1.,     0.166, -0.654],
                             [ 0.759,  0.166,  1.,    -0.615],
                             [-0.187, -0.654, -0.615,  1.   ]])
        first_partial_corr = calculate_partial_correlations(corr_mat)
        second_partial_corr = calculate_regularized_partial_correlations(corr_mat, alpha=0)

        assert_almost_equal(first_partial_corr, second_partial_corr)

        reged_partial_corr = calculate_regularized_partial_correlations(corr_mat, alpha=0.05)

        self.assertGreater((np.abs(first_partial_corr) - np.abs(reged_partial_corr)).sum(), 0)
    """
    SOMEDAY if I implement the sample threshold (which requires this as a helper function)
    I can uncomment this

    def test_overlap_matrix(self):
        data = pd.DataFrame({
            "a": [1, 2, 3, 4, 5, np.nan, np.nan, np.nan],
            "b": [np.nan, np.nan, np.nan, 1, 2, 3, 4, 5],
            "c": [1, np.nan, 2, np.nan, 3, np.nan, 4, 5],
            "d": [1, 2, 3, 4, 5, 6, 7, 8],
            "e": [np.nan, 1, 2, 3, 4, 5, 6, np.nan],
            "f": [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            "g": [1, 2, 3, 4, np.nan, np.nan, 5, 6]
        })

        overlap = get_overlap_matrix(data)
        expected_overlap = np.array([[5, 2, 3, 5, 4, 0, 4],
                                     [2, 5, 3, 5, 4, 0, 3],
                                     [3, 3, 5, 5, 3, 0, 4],
                                     [5, 5, 5, 8, 6, 0, 6],
                                     [4, 4, 3, 6, 6, 0, 4],
                                     [0, 0, 0, 0, 0, 0, 0],
                                     [4, 3, 4, 6, 4, 0, 6]])
        
        self.assertTrue((overlap == expected_overlap).all())

    """

if __name__ == '__main__':
    unittest.main()
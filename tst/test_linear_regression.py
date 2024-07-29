import unittest
import pandas as pd
import numpy as np
from numpy.testing import assert_almost_equal

import sys
sys.path.append('../src')

from linear_regression import *

class TestModuleFunctions(unittest.TestCase):

    def test_lin_reg_imputation(self):
        num_rows = 10000
        feature_data = np.random.rand(num_rows, 5)
        coeffs = np.array([3, 2, 5, 7, 1]).reshape(5, 1)

        target_data = np.matmul(feature_data, coeffs) + 0.5
        data = np.zeros((num_rows, 6))
        data[:,:5] = feature_data
        data[:,5:] = target_data

        column_names = ["feature_0", "feature_1", "feature_2", "feature_3", "feature_4", "target"]
        test_dataframe = pd.DataFrame(data, columns=column_names)

        result = lin_reg_imputation(test_dataframe, ["feature_0", "feature_2", "feature_3", "feature_4"], "target", rank=False)

        # check that the coefficients are correct
        assert_almost_equal(result.params, np.array([1.5, 3, 5, 7, 1]), decimal=1)


        # Generate the first column with random values
        data = np.random.rand(num_rows, 1)

        num_cols = 5

        coeff_array = np.array([[  0,   0,   0,   0],
                                [  2,   0,   0,   0],
                                [0.2, 0.8,   0,   0],
                                [  3, 0.1, 1.2,   0],
                                [0.6, 1.5, 0.9,   2]])
        # Add subsequent columns as linear combinations of previous columns with added noise
        for i in range(1, num_cols):
            noise = np.random.randn(num_rows)  # Noise with mean=0 and std=0.1
            new_column = np.sum(data * coeff_array[i:i+1,0:i], axis=1) + noise  # Linear combination of all previous columns + noise
            data = np.hstack((data, new_column.reshape(-1, 1)))  # Append the new column to the data

        coeffs = np.array([3, 1, 4, 2, 5]).reshape(5, 1)

        noise = np.random.randn(num_rows)
        target = np.matmul(data, coeffs) + noise

        data = np.hstack((data, target))
        column_names = ["feature_0, feature_1, feature_2", "feature_3", "feature_4", "target"]
        test_dataframe = pd.DataFrame(data, columns=column_names)

        result = lin_reg_imputation(test_dataframe, ["feature_0", "feature_1", "feature_2", "feature_3", "feature_4"], "target", rank=False)

        print(result.params)

        assert_almost_equal(result.params, np.array([3, 1, 4, 2, 5]))

        nan_mask = np.random.choice([True, False], size=data.shape, replace=True, p=0.7)
        data = np.where(nan_mask, data, np.nan)

        self.assertAlmostEqual(np.isnan(data).sum() / (num_rows * (num_cols + 1)), 0.3)

        column_names = ["feature_0, feature_1, feature_2", "feature_3", "feature_4", "target"]
        test_dataframe = pd.DataFrame(data, column_names)

        result = lin_reg_imputation(test_dataframe, ["feature_0", "feature_1", "feature_2", "feature_3", "Feature_4"], "target", rank=False)
        assert_almost_equal(result.params, np.array([3, 1, 4, 2, 5]))

if __name__ == "__main__":
    unittest.main()


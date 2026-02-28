"""
this file contains functions that deal with changes in beliefs over the years
"""

from corr_networks import my_pairwise_correlations
from archive.transform_df_to_our_standard import normalize_columns
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

def make_degree_strength_change_dfs(df, meta_df, vars_of_interest, every_x_years=2, normalize_changes=True):

    """
    makes three dataframes, one with var strengths in a particular window, one with var degrees in a particular window, and the last
    with the difference in the var from one window to the next

    TODO: normalize by the range of values of that particular variable
    """
    filtered_df = df.loc[df["YEAR"] < 2019, vars_of_interest]
    filtered_df["SLOT"] = filtered_df["YEAR"] // every_x_years

    if normalize_changes:
        filtered_df = normalize_columns(filtered_df, meta_df, exclude=["YEAR", "SLOT"])

    avg_var_vals = filtered_df.groupby(["SLOT"]).mean()
    avg_var_vals.reset_index()

    change_df = avg_var_vals.iloc[:avg_var_vals.shape[0] - 1,:].reset_index() - avg_var_vals.iloc[1:,:].reset_index()

    years = range(1972, 2019, every_x_years)

    outputs = {}

    for year in years:
        year_sample = [year + i for i in range(every_x_years)]
        df_subset = df[df["YEAR"].isin(year_sample)]        
        variables_list, correlation_matrix = my_pairwise_correlations(vars_of_interest, df_subset,
                                                                            method="spearman", partial=True, 
                                                                            sample_threshold=0, regularization=0.2)
        
        outputs[year] = {"variables": variables_list, "corr_mat": correlation_matrix}

    variable_indices = {var: i for i, var in enumerate(vars_of_interest)}
    degree_matrix = []
    strength_matrix = []

    for year in years:
        correlation_matrix = outputs[year]["corr_mat"]    
        variables_list = outputs[year]["variables"]
        
        strengths = np.abs(correlation_matrix).sum(axis=0) - 1
        degrees = (correlation_matrix != 0).sum(axis=0) - 1
        
        strengths_elongated = np.zeros((len(vars_of_interest),))
        degrees_elongated = np.zeros((len(vars_of_interest),))
        strengths_elongated[[variable_indices[var] for var in variables_list]] = strengths
        degrees_elongated[[variable_indices[var] for var in variables_list]] = degrees

        degree_matrix.append(degrees_elongated)
        strength_matrix.append(strengths_elongated)

    degree_matrix = np.array(degree_matrix)
    strength_matrix = np.array(strength_matrix)

    strength_df = pd.DataFrame({var: strength_matrix[:,i] for i, var in enumerate(vars_of_interest)})
    degree_df = pd.DataFrame({var: degree_matrix[:, i] for i, var in enumerate(vars_of_interest)})

    num_rows = change_df.shape[0]

    strength_df = strength_df.drop("YEAR", axis=1)
    degree_df = degree_df.drop("YEAR", axis=1)
    change_df = change_df.drop("YEAR", axis=1)

    return strength_df.loc[:num_rows - 1,:], degree_df.loc[:num_rows - 1, :], change_df


def difference_first_last_non_nan(column):
    """
    for a dataframe column calculates the difference between the first and last non nan value
    can use with df.apply to do it for every column
    """
    first_non_nan_index = column.first_valid_index()
    last_non_nan_index = column.last_valid_index()
    
    if first_non_nan_index is None or last_non_nan_index is None:
        return np.nan
    
    first_non_nan_value = column[first_non_nan_index]
    last_non_nan_value = column[last_non_nan_index]
    
    return last_non_nan_value - first_non_nan_value

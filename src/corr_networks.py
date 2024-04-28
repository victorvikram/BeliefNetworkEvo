import numpy as np
import pandas as pd
from pingouin import partial_corr
from semopy.polycorr import polychoric_corr

from sklearn.covariance import graphical_lasso

"""
things to do to fix the problem
- unit test all the functions
- see if the problem arises once nans start to get filtered (which variables add problems?)
- possibly require a certain number of samples to include the element in the network 
- see if old code has the same issue 


"""

def my_pairwise_correlations(vars, data, method, partial=True, regularization=0, sample_threshold=0):

    """
    `vars` list of variable names 
    `data` DataFrame with data samples, variable names should be columns in this DataFrame
    `method` one of "spearman", "pearson", or "polychoric"
    `partial` boolean value, whether to calculate partial correlations
    `regularization` the regularization parameter (0 is no regularization)

    calculates all pairwise correlations between the variables in `var` using samples in `data` and one of the
    three methods. If partial, returns partial correlations, if regularization is greater than 0, uses lasso
    regularization on the partials to push some values to 0

    If any pairwise correlations are NaN, variables are removed until the full matrix contains no NaN values

    tests to do:
    - test with known correlations, pearson
    - test with known correlations, spearman
    - test with known correlations, polychoric
    - compare with alt_pairwise correlations
    """
    relevant_df = data.loc[:, vars]

    num_samples = relevant_df.shape[0]
    num_vars = relevant_df.shape[1]

    non_nan_mat = ~np.isnan(np.array(relevant_df))
    sample_count = np.logical_and(non_nan_mat[:, :, np.newaxis], non_nan_mat[:, np.newaxis, :]).sum(axis=0)
    sample_pct = sample_count / num_samples
    # print(sample_pct)
    # print(sample_count)

    if method in ["spearman", "pearson"]:
        corr_mat_pd = relevant_df.corr(method=method)
        corr_mat = np.array(corr_mat_pd)
    elif method == "polychoric":  
        corr_mat = pairwise_polychoric_correlations(vars, data)
        
    if partial:
        corr_mat = np.where(sample_pct < sample_threshold, np.nan, corr_mat) # set variables below the threshold to nan
        corr_mat, i_removed = filter_nans(corr_mat)
        vars = [var for i, var in enumerate(vars) if i not in i_removed]

        # print(corr_mat)

        if regularization == 0 and method != "polychoric":
            corr_mat = corr_mat_to_partial_corr_mat(corr_mat)  
        else:
            corr_mat = cov_mat_to_regularized_partial_corr(corr_mat, alpha=regularization)
            

    return vars, corr_mat

def alt_pairwise_correlations(vars, data, method, partial=True):
    """
    `vars` list of variable names 
    `data` DataFrame with data samples, variable names should be columns in this DataFrame
    `method` one of "spearman", "pearson", or "polychoric"
    `partial` boolean value, whether to calculate partial correlations
  
    calculates all pairwise correlations between `vars`, using the sample DataFrame `data`, and method either
    "spearman" or "pearson". Returns the partial correlations if `partial` is true
    """
    corr_dfs = []
    corr_mat = np.zeros((len(vars), len(vars)))

    for i in range(len(vars)):
        for j in range(i+1, len(vars)):

            if partial:
                covar_list = vars[:i] + vars[i+1:j] + vars[j+1:]
            else:
                covar_list = None
            
            corr_df = partial_corr(data=data, x=vars[i], y=vars[j], covar=covar_list, alternative='two-sided', method=method)
            corr_df["x"] = i
            corr_df["y"] = j
            corr_mat[i, j] = corr_df.loc[method, "r"]
            corr_mat[j, i] = corr_df.loc[method, "r"]
            corr_dfs.append(corr_df)

    corr_info = pd.concat(corr_dfs)
    np.fill_diagonal(corr_mat, 1)
    return corr_info, corr_mat  

def pairwise_polychoric_correlations(vars, data):
    """
    `vars` list of variable names
    `data` are the column names

    calculates a correlation matrix 
    """

    polychor_corr_mat = np.zeros((len(vars), len(vars))) + np.identity(len(vars))
    for i in range(len(vars)):
        for j in range(i + 1, len(vars)):
            subdf = data.loc[:, [vars[i], vars[j]]]
            subdf = subdf[~subdf.isna().any(axis=1)]
            corr = polychoric_corr(subdf.loc[:,vars[i]], subdf.loc[:, vars[j]], x_ints=None, y_ints=None)
            polychor_corr_mat[i, j] = polychor_corr_mat[j, i] = corr

    return polychor_corr_mat

def cov_mat_to_regularized_partial_corr(cov_mat, alpha=0):
    """
    `cov_mat` is a covariance matrix
    `alpha` is the regularization parameter

    takes a covariance matrix and returns the estimated regularized covariances and partial
    correlations. 
    
    Note that a correlation matrix can also be passed in since the correlation matrix is
    simply the covariance of the standardized variables, and the partial correlations between
    the standardized variables should be equal to the partial correlations between the untransformed
    variables

    **tested**
    """
    cov, precision = graphical_lasso(cov_mat, alpha=alpha)
    partial_cor_mat = precision_mat_to_partial_corr(precision)

    return partial_cor_mat


def corr_mat_to_partial_corr_mat(corr_mat):
    """
    `corr_mat` is a numpy array representing a correlation matrix 

    calculates the partial correlation matrix by inverting the correlation matrix 

    **tested**
    """
    precision_mat = np.linalg.inv(corr_mat)
    partial_corr_mat = precision_mat_to_partial_corr(precision_mat)
    return partial_corr_mat

def precision_mat_to_partial_corr(precision_mat):
    """
    `precision_mat` is a numpy array representing the inverse of the correlation matrix

    calculates the partial correlations by correctly scaling the precision matrix

    **tested**
    """
    # Calculate the partial correlation matrix
    outer_product = np.outer(np.diag(precision_mat), np.diag(precision_mat))
    div = np.sqrt(outer_product)
    partial_corr_mat = - precision_mat / div

    # Set diagonal elements to 1
    np.fill_diagonal(partial_corr_mat, 1)

    return partial_corr_mat

def filter_nans(mat):
    """
    `mat` is a symmetric matrix

    the function removes nan values by finding the row/col with the most nans, removes it, and then 
    repeats the process on the shrunken array until there are no more nans

    **tested**
    """
    mat = np.copy(mat)
    is_to_remove = []
    while (nan_sums := np.isnan(mat).sum(axis=0)).sum() > 0:
        i_to_remove = np.argmax(nan_sums)
        mat[i_to_remove,:] = 0
        mat[:, i_to_remove] = 0
        is_to_remove.append(i_to_remove)
    
    mat = np.delete(mat, is_to_remove, axis=0)
    mat = np.delete(mat, is_to_remove, axis=1)

    return mat, is_to_remove
    


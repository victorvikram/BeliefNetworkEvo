import numpy as np
import pandas as pd
from pingouin import partial_corr
from semopy.polycorr import polychoric_corr

from sklearn.covariance import graphical_lasso



def cov_mat_to_regularized_partial_corr(cov_mat, alpha=0):
    """
    takes a covariance matrix and returns the estimated regularized covariances and partial
    correlations. 
    
    note that a corr_mat can also be passed in since the correlation matrix is
    simply the covariance of the standardized variables, and the partial correlations between
    the standardized variables should be equal to the partial correlations between the untransformed
    variables
    """

    cov, precision = graphical_lasso(cov_mat, alpha=alpha)
    partial_cor_mat = precision_mat_to_partial_corr(precision)

    return partial_cor_mat



def precision_mat_to_partial_corr(precision_mat):
    # Calculate the partial correlation matrix
    partial_corr_mat = - precision_mat / np.sqrt(np.outer(np.diag(precision_mat), np.diag(precision_mat)))
    
    # Set diagonal elements to 1
    np.fill_diagonal(partial_corr_mat, 1)

    return partial_corr_mat


def pairwise_polychoric_correlations(vars, data):

    polychor_corr_mat = np.zeros((len(vars), len(vars))) + np.identity(len(vars))
    for i in range(len(vars)):
        for j in range(i + 1, len(vars)):
            corr = polychoric_corr(data.loc[:,vars[i]], data.loc[:, vars[j]], x_ints=None, y_ints=None)
            polychor_corr_mat[i, j] = polychor_corr_mat[j, i] = corr

    return polychor_corr_mat


def my_pairwise_correlations(vars, data, method, partial=True):
    relevant_df = data.loc[:, vars]

    corr_mat_pd = relevant_df.corr(method=method)

    corr_mat = np.array(corr_mat_pd)
    print(corr_mat)


    # ranked_df = relevant_df.rank(axis=0, method="average")
    # corr_mat_test = ranked_df.corr(method="pearson")
    # they should be the same
    # print(corr_mat)
    # print(corr_mat_test)

    if partial:
        corr_mat, i_removed = filter_nans(corr_mat)
        vars = [var for i, var in enumerate(vars) if i not in i_removed]
        precision_mat = np.linalg.inv(corr_mat)
        corr_mat = precision_mat_to_partial_corr(precision_mat)

    return vars, corr_mat

def filter_nans(mat):
    is_to_remove = []
    while (nan_sums := np.isnan(mat).sum(axis=0)).sum() > 0:
        i_to_remove = np.argmax(nan_sums)
        mat[i_to_remove,:] = 0
        mat[:, i_to_remove] = 0
        is_to_remove.append(i_to_remove)

    mat = np.delete(mat, is_to_remove, axis=0)
    mat = np.delete(mat, is_to_remove, axis=1)

    return mat, is_to_remove
    
def pairwise_correlations(vars, data, method, partial=True):
    """
    for list of variables `vars` which are names of columns in the dataframe `data`, calculate the  partial correlations of
    all pairs of variables conditioned on all other variables. use correlation type `method` (either "spearman" or "pearson")

    `partial` is `True` to calculate partial correlations, otherwise it is `False`
    """
    corr_dfs = []
    corr_mat = np.zeros((len(vars), len(vars)))

    for i in range(len(vars)):
        for j in range(i+1, len(vars)):

            if partial:
                covar_list = vars[:i] + vars[i+1:j] + vars[j+1:]
            else:
                covar_list = None

            # print(f"{vars[i]}, {vars[j]}, {covar_list}")
            
            corr_df = partial_corr(data=data, x=vars[i], y=vars[j], covar=covar_list, alternative='two-sided', method=method)
            corr_df["x"] = i
            corr_df["y"] = j
            corr_mat[i, j] = corr_df.loc[method, "r"]
            corr_mat[j, i] = corr_df.loc[method, "r"]
            corr_dfs.append(corr_df)

    corr_info = pd.concat(corr_dfs)
    np.fill_diagonal(corr_mat, 1)
    return corr_info, corr_mat  

def gen_partial_polychor_corr_net():
    return

def lasso_reg_partial_polychor_chor_net():
    return 

def gen_partial_spearman_corr_net(df):
    """
    calculates the partial spearman correlation coefficients between every pair of columns in 
    `df`. Returns that
    """
    pairwise_corr(df, columns=None, covar=None, alternative='two-sided', method='spearman', padjust='none', nan_policy='pairwise')

def gen_full_polychor_corr_net():
    return

def gen_full_spearman_corr_net():
    return


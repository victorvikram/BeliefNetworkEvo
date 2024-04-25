import numpy as np
import pandas as pd
from pingouin import partial_corr

def pairwise_correlations(vars, data, method, partial=True):
    """
    for list of variables `vars` which are names of columns in the dataframe `data`, calculate the  partial correlations of
    all pairs of variables conditioned on all other variables. use correlation type `method` (either "spearman" or "pearson")
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


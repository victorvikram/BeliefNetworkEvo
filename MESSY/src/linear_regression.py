"""
functions that perform different sorts of linear regression
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm

from sklearn.experimental import enable_iterative_imputer 
from sklearn.impute import IterativeImputer

def lin_reg_imputation(table, predictors, response, rank=False):
    """
    does linear regression using data in `table`, using `predictors` as the predictor variables
    and `response` as the response variables (these are the column names). if `rank` is true, does
    regression on the ranked values of the predictors rather than the regular values
    """

    # filter out all the rows where response is nan, isolate the predictor table]
    print(table[response].notna().shape)
    print(table.shape)
    table_pruned = table[table[response].notna()] 
    predictor_table = table_pruned.loc[:,predictors]

    # impute missing values
    imp = IterativeImputer(random_state=0)
    imputed = imp.fit_transform(predictor_table)
    predictor_table = pd.DataFrame(imputed, columns=predictor_table.columns)
    
    # convert to rank if necessary
    if rank:
        table = predictor_table.rank(method="average")
    
    # do the regression
    predictor_arr = predictor_table.values
    y = table_pruned[response].values
    X = sm.add_constant(predictor_arr)
    model = sm.OLS(y, X)
    results = model.fit()

    # print(results.summary())

    return results

    

def make_lin_reg_network(table, relevant_variables, rank=False):

    """
    this function does a linear regression with each variable being the response, with all 
    others being predictors, making an adjacency matrix where entry (i, j) is the coefficient
    of variable j for predicting variable i.
    """

    reg_coefficients = np.zeros(len(relevant_variables), len(relevant_variables))
    for i, resp in enumerate(relevant_variables):
        predictors = [pred for pred in relevant_variables if pred != resp]
        results = lin_reg_imputation(table, predictors, resp, rank)
        coeffs = results.params.values
        reg_coefficients[i,0:i] = coeffs[:i]
        reg_coefficients[i,i+1:] = coeffs[i:]
    
    return reg_coefficients
        
        

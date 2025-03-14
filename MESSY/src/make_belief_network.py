"""
generates belief networks using partial correlations
"""

# Import necessary modules for correlation analysis, graph construction, and visualization.
from corr_networks import my_pairwise_correlations, pairwise_polychoric_correlations, cov_mat_to_regularized_partial_corr
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from graph_tools import create_graph_from_adj_mat

import numpy as np

def make_conditional_belief_network(condition_col, dataframe, condition_method="negpos", 
                                    variables_of_interest=None, years_of_interest=None, method="spearman", is_partial=True, 
                                    threshold=None, sample_threshold=0, regularisation=0):
    
    """
    makes a conditional belief network, conditioning on the column `condition_col`, using the data from `dataframe`. 
    if condition_method is "negpos", it makes two networks, one if `condition_col` is negative, and the other if 
    `condition_col` is positive. if `condition_method` is "unique", it makes a separate network for each unique value
    in `condition_col`. all other parameters function as they do in `make_belief_network()`
    """

    if condition_method == "negpos":
        dataframe["FLAG"] = np.where(dataframe[condition_col] > 0, True, np.where(dataframe[condition_col] < 0, False, np.nan))
    elif condition_method == "unique":
        dataframe["FLAG"] = dataframe[condition_col]

    unique_vals = dataframe["FLAG"].unique()
    vals_to_condition = unique_vals[~np.isnan(unique_vals)]
    # print(vals_to_condition)
    
    outputs = {}
    for val in vals_to_condition:
        sub_df = dataframe.loc[dataframe["FLAG"] == val, :]
        # print(sub_df[condition_col].unique())

        graph, variables_list, correlation_matrix = make_belief_network(sub_df, variables_of_interest, years_of_interest, 
                                                                        method, is_partial, threshold, sample_threshold, regularisation)

        outputs[val] = {"graph": graph, "vars": variables_list, "corr_mat": correlation_matrix}

    return outputs

def convert_graph_to_absolute_value(graph):
    for u, v, data in graph.edges(data=True):
        data['weight'] = abs(data['weight'])
    
    return graph

# Define a function to create a belief network based on correlation analysis of a given DataFrame.
def make_belief_network(dataframe, variables_of_interest=None, years_of_interest=None, method="spearman", 
                        is_partial=True, threshold=0, sample_threshold=0, regularisation=0):
    
    """
    makes a partial correlation network
    `dataframe` is the data we are using 
    `variables_of_interest` will be the nodes in the correlation network (assuming there is enough data to include them)
    `years_of_interest` we only use the rows from the years of interest
    `method` specifies the correlation method, which defaults to "spearman" (which is pearson correlations on the ranks). 
    `is_partial` means that the correlations calculated are partial correlations, which control for the linear effects of other variables,
        using the matrix inversion method
    `threshold` is how high the correlation must be to be included in the network
    `sample_threshold` is the number of samples a variable needs to have that are non-nan for it to be included
    `regularisation` is the regularisation parameter
    """
    # Start with the full dataframe and filter it according to specified years if provided.
    df_subset = dataframe
    if years_of_interest:
        df_subset = dataframe[dataframe["YEAR"].isin(years_of_interest)]
    
    # Further filter the dataframe to only include specified variables if provided.
    if variables_of_interest:
        df_subset = df_subset[variables_of_interest]

    # Extract column names as a list, these represent all variables under consideration.
    all_variables = list(df_subset.columns)
    
    # Calculate pairwise correlations (and associated information) between variables using the specified method.
    variables_list, correlation_matrix = my_pairwise_correlations(all_variables, df_subset, method, partial=is_partial, regularization=regularisation, sample_threshold=sample_threshold)

    # Initialize an undirected graph to represent the belief network.
    graph = create_graph_from_adj_mat(correlation_matrix, variables_list, threshold=threshold)

    # Return the graph, correlation information, and the correlation matrix itself.
    return graph, variables_list, correlation_matrix
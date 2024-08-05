# Import necessary modules for correlation analysis, graph construction, and visualization.
from corr_networks import my_pairwise_correlations, pairwise_polychoric_correlations, cov_mat_to_regularized_partial_corr
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import numpy as np

def make_conditional_belief_network(condition_col, dataframe, condition_method="negpos", 
                                    variables_of_interest=None, years_of_interest=None, method="spearman", is_partial=True, 
                                    threshold=None, sample_threshold=0, regularisation=0):
    

    if condition_method == "negpos":
        dataframe["FLAG"] = np.where(dataframe[condition_col] > 0, True, np.where(dataframe[condition_col] < 0, False, np.nan))

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
                        is_partial=True, threshold=None, sample_threshold=0, regularisation=0):

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
    graph = nx.Graph()
    
    # Add edges between nodes (variables) with significant correlations as per the threshold.
    for i in range(len(variables_list)):
        for j in range(i+1, len(variables_list)):
            if threshold:
                # If a threshold is specified, only add an edge if the absolute correlation exceeds it.
                if abs(correlation_matrix[i, j]) > threshold:
                    graph.add_edge(variables_list[i], variables_list[j], weight=correlation_matrix[i, j])
            else:
                if abs(correlation_matrix[i, j]) > 0:
                    # If no threshold is specified, add an edge for all pairs.
                    graph.add_edge(variables_list[i], variables_list[j], weight=correlation_matrix[i, j])

    # Return the graph, correlation information, and the correlation matrix itself.
    return graph, variables_list, correlation_matrix
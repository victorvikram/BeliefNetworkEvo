from corr_networks import my_pairwise_correlations
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

def make_degree_strength_change_dfs(df, vars_of_interest, every_x_years):

    """
    makes three dataframes, one with var strengths in a particular window, one with var degrees in a particular window, and the last
    with the difference in the var from one window to the next

    TODO: normalize by the range of values of that particular variable
    """
    filtered_df = df.loc[df["YEAR"] < 2019, vars_of_interest]
    filtered_df["SLOT"] = filtered_df["YEAR"] // every_x_years
    avg_var_vals = filtered_df.groupby(["SLOT"]).mean()
    avg_var_vals.reset_index()

    print(filtered_df.max())

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

def make_degree_strength_change_heatmap_figure(strength_df, degree_df, change_df, degree_bin_size=2, strength_bin_size=0.12):
    scatter_fig, scatter_ax = plt.subplots()  # Create a new figure for each column pair
    scatter_ax.set_title(f"Scatterplot of strength vs degree")
    scatter_ax.set_xlabel(f"strength")
    scatter_ax.set_ylabel(f"degree")

    strength_ax = strength_df.max().max()
    degree_ax = degree_df.max().max()
    print(strength_ax)
    print(degree_ax)

    value_count = np.zeros((np.floor(degree_ax / degree_bin_size).astype(int) + 1, np.floor(strength_ax / strength_bin_size).astype(int) + 1))
    value_sum = np.zeros((np.floor(degree_ax / degree_bin_size).astype(int) + 1, np.floor(strength_ax / strength_bin_size).astype(int) + 1))

    for col in strength_df.columns:
        diffs = np.abs(np.array(change_df[col]))
        incs = np.where(~np.isnan(diffs), 1, 0)
        vals = np.where(~np.isnan(diffs), diffs, 0)

        np.add.at(value_count, (np.floor(np.array(degree_df[col] / degree_bin_size)).astype(int), np.floor(np.array(strength_df[col]) / strength_bin_size).astype(int)), incs)
        
        np.add.at(value_sum, (np.floor(np.array(degree_df[col] / degree_bin_size)).astype(int), np.floor(np.array(strength_df[col]) / strength_bin_size).astype(int)), 
                                vals)
        
        scatter_ax.scatter(strength_df[col], degree_df[col], s=change_df[col].abs() * 1000)
        # plt.scatter(trimmed_degree_df[col], change_df[col].abs(), s=trimmed_strength_df[col] * 100)

        means = value_sum / value_count

        cutoff = np.percentile(means[~np.isnan(means)], 90)

        chopped_means = np.where(means > cutoff, cutoff, means)

        heatmap_fig, heatmap_ax = plt.subplots()
        heatmap_ax.imshow(chopped_means, cmap="plasma")

        # Add annotations (numeric values) on top of each heatmap cell
        for i in range(chopped_means.shape[0]):
            for j in range(chopped_means.shape[1]):

                if ~np.isnan(chopped_means[i, j]):
                    plt.text(j, i, f'{means[i, j]:.2f}', ha='center', va='center', color='black', fontsize=6)

        
        return scatter_ax, heatmap_ax


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

"""
this file contains code to analyze how the networks change over time
"""

import pandas as pd

def align_dfs(df1, df2, method='union'):
    """
    reindexes the rows and columns of the dataframe so that they have the same variable set
    and columns of df1 and df2. fills missing rows and columns with 0s

    `method` can be 'union' (default) or 'intersection'.
    `union` makes the variables the union 
    `intersection` makes the variables the intersection
    """
    # Get the union of the indices and columns from both DataFrames
    all_vars = sorted(set(df1.index).union(set(df2.index)))
    
    if method == "union":
        # Reindex both DataFrames to include all variables, filling missing values with zeros
        df1_aligned = df1.reindex(index=all_vars, columns=all_vars, fill_value=0)
        df2_aligned = df2.reindex(index=all_vars, columns=all_vars, fill_value=0)
    elif method == "intersection":
        # Get the intersection of the indices and columns from both DataFrames
        common_vars = sorted(set(df1.index).intersection(set(df2.index)))
        df1_aligned = df1.reindex(index=common_vars, columns=common_vars, fill_value=0)
        df2_aligned = df2.reindex(index=common_vars, columns=common_vars, fill_value=0)

    return df1_aligned, df2_aligned

def align_dfs_multiple(dfs):
    """
    reindexes rows and columns of all dataframes in `dfs` so they all have the same variable set in the rows and columns
    fills missing rows and columns with 0s
    """

    all_vars = sorted(set().union(*(set(df.index).union(set(df.columns)) for df in dfs)))

    dfs_aligned = [df.reindex(index=all_vars, columns=all_vars, fill_value=0) for df in dfs]

    return dfs_aligned

def subtract_dataframes(df1, df2, mismatch_method='intersection'):
    """
    this function takes two dataframes (`df1` and `df2`) and returns a new dataframe that is the result of subtracting `df2` from `df1`
    it aligns the indices and columns of both dataframes, filling any missing values with zeros before performing the subtraction

    these dataframes can be the correlation dataframes that have the variable names as their index and columns. 
    """
    # Align the DataFrames to ensure they have the same indices and columns
    df1_aligned, df2_aligned = align_dfs(df1, df2, method=mismatch_method)
    
    # Subtract the aligned DataFrames
    result = df1_aligned - df2_aligned
    
    return result
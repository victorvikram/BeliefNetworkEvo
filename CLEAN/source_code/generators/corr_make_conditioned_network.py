import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from typing import Optional, Union, Dict, Any, List, Tuple

# Add the root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath("..")))
if project_root not in sys.path:
    sys.path.append(project_root)
from CLEAN.source_code.generators.corr_make_network import CorrelationMethod, EdgeSuppressionMethod, calculate_correlation_matrix



def calculate_conditioned_correlation_matrix(
    df: pd.DataFrame,       # Parameters for regular correlation matrix
    variables_of_interest: Optional[List[str]] = None,
    years_of_interest: Optional[List[int]] = None,
    method: Union[str, CorrelationMethod] = CorrelationMethod.SPEARMAN,
    partial: bool = False,
    edge_suppression: Union[str, EdgeSuppressionMethod] = EdgeSuppressionMethod.NONE,
    suppression_params: Optional[Dict[str, Any]] = None,     

    variable_to_condition: Optional[str] = None,     # Parameters for conditioned correlation matrix                            
    condition: Optional[str] = None,
    value: Optional[Any] = None,
    return_df: bool = False
) -> pd.DataFrame:
    """
    Filter a DataFrame based on a condition applied to a specified column.


    Parameters:
    df (pd.DataFrame): The input DataFrame.
    column (str): The column name to apply the condition on.
    condition (str): The condition to apply. Must be one of ['equal_to', 'less_than_zero', 'greater_than_zero'].
    value (optional): The value to compare against if condition is 'equal_to'.

    Returns:
    pd.DataFrame: The filtered DataFrame.
    """  

    # If no conditioning variable is specified, calculate regular correlation matrix
    if variable_to_condition is None:
        return calculate_correlation_matrix(
            df,
            variables_of_interest=variables_of_interest,
            years_of_interest=years_of_interest,
            method=method,
            partial=partial,
            edge_suppression=edge_suppression,
            suppression_params=suppression_params
        )

    # Check if the variable to condition is in the DataFrame
    if variable_to_condition not in df.columns:
        raise ValueError(f"Column '{variable_to_condition}' not found in DataFrame")
    
    # Check if contains string values
    if df[variable_to_condition].dtype == 'object':
        raise ValueError(f"Column '{variable_to_condition}' contains string values. Are you using the raw dataset by any chance? Make sure to clean the dataset before using this function.")

    # Apply the condition to the DataFrame
    if condition == 'equal_to':
        if value is None:
            raise ValueError("Value must be provided for 'equal' condition")
        df = df[df[variable_to_condition] == value]
    elif condition == 'less_than_zero':
        df = df[df[variable_to_condition] < 0]
    elif condition == 'greater_than_zero':
        df = df[df[variable_to_condition] > 0]

    else:
        raise ValueError("Invalid condition. Must be 'equal_to', 'less_than_zero', or 'greater_than_zero'")

    # Calculate the correlation matrix
    conditioned_corr_matrix = calculate_correlation_matrix(
        df,
        variables_of_interest=variables_of_interest,
        years_of_interest=years_of_interest,
        method=method,
        partial=partial,
        edge_suppression=edge_suppression,
        suppression_params=suppression_params)
    
    # Return the conditioned correlation matrix or the filtered DataFrame
    if return_df:
        return conditioned_corr_matrix, df
    else:
        return conditioned_corr_matrix


# Example usage:
if __name__ == "__main__":
    
    data = {'A': [1, -2, 3, -4, 5], 'B': [10, 20, -30, 40, -50]}
    
    df = pd.DataFrame(data)
    
    conditioned_corr_matrix, conditioned_df = calculate_conditioned_correlation_matrix(
        df, 
        variable_to_condition='A', 
        condition='equal_to',
        value=-2,
        return_df=True)
    print(conditioned_corr_matrix)
    print("\n")
    print(conditioned_df)
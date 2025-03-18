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
from source_code.generators.corr_make_network import CorrelationMethod, EdgeSuppressionMethod, calculate_correlation_matrix



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
    return_df: bool = False,
    verbose: bool = False
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Filter a DataFrame based on a condition applied to a specified column and calculate correlation matrix.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    variables_of_interest (Optional[List[str]]): List of variables to include in the correlation matrix.
    years_of_interest (Optional[List[int]]): List of years to include in the correlation matrix.
    method (Union[str, CorrelationMethod]): The method to use for calculating the correlation matrix.
    partial (bool): Whether to calculate a partial correlation matrix.
    edge_suppression (Union[str, EdgeSuppressionMethod]): The method to use for edge suppression.
    suppression_params (Optional[Dict[str, Any]]): Parameters for edge suppression.
    variable_to_condition (Optional[str]): The variable to condition on.
    condition (Optional[str]): The condition to apply. Must be one of ['equal_to', 'less_than_zero', 'greater_than_zero'].
    value (Optional[Any]): The value to compare against if condition is 'equal_to'.
    return_df (bool): Whether to return the filtered DataFrame.
    verbose (bool): Whether to print information about filtered variables and sample sizes. Default is True.

    Returns:
    Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]: 
        - If return_df=False: Returns only the correlation matrix.
        - If return_df=True: Returns (correlation_matrix, filtered_df).
    """  

    # If no conditioning variable is specified, calculate regular correlation matrix
    if variable_to_condition is None:
        result = calculate_correlation_matrix(
            df,
            variables_of_interest=variables_of_interest,
            years_of_interest=years_of_interest,
            method=method,
            partial=partial,
            edge_suppression=edge_suppression,
            suppression_params=suppression_params,
            verbose=verbose
        )
        
        if return_df:
            return result, df
        else:
            return result

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
        filtered_df = df[df[variable_to_condition] == value]
    elif condition == 'less_than_zero':
        filtered_df = df[df[variable_to_condition] < 0]
    elif condition == 'greater_than_zero':
        filtered_df = df[df[variable_to_condition] > 0]
    else:
        raise ValueError("Invalid condition. Must be 'equal_to', 'less_than_zero', or 'greater_than_zero'")

    # Print conditioning information if verbose is True
    if verbose:
        print("\n" + "="*50)
        print("CONDITIONING INFORMATION")
        print("="*50)
        print(f"Conditioning variable: {variable_to_condition}")
        print(f"Condition: {condition}")
        if condition == 'equal_to':
            print(f"Value: {value}")
        print(f"Filtered samples count: {len(filtered_df)} of {len(df)} ({len(filtered_df)/len(df)*100:.1f}%)")
        print("-"*50)

    # Calculate the correlation matrix
    result = calculate_correlation_matrix(
        filtered_df,
        variables_of_interest=variables_of_interest,
        years_of_interest=years_of_interest,
        method=method,
        partial=partial,
        edge_suppression=edge_suppression,
        suppression_params=suppression_params,
        verbose=verbose
    )
    
    # Return results based on the requested output format
    if return_df:
        return result, filtered_df
    else:
        return result


# Example usage:
if __name__ == "__main__":
    
    data = {'A': [1, -2, 3, -4, 5], 'B': [10, 20, -30, 40, -50]}
    
    df = pd.DataFrame(data)
    
    conditioned_corr_matrix, conditioned_df = calculate_conditioned_correlation_matrix(
        df, 
        variable_to_condition='A', 
        condition='equal_to',
        value=-2,
        return_df=True,
        verbose=True)
    print(conditioned_corr_matrix)
    print("\n")
    print(conditioned_df)
"""
This module creates correlation matrices (belief networks) from pandas DataFrames.
It supports multiple correlation methods and edge suppression techniques for network analysis.

The module provides functions for:
- Calculating correlations between variables using various methods
- Computing partial correlations to identify direct relationships
- Applying edge suppression techniques
- Automatically excludes metadata variables (YEAR, BALLOT, ID) from correlation calculations
  while preserving them in the original DataFrame

Available correlation methods:
- Spearman: Rank correlation for non-linear relationships
- Pearson: Linear correlation between variables

Each method can be used with partial correlations to control for other variables' effects.

Edge suppression techniques:
- Square values: Emphasize stronger relationships
- (Future) Threshold-based: Remove edges below certain values
- (Future) Regularization: L1/L2 penalties on edge weights
- (Future) Statistical significance: Filter by p-values

Example:
    >>> df = pd.DataFrame({'A': [1,2,3], 'B': [2,4,6], 'YEAR': [2020,2020,2020]})
    >>> # Calculate partial Spearman correlations with squared edge suppression
    >>> corr_matrix = calculate_correlation_matrix(
    ...     df, 
    ...     method='spearman',
    ...     partial=True,
    ...     edge_suppression='square'
    ... )
"""

from enum import Enum
from typing import Optional, Union, Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from pingouin import partial_corr

from sklearn.covariance import graphical_lasso


class CorrelationMethod(Enum):
    """Available correlation methods for network analysis."""
    SPEARMAN = 'spearman'  # Non-linear rank correlation
    PEARSON = 'pearson'    # Linear correlation


class EdgeSuppressionMethod(Enum):
    """Methods for suppressing weak or spurious edges in the network."""
    NONE = 'none'          # No edge suppression
    SQUARE = 'square'      # Square correlations to emphasize strong relationships
    THRESHOLD = 'threshold'      # Remove edges below threshold (future)
    REGULARIZATION = 'regularization'  # L1/L2 penalties (future)
    STATISTICAL = 'statistical'    # Statistical significance (future)


def calculate_partial_correlations(correlation_matrix: np.ndarray, 
                                 min_eigenval: float = 1e-10) -> np.ndarray:
    """
    Calculate partial correlations from a correlation matrix.
    
    Partial correlations measure the relationship between two variables while
    controlling for all other variables in the dataset. This helps identify
    direct relationships by removing indirect effects.
    
    Method:
    This implementation uses the matrix inversion method to calculate partial correlations.
    Given a correlation matrix R, the partial correlation ρ_ij|rest between variables
    i and j controlling for all other variables is:
    
    ρ_ij|rest = -P_ij / sqrt(P_ii * P_jj)
    
    where P = R^(-1) is the precision matrix (inverse of correlation matrix).
    
    This method is equivalent to:
    1. Regressing each variable on all other variables
    2. Computing correlations between the residuals
    But is much more computationally efficient for a full partial correlation matrix.
    
    The precision matrix P contains the partial correlation information because its
    elements P_ij are proportional to the partial correlation between variables i and j,
    but need to be properly scaled by the diagonal elements to get correlations.
    
    Args:
        correlation_matrix: Input correlation matrix (Pearson or Spearman)
        min_eigenval: Minimum eigenvalue for numerical stability
        
    Returns:
        np.ndarray: Matrix of partial correlations
        
    Raises:
        ValueError: If correlation matrix is singular or nearly singular
    """
    # Debug: Check input correlation matrix
    if np.any(np.isnan(correlation_matrix)):
        raise ValueError("Input correlation matrix contains NaN values")
    if np.any(np.isinf(correlation_matrix)):
        raise ValueError("Input correlation matrix contains infinite values")
    
    # Check matrix condition
    eigenvals = np.linalg.eigvals(correlation_matrix)
    
    # Debug: Check eigenvalues
    if np.any(np.isnan(eigenvals)):
        raise ValueError("Eigenvalue calculation produced NaN values")
    if np.any(np.isinf(eigenvals)):
        raise ValueError("Eigenvalue calculation produced infinite values")
    
    if np.min(np.abs(eigenvals)) < min_eigenval:
        raise ValueError(
            "Correlation matrix is nearly singular. This usually indicates "
            "highly collinear variables or insufficient data."
        )
    
    # Step 1: Calculate precision matrix (inverse of correlation matrix)
    precision_matrix = np.linalg.inv(correlation_matrix)
    
    # Debug: Check precision matrix
    if np.any(np.isnan(precision_matrix)):
        raise ValueError("Precision matrix contains NaN values")
    if np.any(np.isinf(precision_matrix)):
        raise ValueError("Precision matrix contains infinite values")
    
    # Step 2: Convert precision matrix to partial correlations
    partial_correlations = precision_mat_to_partial_corr(precision_matrix)

    return partial_correlations

def precision_mat_to_partial_corr(precision_matrix):
    """
    `precision_matrix` is a numpy array representing the inverse of the correlation matrix

    calculates the partial correlations by correctly scaling the precision matrix

    **tested**
    """
    diag_precision = np.diag(precision_matrix)
    
    # Debug: Check diagonal elements
    if np.any(np.isnan(diag_precision)):
        raise ValueError("Diagonal of precision matrix contains NaN values")
    if np.any(np.isinf(diag_precision)):
        raise ValueError("Diagonal of precision matrix contains infinite values")
    if np.any(diag_precision <= 0):
        raise ValueError("Diagonal of precision matrix contains non-positive values")
    
    outer_product = np.outer(diag_precision, diag_precision)
    div = np.sqrt(outer_product)
    partial_correlations = -precision_matrix / div
    
    # Debug: Check final partial correlations
    if np.any(np.isnan(partial_correlations)):
        raise ValueError("Partial correlation calculation produced NaN values")
    if np.any(np.isinf(partial_correlations)):
        raise ValueError("Partial correlation calculation produced infinite values")
    
    # Step 3: Ensure proper correlation matrix properties
    np.fill_diagonal(partial_correlations, 1.0)
    
    return partial_correlations


def calculate_regularized_partial_correlations(cov_mat, alpha=0.1):
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

def suppress_edges(
    correlation_matrix: pd.DataFrame,
    method: Union[str, EdgeSuppressionMethod] = EdgeSuppressionMethod.NONE,
    params: Optional[Dict[str, Any]] = None
) -> pd.DataFrame:
    """
    Apply edge suppression to reduce noise in the correlation network.
    
    Args:
        correlation_matrix: Input correlation matrix
        method: Edge suppression method to apply
        params: Additional parameters for the suppression method
        
    Returns:
        DataFrame: Processed correlation matrix with suppressed edges
        
    Raises:
        ValueError: If unknown suppression method is specified
    """
    if isinstance(method, str):
        method = EdgeSuppressionMethod(method)
    
    params = params or {}
    
    if method == EdgeSuppressionMethod.NONE:
        return correlation_matrix
    elif method == EdgeSuppressionMethod.SQUARE:
        return correlation_matrix ** 2
    elif method == EdgeSuppressionMethod.THRESHOLD:
        raise NotImplementedError("Threshold-based edge suppression not yet implemented")
    elif method == EdgeSuppressionMethod.REGULARIZATION:
        # regularization happens upstread of this function
        return correlation_matrix
    elif method == EdgeSuppressionMethod.STATISTICAL:
        raise NotImplementedError("Statistical edge suppression not yet implemented")

    raise ValueError(f"Unknown edge suppression method: {method}")


def get_correlation_columns(df: pd.DataFrame, 
                            base_variable_list: Optional[List[str]] = None
                            ) -> List[str]:
    """
    Get columns to include in correlation calculations, excluding metadata.
    
    If you pass in base_variable_list, it will return those variables 
    excluding the metadata variables 

    Args:
        df: Input DataFrame
        
    Returns:
        List[str]: Column names to use for correlation calculation
    """
    if not base_variable_list:
        base_variable_list = df.columns.tolist()
    
    metadata_vars = {'YEAR', 'BALLOT', 'ID'}  # Using set for O(1) lookup

    return [var for var in base_variable_list if var not in metadata_vars]


def filter_nans(correlation_matrix: np.ndarray) -> Tuple[np.ndarray, List[int]]:
    """
    Remove variables with NaN correlations from a correlation matrix.
    
    This function handles missing correlations by iteratively removing variables
    that have the most NaN correlations with other variables. This is necessary
    for partial correlation calculation which requires a complete correlation matrix
    with no missing values.
    
    Method:
    1. Find the variable (row/column) with the most NaN correlations
    2. Remove that variable from the matrix
    3. Repeat until no NaN values remain
    
    This approach maximizes the number of remaining variables while ensuring
    we have a complete correlation matrix for partial correlation calculation.
    
    Note that this is different from simply dropping NaN values, as we need
    to maintain the symmetry of the correlation matrix and ensure we have
    enough information to calculate meaningful partial correlations.
    
    Args:
        correlation_matrix: Square correlation matrix potentially containing NaN values.
                          Must be symmetric (correlation_matrix[i,j] = correlation_matrix[j,i])
        
    Returns:
        Tuple containing:
        - np.ndarray: Clean correlation matrix with NaN-containing rows/columns removed
        - List[int]: Indices of removed variables in the original matrix
        
    Example:
        >>> corr_matrix = np.array([[1, np.nan, 0.5], 
        ...                         [np.nan, 1, 0.3],
        ...                         [0.5, 0.3, 1]])
        >>> clean_matrix, removed = filter_nans(corr_matrix)
        >>> print(removed)
        [1]  # Second variable was removed due to NaN correlation
    """
    # Create a copy to avoid modifying the input
    working_matrix = np.copy(correlation_matrix)
    removed_indices = []
    
    # Continue removing variables until no NaNs remain
    while np.isnan(working_matrix).any():
        # Count NaN values for each variable
        nan_counts = np.isnan(working_matrix).sum(axis=0)
        
        # Find variable with most NaN correlations
        worst_var_idx = np.argmax(nan_counts)
        
        # Zero out this variable's correlations (temporary step before deletion)
        working_matrix[worst_var_idx, :] = 0
        working_matrix[:, worst_var_idx] = 0
        
        # Keep track of removed variables
        removed_indices.append(worst_var_idx)
    
    # Remove all problematic variables at once
    clean_matrix = np.delete(working_matrix, removed_indices, axis=0)
    clean_matrix = np.delete(clean_matrix, removed_indices, axis=1)
    
    return clean_matrix, removed_indices


def calculate_correlation_matrix(
    df: pd.DataFrame,
    variables_of_interest: Optional[List[str]] = None,
    years_of_interest: Optional[List[int]] = None,
    method: Union[str, CorrelationMethod] = CorrelationMethod.SPEARMAN,
    partial: bool = False,
    edge_suppression: Union[str, EdgeSuppressionMethod] = EdgeSuppressionMethod.NONE,
    suppression_params: Optional[Dict[str, Any]] = None
) -> pd.DataFrame:
    """
    Calculate correlation matrix for network analysis with optional edge suppression.
    
    This function calculates correlations between variables using either Pearson or
    Spearman methods, with the option to compute partial correlations. It automatically
    handles missing values and excludes metadata variables from the calculations.
    
    Method:
    1. Calculate base correlations (Pearson or Spearman)
    2. If partial=True:
       a. Remove variables with NaN correlations
       b. Calculate partial correlations using matrix inversion
    3. Apply edge suppression if specified
    
    The partial correlation calculation controls for all other variables when computing
    the correlation between any two variables. This helps identify direct relationships
    by removing indirect effects through other variables.
    
    Args:
        df: Input DataFrame containing the variables to analyze
        method: Base correlation method ('spearman' for non-linear or 'pearson' for linear)
        partial: Whether to compute partial correlations (default: False)
        edge_suppression: Method to reduce weak or spurious edges
        suppression_params: Additional parameters for edge suppression
        
    Returns:
        DataFrame: Processed correlation matrix. If partial=True, this will be a partial
                 correlation matrix potentially containing fewer variables than the input
                 if some had to be removed due to missing correlations.
        
    Raises:
        ValueError: If fewer than 2 valid variables remain after filtering
                   or if the correlation matrix is singular (for partial correlations)
    """
    if isinstance(method, str):
        method = CorrelationMethod(method)
    
    # Start with the full dataframe and filter it according to specified years if provided.
    df_subset = df

    if years_of_interest:
        df_subset = df[df["YEAR"].isin(years_of_interest)]
    
    # Get non-metadata columns for correlation calculation, and filter the dataset
    # also only uses the variables_of_interest columns
    correlation_cols = get_correlation_columns(df, base_variable_list=variables_of_interest)
    df_subset = df_subset[correlation_cols]

    if len(correlation_cols) < 2:
        raise ValueError("Need at least 2 non-metadata columns for correlation analysis")
    
    # Calculate base correlations using specified method
    correlation_matrix = df_subset.corr(method=method.value)
    
    # Handle partial correlations if requested
    if partial:
        try:
            # Remove variables with NaN correlations
            clean_matrix, removed_indices = filter_nans(correlation_matrix.values)
            
            # Track remaining variables after NaN removal
            correlation_cols = [col for i, col in enumerate(correlation_cols) 
                            if i not in removed_indices]
            
            if len(correlation_cols) < 2:
                raise ValueError(
                    "Too many variables removed due to missing correlations. "
                    "Need at least 2 variables to calculate partial correlations."
                )
            
            # Calculate partial correlations on the clean matrix
            if edge_suppression == EdgeSuppressionMethod.REGULARIZATION:
                if suppression_params is None or "regularization" not in suppression_params:
                    raise ValueError("Regularization parameter 'regularization' must be provided in suppression_params")
                alpha = suppression_params["regularization"]
                partial_correlations = calculate_regularized_partial_correlations(clean_matrix, alpha=alpha)
            else:
                partial_correlations = calculate_partial_correlations(clean_matrix)
            
            # Create new DataFrame with the partial correlations
            correlation_matrix = pd.DataFrame(
                partial_correlations,
                index=correlation_cols,
                columns=correlation_cols
            )
        except ValueError as e:
            raise ValueError(f"Failed to calculate partial correlations: {str(e)}")
    
    # Set diagonal to 0 for network analysis
    # (self-correlations aren't meaningful for network visualization)
    np.fill_diagonal(correlation_matrix.values, 0)
    
    # Apply any requested edge suppression
    return suppress_edges(
        correlation_matrix,
        method=edge_suppression,
        params=suppression_params
    )

def alternative_calculate_pairwise_correlations(vars, data, method, partial=True):
    """
    `vars` list of variable names 
    `data` DataFrame with data samples, variable names should be columns in this DataFrame
    `method` one of "spearman", "pearson", or "polychoric"
    `partial` boolean value, whether to calculate partial correlations
  
    calculates all pairwise correlations between `vars`, using the sample DataFrame `data`, and method either
    "spearman" or "pearson". Returns the partial correlations if `partial` is true
    
    This is for testing purposes to maek sure that the other vectorized function works
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
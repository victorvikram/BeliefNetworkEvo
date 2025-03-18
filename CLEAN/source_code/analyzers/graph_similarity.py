from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, Union, List, Set
from dataclasses import dataclass

@dataclass
class SimilarityResult:
    """
    Data class to store similarity computation results.
    
    Attributes:
        score (float): The computed similarity/distance score
        method (str): Name of the method used
        metadata (Dict): Additional information about the computation
        normalized_score (float, optional): Score normalized by theoretical maximum
    """
    score: float
    method: str
    metadata: Dict = None
    normalized_score: Optional[float] = None

    def __repr__(self) -> str:
        """Custom string representation for better readability."""
        metadata_str = ', '.join(f"{key}: {value}" for key, value in (self.metadata or {}).items())
        normalized_str = f"  Normalized_score: {self.normalized_score:.4f},\n" if self.normalized_score is not None else ""
        return (f"SimilarityResult(\n"
                f"  Score: {self.score:.4f},\n"
                f"{normalized_str}"
                f"  Method: '{self.method}',\n"
                f"  Metadata: {{{metadata_str}}}\n"
                f")")

class GraphSimilarityBase(ABC):
    """Abstract base class for graph similarity measures."""
    
    @property
    @abstractmethod
    def required_parameters(self) -> Set[str]:
        """Required parameters for this similarity measure."""
        pass
    
    @property
    @abstractmethod
    def optional_parameters(self) -> Set[str]:
        """Optional parameters for this similarity measure."""
        pass
    
    def validate_parameters(self, params: Dict) -> None:
        """Validate that all required parameters are present."""
        missing = self.required_parameters - set(params.keys())
        if missing:
            raise ValueError(
                f"{self.__class__.__name__} requires parameters: {missing}"
            )
        
        invalid = set(params.keys()) - (self.required_parameters | self.optional_parameters)
        if invalid:
            raise ValueError(
                f"Invalid parameters for {self.__class__.__name__}: {invalid}"
            )
    
    @abstractmethod
    def compute(self, 
                matrix1: pd.DataFrame, 
                matrix2: pd.DataFrame, 
                **kwargs) -> Union[float, Tuple[float, Optional[float]]]:
        """
        Compute similarity between two correlation matrices.
        
        Returns:
            Either a float score or a tuple of (score, normalized_score)
        """
        pass
    
    def _validate_input(self, 
                       matrix1: pd.DataFrame, 
                       matrix2: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Validate and preprocess input matrices."""
        # Ensure both matrices have the same columns
        cols1 = set(matrix1.columns)
        cols2 = set(matrix2.columns)
        
        # Find missing columns in matrix1 and add NaN columns
        for col in cols2 - cols1:
            matrix1[col] = np.nan
        
        # Find missing columns in matrix2 and add NaN columns
        for col in cols1 - cols2:
            matrix2[col] = np.nan
        
        # Ensure both matrices have the same rows
        index1 = set(matrix1.index)
        index2 = set(matrix2.index)
        
        # Find missing rows in matrix1 and add NaN rows
        for idx in index2 - index1:
            matrix1.loc[idx] = np.nan
        
        # Find missing rows in matrix2 and add NaN rows
        for idx in index1 - index2:
            matrix2.loc[idx] = np.nan
        
        # Reorder both matrices to match the same row and column order
        matrix1 = matrix1.reindex(index=matrix2.index, columns=matrix2.columns)
        matrix2 = matrix2.reindex(index=matrix1.index, columns=matrix1.columns)
        
        m1 = matrix1.to_numpy()
        m2 = matrix2.to_numpy()
        
        if not (m1.shape == m2.shape):
            print(m1.shape, m2.shape)
            raise ValueError("Matrices must have the same dimensions")
        
        # Check for symmetry, treating np.nan as equal
        if not (np.allclose(m1, m1.T, equal_nan=True) and np.allclose(m2, m2.T, equal_nan=True)):
            raise ValueError("Matrices must be symmetric")
        return m1, m2
    

class GraphEditDistance(GraphSimilarityBase):
    """Graph Edit Distance similarity measure."""
    
    @property
    def required_parameters(self) -> Set[str]:
        return {'edge_threshold'}
    
    @property
    def optional_parameters(self) -> Set[str]:
        return set()  # No optional parameters for this method
    
    def compute(self, 
                matrix1: pd.DataFrame, 
                matrix2: pd.DataFrame,
                **kwargs) -> Tuple[float, Optional[float]]:
        """
        Compute Graph Edit Distance between two correlation matrices.
        
        Args:
            matrix1, matrix2: Correlation matrices to compare
            edge_threshold: Threshold for considering an edge present
        Returns:
            Tuple[float, float]: GED score and normalized score
        """
        self.validate_parameters(kwargs)
        edge_threshold = kwargs['edge_threshold']
        
        m1, m2 = self._validate_input(matrix1, matrix2)
        
        # Convert to adjacency matrices
        adj1 = (np.abs(m1) >= edge_threshold).astype(int)
        adj2 = (np.abs(m2) >= edge_threshold).astype(int)
        
        # Compute normalized edge differences
        edge_diff = np.sum(np.abs(adj1 - adj2)) / 2
        
        # Calculate theoretical maximum difference (total possible edges in a graph)
        max_possible_diff = m1.shape[0] * (m1.shape[0] - 1) / 2
        normalized_edge_diff = edge_diff / max_possible_diff if max_possible_diff > 0 else 0

        return edge_diff, normalized_edge_diff



class SpectralSimilarity(GraphSimilarityBase):
    """Spectral similarity measure using eigenvalues.
    
    This method is based on the idea that the eigenvalues of a matrix are a good representation of the network structure.
    The similarity is computed by taking the absolute difference between the eigenvalues of the two matrices and then normalizing by the maximum possible difference.

    """
    
    @property
    def required_parameters(self) -> Set[str]:
        return {'num_eigenvalues'}
    
    @property
    def optional_parameters(self) -> Set[str]:
        return {'abs_values'}
    
    def compute(self,
                matrix1: pd.DataFrame,
                matrix2: pd.DataFrame,
                **kwargs) -> float:
        """
        Compute spectral similarity between matrices.
        
        Args:
            matrix1, matrix2: Correlation matrices to compare
            num_eigenvalues: Number of eigenvalues to compare
            abs_values: Whether to use absolute eigenvalues (default: True)
            
        Returns:
            float: Spectral similarity score
        """
        self.validate_parameters(kwargs)
        n = kwargs['num_eigenvalues']
        use_abs = kwargs.get('abs_values', True)
        
        m1, m2 = self._validate_input(matrix1, matrix2)
        
        eig1 = np.linalg.eigvals(m1)
        eig2 = np.linalg.eigvals(m2)
        
        if use_abs:
            eig1, eig2 = np.abs(eig1), np.abs(eig2)
        
        # Sort in descending order and take top n
        eig1 = -np.sort(-eig1)[:n]
        eig2 = -np.sort(-eig2)[:n]
        
        return np.mean(np.abs(eig1 - eig2))

# Dictionary of available similarity methods
SIMILARITY_METHODS = {
    'graph_edit_distance': GraphEditDistance(),
    'spectral': SpectralSimilarity(),
    # Add new methods here
}

def get_method_parameters(method_name: str) -> Tuple[Set[str], Set[str]]:
    """Get required and optional parameters for a similarity method."""
    if method_name not in SIMILARITY_METHODS:
        raise ValueError(f"Unknown method: {method_name}")
    method = SIMILARITY_METHODS[method_name]
    return method.required_parameters, method.optional_parameters

def graph_similarity(matrix1: pd.DataFrame, 
                    matrix2: pd.DataFrame, 
                    similarity_method: str = "graph_edit_distance",
                    **kwargs) -> SimilarityResult:
    """
    Compute similarity between two correlation matrices.
    
    Args:
        matrix1, matrix2: Correlation matrices to compare
        similarity_method: Name of similarity measure to use
        **kwargs: Parameters specific to the chosen similarity method
    
    Returns:
        SimilarityResult object containing similarity score and metadata
    
    Examples:
        >>> # Using graph edit distance
        >>> result = graph_similarity(matrix1, matrix2, 
                                    similarity_method="graph_edit_distance",
                                    edge_threshold=0.5)
        
        >>> # Using spectral similarity
        >>> result = graph_similarity(matrix1, matrix2,
                                    similarity_method="spectral",
                                    num_eigenvalues=3,
                                    abs_values=True)
    """
    if similarity_method not in SIMILARITY_METHODS:
        raise ValueError(f"Unknown similarity method: {similarity_method}. "
                        f"Available methods: {list(SIMILARITY_METHODS.keys())}")
    
    method = SIMILARITY_METHODS[similarity_method]
    
    if similarity_method == "graph_edit_distance":
        score, normalized_score = method.compute(matrix1, matrix2, **kwargs)
        return SimilarityResult(
            score=score,
            method=similarity_method,
            metadata={'parameters': kwargs},
            normalized_score=normalized_score
        )
    else:
        score = method.compute(matrix1, matrix2, **kwargs)
        return SimilarityResult(
            score=score,
            method=similarity_method,
            metadata={'parameters': kwargs}
        )

# Example usage:
if __name__ == "__main__":
    # Create sample correlation matrices
    matrix1 = pd.DataFrame(np.array([[1.0, 0.8, 0.3],
                                   [0.8, 1.0, 0.5],
                                   [0.3, 0.5, 1.0]]))
    
    matrix2 = pd.DataFrame(np.array([[1.0, 0.7, 0.2],
                                   [0.7, 1.0, 0.6],
                                   [0.2, 0.6, 1.0]]))
    
    # Print required parameters for each method
    for method_name in SIMILARITY_METHODS:
        req, opt = get_method_parameters(method_name)
        print(f"\n{method_name}:")
        print(f"  Required parameters: {req}")
        print(f"  Optional parameters: {opt}")
    
    # Example with graph edit distance
    result1 = graph_similarity(matrix1, matrix2, 
                             similarity_method="graph_edit_distance",
                             edge_threshold=0.5)
    print(f"\nGED score: {result1}")
    
    # Example with spectral similarity
    result2 = graph_similarity(matrix1, matrix2,
                             similarity_method="spectral",
                             num_eigenvalues=2)
    print(f"Spectral score: {result2}")
    
    # This would raise an error due to missing required parameter:
    # result3 = graph_similarity(matrix1, matrix2, similarity_method="spectral")
    
    # This would raise an error due to invalid parameter:
    # result4 = graph_similarity(matrix1, matrix2,
    #                          similarity_method="graph_edit_distance",
    #                          invalid_param=1)
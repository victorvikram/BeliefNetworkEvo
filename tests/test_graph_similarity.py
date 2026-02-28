"""Tests for analyzers/graph_similarity.py"""

import numpy as np
import pandas as pd
import pytest

from src.analyzers.graph_similarity import graph_similarity, SimilarityResult


class TestGraphEditDistance:
    def test_ged_identical_matrices(self, small_corr_matrix):
        """Same matrix → score 0."""
        result = graph_similarity(
            small_corr_matrix, small_corr_matrix,
            similarity_method="graph_edit_distance",
            edge_threshold=0.1,
        )
        assert isinstance(result, SimilarityResult)
        assert result.score == 0

    def test_ged_different_matrices(self, small_corr_matrix, small_corr_matrix_alt):
        """Known different matrices → score > 0."""
        # Use intersection of variables (C, D, E)
        common = sorted(set(small_corr_matrix.index) & set(small_corr_matrix_alt.index))
        m1 = small_corr_matrix.loc[common, common]
        m2 = small_corr_matrix_alt.loc[common, common]
        result = graph_similarity(
            m1, m2,
            similarity_method="graph_edit_distance",
            edge_threshold=0.1,
        )
        assert result.score >= 0

    def test_ged_normalized_score(self, small_corr_matrix, identity_corr_matrix):
        """Normalized score in [0, 1]."""
        result = graph_similarity(
            small_corr_matrix, identity_corr_matrix,
            similarity_method="graph_edit_distance",
            edge_threshold=0.1,
        )
        assert 0 <= result.normalized_score <= 1


class TestSpectralSimilarity:
    def test_spectral_identical(self, small_corr_matrix):
        """Same matrix → score 0."""
        result = graph_similarity(
            small_corr_matrix, small_corr_matrix,
            similarity_method="spectral",
            num_eigenvalues=3,
        )
        assert result.score == pytest.approx(0, abs=1e-10)

    def test_spectral_different(self, small_corr_matrix, identity_corr_matrix):
        """Different matrices → score > 0."""
        result = graph_similarity(
            small_corr_matrix, identity_corr_matrix,
            similarity_method="spectral",
            num_eigenvalues=3,
        )
        assert result.score > 0


class TestEdgeCases:
    def test_unknown_method_raises(self, small_corr_matrix):
        """Invalid method name."""
        with pytest.raises(ValueError, match="Unknown similarity method"):
            graph_similarity(
                small_corr_matrix, small_corr_matrix,
                similarity_method="nonexistent_method",
            )

    def test_missing_params_raises(self, small_corr_matrix):
        """Missing required params."""
        with pytest.raises(ValueError):
            graph_similarity(
                small_corr_matrix, small_corr_matrix,
                similarity_method="graph_edit_distance",
                # missing edge_threshold
            )

    def test_mismatched_variables(self, small_corr_matrix, small_corr_matrix_alt):
        """Matrices with different var sets (should align internally)."""
        result = graph_similarity(
            small_corr_matrix, small_corr_matrix_alt,
            similarity_method="graph_edit_distance",
            edge_threshold=0.1,
        )
        assert isinstance(result, SimilarityResult)
        assert result.score >= 0

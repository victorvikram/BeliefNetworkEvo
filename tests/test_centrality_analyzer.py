"""Tests for analyzers/centrality_analyzer.py"""

import numpy as np
import pandas as pd
import pytest

from source_code.analyzers.centrality_analyzer import (
    betweenness_centrality_weighted,
    calculate_centrality_measures,
    create_centrality_dataframes,
)


class TestBetweennessCentrality:
    def test_betweenness_star_graph(self):
        """Star graph: center node should have highest BC."""
        # Star graph with 5 nodes: node 0 is center, connected to 1-4
        n = 5
        adj = np.zeros((n, n))
        for i in range(1, n):
            adj[0, i] = 1.0
            adj[i, 0] = 1.0
        bc = betweenness_centrality_weighted(adj)
        # Center node (0) should have the highest BC
        assert bc[0] == bc.max()
        assert bc[0] > 0

    def test_betweenness_disconnected(self):
        """Disconnected graph: isolated nodes have BC=0."""
        # Two disconnected pairs
        adj = np.zeros((4, 4))
        adj[0, 1] = 1.0
        adj[1, 0] = 1.0
        adj[2, 3] = 1.0
        adj[3, 2] = 1.0
        bc = betweenness_centrality_weighted(adj)
        assert (bc == 0).all()


@pytest.fixture
def nonneg_corr_matrix():
    """5x5 non-negative symmetric matrix suitable for Dijkstra-based betweenness."""
    np.random.seed(42)
    raw = np.random.uniform(0, 1, size=(5, 5))
    sym = (raw + raw.T) / 2
    np.fill_diagonal(sym, 0)
    labels = list("ABCDE")
    return pd.DataFrame(sym, index=labels, columns=labels)


class TestCalculateCentralityMeasures:
    def test_calculate_centrality_measures_shapes(self, nonneg_corr_matrix):
        """Output shapes match input dimensions."""
        bc, degrees, strengths, variables = calculate_centrality_measures(nonneg_corr_matrix)
        n = len(nonneg_corr_matrix)
        assert len(bc) == n
        assert len(degrees) == n
        assert len(strengths) == n
        assert len(variables) == n

    def test_degrees_count(self, nonneg_corr_matrix):
        """Degree counts match expected non-zero entries."""
        bc, degrees, strengths, variables = calculate_centrality_measures(nonneg_corr_matrix)
        # Manually count non-zero off-diagonal entries per column
        expected_degrees = np.sum(nonneg_corr_matrix != 0, axis=0) - 1
        np.testing.assert_array_equal(degrees.values, expected_degrees.values)


class TestCreateCentralityDataframes:
    def test_create_centrality_dataframes(self, nonneg_corr_matrix):
        """Output DataFrames have correct columns, sorted descending, top_n works."""
        bc, degrees, strengths, variables = calculate_centrality_measures(nonneg_corr_matrix)
        bc_df, deg_df, str_df = create_centrality_dataframes(
            nonneg_corr_matrix, bc, degrees, strengths, top_n=3
        )
        # Check shapes are limited to top_n
        assert len(bc_df) <= 3
        assert len(deg_df) <= 3
        assert len(str_df) <= 3

        # Check sorting (descending)
        assert (bc_df["Betweenness Centrality"].diff().dropna() <= 0).all()
        assert (deg_df["Degree"].diff().dropna() <= 0).all()
        assert (str_df["Total Correlation"].diff().dropna() <= 0).all()

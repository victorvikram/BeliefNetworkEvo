"""Tests for analyzers/frustration_analyzer.py"""

import numpy as np
import pandas as pd
import pytest

from src.analyzers.frustration_analyzer import (
    calculate_frustration,
    get_satisfaction_mats,
    get_frust_percentage,
)


class TestGetSatisfactionMats:
    def test_satisfaction_mats(self):
        """Direct test of get_satisfaction_mats with known vectors."""
        # Simple 3-node network
        adj_mat = np.array([
            [0, 1, -1],
            [1, 0, 1],
            [-1, 1, 0],
        ], dtype=float)
        # Two belief vectors
        vectors = np.array([
            [1, 1, 1],
            [1, 1, -1],
        ], dtype=float)
        sat = get_satisfaction_mats(adj_mat, vectors)
        assert sat.shape == (2, 3, 3)

        # For vector [1,1,1]: sat[0,0,1] = 1*1*1 = 1 (satisfied)
        assert sat[0, 0, 1] == 1.0
        # For vector [1,1,1]: sat[0,0,2] = 1*1*(-1) = -1 (frustrated)
        assert sat[0, 0, 2] == -1.0
        # For vector [1,1,-1]: sat[1,0,2] = 1*(-1)*(-1) = 1 (satisfied)
        assert sat[1, 0, 2] == 1.0


class TestGetFrustPercentage:
    def test_frust_percentage(self):
        """Direct test of get_frust_percentage with known satisfaction matrices."""
        # 3 measurements, 2x2 satisfaction values
        sat_mats = np.array([
            [[1, -1], [-1, 1]],
            [[1, 1], [1, 1]],
            [[1, -1], [-1, 1]],
        ], dtype=float)
        frust = get_frust_percentage(sat_mats)
        assert frust.shape == (2, 2)
        # Position [0,1]: -1 appears 2 out of 3 times
        assert frust[0, 1] == pytest.approx(2.0 / 3.0)
        # Position [0,0]: never negative
        assert frust[0, 0] == 0.0


@pytest.mark.slow
class TestCalculateFrustration:
    def test_frustration_shape(self):
        """Output shape matches input."""
        np.random.seed(42)
        labels = list("ABC")
        adj = np.array([[0, 0.5, -0.3], [0.5, 0, 0.2], [-0.3, 0.2, 0]], dtype=float)
        adj_df = pd.DataFrame(adj, index=labels, columns=labels)
        result = calculate_frustration(adj_df, optimizer="multi_pass")
        assert result.shape == (3, 3)
        assert list(result.index) == labels
        assert list(result.columns) == labels

    def test_frustration_values_range(self):
        """All values in [0, 1]."""
        np.random.seed(42)
        labels = list("ABC")
        adj = np.array([[0, 0.5, -0.3], [0.5, 0, 0.2], [-0.3, 0.2, 0]], dtype=float)
        adj_df = pd.DataFrame(adj, index=labels, columns=labels)
        result = calculate_frustration(adj_df, optimizer="multi_pass")
        assert (result.values >= 0).all()
        assert (result.values <= 1).all()

    def test_frustration_consistent_network(self):
        """All-positive-edge network → frustration near 0."""
        np.random.seed(42)
        labels = list("ABC")
        adj = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=float)
        adj_df = pd.DataFrame(adj, index=labels, columns=labels)
        result = calculate_frustration(adj_df, optimizer="multi_pass")
        # Off-diagonal entries should be near 0 (all edges satisfiable)
        off_diag = result.values[np.triu_indices(3, k=1)]
        assert np.all(off_diag < 0.1)

    def test_frustration_fully_frustrated(self):
        """Triangle with (+, +, -) edges → non-zero frustration."""
        np.random.seed(42)
        labels = list("ABC")
        # A-B positive, B-C positive, A-C negative → one edge always frustrated
        adj = np.array([[0, 1, -1], [1, 0, 1], [-1, 1, 0]], dtype=float)
        adj_df = pd.DataFrame(adj, index=labels, columns=labels)
        result = calculate_frustration(adj_df, optimizer="multi_pass")
        off_diag = result.values[np.triu_indices(3, k=1)]
        assert np.any(off_diag > 0)

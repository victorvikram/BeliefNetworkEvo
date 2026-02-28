"""
Shared fixtures for the test suite.
All fixtures use np.random.seed(42) for reproducibility.
"""

import numpy as np
import pandas as pd
import networkx as nx
import pytest


@pytest.fixture
def small_corr_matrix():
    """5x5 symmetric correlation-like DataFrame with named variables (A-E), values in [-1,1], diagonal=0."""
    np.random.seed(42)
    raw = np.random.uniform(-1, 1, size=(5, 5))
    sym = (raw + raw.T) / 2
    np.fill_diagonal(sym, 0)
    labels = list("ABCDE")
    return pd.DataFrame(sym, index=labels, columns=labels)


@pytest.fixture
def small_corr_matrix_alt():
    """Different 5x5 matrix with partially overlapping variables (C,D,E,F,G)."""
    np.random.seed(99)
    raw = np.random.uniform(-1, 1, size=(5, 5))
    sym = (raw + raw.T) / 2
    np.fill_diagonal(sym, 0)
    labels = list("CDEFG")
    return pd.DataFrame(sym, index=labels, columns=labels)


@pytest.fixture
def identity_corr_matrix():
    """5x5 zero matrix (no correlations)."""
    labels = list("ABCDE")
    return pd.DataFrame(np.zeros((5, 5)), index=labels, columns=labels)


@pytest.fixture
def synthetic_cleaned_df():
    """1000-row DataFrame mimicking cleaned GSS data: YEAR column (2000-2004), 10 belief variables with values in [-1,1], includes NaN gaps."""
    np.random.seed(42)
    n = 1000
    years = np.random.choice([2000, 2001, 2002, 2003, 2004], size=n)
    data = {"YEAR": years}
    for var in [f"VAR_{i}" for i in range(10)]:
        vals = np.random.uniform(-1, 1, size=n)
        # Introduce ~10% NaN gaps
        mask = np.random.random(n) < 0.1
        vals[mask] = np.nan
        data[var] = vals
    return pd.DataFrame(data)


@pytest.fixture
def small_networkx_graph(small_corr_matrix):
    """Weighted NetworkX graph built from small_corr_matrix."""
    G = nx.Graph()
    labels = small_corr_matrix.columns.tolist()
    G.add_nodes_from(labels)
    for i, row_label in enumerate(labels):
        for j, col_label in enumerate(labels):
            if i < j:
                w = small_corr_matrix.iloc[i, j]
                if w != 0:
                    G.add_edge(row_label, col_label, weight=w)
    return G

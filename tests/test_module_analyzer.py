"""Tests for analyzers/module_analyzer.py"""

import networkx as nx
import numpy as np
import pytest

from source_code.analyzers.module_analyzer import calculate_interaction_strength


class TestInteractionStrength:
    def test_interaction_strength_basic(self, small_networkx_graph):
        """Known graph, known node sets → expected sum."""
        G = small_networkx_graph
        nodes = list(G.nodes())
        set_1 = {nodes[0]}
        set_2 = {nodes[1], nodes[2]}
        strength = calculate_interaction_strength(G, set_1, set_2)
        # Manually compute expected sum of |weights| from set_1 to set_2
        expected = 0
        for u in set_1:
            for _, v, d in G.edges(u, data=True):
                if v in set_2:
                    expected += np.abs(d["weight"])
        assert strength == pytest.approx(expected)

    def test_interaction_strength_no_edges(self):
        """Disjoint sets with no connecting edges → 0."""
        G = nx.Graph()
        G.add_edge("A", "B", weight=0.5)
        G.add_edge("C", "D", weight=0.8)
        strength = calculate_interaction_strength(G, {"A", "B"}, {"C", "D"})
        assert strength == 0

    def test_interaction_strength_uses_abs(self):
        """Negative weights contribute positively."""
        G = nx.Graph()
        G.add_edge("A", "B", weight=-0.7)
        strength = calculate_interaction_strength(G, {"A"}, {"B"})
        assert strength == pytest.approx(0.7)

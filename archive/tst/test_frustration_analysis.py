import unittest
import pandas as pd
import numpy as np
import networkx as nx
from numpy.testing import assert_almost_equal
import networkx.algorithms.isomorphism as iso


import sys
sys.path.append('../src')

from frustration_analysis import *

class TestModuleFunctions(unittest.TestCase):
    def test_reverse_node(self):
        # four node graph with all nodes connected except "A" and "B"
        G = nx.Graph()
        G.add_edge("A", "C", weight=-1)
        G.add_edge("C", "B", weight=1)
        G.add_edge("B", "D", weight=1)
        G.add_edge("A", "D", weight=-1)
        G.add_edge("C", "D", weight=1)
        G_cons = get_consistency_graph(G)

        reverse_node(G, G_cons, "A")

        self.assertEqual(G.get_edge_data("A", "C")["weight"], 1)
        self.assertEqual(G.get_edge_data("A", "D")["weight"], 1)

        # graph with an unstable square
        G = nx.Graph()
        G.add_edge(0, 1, weight=-1)
        G.add_edge(0, 2, weight=-1)
        G.add_edge(1, 2, weight=1)
        G.add_edge(1, 4, weight=1)
        G.add_edge(2, 3, weight=1)
        G.add_edge(3, 4, weight=-1)
        G.add_edge(4, 5, weight=1)
        G.add_edge(3, 5, weight=1)
        G_cons = get_consistency_graph(G)

        reverse_node(G, G_cons, 3)
        self.assertEqual(G.get_edge_data(3, 2)["weight"], -1)
        self.assertEqual(G.get_edge_data(3, 4)["weight"], 1)
        self.assertEqual(G.get_edge_data(3, 5)["weight"], -1)

    
    def test_get_consistency_graph(self):
        # simple two-node graphs
        G = nx.Graph()
        G.add_edge(1, 2, weight=1)
        G_cons = get_consistency_graph(G)

        exp_cons = nx.MultiGraph()
        exp_cons.add_edge(1, 2, weight=1)

        self.assertTrue(graphs_equal(G_cons, exp_cons))
        
        G = nx.Graph()
        G.add_edge(1, 2, weight=-1)
        G_cons = get_consistency_graph(G)

        exp_cons = nx.MultiGraph()
        exp_cons.add_edge(1, 2, weight=-1)  
        self.assertTrue(graphs_equal(G_cons, exp_cons))

        # triangle graph with spurs
        G = nx.Graph()
        G.add_edge(1, 2, weight=1)
        G.add_edge(2, 3, weight=-1)
        G.add_edge(1, 4, weight=1)
        G.add_edge(3, 5, weight=1)
        G_cons = get_consistency_graph(G)

        exp_cons = nx.MultiGraph()
        exp_cons.add_edge(1, 2, weight=1)
        exp_cons.add_edge(1, 3, weight=-1)
        exp_cons.add_edge(2, 3, weight=-1)
        exp_cons.add_edge(3, 4, weight=-1)
        exp_cons.add_edge(3, 4, weight=1)
        exp_cons.add_edge(3, 5, weight=1)
        exp_cons.add_edge(1, 4, weight=1)
        exp_cons.add_edge(1, 5, weight=1)
        exp_cons.add_edge(1, 5, weight=-1)
        exp_cons.add_edge(2, 4, weight=1)
        exp_cons.add_edge(2, 5, weight=-1)
        exp_cons.add_edge(4, 5, weight=1)
        exp_cons.add_edge(4, 5, weight=-1)

        self.assertTrue(graphs_equal(G_cons, exp_cons))

        # now get rid of the double edges
        G.add_edge(1, 3, weight=-1)
        G_cons = get_consistency_graph(G)

        exp_cons = nx.MultiGraph()
        exp_cons.add_edge(1, 2, weight=1)
        exp_cons.add_edge(1, 3, weight=-1)
        exp_cons.add_edge(2, 3, weight=-1)
        exp_cons.add_edge(3, 4, weight=-1)
        exp_cons.add_edge(3, 5, weight=1)
        exp_cons.add_edge(1, 4, weight=1)
        exp_cons.add_edge(1, 5, weight=-1)
        exp_cons.add_edge(2, 4, weight=1)
        exp_cons.add_edge(2, 5, weight=-1)
        exp_cons.add_edge(4, 5, weight=1)
        exp_cons.add_edge(4, 5, weight=-1)

        self.assertTrue(graphs_equal(G_cons, exp_cons))

        # unstable square
        G = nx.Graph()
        G.add_edge(1, 2, weight=1)
        G.add_edge(2, 3, weight=1)
        G.add_edge(3, 4, weight=1)
        G.add_edge(4, 1, weight=-1)
        G_cons = get_consistency_graph(G) 

        exp_cons = nx.MultiGraph()
        exp_cons.add_edge(1, 2, weight=1)
        exp_cons.add_edge(2, 3, weight=1)
        exp_cons.add_edge(3, 4, weight=1)
        exp_cons.add_edge(4, 1, weight=-1)

        self.assertTrue(graphs_equal(G_cons, exp_cons))
    
    def test_check_for_consistency(self):
        # four node graph with all nodes connected except "A" and "B"
        G = nx.Graph()
        G.add_edge("A", "C", weight=-1)
        G.add_edge("C", "B", weight=1)
        G.add_edge("B", "D", weight=1)
        G.add_edge("A", "D", weight=-1)
        G.add_edge("C", "D", weight=1)

        # all of them are consistent
        pos, neg = check_for_consistency(G, "A", "B")
        self.assertFalse(pos)
        self.assertTrue(neg)

        pos, neg = check_for_consistency(G, "C", "D")
        self.assertTrue(pos)
        self.assertFalse(neg)

        pos, neg = check_for_consistency(G, "B", "D")
        self.assertTrue(pos)
        self.assertFalse(neg)

        pos, neg = check_for_consistency(G, "A", "C")
        self.assertFalse(pos)
        self.assertTrue(neg)

        pos, neg = check_for_consistency(G, "B", "C")
        self.assertTrue(pos)
        self.assertFalse(neg)

        pos, neg = check_for_consistency(G, "A", "D")
        self.assertFalse(pos)
        self.assertTrue(neg)

        # same graph as before except reverse the sign of "C" and "D"
        G = nx.Graph()
        G.add_edge("A", "C", weight=-1)
        G.add_edge("C", "B", weight=1)
        G.add_edge("B", "D", weight=1)
        G.add_edge("A", "D", weight=-1)
        G.add_edge("C", "D", weight=-1)

        # only one is consistent
        pos, neg = check_for_consistency(G, "A", "B")
        self.assertFalse(pos)
        self.assertTrue(neg)

        # the rest inconsistent
        pos, neg = check_for_consistency(G, "C", "D")
        self.assertFalse(pos)
        self.assertFalse(neg)

        pos, neg = check_for_consistency(G, "B", "D")
        self.assertFalse(pos)
        self.assertFalse(neg)

        pos, neg = check_for_consistency(G, "A", "C")
        self.assertFalse(pos)
        self.assertFalse(neg)

        pos, neg = check_for_consistency(G, "B", "C")
        self.assertFalse(pos)
        self.assertFalse(neg)

        pos, neg = check_for_consistency(G, "A", "D")
        self.assertFalse(pos)
        self.assertFalse(neg)

        # graph with an unstable square
        G = nx.Graph()
        G.add_edge(0, 1, weight=-1)
        G.add_edge(0, 2, weight=-1)
        G.add_edge(1, 2, weight=1)
        G.add_edge(1, 4, weight=1)
        G.add_edge(2, 3, weight=1)
        G.add_edge(3, 4, weight=-1)
        G.add_edge(4, 5, weight=1)
        G.add_edge(3, 5, weight=1)

        # consistent edges

        pos, neg = check_for_consistency(G, 1, 2)
        self.assertTrue(pos)
        self.assertFalse(neg)

        pos, neg = check_for_consistency(G, 2, 3)
        self.assertTrue(pos)
        self.assertFalse(neg)

        pos, neg = check_for_consistency(G, 1, 4)
        self.assertTrue(pos)
        self.assertFalse(neg)

        pos, neg = check_for_consistency(G, 0, 1)
        self.assertFalse(pos)
        self.assertTrue(neg)
        
        pos, neg = check_for_consistency(G, 0, 2)
        self.assertFalse(pos)
        self.assertTrue(neg)

        # inconsistent edges
        pos, neg = check_for_consistency(G, 3, 4)
        self.assertFalse(pos)
        self.assertFalse(neg)

        pos, neg = check_for_consistency(G, 4, 5)
        self.assertFalse(pos)
        self.assertFalse(neg)

        pos, neg = check_for_consistency(G, 3, 5)
        self.assertFalse(pos)
        self.assertFalse(neg)

        # unstable square graph with no other edges
        G = nx.Graph()

        G.add_edge(1, 2, weight=1)
        G.add_edge(2, 3, weight=1)
        G.add_edge(3, 4, weight=-1)
        G.add_edge(1, 4, weight=1)

        pos, neg = check_for_consistency(G, 1, 2)
        self.assertTrue(pos)
        self.assertFalse(neg)

        pos, neg = check_for_consistency(G, 2, 3)
        self.assertTrue(pos)
        self.assertFalse(neg)

        pos, neg = check_for_consistency(G, 3, 4)
        self.assertFalse(pos)
        self.assertTrue(neg)

        pos, neg = check_for_consistency(G, 1, 4)
        self.assertTrue(pos)
        self.assertFalse(neg)

        # cross edges are inconsistent
        pos, neg = check_for_consistency(G, 1, 3)
        self.assertFalse(pos)
        self.assertFalse(neg)

        pos, neg = check_for_consistency(G, 2, 4)
        self.assertFalse(pos)
        self.assertFalse(neg)

        # now add a single cross edge, which creates deconsistification
        G.add_edge(2, 4, weight=1)

        pos, neg = check_for_consistency(G, 1, 2)
        self.assertTrue(pos)
        self.assertFalse(neg)

        pos, neg = check_for_consistency(G, 1, 4)
        self.assertTrue(pos)
        self.assertFalse(neg)

        # all other edges inconsistent
        pos, neg = check_for_consistency(G, 2, 3)
        self.assertFalse(pos)
        self.assertFalse(neg)

        pos, neg = check_for_consistency(G, 3, 4)
        self.assertFalse(pos)
        self.assertFalse(neg)

        
        pos, neg = check_for_consistency(G, 1, 3)
        self.assertFalse(pos)
        self.assertFalse(neg)

        pos, neg = check_for_consistency(G, 2, 4)
        self.assertFalse(pos)
        self.assertFalse(neg)
    
    def test_graphs_equal(self):
        # isomorphic but nodes don't match 
        G1 = nx.Graph()
        G1.add_edge(1, 2)
        G1.add_edge(2, 3)

        G2 = nx.Graph()
        G2.add_edge(2, 3)
        G2.add_edge(3, 1)
        self.assertFalse(graphs_equal(G1, G2))

        # nodes match but weights don't 
        G1 = nx.Graph()
        G1.add_edge(1, 2, weight=1)
        G1.add_edge(2, 3, weight=2)

        G2 = nx.Graph()
        G2.add_edge(1, 2, weight=2)
        G2.add_edge(2, 3, weight=1)
        self.assertFalse(graphs_equal(G1, G2))

        # everything matches
        G1 = nx.Graph()
        G1.add_edge(1, 2, weight=1)
        G1.add_edge(2, 3, weight=2)

        G2 = nx.Graph()
        G2.add_edge(1, 2, weight=1)
        G2.add_edge(2, 3, weight=2)
        self.assertTrue(graphs_equal(G1, G2))

        # MultiGraph that doesn't match
        G1 = nx.MultiGraph()
        G1.add_edge(1, 2, weight=1)
        G1.add_edge(1, 2, weight=2)
        G1.add_edge(2, 3, weight=3)

        G2 = nx.MultiGraph()
        G2.add_edge(1, 2, weight=1)
        G2.add_edge(1, 2, weight=1)
        G2.add_edge(2, 3, weight=3)
        self.assertFalse(graphs_equal(G1, G2))

        # MultiGraph that does match 
        G1 = nx.MultiGraph()
        G1.add_edge(1, 2, weight=1)
        G1.add_edge(1, 2, weight=2)
        G1.add_edge(2, 3, weight=3)

        G2 = nx.MultiGraph()
        G2.add_edge(1, 2, weight=1)
        G2.add_edge(1, 2, weight=2)
        G2.add_edge(2, 3, weight=3)
        self.assertTrue(graphs_equal(G1, G2))


def graphs_equal(G1, G2):
    G1_int = nx.convert_node_labels_to_integers(G1, label_attribute='label')
    G2_int = nx.convert_node_labels_to_integers(G2, label_attribute='label')

    if isinstance(G1, nx.MultiGraph):
        edge_matcher = iso.numerical_multiedge_match('weight', 1)
    else:
        edge_matcher = lambda x, y: x == y

    return nx.is_isomorphic(G1_int, G2_int, node_match=lambda x, y: x == y, edge_match=edge_matcher)



if __name__ == '__main__':
    unittest.main()





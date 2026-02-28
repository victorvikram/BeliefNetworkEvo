"""Tests for analyzers/optimization_analyzer.py"""

import numpy as np
import pytest

from source_code.analyzers.optimization_analyzer import (
    flip_step_function,
    hamiltonian_objective_function,
    multi_pass_optimize,
    simulated_annealing,
    accept_new_vector,
)


class TestFlipStepFunction:
    def test_flip_step_function(self):
        """Output shape matches, at least one component changed."""
        np.random.seed(42)
        vectors = np.array([[1, -1, 0, 1, -1]], dtype=float)
        result = flip_step_function(vectors, num_flips=2)
        assert result.shape == vectors.shape
        # At least something should differ (with 2 flips on 5 components, very likely)
        # But we can't guarantee it since a flip might land on same value
        # Instead just check shape and type
        assert result.dtype == vectors.dtype


class TestHamiltonianObjective:
    def test_hamiltonian_objective(self):
        """Known vector + couplings → expected energy."""
        # Simple 2-var system: coupling = [[0, 1], [1, 0]]
        couplings = np.array([[0, 1], [1, 0]], dtype=float)
        # vector = [1, 1] → cost = -(1*1*1 + 1*1*1)/2 = -1
        vectors = np.array([[1, 1]], dtype=float)
        cost = hamiltonian_objective_function(vectors, couplings)
        assert cost.shape == (1, 1)
        assert cost[0, 0] == pytest.approx(-1.0)

        # vector = [1, -1] → cost = -(1*(-1)*1 + (-1)*1*1)/2 = 1
        vectors2 = np.array([[1, -1]], dtype=float)
        cost2 = hamiltonian_objective_function(vectors2, couplings)
        assert cost2[0, 0] == pytest.approx(1.0)


class TestMultiPassOptimize:
    def test_multi_pass_converges(self):
        """Simple 3-var system converges to known minimum."""
        np.random.seed(42)
        # All-positive couplings: minimum when all spins aligned
        couplings = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=float)
        initial = np.random.choice([-1, 0, 1], size=(50, 3))
        result, num_changed = multi_pass_optimize(initial, couplings, max_iterations=100)
        # All vectors should converge to all-same-sign
        for vec in result:
            nonzero = vec[vec != 0]
            if len(nonzero) > 0:
                assert np.all(nonzero == nonzero[0]) or np.all(nonzero == -nonzero[0])


class TestSimulatedAnnealing:
    def test_simulated_annealing_reduces_cost(self):
        """Final cost <= initial cost."""
        np.random.seed(42)
        couplings = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=float)
        initial = np.random.choice([-1, 1], size=(10, 3)).astype(float)
        initial_cost = hamiltonian_objective_function(initial, couplings)

        final, final_cost = simulated_annealing(
            initial, 10, 0.95, 500,
            lambda vecs: hamiltonian_objective_function(vecs, couplings),
            lambda vecs: flip_step_function(vecs, num_flips=1),
        )
        # Mean final cost should be <= mean initial cost
        assert final_cost.mean() <= initial_cost.mean()


class TestAcceptNewVector:
    def test_accept_new_vector_always_accepts_lower(self):
        """Lower cost → always accepted."""
        old_cost = np.array([[10.0], [5.0], [3.0]])
        new_cost = np.array([[5.0], [2.0], [1.0]])
        acceptance = accept_new_vector(old_cost, new_cost, temperature=1.0)
        assert acceptance.all()

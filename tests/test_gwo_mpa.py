"""Unit tests for GWO and MPA. Run: python -m unittest discover -s tests -v"""
import contextlib
import io
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from optimizers.gwo import GreyWolfOptimizer
from optimizers.mpa import MarinePredatorsOptimizer, levy_steps


class ScriptedFitness:
    """Returns pre-set fitness vectors call by call (last one repeats); records inputs."""
    def __init__(self, values):
        self.values = [torch.tensor(v, dtype=torch.float32) for v in values]
        self.inputs = []

    def __call__(self, candidates):
        self.inputs.append(candidates.detach().clone())
        return self.values[min(len(self.inputs) - 1, len(self.values) - 1)].clone()


def sphere(target):
    return lambda x: -((x - target) ** 2).sum(1)


def quiet(fn, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


class GWO(unittest.TestCase):
    def test_solves_sphere(self):
        torch.manual_seed(0)
        target = torch.full((5,), 0.3)
        best, history = quiet(GreyWolfOptimizer(20, 5, sphere(target), "cpu").train, max_iter=200, patience=1000)
        self.assertGreater(history[-1], -1e-3)
        torch.testing.assert_close(best, target, atol=0.05, rtol=0)

    def test_history_is_monotone(self):
        torch.manual_seed(1)
        _, history = quiet(GreyWolfOptimizer(10, 3, sphere(torch.ones(3)), "cpu").train, max_iter=30, patience=100)
        self.assertTrue(all(b >= a for a, b in zip(history, history[1:])))

    def test_returns_snapshot_of_best(self):
        torch.manual_seed(0)
        fit = ScriptedFitness([[5.0, 1.0, 0.0, 0.0], [0.0] * 4])
        best, history = quiet(GreyWolfOptimizer(4, 3, fit, "cpu").train, max_iter=3, patience=10)
        torch.testing.assert_close(best, fit.inputs[0][0])
        self.assertEqual(history, [5.0] * 4)

    def test_patience(self):
        fit = ScriptedFitness([[1.0] * 4])
        _, history = quiet(GreyWolfOptimizer(4, 2, fit, "cpu").train, max_iter=50, patience=3)
        self.assertEqual(len(history), 4)  # initial + 3 non-improving iterations

    def test_rejects_tiny_population(self):
        with self.assertRaises(ValueError):
            GreyWolfOptimizer(2, 3, sphere(torch.zeros(3)), "cpu")


class MPA(unittest.TestCase):
    def test_solves_sphere(self):
        torch.manual_seed(0)
        target = torch.full((5,), 0.3)
        best, history = quiet(MarinePredatorsOptimizer(20, 5, sphere(target), "cpu").train, max_iter=200, patience=1000)
        self.assertGreater(history[-1], -1e-2)
        torch.testing.assert_close(best, target, atol=0.1, rtol=0)

    def test_history_is_monotone(self):
        torch.manual_seed(1)
        _, history = quiet(MarinePredatorsOptimizer(10, 3, sphere(torch.ones(3)), "cpu").train, max_iter=30, patience=100)
        self.assertTrue(all(b >= a for a, b in zip(history, history[1:])))

    def test_returns_snapshot_of_best(self):
        torch.manual_seed(0)
        fit = ScriptedFitness([[5.0, 1.0, 0.0, 0.0], [0.0] * 4])
        best, _ = quiet(MarinePredatorsOptimizer(4, 3, fit, "cpu").train, max_iter=6, patience=10)
        torch.testing.assert_close(best, fit.inputs[0][0])

    def test_two_calls_per_iteration_and_patience(self):
        fit = ScriptedFitness([[1.0] * 4])
        _, history = quiet(MarinePredatorsOptimizer(4, 2, fit, "cpu").train, max_iter=50, patience=3)
        self.assertEqual(len(history), 4)      # first iteration improves from -inf, then 3 without
        self.assertEqual(len(fit.inputs), 8)

    def test_memory_never_loses_best_prey(self):
        # After memory saving, every prey is at least as good as its previous position.
        torch.manual_seed(2)
        calls = []
        def recording(x):
            values = -(x ** 2).sum(1)
            calls.append(values.max().item())
            return values
        _, history = quiet(MarinePredatorsOptimizer(8, 3, recording, "cpu").train, max_iter=20, patience=100)
        self.assertAlmostEqual(history[-1], max(calls))

    def test_levy_steps_shape_and_heavy_tail(self):
        torch.manual_seed(0)
        steps = levy_steps((20000,), 1.5, "cpu")
        self.assertEqual(steps.shape, (20000,))
        self.assertTrue(torch.isfinite(steps).all())
        self.assertGreater(steps.abs().max().item(), 10 * steps.abs().median().item())


if __name__ == "__main__":
    unittest.main()

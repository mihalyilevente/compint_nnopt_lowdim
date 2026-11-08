"""Regression tests for the P0 fixes. Run: python -m unittest discover -s tests -v"""
import contextlib
import io
import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from optimizers.ga import GeneticAlgorithmOptimizer
from optimizers.pso import ParticleSwarmOptimizer
from utils.data_split import stratified_split_indices
from utils.ga_utils import arithmetic_crossover, truncation_selection
from utils.train_model import train_model


class ScriptedFitness:
    """Returns pre-set fitness vectors call by call (last one repeats); records inputs."""
    def __init__(self, values):
        self.values = [torch.tensor(v, dtype=torch.float32) for v in values]
        self.inputs = []

    def __call__(self, candidates):
        self.inputs.append(candidates.detach().clone())
        return self.values[min(len(self.inputs) - 1, len(self.values) - 1)].clone()


def quiet(fn, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


class TrainModelCheckpoint(unittest.TestCase):
    def test_restores_best_epoch_not_last(self):
        torch.manual_seed(0)
        model = nn.Linear(1, 1, bias=False)
        schedule = iter([1.0, 3.0, 5.0])  # weight after each epoch; target weight is 1

        class ScriptedOptimizer:
            def zero_grad(self):
                pass
            def step(self):
                with torch.no_grad():
                    model.weight.fill_(next(schedule))

        class NoScheduler:
            def step(self, _):
                pass

        x = torch.ones(4, 1)
        loader = [(x, x.clone())]  # MSE is zero only when weight == 1
        quiet(train_model, model, "cpu", loader, loader, nn.MSELoss(),
              ScriptedOptimizer(), NoScheduler(), epochs=3, patience=10)
        self.assertEqual(model.weight.item(), 1.0)


class PSOFixes(unittest.TestCase):
    def test_initial_winner_improving_is_recorded(self):
        fit = ScriptedFitness([[1.0, 0.0], [3.0, 0.0]])
        _, history = quiet(ParticleSwarmOptimizer(2, 1, fit, "cpu").train, max_iter=1)
        self.assertEqual(history, [1.0, 3.0])

    def test_returned_position_is_snapshot_of_best(self):
        torch.manual_seed(0)
        fit = ScriptedFitness([[1.0, 0.0], [3.0, 0.0], [0.0, 0.0]])
        best, history = quiet(ParticleSwarmOptimizer(2, 1, fit, "cpu").train, max_iter=4, patience=10)
        self.assertEqual(max(history), 3.0)
        torch.testing.assert_close(best, fit.inputs[1][0])

    def test_patience_counts_when_nobody_improves(self):
        fit = ScriptedFitness([[1.0, 1.0]])
        _, history = quiet(ParticleSwarmOptimizer(2, 1, fit, "cpu").train, max_iter=10, patience=2)
        self.assertEqual(len(history), 3)  # initial + 2 non-improving iterations

    def test_other_particle_wins(self):
        fit = ScriptedFitness([[2.0, 1.0], [2.0, 5.0], [0.0, 0.0]])
        best, history = quiet(ParticleSwarmOptimizer(2, 1, fit, "cpu").train, max_iter=3, patience=10)
        self.assertEqual(history, [2.0, 5.0, 5.0, 5.0])
        torch.testing.assert_close(best, fit.inputs[1][1])


class GAFixes(unittest.TestCase):
    def test_best_genome_survives_in_place_callbacks(self):
        torch.manual_seed(0)
        fit = ScriptedFitness([[5.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
        ga = GeneticAlgorithmOptimizer(4, 3, fit, "cpu")
        best, history = quiet(ga.train,
                              selection_fn=lambda f: torch.arange(f.shape[0]),
                              crossover_fn=lambda g, p: g,
                              mutation_fn=lambda g: g.add_(1.0),  # in-place on purpose
                              max_iter=3, patience=10)
        torch.testing.assert_close(best, fit.inputs[0][0])
        self.assertEqual(history, [5.0, 5.0, 5.0])

    def test_wrong_parent_count_rejected(self):
        ga = GeneticAlgorithmOptimizer(4, 2, ScriptedFitness([[0.0] * 4]), "cpu")
        with self.assertRaises(ValueError):
            quiet(ga.train, selection_fn=lambda f: torch.arange(2),
                  crossover_fn=lambda g, p: g, mutation_fn=lambda g: g, max_iter=1)


class SelectionAndCrossover(unittest.TestCase):
    def test_truncation_keeps_the_best(self):
        torch.manual_seed(0)
        parents = truncation_selection(torch.tensor([0.0, 1.0, 2.0, 3.0]), keep_frac=0.5)
        self.assertEqual(len(parents), 4)
        self.assertEqual(sorted(parents.tolist()), [2, 2, 3, 3])

    def test_truncation_rejects_bad_fraction(self):
        for frac in (0, -0.1, 1.5, 2.5):
            with self.assertRaises(ValueError):
                truncation_selection(torch.zeros(4), keep_frac=frac)

    def test_crossover_rejects_odd_population(self):
        with self.assertRaises(ValueError):
            arithmetic_crossover(torch.zeros(3, 2), torch.arange(3), "cpu")


class ValidationSplit(unittest.TestCase):
    def setUp(self):
        self.labels = torch.arange(10).repeat_interleave(100)

    def test_disjoint_complete_stratified(self):
        train, val = stratified_split_indices(self.labels, 0.1, seed=1)
        self.assertEqual(len(set(train.tolist()) & set(val.tolist())), 0)
        self.assertEqual(len(train) + len(val), 1000)
        self.assertEqual(torch.bincount(self.labels[val]).tolist(), [10] * 10)

    def test_seeded(self):
        a = stratified_split_indices(self.labels, 0.1, seed=1)[1]
        b = stratified_split_indices(self.labels, 0.1, seed=1)[1]
        c = stratified_split_indices(self.labels, 0.1, seed=2)[1]
        self.assertTrue(torch.equal(a, b))
        self.assertFalse(torch.equal(a, c))


if __name__ == "__main__":
    unittest.main()

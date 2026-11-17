"""End-to-end tests of run.py on a tiny synthetic dataset (CPU, a few seconds)."""
import contextlib
import csv
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run
from utils.fitness_function import FitnessFunction
from models.vector_mlp import VectorMLP
from torch.utils.data import DataLoader, TensorDataset


def write_fake_mnist(data_root):
    rng = np.random.default_rng(0)
    folder = data_root / "mnist"
    folder.mkdir(parents=True)
    header = "label," + ",".join(f"pixel{i}" for i in range(784))
    for split, n in (("train", 200), ("test", 50)):
        rows = np.concatenate([(np.arange(n) % 10)[:, None], rng.integers(0, 256, (n, 784))], axis=1)
        np.savetxt(folder / f"mnist_{split}.csv", rows, fmt="%d", delimiter=",", header=header, comments="")


class RunnerEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        write_fake_mnist(cls.root / "data")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def run_cli(self, *extra):
        out = self.root / "out" / self.id().rsplit(".", 1)[-1]
        argv = ["--data-root", str(self.root / "data"), "--out", str(out), "--device", "cpu", "--eval-test",
                "--fitness-samples", "50", "--fitness-batch", "25", "--chunk-size", "2", *extra]
        with contextlib.redirect_stdout(io.StringIO()):
            code = run.main(argv)
        runs = sorted(out.iterdir())
        summaries = []
        for run_dir in runs:
            summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["status"], "completed", summary.get("traceback"))
            config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
            self.assertEqual(config["data"]["n_val"], 20)
            with open(run_dir / "metrics.csv", newline="", encoding="utf-8") as fh:
                self.assertGreater(len(list(csv.DictReader(fh))), 0)
            self.assertIn("test_acc", summary)
            summaries.append(summary)
        self.assertEqual(code, 0)
        return runs, summaries

    def test_adam_subspace_sweep(self):
        runs, _ = self.run_cli("--optimizer", "adam", "--dims", "4", "8", "--seeds", "0", "1", "--epochs", "2")
        self.assertEqual(len(runs), 4)
        self.assertTrue(all((r / "z.pt").exists() for r in runs))

    def test_full_adam(self):
        runs, summaries = self.run_cli("--mode", "full", "--optimizer", "adam", "--epochs", "1")
        self.assertEqual(len(runs), 1)
        self.assertTrue((runs[0] / "model.pt").exists())
        self.assertEqual(summaries[0]["num_trainable"], summaries[0]["num_weights"])

    def test_pso(self):
        _, summaries = self.run_cli("--optimizer", "pso", "--dims", "4", "--population", "4", "--max-iter", "2")
        self.assertEqual(summaries[0]["candidate_evals"], 12)  # initial + 2 iterations

    def test_ga(self):
        _, summaries = self.run_cli("--optimizer", "ga", "--dims", "4", "--population", "4", "--max-iter", "2",
                                    "--ga-selection", "truncation")
        self.assertEqual(summaries[0]["candidate_evals"], 8)

    def test_norm_options_recorded(self):
        _, summaries = self.run_cli("--optimizer", "ga", "--dims", "12", "--population", "4", "--max-iter", "1",
                                    "--init-norm", "2", "--ga-sigma-norm", "6")
        self.assertAlmostEqual(summaries[0]["init_scale_effective"], 1.0)
        self.assertAlmostEqual(summaries[0]["ga_sigma_effective"], 6 / 12 ** 0.5)

    def test_same_seed_is_reproducible(self):
        _, a = self.run_cli("--optimizer", "adam", "--dims", "4", "--epochs", "2", "--seeds", "3", "3")
        self.assertEqual(a[0]["val_loss"], a[1]["val_loss"])


class NormalisedScales(unittest.TestCase):
    def args(self, *extra):
        return run.parse_args(list(extra))

    def test_defaults_keep_per_coordinate_values(self):
        self.assertEqual(run.effective_scales(self.args("--init-scale", "3", "--ga-sigma", "1"), 25), (3.0, 1.0))

    def test_norm_options_match_d25_settings_and_fix_lengths(self):
        args = self.args("--init-norm", str(3 * (25 / 3) ** 0.5), "--ga-sigma-norm", "5")
        scale, sigma = run.effective_scales(args, 25)
        self.assertAlmostEqual(scale, 3.0)
        self.assertAlmostEqual(sigma, 1.0)
        torch.manual_seed(0)
        for dim in (25, 500):
            scale, sigma = run.effective_scales(args, dim)
            z = (torch.rand(2000, dim) * 2 - 1) * scale
            dz = torch.randn(2000, dim) * sigma
            self.assertAlmostEqual(z.norm(dim=1).pow(2).mean().sqrt().item(), args.init_norm, delta=0.2)
            self.assertAlmostEqual(dz.norm(dim=1).pow(2).mean().sqrt().item(), 5.0, delta=0.1)


class NegCrossEntropyFitness(unittest.TestCase):
    def test_matches_serial_cross_entropy(self):
        torch.manual_seed(0)
        model = VectorMLP(torch.nn.Sequential(torch.nn.Linear(784, 10)), lower_dim=3, proj_seed=0)
        x, y = torch.rand(10, 784), torch.arange(10)
        loader = DataLoader(TensorDataset(x, y), batch_size=4)
        zs = torch.randn(5, 3)
        got = FitnessFunction(model, loader, 2, "cpu", metric="neg_ce")(zs)
        with torch.no_grad():
            want = torch.stack([-torch.nn.functional.cross_entropy(model.parallel_eval(z, x), y) for z in zs])
        torch.testing.assert_close(got, want)


if __name__ == "__main__":
    unittest.main()

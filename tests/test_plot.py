"""Plotting script on synthetic run folders (no torch needed)."""
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import plot_runs
from summarize_runs import collect

ARGS = dict(dataset="mnist", model="mlp", mode="subspace", lr=0.03, population=4, init_scale=1.0, init_norm=8.66,
            fitness="neg_ce", ga_sigma=0.1, ga_sigma_norm=5.0, ga_selection="tournament", pso_w=0.7, pso_cp=1.5,
            pso_cg=1.5, max_iter=3)


def fake_run(root, name, optimizer, dim, seed, val_acc, fitness=(), dirty=False):
    folder = root / name
    folder.mkdir(parents=True)
    args = dict(ARGS, optimizer=optimizer, mode="full" if dim is None else "subspace")
    (folder / "config.json").write_text(json.dumps(dict(run=name, dim=dim, seed=seed, args=args,
                                                         environment=dict(git_commit="abc", git_dirty=dirty))))
    (folder / "summary.json").write_text(json.dumps(dict(status="completed", val_acc=val_acc)))
    lines = ["kind,step,best_fitness,candidate_evals"]
    lines += [f"iteration,{i},{f},{4 * (i + 1)}" for i, f in enumerate(fitness)]
    (folder / "metrics.csv").write_text("\n".join(lines) + "\n")


class PlotRuns(unittest.TestCase):
    def test_figures_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake_run(root / "out", "full", "adam", None, 0, 0.97)
            for seed in (0, 1):
                fake_run(root / "out", f"adam{seed}", "adam", 10, seed, 0.2)
                fake_run(root / "out", f"pso{seed}", "pso", 10, seed, 0.19, fitness=[-2.3, -2.2, -2.1][: 3 - seed])
                fake_run(root / "out", f"ga{seed}", "ga", 25, seed, 0.3, fitness=[-2.3, -2.0], dirty=True)
            frame = collect(root / "out")
            self.assertIn("it=3", frame[frame.optimizer == "pso"].setting.iloc[0])
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(plot_runs.main(["--runs", str(root / "out"), "--fig-dir", str(root / "fig")]), 0)
            names = sorted(p.name for p in (root / "fig").iterdir())
            self.assertEqual(names, ["accuracy_vs_d.png", "convergence_d10.png", "convergence_d25.png"])
            with contextlib.redirect_stdout(io.StringIO()):
                plot_runs.main(["--runs", str(root / "out"), "--fig-dir", str(root / "clean"), "--clean-only"])
            self.assertEqual(sorted(p.name for p in (root / "clean").iterdir()), ["accuracy_vs_d.png", "convergence_d10.png"])

    def test_curves_padded_with_last_value(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake_run(root, "a", "pso", 10, 0, 0.1, fitness=[-3, -2, -1])
            fake_run(root, "b", "pso", 10, 1, 0.1, fitness=[-3, -2.5])
            evals, values = plot_runs.mean_curve([root / "a", root / "b"])
            self.assertEqual(evals.tolist(), [4, 8, 12])
            self.assertEqual(values[1].tolist(), [-3, -2.5, -2.5])


if __name__ == "__main__":
    unittest.main()

"""Plot run.py results.

Figures (PNG) written to --fig-dir:
  accuracy_vs_d.png     final validation accuracy vs subspace dimension d (mean +/- std over
                        seeds), one line per optimiser setting, plus the full-network baseline
                        and its 90% level.
  convergence_d<d>.png  population methods at dimension d: best training loss (-fitness)
                        vs candidate evaluations, mean over seeds with a min-max band.

Usage: python scripts/plot_runs.py --runs out out_budget [--fig-dir figures] [--clean-only]
"""
import argparse
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from summarize_runs import collect

ROOT = Path(__file__).resolve().parents[1]


def load(run_folders, clean_only):
    frames = [f for f in (collect(Path(folder)) for folder in run_folders) if not f.empty]
    if not frames:
        return pd.DataFrame()
    with warnings.catch_warnings():  # all-NA columns (e.g. test_acc) are expected
        warnings.simplefilter("ignore", FutureWarning)
        frame = pd.concat(frames, ignore_index=True)
    if frame.empty:
        return frame
    frame = frame[frame.status == "completed"]
    dirty = frame.git_dirty.fillna(True).astype(bool)
    if clean_only:
        frame = frame[~dirty]
    elif dirty.any():
        print(f"note: {int(dirty.sum())} runs used uncommitted code (use --clean-only to drop them)")
    return frame


def plot_accuracy_vs_d(frame, path):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    sub = frame[frame.dim.notna()]
    for (opt, setting), grp in sub.groupby(["optimizer", "setting"]):
        stats = grp.groupby("dim").val_acc.agg(["mean", "std", "count"]).sort_index()
        ax.errorbar(stats.index, stats["mean"], yerr=stats["std"].fillna(0), marker="o", ms=4, capsize=3,
                    label=f"{opt}: {setting} (seeds {int(stats['count'].min())}-{int(stats['count'].max())})")
    full = frame[frame.dim.isna()]
    if len(full):
        base = full.val_acc.mean()
        ax.axhline(base, color="black", ls="--", lw=1, label=f"full network ({base:.3f})")
        ax.axhline(0.9 * base, color="gray", ls=":", lw=1, label=f"90% of full ({0.9 * base:.3f})")
    ax.set_xscale("log")
    ax.set_xlabel("subspace dimension d")
    ax.set_ylabel("validation accuracy")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def mean_curve(run_dirs):
    """Best-so-far fitness per fitness call, aligned by call index; stopped runs keep their last value."""
    curves = []
    for run_dir in run_dirs:
        metrics = pd.read_csv(Path(run_dir) / "metrics.csv")
        rows = metrics[metrics.kind == "iteration"]
        curves.append(rows[["candidate_evals", "best_fitness"]].to_numpy(float))
    longest = max(curves, key=len)
    values = np.stack([np.concatenate([c[:, 1], np.full(len(longest) - len(c), c[-1, 1])]) for c in curves])
    return longest[:, 0], values


def plot_convergence(frame, fig_dir):
    pop = frame[frame.dim.notna() & (frame.optimizer != "adam")]
    paths = []
    for dim, dim_grp in pop.groupby("dim"):
        fig, ax = plt.subplots(figsize=(9, 5.5))
        for (opt, setting), grp in dim_grp.groupby(["optimizer", "setting"]):
            evals, values = mean_curve(grp.run_dir)
            loss = -values
            line, = ax.plot(evals, loss.mean(0), label=f"{opt}: {setting} (n={len(grp)})")
            ax.fill_between(evals, loss.min(0), loss.max(0), color=line.get_color(), alpha=0.15)
        ax.set_xlabel("candidate evaluations")
        ax.set_ylabel("best training loss (-fitness)")
        ax.set_title(f"d = {int(dim)}")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
        fig.tight_layout()
        path = fig_dir / f"convergence_d{int(dim)}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)
    return paths


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, nargs="+", default=[ROOT / "out"], help="folders holding run directories")
    parser.add_argument("--fig-dir", type=Path, default=ROOT / "figures")
    parser.add_argument("--clean-only", action="store_true", help="drop runs made with uncommitted code")
    args = parser.parse_args(argv)
    frame = load(args.runs, args.clean_only)
    if frame.empty:
        print("no completed runs found")
        return 1
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    plot_accuracy_vs_d(frame, args.fig_dir / "accuracy_vs_d.png")
    paths = [args.fig_dir / "accuracy_vs_d.png", *plot_convergence(frame, args.fig_dir)]
    print("\n".join(str(p) for p in paths))
    return 0


if __name__ == "__main__":
    sys.exit(main())

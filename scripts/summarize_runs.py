"""Collect every run folder under out/ into one CSV and print mean/std per setting.

Usage: python scripts/summarize_runs.py [--out out] [--csv out/summary.csv]
"""
import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
COLUMNS = ["run", "status", "dataset", "model", "mode", "dim", "optimizer", "setting", "seed", "val_acc", "val_loss",
           "test_acc", "test_loss", "epochs_run", "iterations", "candidate_evals", "examples_processed",
           "elapsed_s", "peak_gpu_mem_mb", "git_commit", "git_dirty"]


def setting_label(args):
    """Short label of the hyperparameters that distinguish runs of one optimiser."""
    if args["optimizer"] == "adam":
        return f"lr={args['lr']}"
    init = f"norm={args['init_norm']}" if args.get("init_norm") is not None else args["init_scale"]
    label = f"pop={args['population']} init={init} fit={args['fitness']}"
    if args["optimizer"] == "ga":
        sigma = f"norm={args['ga_sigma_norm']}" if args.get("ga_sigma_norm") is not None else args["ga_sigma"]
        label += f" sigma={sigma} sel={args['ga_selection']}"
    if args["optimizer"] == "pso":
        label += f" w={args['pso_w']} cp={args['pso_cp']} cg={args['pso_cg']}"
    return label


def collect(out):
    rows = []
    for summary_path in sorted(out.glob("*/summary.json")):
        config_path = summary_path.with_name("config.json")
        if not config_path.exists():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        config = json.loads(config_path.read_text(encoding="utf-8"))
        args, env = config["args"], config["environment"]
        rows.append(dict(summary, run=config["run"], dataset=args["dataset"], model=args["model"],
                         mode=args["mode"], dim=config["dim"], optimizer=args["optimizer"],
                         lr=args["lr"] if args["optimizer"] == "adam" else None, seed=config["seed"],
                         setting=setting_label(args),
                         git_commit=env.get("git_commit"), git_dirty=env.get("git_dirty")))
    frame = pd.DataFrame(rows)
    return frame.reindex(columns=[c for c in COLUMNS if c in frame.columns] + [c for c in frame.columns if c not in COLUMNS and c != "traceback"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "out")
    parser.add_argument("--csv", type=Path, default=None)
    args = parser.parse_args()
    frame = collect(args.out)
    if frame.empty:
        print(f"no runs found in {args.out}")
        return
    target = args.csv or args.out / "summary.csv"
    frame.to_csv(target, index=False)
    print(f"{len(frame)} runs -> {target}")
    done = frame[frame.status == "completed"].copy()
    if done.empty:
        return
    done["dim"] = done["dim"].fillna(-1).astype(int)  # -1 = full parameter space
    table = (done.groupby(["dataset", "model", "optimizer", "setting", "dim"])["val_acc"]
                 .agg(["count", "mean", "std", "min", "max"]).round(4))
    with pd.option_context("display.max_rows", None, "display.width", 200):
        print(table)


if __name__ == "__main__":
    main()

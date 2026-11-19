"""Run low-dimensional neural-network optimisation experiments.

One run = one (dataset, model, mode, d, optimizer, seed). Every run writes a fresh
folder under --out containing config.json, metrics.csv, summary.json and the final
weights. Lists given to --dims and --seeds run every combination in sequence.

Examples (PowerShell, repository root):
  .\\.venv\\Scripts\\python.exe run.py --mode full --optimizer adam
  .\\.venv\\Scripts\\python.exe run.py --optimizer adam --dims 10 25 50 100 250 500 1000 --seeds 0 1 2
  .\\.venv\\Scripts\\python.exe run.py --optimizer pso --dims 50 --population 256 --max-iter 100

Model selection uses a validation split of the official training set. The official
test set is evaluated only with --eval-test; use it once choices are frozen.
"""
import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from models.lenet5 import LeNet5
from models.mnist_mlp import MNIST_MLP
from models.vector_mlp import VectorMLP
from optimizers.ga import GeneticAlgorithmOptimizer
from optimizers.gwo import GreyWolfOptimizer
from optimizers.mpa import MarinePredatorsOptimizer
from optimizers.pso import ParticleSwarmOptimizer
from utils.data_split import stratified_split_indices
from utils.fitness_function import FitnessFunction
from utils.ga_utils import arithmetic_crossover, gaussian_mutation, tournament_selection, truncation_selection
from utils.mnist_dataloader import MNISTDataset
from utils.train_model import _train_one_epoch, _validate

DATASETS = {
    "mnist": ("mnist/mnist_train.csv", "mnist/mnist_test.csv"),
    "fashionmnist": ("fashionmnist/fashion-mnist_train.csv", "fashionmnist/fashion-mnist_test.csv"),
}
# Offsets keep the init / projection / optimiser RNG streams distinct for one --seeds value.
PROJ_SEED_OFFSET = 10_000
OPT_SEED_OFFSET = 20_000


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_argument_group("experiment")
    g.add_argument("--dataset", choices=DATASETS, default="mnist")
    g.add_argument("--model", choices=["mlp", "lenet"], default="mlp")
    g.add_argument("--hidden", type=int, nargs="+", default=[200, 200], help="MLP hidden sizes")
    g.add_argument("--mode", choices=["subspace", "full"], default="subspace")
    g.add_argument("--dims", type=int, nargs="+", default=[100], help="subspace dimensions d")
    g.add_argument("--optimizer", choices=["adam", "pso", "ga", "gwo", "mpa"], default="adam")
    g.add_argument("--seeds", type=int, nargs="+", default=[0])
    g.add_argument("--split-seed", type=int, default=0, help="validation split seed (kept fixed across runs)")
    g.add_argument("--val-frac", type=float, default=0.1)
    g.add_argument("--eval-test", action="store_true", help="also evaluate the official test set")
    g.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    g.add_argument("--data-root", type=Path, default=ROOT / "data")
    g.add_argument("--out", type=Path, default=ROOT / "out")
    g.add_argument("--eval-batch", type=int, default=2048)

    g = p.add_argument_group("adam")
    g.add_argument("--epochs", type=int, default=20)
    g.add_argument("--lr", type=float, default=1e-3)
    g.add_argument("--batch-size", type=int, default=128)
    g.add_argument("--patience", type=int, default=5, help="early-stopping patience in epochs")

    g = p.add_argument_group("population methods (pso, ga, gwo, mpa)")
    g.add_argument("--population", type=int, default=256)
    g.add_argument("--max-iter", type=int, default=100)
    g.add_argument("--pop-patience", type=int, default=15)
    g.add_argument("--init-scale", type=float, default=1.0, help="initial z uniform in [-s, s]^d")
    g.add_argument("--init-norm", type=float, default=None,
                   help="expected initial ||z|| (= distance from theta0, A has unit columns); "
                        "overrides --init-scale with s = R*sqrt(3/d), so the spread is the same for every d")
    g.add_argument("--fitness", choices=["neg_ce", "accuracy"], default="neg_ce")
    g.add_argument("--fitness-samples", type=int, default=10000, help="fixed training subset used for fitness")
    g.add_argument("--fitness-batch", type=int, default=2000)
    g.add_argument("--chunk-size", type=int, default=64, help="candidates evaluated in parallel (GPU memory)")
    g.add_argument("--pso-w", type=float, default=0.7)
    g.add_argument("--pso-cp", type=float, default=1.5)
    g.add_argument("--pso-cg", type=float, default=1.5)
    g.add_argument("--ga-selection", choices=["tournament", "truncation"], default="tournament")
    g.add_argument("--ga-tournament", type=int, default=5)
    g.add_argument("--ga-keep-frac", type=float, default=0.5)
    g.add_argument("--ga-sigma", type=float, default=0.1, help="per-coordinate mutation std")
    g.add_argument("--mpa-fads", type=float, default=0.2, help="MPA FADs effect probability")
    g.add_argument("--mpa-p", type=float, default=0.5, help="MPA step-size constant P")
    g.add_argument("--ga-sigma-norm", type=float, default=None,
                   help="expected mutation step length ||dz||; overrides --ga-sigma with S/sqrt(d)")
    return p.parse_args(argv)


# ---------------------------------------------------------------- bookkeeping
def _git(*cmd):
    try:
        res = subprocess.run(["git", *cmd], cwd=ROOT, capture_output=True, text=True, timeout=60)
        return res.stdout.strip() if res.returncode == 0 else None
    except Exception:
        return None


def environment():
    status = _git("status", "--porcelain", "-uno")
    return dict(git_commit=_git("rev-parse", "HEAD"), git_dirty=None if status is None else bool(status),
                python=sys.version, platform=platform.platform(), torch=torch.__version__,
                numpy=np.__version__, cuda=torch.version.cuda,
                gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


class MetricsLog:
    FIELDS = ["kind", "step", "train_loss", "train_acc", "val_loss", "val_acc", "best_fitness",
              "mean_fitness", "candidate_evals", "examples_processed", "lr", "elapsed_s"]

    def __init__(self, path):
        self.fh = open(path, "w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.fh, self.FIELDS)
        self.writer.writeheader()

    def row(self, **values):
        self.writer.writerow(values)
        self.fh.flush()

    def close(self):
        self.fh.close()


# ---------------------------------------------------------------- data / model
def load_data(args):
    train_path, test_path = (args.data_root / rel for rel in DATASETS[args.dataset])
    train, test = MNISTDataset(train_path), MNISTDataset(test_path)
    train_idx, val_idx = stratified_split_indices(train.label, args.val_frac, args.split_seed)
    info = dict(train_csv=str(train_path), train_csv_sha256=sha256(train_path), test_csv_sha256=sha256(test_path),
                n_train=len(train_idx), n_val=len(val_idx), n_test=len(test),
                val_indices_sha256=hashlib.sha256(val_idx.numpy().tobytes()).hexdigest())
    return dict(train=Subset(train, train_idx.tolist()), val=Subset(train, val_idx.tolist()), test=test,
                train_idx=train_idx, info=info)


def build_model(args, dim, seeds):
    torch.manual_seed(seeds["init"])
    base = MNIST_MLP(args.hidden) if args.model == "mlp" else LeNet5()
    if args.mode == "full":
        return base
    return VectorMLP(base, lower_dim=dim, proj_seed=seeds["proj"])


# ---------------------------------------------------------------- optimisers
def run_adam(model, args, data, device, log, seeds, t0):
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=max(1, args.patience // 2))
    train_loader = DataLoader(data["train"], batch_size=args.batch_size, shuffle=True,
                              generator=torch.Generator().manual_seed(seeds["opt"]))
    val_loader = DataLoader(data["val"], batch_size=args.eval_batch)
    criterion = nn.CrossEntropyLoss()

    def snapshot():  # only trainable tensors: avoids copying the (large) projection matrix
        return {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}

    best_loss, best_state, wait, examples, epochs_run = float("inf"), snapshot(), 0, 0, 0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = _train_one_epoch(model, device, train_loader, criterion, optimizer)
        examples += len(data["train"])
        epochs_run = epoch
        val_loss, val_acc = _validate(model, device, val_loader, criterion)
        scheduler.step(val_loss)
        log.row(kind="epoch", step=epoch, train_loss=train_loss, train_acc=train_acc, val_loss=val_loss,
                val_acc=val_acc, examples_processed=examples, lr=optimizer.param_groups[0]["lr"],
                elapsed_s=time.perf_counter() - t0)
        print(f"  epoch {epoch:3d}  train {train_loss:.4f}/{train_acc:.4f}  val {val_loss:.4f}/{val_acc:.4f}", flush=True)
        if val_loss < best_loss:
            best_loss, best_state, wait = val_loss, snapshot(), 0
        else:
            wait += 1
            if wait >= args.patience:
                print("  early stopping", flush=True)
                break

    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in best_state:
                param.copy_(best_state[name])
    return dict(epochs_run=epochs_run, examples_processed=examples, gradient_steps=epochs_run * len(train_loader),
                best_val_loss_during_training=best_loss)


class LoggedFitness:
    """Wraps a fitness function: counts evaluations and logs best-so-far per call."""

    def __init__(self, fn, log, samples_per_eval, t0):
        self.fn, self.log, self.samples, self.t0 = fn, log, samples_per_eval, t0
        self.calls, self.evals, self.best = 0, 0, -float("inf")

    def __call__(self, candidates):
        values = self.fn(candidates)
        self.evals += candidates.shape[0]
        self.best = max(self.best, values.max().item())
        self.log.row(kind="iteration", step=self.calls, best_fitness=self.best, mean_fitness=values.mean().item(),
                     candidate_evals=self.evals, examples_processed=self.evals * self.samples,
                     elapsed_s=time.perf_counter() - self.t0)
        self.calls += 1
        return values


def effective_scales(args, dim):
    """Per-coordinate init range and GA sigma; *-norm options keep lengths fixed across d."""
    # z ~ U[-s, s]^d has E||z||^2 = d s^2 / 3; Gaussian mutation has E||dz||^2 = d sigma^2.
    init_scale = args.init_scale if args.init_norm is None else args.init_norm * (3 / dim) ** 0.5
    ga_sigma = args.ga_sigma if args.ga_sigma_norm is None else args.ga_sigma_norm / dim ** 0.5
    return init_scale, ga_sigma


def run_population(model, args, dim, data, device, log, seeds, t0):
    if args.optimizer == "ga" and args.population % 2:
        raise ValueError("GA with arithmetic crossover needs an even --population")
    pool = data["train_idx"]
    n = min(args.fitness_samples, len(pool))
    chosen = pool[torch.randperm(len(pool), generator=torch.Generator().manual_seed(seeds["opt"]))[:n]]
    fit_loader = DataLoader(Subset(data["train"].dataset, chosen.tolist()), batch_size=args.fitness_batch)
    fitness = LoggedFitness(FitnessFunction(model, fit_loader, args.chunk_size, device, metric=args.fitness),
                            log, n, t0)

    init_scale, ga_sigma = effective_scales(args, dim)
    torch.manual_seed(seeds["opt"])
    if args.optimizer == "pso":
        z, _ = ParticleSwarmOptimizer(args.population, dim, fitness, device).train(
            w=args.pso_w, cp=args.pso_cp, cg=args.pso_cg, max_iter=args.max_iter,
            patience=args.pop_patience, init_scale=init_scale)
    elif args.optimizer == "gwo":
        z, _ = GreyWolfOptimizer(args.population, dim, fitness, device).train(
            max_iter=args.max_iter, patience=args.pop_patience, init_scale=init_scale)
    elif args.optimizer == "mpa":
        z, _ = MarinePredatorsOptimizer(args.population, dim, fitness, device).train(
            max_iter=args.max_iter, patience=args.pop_patience, init_scale=init_scale,
            fads=args.mpa_fads, p=args.mpa_p)
    else:
        if args.ga_selection == "tournament":
            selection = lambda f: tournament_selection(f, device, args.ga_tournament)
        else:
            selection = lambda f: truncation_selection(f, args.ga_keep_frac)
        z, _ = GeneticAlgorithmOptimizer(args.population, dim, fitness, device).train(
            selection, lambda g, p: arithmetic_crossover(g, p, device),
            lambda g: gaussian_mutation(g, device, ga_sigma),
            max_iter=args.max_iter, patience=args.pop_patience, init_scale=init_scale)

    with torch.no_grad():
        model.lower_weights.copy_(z.to(model.lower_weights.device))
    fit_loss, fit_acc = _validate(model, device, fit_loader, nn.CrossEntropyLoss())
    return dict(iterations=fitness.calls, candidate_evals=fitness.evals, examples_processed=fitness.evals * n,
                best_fitness=fitness.best, fitness_samples=n, init_scale_effective=init_scale,
                ga_sigma_effective=ga_sigma if args.optimizer == "ga" else None, fitness_subset_loss=fit_loss,
                fitness_subset_acc=fit_acc)


# ---------------------------------------------------------------- one run
def make_run_dir(out, tag):
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    for k in range(1000):
        run_dir = out / (f"{stamp}_{tag}" + (f"_{k}" if k else ""))
        try:
            run_dir.mkdir(parents=True, exist_ok=False)
            return run_dir
        except FileExistsError:
            continue
    raise RuntimeError("could not create a unique run directory")


def run_one(args, dim, seed, data, env):
    seeds = dict(split=args.split_seed, init=seed, proj=seed + PROJ_SEED_OFFSET, opt=seed + OPT_SEED_OFFSET)
    tag = f"{args.dataset}_{args.model}_{args.mode}" + (f"_d{dim}" if dim else "") + f"_{args.optimizer}_s{seed}"
    run_dir = make_run_dir(args.out, tag)
    device = torch.device(args.device)
    print(f"=== {run_dir.name}", flush=True)

    config = dict(run=run_dir.name, dim=dim, seed=seed, seeds=seeds,
                  args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  data=data["info"], environment=env)
    write_json(run_dir / "config.json", config)
    summary = dict(status="running", started=datetime.now().isoformat(timespec="seconds"))
    write_json(run_dir / "summary.json", summary)

    log = MetricsLog(run_dir / "metrics.csv")
    t0 = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model = None
    try:
        model = build_model(args, dim, seeds).to(device)
        summary["num_weights"] = sum(p.numel() for p in (model.model if dim else model).parameters())
        summary["num_trainable"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if dim:
            summary["projection_mb"] = model.proj_matrix.numel() * model.proj_matrix.element_size() / 2**20

        if args.optimizer == "adam":
            result = run_adam(model, args, data, device, log, seeds, t0)
        else:
            result = run_population(model, args, dim, data, device, log, seeds, t0)

        criterion = nn.CrossEntropyLoss()
        val_loss, val_acc = _validate(model, device, DataLoader(data["val"], batch_size=args.eval_batch), criterion)
        result.update(val_loss=val_loss, val_acc=val_acc)
        if args.eval_test:
            test_loss, test_acc = _validate(model, device, DataLoader(data["test"], batch_size=args.eval_batch), criterion)
            result.update(test_loss=test_loss, test_acc=test_acc)
        log.row(kind="final", val_loss=val_loss, val_acc=val_acc, elapsed_s=time.perf_counter() - t0)

        if dim:
            torch.save(dict(z=model.lower_weights.detach().cpu(), dim=dim, seeds=seeds), run_dir / "z.pt")
        else:
            torch.save(model.state_dict(), run_dir / "model.pt")
        summary.update(status="completed", **result)
        print(f"  done: val acc {val_acc:.4f}  ({time.perf_counter() - t0:.1f}s)", flush=True)
    except Exception as exc:
        summary.update(status="failed", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        print(f"  FAILED: {summary['error']}", flush=True)
    finally:
        log.close()
        if summary["status"] == "running":
            summary["status"] = "interrupted"
        summary.update(finished=datetime.now().isoformat(timespec="seconds"), elapsed_s=time.perf_counter() - t0,
                       peak_gpu_mem_mb=torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None)
        write_json(run_dir / "summary.json", summary)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return summary["status"] == "completed"


def main(argv=None):
    args = parse_args(argv)
    if args.mode == "full" and args.optimizer != "adam":
        raise SystemExit("--mode full only supports --optimizer adam")
    dims = [None] if args.mode == "full" else args.dims
    data, env = load_data(args), environment()
    if env["git_dirty"]:
        print("warning: uncommitted changes to tracked files; recorded as git_dirty=true", flush=True)
    results = [run_one(args, dim, seed, data, env) for dim in dims for seed in args.seeds]
    print(f"{sum(results)}/{len(results)} runs completed; results in {args.out}", flush=True)
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())

# Low-dimensional neural network optimization

Original research snapshot: `e107b34`. Setup additions preserve all original notebooks,
models, optimizers and utilities. See [implementation review](docs/REVIEW.md),
[development plan](docs/TODO.md), and [local verification](docs/SETUP_RESULTS.md).

## Windows setup (PowerShell, repository root)

Tested target: Python 3.11.9, Windows 11 x64, RTX 3060 Laptop GPU (6 GB).
The original repository declares no Python version or dependencies. Notebook metadata
says 3.14.8, but that is not a validated requirement. `torch.func` requires PyTorch 2.x;
this setup pins a compatible PyTorch/torchvision pair and Python 3.11.

```powershell
git clone https://github.com/palmarbg/compint_nnopt_lowdim.git
cd compint_nnopt_lowdim
git switch -c feature/local-setup-review
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe scripts/prepare_data.py
.\.venv\Scripts\python.exe scripts/smoke_test.py
```

These setup files must be present locally before running their commands; no setup branch
has been pushed. Activation is optional; explicit interpreter paths avoid PowerShell
execution-policy changes. CUDA runtime libraries come with the wheel; `nvcc` is not
needed. The driver's reported CUDA version is its capability, not the installed wheel
runtime. The version pair and index follow [PyTorch's official installation commands](https://pytorch.org/get-started/previous-versions/).

For exact reproduction of this Windows/Python 3.11 environment, install the same CUDA
pair first, then `python -m pip install -r requirements-lock-win-py311.txt` using the venv
interpreter. `requirements.txt` pins direct dependencies; the lock records transitives.
For CPU-only machines replace the first install's index with
`https://download.pytorch.org/whl/cpu` and use `requirements.txt`, not the CUDA lock.
CPU-only setup is not separately installed here; smoke checks also exercise CPU.

## Data

All four CSVs were missing in the fresh checkout. `prepare_data.py` downloads original
IDX archives, uses torchvision's published MD5 checks, and exports label-first CSVs
with a header and 784 integer pixel columns (0..255). The existing loader divides by
255 exactly once. Each dataset has 60,000 training and 10,000 test rows.

| Dataset | Files expected by `main.ipynb` | Source |
| --- | --- | --- |
| MNIST | `data/mnist/mnist_train.csv`, `mnist_test.csv` | [Official torchvision loader's OSSCI mirror of original MNIST](https://docs.pytorch.org/vision/0.22/_modules/torchvision/datasets/mnist.html) |
| Fashion-MNIST | `data/fashionmnist/fashion-mnist_train.csv`, `fashion-mnist_test.csv` | [Dataset authors' repository](https://github.com/zalandoresearch/fashion-mnist/tree/master/data/fashion) |

Use `--dataset mnist` or `--dataset fashionmnist` for individual downloads. Raw files
are retained under `data/raw/`. Existing CSVs are preserved, not regenerated or validated
by this command. Allow roughly 0.5 GB for data and several GB for the CUDA environment
and pip cache. Data and environment files are ignored by Git.

The original `data.ipynb` uses Kaggle downloads and a Bash cleanup cell. Its imports
are installed for compatibility, but it is unnecessary for this official-source path.
The Bash cleanup cell deletes non-CSV files and is not portable to native PowerShell.

## Development and execution

```powershell
.\.venv\Scripts\python.exe -m ipykernel install --sys-prefix --name nnopt-lowdim --display-name "NNOpt (.venv)"
.\.venv\Scripts\python.exe -m jupyterlab
```

Select `NNOpt (.venv)` or `.venv/Scripts/python.exe` in the notebook editor. Start Jupyter
from the repository root so imports and relative data paths resolve. `main.ipynb` loads
both datasets, trains full MLP/LeNet, then subspace MLP/LeNet with Adam, then PSO and GA.
There is no original CLI. The population sections use the most recently assigned model
(subspace LeNet in sequential execution). `out/` is not created automatically; when
planning experiments, create it and change output paths in a separate notebook copy to
a fresh run directory to avoid overwriting checkpoints.

Do not use Run All as a setup check: notebook defaults include 4096-sample batches,
4096 PSO particles, population chunk size 64, and up to 100 iterations. These can exceed
6 GB VRAM and take substantial time. `smoke_test.py` imports every module, validates and
compiles notebook cells, executes only their import statements, and takes one Adam step
on 32 MNIST training examples per model on CPU and available CUDA. It also compares
vectorized versus serial fitness on two candidates. It writes no research outputs.
Its accuracy is a plumbing check, not an experimental result.

Dependencies: torch for models/optimization; pandas/numpy for CSVs and display helpers;
matplotlib for plots; torchsummary for notebook summaries; kagglehub for the legacy data
notebook; torchvision for official data loading; JupyterLab/ipykernel/nbformat for
notebooks and validation. No GWO/MPA libraries are needed for existing code.

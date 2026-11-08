# Local setup verification — 2026-10-23

Checkout: `D:\Uni\3.semester\ComputationalInt\nnWeights\compint_nnopt_lowdim`.
Branch: `feature/local-setup-review`, based on `e107b34`. No push or PR.

Environment: Windows 11 Pro x64; Intel i5-12500H (12 cores, 16 logical processors),
approximately 63.7 GiB RAM; NVIDIA RTX 3060 Laptop GPU, 6144 MiB VRAM, driver 610.62.
Python 3.11.9 in `.venv`; Python 3.12 also present but not used. PyTorch
2.7.1+cu126, torchvision 0.22.1+cu126, wheel CUDA runtime 12.6. `nvcc` was not found
on PATH and is not required for this project. CUDA execution succeeded.

- `pip check`: no broken requirements.
- JupyterLab 4.4.3 installed; `NNOpt (.venv)` kernel registered with `--sys-prefix`.
- All 12 original Python modules imported.
- Both notebooks validated and all code cells compiled after IPython transformation;
  executed 14 top-level import statements from `main.ipynb` and 3 from `data.ipynb`.
  Training, downloads and Bash cells were not executed from the notebooks.
- Official IDX download/conversion completed for both datasets: 60,000 train and
  10,000 test rows each. Rerunning preparation preserved all four CSV files.
  A full chunked CSV scan verified all row counts, 785 columns, labels 0..9 and
  pixel values 0..255.
- Smoke: first 32 MNIST training examples, seed 42, batch size 32, one Adam step per
  model/device, d=8 for subspace wrappers. Finite loss/gradients, changed trainable
  weights, unchanged projection/offset buffers and frozen base parameters passed.
- Two-candidate vectorized fitness matched serial accuracy for both wrapped models
  on CPU and CUDA. Validation forward passes also produced finite losses.

| Model | CPU initial batch CE | CUDA initial batch CE |
| --- | --- | --- |
| MLP [200,200] | 2.2999349 | 2.2999351 |
| LeNet5 | 2.2834523 | 2.2834518 |
| Subspace MLP, d=8 | 2.3018441 | 2.3018436 |
| Subspace LeNet5, d=8 | 2.3014030 | 2.3014023 |

All eight single-step checks passed. These losses are pre-update batch values, not
evidence of convergence. No expensive experiments or full notebook execution occurred.
The 6 GB GPU is sufficient for these checks; full notebook populations, large batches
and large dimension sweeps remain unvalidated and may exceed available VRAM.

Additional bounded review diagnostics confirmed checkpoint tensor aliasing, PSO
patience/history errors, and reversed truncation selection; see REVIEW.md. They used
tiny synthetic tensors and no research training. They document defects rather than
changing the original algorithms.

Added files: README, direct requirements, Windows/Python dependency lock, two setup
scripts, and these three documentation files. `.gitignore` additionally excludes
the virtual environment and notebook checkpoints. Original notebooks, modules and
saved outputs are unchanged. Local data, caches, kernel files and `.venv` are ignored.

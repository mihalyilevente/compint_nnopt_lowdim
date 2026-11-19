# Final comparison protocol (MNIST, 784-200-200-10 MLP).
# Population methods: population 256, initial ||z|| = 8.66, a fixed budget of
# ~256k fitness evaluations and no early stopping (GWO/MPA schedules depend on
# max_iter). MPA makes two fitness calls per iteration, so it gets 500 iterations.
# Every run also evaluates the official test set. Results go to out_final/.
#
# Usage (repository root):
#   powershell -ExecutionPolicy Bypass -File scripts\final_runs.ps1                 # everything
#   powershell -ExecutionPolicy Bypass -File scripts\final_runs.ps1 -Dims 25,100    # a subset
#   ... -SkipFull to skip the full-network baseline when resuming
param(
    [int[]]$Dims = @(25, 100, 250, 500, 1000),
    [switch]$SkipFull
)
$ErrorActionPreference = "Stop"
$py = ".\.venv\Scripts\python.exe"
$out = "out_final"
$seeds = @("--seeds", "0", "1", "2")
$pop = @("--population", "256", "--init-norm", "8.66", "--pop-patience", "100000", "--eval-test", "--out", $out) + $seeds

if (-not $SkipFull) {
    & $py run.py --mode full --optimizer adam --epochs 30 --eval-test --out $out @seeds
}
foreach ($d in $Dims) {
    Write-Host "===== d = $d =====" -ForegroundColor Cyan
    & $py run.py --optimizer adam --lr 0.03 --epochs 30 --dims $d --eval-test --out $out @seeds
    & $py run.py --optimizer pso --dims $d --max-iter 1000 @pop
    & $py run.py --optimizer ga --dims $d --max-iter 1000 --ga-sigma-norm 5 @pop
    & $py run.py --optimizer gwo --dims $d --max-iter 1000 @pop
    & $py run.py --optimizer mpa --dims $d --max-iter 500 @pop
}
& $py scripts/summarize_runs.py --out $out
& $py scripts/plot_runs.py --runs $out --fig-dir figures/final --clean-only

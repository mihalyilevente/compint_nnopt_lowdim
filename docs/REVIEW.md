# Implementation review

Reviewed all 15 tracked files at `e107b34` (two commits: initial code and missing
models). Findings below describe the original code, which was not edited. Notebook
cell references are zero-based JSON cell indices. Saved outputs are historical
evidence, not rerun results or proof that current cell source produced them.

## What exists

| Component | Implementation |
| --- | --- |
| `models/mnist_mlp.py` | Configurable ReLU MLP; notebook uses 784-200-200-10, 199,210 parameters; expects flattened inputs. |
| `models/lenet5.py` | Sigmoid convolutional network with average pooling and 120/84-unit dense layers; 61,706 parameters; reshapes flat input to images. One shape comment before Flatten is inaccurate, but execution is correct. |
| `models/vector_mlp.py` | Generic wrapper despite its name; freezes base parameters, stores dense Gaussian A and offset as buffers, learns zero-initialized z, reconstructs named tensors with differentiable `functional_call`; supports both models and `vmap`. |
| `utils/train_model.py` | Adam-compatible epoch training, sample-weighted CE/accuracy, validation, plateau scheduler, early stopping and attempted best-state restoration. |
| `optimizers/pso.py` | Uniform initial positions, random velocities, personal/global best tracking, inertia/cognitive/social updates, history and patience. |
| `optimizers/ga.py`, `utils/ga_utils.py` | Tournament/truncation selection, arithmetic crossover, additive Gaussian mutation, external best genome and patience. |
| `utils/fitness_function.py` | No-gradient, population-vectorized accuracy over every sample in a supplied loader; chunking controls vmap computation. Maximization agrees with PSO/GA. |
| Other utilities | CSV loading/scaling and image/inference displays. |
| `data.ipynb`, `main.ipynb` | Kaggle data preparation and sequential interactive experiments; no configuration-driven runner. |

## Experiments evidenced by saved outputs

| Notebook cells | Saved evidence |
| --- | --- |
| `data.ipynb` 4, 8 | Fashion-MNIST and MNIST Kaggle download logs. |
| `main.ipynb` 26 | Full MLP, 50 epochs; final displayed validation accuracy 97.26%. |
| 35 | Full LeNet, 50 epochs; final 91.95%. |
| 41, 45 | Subspace MLP, current source d=1000; 50 epochs; final 91.89%. |
| 50, 54 | Subspace LeNet, current source d=1000; 50 epochs; final 95.31%. |
| 59, 61 | PSO, current source population 4096; 100 iterations; last displayed best training fitness 59.51%. Sequential execution targets subspace LeNet. |
| 66, 68 | GA, current source population 1024, tournament size 5; output through iteration 12, fitness 26.43%, then KeyboardInterrupt. No completed GA result established. |

Fashion-MNIST is loaded and displayed, but no saved training section uses it. There
are no tracked checkpoints, structured logs, dimension sweeps, seeds, or GWO/MPA code.
`data/` and `out/` are ignored, so absence from Git does not establish that the teammate
never produced other artifacts. Saved summaries show batch dimension 8192 while the
current constant is 4096: outputs and source are not a clean execution record.

## Confirmed defects

1. **Checkpoint aliases live parameters** (`utils/train_model.py:85`). `state_dict()`
   is not a snapshot copy. Later optimizer steps mutate the saved tensors; loading
   them at the end does not restore the best epoch. A live tensor-mutation diagnostic
   confirmed shared storage. Fix with a deep copy or detached cloned tensors and test
   a deliberately worsening validation trajectory.
2. **PSO initial global records alias mutable personal records** (`optimizers/pso.py:18-21`,
   updates at 38-39). Improvement of the initial winning particle changes the global
   value before the comparison, corrupts historical entries, and can fail to reset
   patience. A deterministic two-particle increasing-fitness probe produced history
   `[3,3,3]` instead of recording initial fitness 1 and stopped despite improvement.
   Later boolean-indexed assignments allocate copies; the initial alias is the problem.
3. **PSO stagnation does not advance patience when no particle improves** (41-48).
   Constant fitness with `max_iter=5, patience=2` ran all five iterations (history length
   six). Count every iteration without a global improvement, including `mask.any()==False`.
4. **Truncation selection favors low fitness and has invalid percentage semantics**
   (`utils/ga_utils.py:3-7`). On scores `[0,1,2,3]`, fraction `.25` returns parents
   `[0,1,2,0]`, discarding the best and duplicating the worst. Default `2.5` returns all
   four indices here, doing no truncation. Zero cut produces no parents. Notebook GA
   uses tournament selection, so this is not evidence explaining its saved result.
5. **Test set influences model selection** (`main.ipynb` 26/35/45/54). Official MNIST
   test data is passed as validation, driving the scheduler and early stopping.
   These reported values are not untouched final-test estimates. Training fitness
   itself uses the training loader, so there is no demonstrated test-label leakage
   inside PSO/GA fitness. Create validation from training data and reserve test data.
6. **Interrupted GA can reuse PSO output** (68 then 70). Failed tuple assignment leaves
   the previous `best_weights` binding intact. Continuing sequentially after the saved
   interrupt can display PSO weights under the GA heading. Saved images alone cannot
   establish which weights they show.

Fresh execution also needs an `out/` directory before save cells, CUDA because of the
hard-coded device, and Bash for the legacy data cleanup. These are workflow defects
or portability limits documented in the setup README.

## Methodological differences and conditional risks

The affine construction, frozen A/offset and z=0 initialization match the core idea.
However, Li et al. normalize projection **columns** to unit length and start from a
well-conditioned network initialization. Here A is unnormalized N(0,1), with column
norm near sqrt(D), and the offset is independently sampled N(0,1/D) rather than using
the base model's initialized weights. These are confirmed differences; their effects
on final performance require controlled experiments. Normalization changes coordinate
scale, not the column span. The paper's MLP has the same 199,210 parameters, but its
reported LeNet has 44,426 rather than this implementation's 61,706. It defines d_int90
relative to a directly trained baseline; one d=1000 experiment does not estimate that
threshold. See [Li et al., sections 2 and 3.1](https://arxiv.org/html/1804.08838v1).

- **Unequal objectives/budgets:** Adam minimizes cross-entropy; population algorithms
  maximize piecewise-constant accuracy. Equal iterations are not equal computation.
  PSO evaluates population once before the loop and once per iteration; GA once per
  iteration. At notebook limits, PSO uses 413,696 candidate dataset evaluations versus
  102,400 for GA. This is a comparison-design weakness, not a tensor implementation bug.
- **Starting scale:** Adam starts at z=0; PSO/GA sample z uniformly in [-1,1]. With
  unnormalized A, their initial high-dimensional perturbations are much larger. Both
  population routines create fresh populations; using an Adam-trained wrapper does
  not warm-start them from its z because `parallel_eval` substitutes candidates.
- **GA ownership:** `best_genom=genomes[best_idx]` is a view. Supplied crossover and
  mutation allocate new tensors, so corruption is not demonstrated in the notebook
  path. In-place callbacks would break it; clone as a defensive contract. No elitism,
  bounds, or mutation schedule is a design choice, not automatically a bug. Odd
  population sizes are unsupported by the pairing expression and should be rejected
  or handled explicitly.
- **Fitness:** totals are correctly sample-weighted, and local CPU/CUDA checks matched
  vmap to serial evaluation. Accuracy plateaus can obstruct search. Eval mode is set
  only in the constructor; later `model.train()` would change behavior for future
  dropout/batchnorm models. Current models contain neither. Empty loaders divide by
  zero. No evaluation counter, runtime, validation fitness, or CE fitness is recorded.
- **Memory:** dense A costs 4*D*d bytes in float32: approximately 760 MiB for this MLP
  at d=1000 and 235 MiB for LeNet. Candidate activations add much more. vmap chunking
  still returns the full population logits for each data batch; 4096 candidates by
  4096 samples by 10 classes alone occupy 640 MiB. Full runs were not tested on 6 GB.
- **Reproducibility:** no seed/split manifests, restartable optimizer state, shared
  objective contract, run IDs or robust checkpoint provenance. `torchsummary` output
  from functional substitution does not faithfully count the d trainable coordinates
  or projection-buffer storage. Inspect named parameters/buffers directly instead.

No research fixes or roadmap implementations were made during this setup.

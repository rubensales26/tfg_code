# Running the experiments and validation scripts

Run everything from the project root. Requires Python with `torch`, `numpy`, and `matplotlib`, and the KMNIST dataset unpacked at `datasets/KMNIST/raw/` (four `.npz` files).

## Experiments (`experiments/`)

| Script | What it does | Run |
|---|---|---|
| `exp_baseline.py` | Trains one fixed hyperparameter configuration on the full KMNIST train/test split (no search) | `python experiments/exp_baseline.py --plot` |
| `exp_random_search.py` | Random Search HPO | `python experiments/exp_random_search.py --time-budget 3600 --plot` |
| `exp_grid_search.py` | Grid Search HPO (progressive deepening) | `python experiments/exp_grid_search.py --time-budget 3600 --plot` |
| `exp_successive_halving.py` | Successive Halving HPO | `python experiments/exp_successive_halving.py --time-budget 3600 --plot` |
| `exp_genetic.py` | Genetic Algorithm HPO | `python experiments/exp_genetic.py --time-budget 3600 --plot` |

All scripts accept `--quick` (small data subset, few epochs, for a fast smoke run) and `--results-dir` (default `./results`). Results are saved as `results/<name>.json`; `--plot` also saves figures to `figures/6_chapter/`.

## Validation (`validation/`)

Each script prints PASS/FAIL lines and exits 0 (all passed) or 1 (something failed).

| Script | What it checks | Run |
|---|---|---|
| `validate_config.py` | `HPConfig` round-trips, type coercion, immutability | `python validation/validate_config.py` |
| `validate_search_space.py` | Sampling, gridding, mutation, and crossover of the search space domains | `python validation/validate_search_space.py` |
| `validate_models.py` | `KMnistMLP` architecture, losses, and optimizer | `python validation/validate_models.py` |
| `validate_resampling.py` | Nested resampling partition invariants, plus a visual check on 10 real images | `python validation/validate_resampling.py` |
| `validate_objective.py` | `Objective` matches an independent, hand-written train/eval loop bit-for-bit | `python validation/validate_objective.py` |
| `validate_pipeline.py` | Runs all four HPO scripts end-to-end with `--quick` and checks their output | `python validation/validate_pipeline.py` |
| `run_all.py` | Runs all of the above in order and prints a combined summary | `python validation/run_all.py` |

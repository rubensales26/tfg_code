"""
Grid Search (progressive deepening).

    for resolution = 1, 2, ... up to the maximum (while the budget lasts):
        build the Cartesian grid at this resolution
        for each configuration lambda not already evaluated:
            evaluate the estimated GE  c(lambda)
            keep it if it is the best so far
    return the incumbent; retrain it on outer_train and test once on outer_test

Coarse levels are evaluated first, then refined: resolution 1 is the single
central point per continuous domain, resolution 2 adds the endpoints, and so on.
Points shared between levels are evaluated only once. Every evaluation is a full
estimate of c(lambda), so the incumbent is valid even if the budget runs out in
the middle of a level. With l hyperparameters a level costs up to r^l
evaluations, so the maximum resolution should stay small.

Run (from the project root):
    python experiments/exp_grid_search.py --grid-resolution 4 --time-budget 7200 --plot
    python experiments/exp_grid_search.py --quick
"""
from __future__ import annotations

import time
from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from hpo_lib.experiment import (common_parser, finalize, load_outer_fold,
                                final_evaluation, print_best, save_results,
                                plot_trials, plot_convergence, still_running)
from hpo_lib.objective import Objective
from hpo_lib.search_space import default_search_space
from hpo_lib.trainer import pick_device

NAME = "grid"


def main():
    parser = common_parser("Grid Search HPO on KMNIST")
    parser.add_argument("--grid-resolution", type=int, default=3,
                        help="maximum grid points per continuous domain")
    args = finalize(parser)
    max_resolution = 1 if args.quick else args.grid_resolution

    device = pick_device()
    fold = load_outer_fold(args)
    objective = Objective(fold, default_epochs=args.epochs, device=device,
                          seed=args.seed)
    space = default_search_space()

    # GRID SEARCH (progressive deepening)
    evaluated = set()     # config keys already evaluated (dedup across levels)
    incumbent, incumbent_error = None, float("inf")
    trials = []
    start = time.time()
    done = False
    for resolution in range(1, max_resolution + 1):
        if done:
            break
        grid = space.grid(resolution)
        new_configs = [c for c in grid
                       if tuple(c[n] for n in space.names) not in evaluated]
        print(f"  resolution {resolution}: {len(grid)} grid points, "
              f"{len(new_configs)} new")
        for config in new_configs:
            if not still_running(start, args.time_budget, len(trials), args.trials):
                done = True
                break
            key = tuple(config[n] for n in space.names)
            evaluated.add(key)
            error = objective(config)             # estimate GE c(lambda)
            new_inc = error < incumbent_error
            if new_inc:
                incumbent, incumbent_error = config, error
            trials.append({
                "trial":         len(trials) + 1,
                "time_s":        time.time() - start,
                "config":        dict(config),
                "error":         error,
                "resolution":    resolution,
                "new_incumbent": new_inc,
            })
            print(f"  trial {len(trials):03d}  error={error:.4f}  "
                  f"incumbent={incumbent_error:.4f}  "
                  f"t={trials[-1]['time_s']:.1f}s")

    # Final model: retrain on outer_train, test once on outer_test
    reached_resolution = trials[-1]["resolution"] if trials else 0

    outer = final_evaluation(incumbent, fold, args.epochs, device, args.seed)
    print_best(NAME, incumbent, incumbent_error, outer["error_rate"])
    print(f"  grid evaluations: {len(trials)}    deepest resolution reached: "
          f"{reached_resolution} of {max_resolution}")

    save_results(NAME, {
        "algorithm":          NAME,
        "max_resolution":     max_resolution,
        "reached_resolution": reached_resolution,
        "incumbent":          incumbent,
        "inner_error":        incumbent_error,
        "outer_error":        outer["error_rate"],
        "outer_accuracy":     outer["accuracy"],
        "n_evaluations":      len(trials),
        "total_time_s":    trials[-1]["time_s"] if trials else 0.0,
        "trials":          trials,
        "history":         outer["history"],
    }, args.results_dir)

    if args.plot:
        plot_trials(NAME, trials)
        plot_convergence(NAME, outer["history"])


if __name__ == "__main__":
    main()

"""
Random Search.

    repeat until the budget is spent:
        sample a configuration lambda uniformly from the search space
        evaluate the estimated GE  c(lambda)
        keep it if it is the best so far
    return the incumbent; retrain it on outer_train and test once on outer_test

Run (from the project root):
    python experiments/exp_random_search.py --time-budget 7200 --plot
    python experiments/exp_random_search.py --quick
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
                                plot_trials, plot_convergence, still_running,
                                make_search_rng)
from hpo_lib.objective import Objective
from hpo_lib.search_space import default_search_space
from hpo_lib.trainer import pick_device

NAME = "random"


def main():
    args = finalize(common_parser("Random Search HPO on KMNIST"))
    if args.time_budget is None and args.trials is None:
        raise SystemExit("give --time-budget and/or --trials")

    device = pick_device()
    fold = load_outer_fold(args)
    objective = Objective(fold, default_epochs=args.epochs, device=device,
                          seed=args.seed)
    space = default_search_space()
    rng = make_search_rng(args.seed, NAME)

    # ---- RANDOM SEARCH ----
    incumbent, incumbent_error = None, float("inf")
    trials = []
    start = time.time()
    while still_running(start, args.time_budget, len(trials), args.trials):
        config = space.sample(rng)                # propose lambda
        error = objective(config)                 # estimate GE c(lambda)
        new_inc = error < incumbent_error
        if new_inc:
            incumbent, incumbent_error = config, error
        trials.append({
            "trial":         len(trials) + 1,
            "time_s":        time.time() - start,
            "config":        dict(config),
            "error":         error,
            "new_incumbent": new_inc,
        })
        print(f"  trial {len(trials):03d}  error={error:.4f}  "
              f"incumbent={incumbent_error:.4f}  t={trials[-1]['time_s']:.1f}s")

    # ---- final model: retrain on outer_train, test once on outer_test ----
    outer = final_evaluation(incumbent, fold, args.epochs, device, args.seed)
    print_best(NAME, incumbent, incumbent_error, outer["error_rate"])
    print(f"  random evaluations: {len(trials)}")

    save_results(NAME, {
        "algorithm":       NAME,
        "incumbent":       incumbent,
        "inner_error":     incumbent_error,
        "outer_error":     outer["error_rate"],
        "outer_accuracy":  outer["accuracy"],
        "n_evaluations":   len(trials),
        "total_time_s":    trials[-1]["time_s"] if trials else 0.0,
        "trials":          trials,
        "history":         outer["history"],
    }, args.results_dir)

    if args.plot:
        plot_trials(NAME, trials)
        plot_convergence(NAME, outer["history"])


if __name__ == "__main__":
    main()

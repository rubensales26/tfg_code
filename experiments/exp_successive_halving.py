"""
Successive Halving.

    rungs = [r_min, r_min*eta, ..., r_max]      (the last rung clamped to r_max)
    repeat (one bracket if no time budget, else until the budget is spent):
        sample n_init = eta^K configurations
        for each rung at budget r:
            evaluate every survivor at budget r epochs
            keep the best 1/eta fraction, promote them to the next rung
        only full-budget (r_max) evaluations may become the incumbent
    return the incumbent; retrain it on outer_train and test once on outer_test

The budget r is the number of training epochs, so a low rung is a cheap,
low-fidelity proxy of the objective. The bracket always runs to completion;
the time budget is checked only between brackets, so the final bracket may
overshoot by at most one bracket's worth of evaluations.

Run (from the project root):
    python experiments/exp_successive_halving.py --epochs 10 --eta 2 --time-budget 7200 --plot
    python experiments/exp_successive_halving.py --quick
"""
from __future__ import annotations

import math
import time
from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from hpo_lib.experiment import (common_parser, finalize, load_outer_fold,
                                final_evaluation, print_best, save_results,
                                plot_trials, plot_convergence, make_search_rng)
from hpo_lib.objective import Objective
from hpo_lib.search_space import default_search_space
from hpo_lib.trainer import pick_device

NAME = "sh"


def main():
    parser = common_parser("Successive Halving HPO on KMNIST")
    parser.add_argument("--eta", type=int, default=2, help="halving factor")
    args = finalize(parser)

    device = pick_device()
    fold = load_outer_fold(args)
    objective = Objective(fold, default_epochs=args.epochs, device=device,
                          seed=args.seed)
    space = default_search_space()
    rng = make_search_rng(args.seed, NAME)

    # rung ladder: geometric, with the final rung clamped to the full budget
    eta, r_min, r_max = args.eta, 1, args.epochs
    K = int(math.floor(math.log(r_max / r_min) / math.log(eta)))
    rungs = [r_min * eta ** k for k in range(K)] + [r_max]
    n_init = eta ** K
    print(f"  eta={eta}  rungs={rungs}  n_init={n_init}")

    # ---- SUCCESSIVE HALVING ----
    incumbent, incumbent_error = None, float("inf")
    trials = []
    start = time.time()
    bracket = 0
    while True:
        survivors = [space.sample(rng) for _ in range(n_init)]
        for stage, budget in enumerate(rungs):
            scored = []
            for config in survivors:
                error = objective(config, budget=budget)
                # only full-fidelity evaluations may set the incumbent
                new_inc = budget == r_max and error < incumbent_error
                if new_inc:
                    incumbent, incumbent_error = config, error
                trials.append({
                    "trial":         len(trials) + 1,
                    "time_s":        time.time() - start,
                    "config":        dict(config),
                    "error":         error,
                    "fidelity":      budget,
                    "new_incumbent": new_inc,
                })
                scored.append((error, config))
            scored.sort(key=lambda ec: ec[0])
            keep = max(1, len(survivors) // eta)
            survivors = [c for _, c in scored[:keep]]
            print(f"  bracket {bracket} rung {stage}: {len(scored)} configs "
                  f"@ {budget} epoch(s) -> keep {len(survivors)} "
                  f"(best={scored[0][0]:.4f})")
        bracket += 1
        if args.time_budget is None or time.time() - start >= args.time_budget:
            break

    # ---- final model: retrain on outer_train, test once on outer_test ----
    n_high_fidelity = sum(1 for t in trials if t["fidelity"] == r_max)

    outer = final_evaluation(incumbent, fold, args.epochs, device, args.seed)
    print_best(NAME, incumbent, incumbent_error, outer["error_rate"])
    print(f"  brackets: {bracket}    total evaluations: {len(trials)}    "
          f"full-fidelity (r_max={r_max}) evaluations: {n_high_fidelity}")

    save_results(NAME, {
        "algorithm":       NAME,
        "eta":             eta,
        "rungs":           rungs,
        "brackets":        bracket,
        "incumbent":       incumbent,
        "inner_error":     incumbent_error,
        "outer_error":     outer["error_rate"],
        "outer_accuracy":  outer["accuracy"],
        "n_evaluations":   len(trials),
        "n_high_fidelity": n_high_fidelity,
        "total_time_s":    trials[-1]["time_s"] if trials else 0.0,
        "trials":          trials,
        "history":         outer["history"],
    }, args.results_dir)

    if args.plot:
        plot_trials(NAME, trials)
        plot_convergence(NAME, outer["history"])


if __name__ == "__main__":
    main()

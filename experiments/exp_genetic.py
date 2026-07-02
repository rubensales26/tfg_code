"""
Genetic Algorithm.

    start from a random population
    repeat until the budget is spent:
        evaluate the estimated GE c(lambda) of every individual
        breed the next generation:
            elitism      -- carry the best individual over unchanged
            selection    -- pick parents by tournament
            crossover    -- mix two parents gene by gene
            mutation     -- perturb each gene with probability mutation_rate
        stagnation check -- if the incumbent has not improved for
                            stagnation_patience consecutive generations,
                            replace half the non-elite population with fresh
                            random configurations (immigrant injection)
    return the incumbent; retrain it on outer_train and test once on outer_test

Run (from the project root):
    python experiments/exp_genetic.py --time-budget 7200 --population 8 --plot
    python experiments/exp_genetic.py --quick
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

NAME = "genetic"


def tournament(scored, rng, k):
    """Pick the best of k random (error, config) contestants."""
    k = min(k, len(scored))
    contestants = rng.sample(scored, k)
    return min(contestants, key=lambda ec: ec[0])[1]


def main():
    parser = common_parser("Genetic Algorithm HPO on KMNIST")
    parser.add_argument("--population", type=int, default=8)
    parser.add_argument("--mutation-rate", type=float, default=0.2)
    parser.add_argument("--tournament-size", type=int, default=2)
    parser.add_argument("--scale", type=float, default=0.1)
    parser.add_argument("--stagnation-patience", type=int, default=3,
                        help="generations without improvement before immigrant injection")
    args = finalize(parser)
    if args.time_budget is None and args.trials is None:
        raise SystemExit("give --time-budget and/or --trials")
    pop_size = 4 if args.quick else args.population

    device = pick_device()
    fold = load_outer_fold(args)
    objective = Objective(fold, default_epochs=args.epochs, device=device,
                          seed=args.seed)
    space = default_search_space()
    rng = make_search_rng(args.seed, NAME)

    # GENETIC ALGORITHM
    population = [space.sample(rng) for _ in range(pop_size)]
    incumbent, incumbent_error = None, float("inf")
    trials = []
    generation = 0
    no_improve = 0        # consecutive generations without improvement
    start = time.time()
    while still_running(start, args.time_budget, len(trials), args.trials):
        # evaluate the current population
        scored = []
        prev_incumbent_error = incumbent_error
        for config in population:
            if not still_running(start, args.time_budget, len(trials), args.trials):
                break
            error = objective(config)
            new_inc = error < incumbent_error
            if new_inc:
                incumbent, incumbent_error = config, error
            trials.append({
                "trial":         len(trials) + 1,
                "time_s":        time.time() - start,
                "config":        dict(config),
                "error":         error,
                "generation":    generation,
                "new_incumbent": new_inc,
            })
            scored.append((error, config))
        if len(scored) < len(population):         # budget hit mid-generation
            break

        # stagnation tracking
        if incumbent_error < prev_incumbent_error:
            no_improve = 0
        else:
            no_improve += 1

        print(f"  generation {generation:02d}: "
              f"best={min(e for e, _ in scored):.4f}  "
              f"incumbent={incumbent_error:.4f}  evals={len(trials)}  "
              f"no_improve={no_improve}")

        # breed the next generation
        scored.sort(key=lambda ec: ec[0])
        elite = scored[0][1]
        next_population = [dict(elite)]           # elitism: carry the best over

        # stagnation response: inject random immigrants to restore diversity
        if no_improve >= args.stagnation_patience:
            n_immigrants = max(1, (pop_size - 1) // 2)
            next_population.extend([space.sample(rng) for _ in range(n_immigrants)])
            no_improve = 0
            print(f"  [stagnation] injecting {n_immigrants} random immigrants")

        while len(next_population) < pop_size:
            parent_a = tournament(scored, rng, args.tournament_size)
            parent_b = tournament(scored, rng, args.tournament_size)
            child = space.crossover(parent_a, parent_b, rng)
            child = space.mutate(child, rng, args.mutation_rate, args.scale)
            next_population.append(child)
        population = next_population
        generation += 1

    # Final model: retrain on outer_train, test once on outer_test
    outer = final_evaluation(incumbent, fold, args.epochs, device, args.seed)
    print_best(NAME, incumbent, incumbent_error, outer["error_rate"])
    print(f"  generations completed: {generation}    total evaluations: {len(trials)}")

    save_results(NAME, {
        "algorithm":       NAME,
        "population_size": pop_size,
        "generations":     generation,
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

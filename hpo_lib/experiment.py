"""
Shared helpers for the per-algorithm experiment scripts.

These are NOT algorithm logic -- just the common plumbing: data setup, the final
retrain-and-test of the chosen configuration, plotting, saving, and CLI parsing.
Each experiment script (experiments/exp_*.py) imports these and supplies its own
explicit search loop, so the script reads like the algorithm's pseudocode while
the verified primitives (data, objective, model, trainer) stay shared.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
import zlib
from pathlib import Path

import torch
from torch.utils.data import Subset

from .data import KMnistDataset, NestedResampling
from .hp_config import HPConfig
from .models import KMnistMLP
from .objective import Objective
from .trainer import Trainer, pick_device

_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_DATA = str(_ROOT / "datasets" / "KMNIST" / "raw")

# Fixed experimental design: 10 000 outer test images, 2 inner folds (four-set scheme)
N_OUTER_TEST = 10_000
N_INNER_FOLDS = 2


# ---------------------------------------------------------------------------
# setup
# ---------------------------------------------------------------------------

def load_outer_fold(args):
    """Build the single outer fold (disjoint hold-out + disjoint inner sets)."""
    dataset = KMnistDataset(args.data_dir).load_train()
    if args.subset is not None:
        g = torch.Generator().manual_seed(args.seed)
        idx = torch.randperm(len(dataset), generator=g)[:args.subset].tolist()
        dataset = Subset(dataset, idx)
    n = len(dataset)
    blk = 2 * N_INNER_FOLDS  # inner divisibility requirement (= 4)
    if n <= N_OUTER_TEST + blk:
        # dataset is smaller than the standard outer test (quick / subset mode)
        test_size = max(blk, (n // 5) // blk * blk)
    else:
        test_size = N_OUTER_TEST
    resampling = NestedResampling(dataset, test_size, N_INNER_FOLDS,
                                  batch_size=args.batch_size, seed=args.seed)
    print(resampling)
    return resampling.outer_fold(0)


def make_search_rng(seed, name):
    """Independent, reproducible search RNG for one algorithm.

    The data/partition seed is shared across algorithms so the objective
    c(lambda) is identical for all of them (a fair comparison). The *search*
    stream, however, is offset per algorithm so that no two algorithms are
    forced to examine the same sequence of candidate configurations. The offset
    is a stable hash of the algorithm name (zlib.crc32, not the process-salted
    built-in hash()), so a single master seed still reproduces every run.
    """
    return random.Random(seed + zlib.crc32(name.encode()))


def final_evaluation(incumbent, fold, epochs, device, seed):
    """Retrain the incumbent on all of outer_train, test once on outer_test.

    Returns the test result dict, including the per-epoch convergence under
    "history".
    """
    torch.manual_seed(seed)
    trainer = Trainer(max_epochs=epochs, device=device)
    test_loader = fold.get_outer_test()
    model = trainer.fit(KMnistMLP(HPConfig.from_dict(incumbent)),
                        fold.get_outer_train(), val_loader=test_loader)
    result = trainer.evaluate(model, test_loader)
    result["history"] = trainer.history
    return result


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------

def print_best(name, incumbent, inner_error, outer_error):
    print(f"\nBest configuration found by {name}:")
    print(f"  inner c(lambda) = {inner_error:.4f}    OUTER GE = {outer_error:.4f}")
    for k, v in incumbent.items():
        print(f"    {k:<14} = {v}")


def save_results(name, payload, results_dir="./results"):
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, f"{name}.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  saved results to {path}")


# ---------------------------------------------------------------------------
# plotting
# ---------------------------------------------------------------------------

def plot_trials(name, trials, out_dir="figures/6_chapter"):
    """Estimated GE c(lambda) of every tried config vs wall-clock, with the
    incumbent (running best) as a step line.

    ``trials`` is a list of dicts with keys ``time_s`` and ``error``.
    """
    import matplotlib.pyplot as plt

    if not trials:
        return
    os.makedirs(out_dir, exist_ok=True)
    times = [t["time_s"] for t in trials]
    errs  = [t["error"]  for t in trials]

    best, bx, by = float("inf"), [], []
    for t, e in zip(times, errs):
        best = min(best, e)
        bx.append(t)
        by.append(best)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(times, errs, s=18, alpha=0.5, color="#7570b3", label="tried config")
    ax.step(bx, by, where="post", color="#d95f02", linewidth=2, label="incumbent")
    ax.set_xlabel("cumulative wall-clock time (s)")
    ax.set_ylabel("estimated GE  c(lambda) = 1 - accuracy")
    ax.set_title(f"Estimated GE of each tried config: {name}")
    ax.legend(framealpha=0.9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"trials_{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print(f"  saved per-trial plot to {out_dir}/trials_{name}.*")


def plot_convergence(name, history, out_dir="figures/6_chapter"):
    """Final model's per-epoch train loss + outer-test error."""
    import matplotlib.pyplot as plt

    if not history or not history["train_loss"]:
        return
    os.makedirs(out_dir, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    l1, = ax1.plot(epochs, history["train_loss"], "-o", color="#2c7fb8",
                   label="train loss (cross-entropy)")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("train loss")
    lines = [l1]
    if history["val_error"]:
        ax2 = ax1.twinx()
        l2, = ax2.plot(epochs, history["val_error"], "-s", color="#d95f02",
                       label="outer-test error (1 - accuracy)")
        ax2.set_ylabel("outer-test error")
        lines.append(l2)
    ax1.set_title(f"Final model convergence: {name}")
    ax1.legend(lines, [l.get_label() for l in lines], loc="best", framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"convergence_{name}.{ext}"),
                    bbox_inches="tight")
    plt.close(fig)
    print(f"  saved convergence plot to {out_dir}/convergence_{name}.*")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def common_parser(description):
    """Argument parser shared by every experiment script."""
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--data-dir", default=_DEFAULT_DATA)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=10, help="full-fidelity budget")
    p.add_argument("--time-budget", type=float, default=None,
                   help="wall-clock seconds (any-time stopping)")
    p.add_argument("--trials", type=int, default=None,
                   help="max objective evaluations (count budget)")
    p.add_argument("--subset", type=int, default=None,
                   help="use only this many training images (for quick runs)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--results-dir", default="./results")
    p.add_argument("--plot", action="store_true")
    p.add_argument("--quick", action="store_true",
                   help="tiny subset / few epochs for a fast smoke run")
    return p


def finalize(parser):
    """Parse args and apply the --quick presets (kept divisibility-safe)."""
    args = parser.parse_args()
    if args.quick:
        args.subset = args.subset or 2_000
        args.subset -= args.subset % (2 * N_INNER_FOLDS)  # must be divisible by 4
        args.epochs = min(args.epochs, 2)
        if args.time_budget is None and args.trials is None:
            args.trials = 6
    return args


def still_running(start_time, time_budget, n_done, max_trials):
    """True while neither the wall-clock nor the evaluation budget is spent."""
    if time_budget is not None and (time.time() - start_time) >= time_budget:
        return False
    if max_trials is not None and n_done >= max_trials:
        return False
    return True

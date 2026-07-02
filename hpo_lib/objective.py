"""
The objective function c(lambda).

This is the black box the search strategies minimize. One call trains and scores
a config on every inner fold of the bound outer fold and returns the mean:

    for each inner fold j:
        train model on inner_train_j      (cross-entropy)
        e_j = error on inner_test_j       (1 - accuracy)
    c(lambda) = mean(e_j)

The number of training epochs is the fidelity (budget): full budget gives the
true objective, a reduced budget gives the cheap proxy used by Successive
Halving. Every call is appended to self.archive (config, budget, per-fold
errors, and their mean).
"""

from __future__ import annotations

import torch

from .data import OuterFold
from .hp_config import HPConfig
from .models import KMnistMLP
from .trainer import Trainer


class Objective:
    """Callable estimator of the generalization error of a configuration."""

    def __init__(
        self,
        fold: OuterFold,
        default_epochs: int = 10,
        device: torch.device | None = None,
        seed: int = 0,
    ):
        """
        Args:
            fold           : the OuterFold this objective is bound to; c(lambda)
                             averages the error over its inner folds.
            default_epochs : budget used when a call supplies none (full fidelity).
            device         : torch device (auto-detected by the Trainer if None).
            seed           : base seed; reseeded per fold so re-evaluating the same
                             config at the same budget is reproducible.
        """
        self.fold = fold
        self.default_epochs = default_epochs
        self.device = device
        self.seed = seed
        self.archive: list[dict] = []

    def __call__(self, config, budget: int | None = None) -> float:
        """
        Evaluate c(config) at the given epoch budget.

        ``config`` may be a ``HPConfig`` or a plain dict of hyperparameters.
        """
        cfg = config if isinstance(config, HPConfig) \
            else HPConfig.from_dict(config)
        epochs = self.default_epochs if budget is None else int(budget)

        fold_errors: list[float] = []
        fold_accs: list[float] = []
        for j in range(self.fold.num_inner_folds):
            # deterministic per (seed, fold): reproducible "pull" of this arm
            torch.manual_seed(self.seed + j)
            train_loader, test_loader = self.fold.get_inner_fold(j)

            trainer = Trainer(max_epochs=epochs, device=self.device)
            model = trainer.fit(KMnistMLP(cfg), train_loader)
            res = trainer.evaluate(model, test_loader)
            fold_errors.append(res["error_rate"])
            fold_accs.append(res["accuracy"])

        mean_error = sum(fold_errors) / len(fold_errors)

        self.archive.append({
            **cfg.to_dict(),
            "budget": epochs,
            "fold_errors": fold_errors,
            "fold_accuracies": fold_accs,
            "mean_error": mean_error,
        })
        return mean_error

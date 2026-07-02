"""
hpo_lib -- a small, serial hyperparameter-optimization library.

A clean re-implementation of the HPO setup in the thesis, structurally based on
the HPO API of Dive into Deep Learning but with no parallelism.

Pipeline (nested resampling):

    KMNIST
      outer split -> outer_train + outer_test          (disjoint)
        inner split -> four disjoint sets:
                         inner_train_1, inner_test_1,
                         inner_train_2, inner_test_2

    inner loss = cross-entropy   (fits models)
    outer loss = 1 - accuracy    (scores models)

    objective c(lambda): fit on each inner_train_j, score on inner_test_j, and
    return the mean error.

This package holds the verified primitives (data, search space, objective, model,
trainer). Each search strategy lives in its own readable script under
experiments/ (exp_random_search, exp_grid_search, exp_genetic,
exp_successive_halving): the script proposes configs, evaluates the objective,
tracks the incumbent, then retrains it on all of outer_train and scores it once
on outer_test.
"""

from .hp_config import HPConfig
from .search_space import SearchSpace, Continuous, Discrete, default_search_space
from .data import KMnistDataset, NestedResampling, OuterFold
from .models import BaseModel, KMnistMLP
from .trainer import Trainer, pick_device
from .objective import Objective

__all__ = [
    "HPConfig",
    "SearchSpace", "Continuous", "Discrete", "default_search_space",
    "KMnistDataset", "NestedResampling", "OuterFold",
    "BaseModel", "KMnistMLP",
    "Trainer", "pick_device",
    "Objective",
]

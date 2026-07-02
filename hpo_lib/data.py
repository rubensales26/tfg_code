"""
Data loading and nested resampling.

The partition used throughout the thesis:

    OUTER   DisjointHoldoutOuter -- single hold-out of 10 000 images
    INNER   DisjointHoldoutInner -- the four-set scheme (n_folds = 2):
                inner_train_1, inner_test_1,
                inner_train_2, inner_test_2

The objective averages the error over the inner folds of the outer fold.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset


# ==== DATASET LOADING ====

class KMnistDataset:
    """Loads the Kuzushiji-MNIST dataset from the downloaded ``.npz`` files."""

    def __init__(self, data_dir: str = "./datasets/KMNIST/raw"):
        self.data_dir = data_dir

    def load_train(self) -> TensorDataset:
        return self._load(
            f"{self.data_dir}/kmnist-train-imgs.npz",
            f"{self.data_dir}/kmnist-train-labels.npz",
        )

    def load_test(self) -> TensorDataset:
        return self._load(
            f"{self.data_dir}/kmnist-test-imgs.npz",
            f"{self.data_dir}/kmnist-test-labels.npz",
        )

    @staticmethod
    def _load(imgs_path: str, labels_path: str) -> TensorDataset:
        imgs = torch.tensor(np.load(imgs_path)["arr_0"], dtype=torch.float32) / 255.0
        labels = torch.tensor(np.load(labels_path)["arr_0"], dtype=torch.long)
        return TensorDataset(imgs.unsqueeze(1), labels)


# ==== RESAMPLING ====

Fold = tuple[np.ndarray, np.ndarray]   # (train_idx, test_idx)


class DisjointHoldoutOuter:
    """Single outer hold-out: randomly splits n samples into (train, test)."""

    def __init__(self, test_size: int, seed: int = 0):
        self.test_size = test_size
        self.seed = seed

    def splits(self, n_samples: int) -> list[Fold]:
        if not 0 < self.test_size < n_samples:
            raise ValueError(f"test_size={self.test_size} invalid for n={n_samples}")
        perm = np.random.default_rng(self.seed).permutation(n_samples)
        test_idx = perm[: self.test_size]
        train_idx = perm[self.test_size :]
        return [(train_idx, test_idx)]


class DisjointHoldoutInner:
    """
    The four-set scheme generalized to n_folds inner folds.

    The outer-train indices are shuffled and cut into 2 * n_folds pairwise-
    disjoint, equal-sized blocks. Fold j uses block 2j for training and block
    2j+1 for testing. At the default n_folds = 2 this is exactly
    (inner_train_1, inner_test_1, inner_train_2, inner_test_2).
    """

    def __init__(self, n_folds: int = 2, seed: int = 0):
        if n_folds < 1:
            raise ValueError("n_folds must be >= 1")
        self.n_folds = n_folds
        self.seed = seed

    def folds(self, indices: np.ndarray) -> list[Fold]:
        n_blocks = 2 * self.n_folds
        if len(indices) % n_blocks != 0:
            raise ValueError(
                f"{len(indices)} inner samples not divisible by {n_blocks} "
                f"(2 x n_folds); the disjoint blocks would be unequal."
            )
        idx = indices.copy()
        np.random.default_rng(self.seed).shuffle(idx)
        blocks = np.split(idx, n_blocks)
        return [(blocks[2 * j], blocks[2 * j + 1]) for j in range(self.n_folds)]


# ==== NESTED RESAMPLING ====

class OuterFold:
    """
    One outer fold: its held-out test set, its training set, and the inner folds
    carved from that training set. This is what an Objective is bound to.
    """

    def __init__(
        self,
        dataset: Dataset,
        train_idx: np.ndarray,
        test_idx: np.ndarray,
        inner_folds: list[Fold],
        batch_size: int,
        seed: int,
        index: int,
    ):
        self.dataset = dataset
        self.train_idx = train_idx
        self.test_idx = test_idx
        self._inner_folds = inner_folds
        self.batch_size = batch_size
        self.seed = seed
        self.index = index

    @property
    def num_inner_folds(self) -> int:
        return len(self._inner_folds)

    def _loader(self, idx: np.ndarray, shuffle: bool) -> DataLoader:
        gen = torch.Generator().manual_seed(self.seed) if shuffle else None
        return DataLoader(
            Subset(self.dataset, [int(i) for i in idx]),
            batch_size=self.batch_size,
            shuffle=shuffle,
            generator=gen,
        )

    def get_inner_fold(self, j: int) -> tuple[DataLoader, DataLoader]:
        """(train_loader, test_loader) for inner fold j."""
        if not 0 <= j < self.num_inner_folds:
            raise IndexError(f"inner fold {j} out of range [0,{self.num_inner_folds})")
        tr, te = self._inner_folds[j]
        return self._loader(tr, shuffle=True), self._loader(te, shuffle=False)

    def get_outer_train(self) -> DataLoader:
        """Full outer-train loader, used to retrain the incumbent after HPO."""
        return self._loader(self.train_idx, shuffle=True)

    def get_outer_test(self) -> DataLoader:
        """Held-out outer-test loader, touched only for final evaluation."""
        return self._loader(self.test_idx, shuffle=False)

    def inner_index_sets(self) -> list[tuple[set[int], set[int]]]:
        """The inner folds as (train_set, test_set) index sets (for tests)."""
        return [(set(tr.tolist()), set(te.tolist())) for tr, te in self._inner_folds]


class NestedResampling:
    """
    Applies the thesis partition to a dataset.

    Args:
        dataset      : Full dataset to partition.
        test_size    : Number of samples held out as the outer test set.
        n_inner_folds: Number of inner folds (2 gives the four-set scheme).
        batch_size   : DataLoader batch size.
        seed         : Master seed for both splits.
    """

    def __init__(
        self,
        dataset: Dataset,
        test_size: int,
        n_inner_folds: int = 2,
        batch_size: int = 64,
        seed: int = 42,
    ):
        self.test_size = test_size
        self.n_inner_folds = n_inner_folds

        outer = DisjointHoldoutOuter(test_size, seed)
        inner = DisjointHoldoutInner(n_inner_folds, seed)

        self._folds: list[OuterFold] = []
        for b, (train_idx, test_idx) in enumerate(outer.splits(len(dataset))):
            self._folds.append(OuterFold(
                dataset, train_idx, test_idx, inner.folds(train_idx),
                batch_size, seed, index=b,
            ))

    @property
    def num_outer_folds(self) -> int:
        return len(self._folds)

    def outer_fold(self, b: int) -> OuterFold:
        return self._folds[b]

    def __iter__(self):
        return iter(self._folds)

    def __repr__(self) -> str:
        f0 = self._folds[0]
        return (
            f"NestedResampling(test_size={self.test_size}, "
            f"n_inner_folds={self.n_inner_folds}, "
            f"train={len(f0.train_idx)}, test={len(f0.test_idx)})"
        )
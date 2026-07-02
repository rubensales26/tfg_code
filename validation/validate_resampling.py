"""Validation: the nested-resampling scheme in hpo_lib/data.py implements the
thesis partition correctly -- disjoint outer hold-out + four pairwise-disjoint
inner sets -- and the splits flow through to the actual DataLoaders.

Two parts:

    PART 1  (rigorous)  programmatic invariant checks on a 1000-sample dataset:
                        outer train/test disjoint & covering, the four inner
                        blocks subset-of-outer-train / equal-sized / pairwise
                        disjoint, loader counts, and seed reproducibility.

    PART 2  (visual)    the same splits on 10 real KMNIST images, rendered to
                        validation/resampling_split.png so the disjointness can
                        be checked by eye (the held-out outer-test images appear
                        in NO inner row).

Run:  python validation/validate_resampling.py
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from hpo_lib.data import NestedResampling

DATA = _ROOT / "datasets" / "KMNIST" / "raw"
OUT = Path(__file__).resolve().parent / "resampling_split.png"

_ok = True


def check(cond, msg):
    global _ok
    print(("PASS" if cond else "FAIL"), "-", msg)
    _ok = _ok and bool(cond)


# ---------------------------------------------------------------------------
# PART 1: rigorous invariant checks
# ---------------------------------------------------------------------------

def part1_invariants():
    print("=== PART 1: partition invariants (N=1000) ===")
    N, TEST_SIZE, INNER_FOLDS, SEED = 1000, 200, 2, 42
    ds = TensorDataset(torch.arange(N).float().unsqueeze(1),
                       torch.zeros(N).long())
    nr = NestedResampling(ds, TEST_SIZE, INNER_FOLDS, batch_size=64, seed=SEED)

    check(nr.num_outer_folds == 1, f"num_outer_folds == 1 (got {nr.num_outer_folds})")
    fold = nr.outer_fold(0)
    tr, te = set(fold.train_idx.tolist()), set(fold.test_idx.tolist())

    check(len(te) == TEST_SIZE, f"outer test size == {TEST_SIZE} (got {len(te)})")
    check(len(tr) == N - TEST_SIZE, f"outer train size == {N - TEST_SIZE} (got {len(tr)})")
    check(tr.isdisjoint(te), "outer train and outer test are DISJOINT")
    check(tr | te == set(range(N)), "outer train + test cover the dataset")

    check(fold.num_inner_folds == INNER_FOLDS,
          f"num_inner_folds == {INNER_FOLDS} (got {fold.num_inner_folds})")
    blocks = [b for pair in fold.inner_index_sets() for b in pair]   # tr1,te1,tr2,te2
    expected = (N - TEST_SIZE) // (2 * INNER_FOLDS)
    for k, b in enumerate(blocks):
        check(b <= tr, f"inner block {k} subset of outer-train")
    check(all(len(b) == expected for b in blocks),
          f"all 4 inner blocks have equal size {expected}")
    pairwise = all(blocks[i].isdisjoint(blocks[j])
                   for i in range(4) for j in range(i + 1, 4))
    check(pairwise, "all four inner sets are PAIRWISE DISJOINT")

    tl, vl = fold.get_inner_fold(0)
    n_tl = sum(len(x) for x, _ in tl)
    n_vl = sum(len(x) for x, _ in vl)
    check(n_tl == expected and n_vl == expected,
          f"inner-fold-0 loaders yield {expected}/{expected} (got {n_tl}/{n_vl})")
    n_otr = sum(len(x) for x, _ in fold.get_outer_train())
    n_ote = sum(len(x) for x, _ in fold.get_outer_test())
    check(n_otr == N - TEST_SIZE and n_ote == TEST_SIZE,
          f"outer loaders yield {N - TEST_SIZE}/{TEST_SIZE} (got {n_otr}/{n_ote})")

    nr2 = NestedResampling(ds, TEST_SIZE, INNER_FOLDS, batch_size=64, seed=SEED)
    check(set(nr2.outer_fold(0).test_idx.tolist()) == te,
          "same seed reproduces the identical outer split")


# ---------------------------------------------------------------------------
# PART 2: visual validation on 10 KMNIST images
# ---------------------------------------------------------------------------

def part2_visual():
    print("\n=== PART 2: visual split on 10 KMNIST images ===")
    N, TEST_SIZE, INNER_FOLDS, SEED = 10, 2, 2, 7
    imgs = np.load(DATA / "kmnist-train-imgs.npz")["arr_0"][:N]
    labels = np.load(DATA / "kmnist-train-labels.npz")["arr_0"][:N]
    X = torch.tensor(imgs, dtype=torch.float32).unsqueeze(1) / 255.0
    y = torch.tensor(labels, dtype=torch.long)
    ds = TensorDataset(X, y)

    nr = NestedResampling(ds, TEST_SIZE, INNER_FOLDS, batch_size=4, seed=SEED)
    fold = nr.outer_fold(0)
    outer_train = fold.train_idx.tolist()
    outer_test = fold.test_idx.tolist()
    (itr1, ite1), (itr2, ite2) = fold.inner_index_sets()
    itr1, ite1, itr2, ite2 = (sorted(s) for s in (itr1, ite1, itr2, ite2))

    print("OUTER train (8):", outer_train)
    print("OUTER test  (2):", outer_test)
    print("  inner_train_1:", itr1, " inner_test_1:", ite1)
    print("  inner_train_2:", itr2, " inner_test_2:", ite2)

    # the held-out outer-test images must appear in NO inner block
    inner_all = set(itr1) | set(ite1) | set(itr2) | set(ite2)
    check(inner_all.isdisjoint(set(outer_test)),
          "outer-test images appear in NO inner block (test isolation)")
    check(inner_all == set(outer_train),
          "the four inner blocks exactly partition the outer-train images")

    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm"})

    def img(i):
        return ds[i][0].squeeze().numpy()

    groups = [
        ("ALL 10 (unsplit)", list(range(N)), "0.3"),
        ("OUTER train (8)", outer_train, "#2c7fb8"),
        ("OUTER test (2)",  outer_test,  "#d95f02"),
        ("inner_train_1",   itr1,        "#1b9e77"),
        ("inner_test_1",    ite1,        "#7570b3"),
        ("inner_train_2",   itr2,        "#66a61e"),
        ("inner_test_2",    ite2,        "#e7298a"),
    ]
    ncols = N
    fig, axes = plt.subplots(len(groups), ncols,
                             figsize=(ncols * 1.1, len(groups) * 1.25))
    fig.suptitle("data.py nested-resampling split on 10 KMNIST images "
                 "(L = label, # = original index)", fontsize=11)
    for r, (title, idxs, color) in enumerate(groups):
        for c in range(ncols):
            ax = axes[r, c]
            ax.axis("off")
            if c < len(idxs):
                i = idxs[c]
                ax.imshow(img(i), cmap="gray_r")
                ax.set_title(f"#{i}  L={int(y[i])}", fontsize=7, color=color)
                ax.axis("on")
                ax.set_xticks([]); ax.set_yticks([])
                for s in ax.spines.values():
                    s.set_visible(True); s.set_color(color); s.set_linewidth(2)
        axes[r, 0].set_ylabel(title, fontsize=8, color=color, rotation=0,
                              ha="right", va="center", labelpad=42)
    fig.tight_layout(rect=[0.06, 0, 1, 0.96])
    fig.savefig(OUT, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("saved visual ->", OUT)


def main():
    part1_invariants()
    part2_visual()
    print("\nRESULT:", "VALIDATION PASSED" if _ok else "VALIDATION FAILED")
    sys.exit(0 if _ok else 1)


if __name__ == "__main__":
    main()
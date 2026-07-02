"""Chapter 5 nested-resampling illustration.

Visualises the thesis resampling scheme on a handful of real KMNIST images,
driven ENTIRELY by the production splits in ``hpo_lib/data.py`` (so the figure
cannot drift from the code it documents):

        10 images
          outer:  8 train  +  2 test                     (DisjointHoldoutOuter)
          inner (from the 8 outer-train images,          (DisjointHoldoutInner,
                 n_folds = 2):                             n_folds = 2)
                  inner_train_1 (2)   inner_test_1 (2)
                  inner_train_2 (2)   inner_test_2 (2)

The script also asserts every partition invariant (outer train/test disjoint and
covering; the four inner blocks subset of outer-train, equal-sized, and pairwise
disjoint) before drawing, so running it doubles as a correctness check.

The specific indices depend on ``SEED``; the structural guarantees hold for any
seed. Output (PDF + PNG) is written next to this file.
"""
from pathlib import Path
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset

# --- make ``hpo_lib`` importable regardless of the current directory -----
_ROOT = Path(__file__).resolve().parents[2]      # figures/B_chapter -> root
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from hpo_lib.data import NestedResampling

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "axes.labelsize": 11,
    "font.size": 10,
})

OUT = Path(__file__).resolve().parent
DATA = _ROOT / "datasets" / "KMNIST" / "raw"
N, TEST_SIZE, INNER_FOLDS, SEED = 10, 2, 2, 7


def load_dummy_dataset() -> TensorDataset:
    """First ``N`` KMNIST training images as a tiny in-memory dataset."""
    imgs = np.load(DATA / "kmnist-train-imgs.npz")["arr_0"][:N]
    labels = np.load(DATA / "kmnist-train-labels.npz")["arr_0"][:N]
    X = torch.tensor(imgs, dtype=torch.float32).unsqueeze(1) / 255.0
    y = torch.tensor(labels, dtype=torch.long)
    return TensorDataset(X, y)


def build_split(ds: TensorDataset):
    """Run the production splits and return the index groups to plot."""
    nr = NestedResampling(ds, TEST_SIZE, INNER_FOLDS, batch_size=4, seed=SEED)
    fold = nr.outer_fold(0)
    outer_train = fold.train_idx.tolist()
    outer_test = fold.test_idx.tolist()
    (itr1, ite1), (itr2, ite2) = fold.inner_index_sets()
    itr1, ite1, itr2, ite2 = (sorted(s) for s in (itr1, ite1, itr2, ite2))
    return outer_train, outer_test, itr1, ite1, itr2, ite2


def check_invariants(outer_train, outer_test, itr1, ite1, itr2, ite2):
    """Assert the thesis partition properties (correctness self-test)."""
    tr, te = set(outer_train), set(outer_test)
    assert tr.isdisjoint(te), "outer train/test overlap"
    assert tr | te == set(range(N)), "outer split does not cover the data"
    blocks = [set(itr1), set(ite1), set(itr2), set(ite2)]
    for b in blocks:
        assert b <= tr, "inner block not subset of outer-train"
        assert len(b) == (N - TEST_SIZE) // (2 * INNER_FOLDS), "inner block size"
    for i in range(4):
        for j in range(i + 1, 4):
            assert blocks[i].isdisjoint(blocks[j]), "inner blocks not disjoint"


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", dpi=200, bbox_inches="tight")
    print("saved", name)


def main():
    ds = load_dummy_dataset()
    outer_train, outer_test, itr1, ite1, itr2, ite2 = build_split(ds)

    print("original image indices : 0..%d" % (N - 1))
    print("OUTER train (%d):" % len(outer_train), outer_train)
    print("OUTER test  (%d):" % len(outer_test), outer_test)
    print("  inner_train_1 :", itr1)
    print("  inner_test_1  :", ite1)
    print("  inner_train_2 :", itr2)
    print("  inner_test_2  :", ite2)
    check_invariants(outer_train, outer_test, itr1, ite1, itr2, ite2)
    print("all partition invariants hold.")

    y = ds.tensors[1]

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
    ncols = N   # widest row is the unsplit one (all N images)
    fig, axes = plt.subplots(len(groups), ncols,
                             figsize=(ncols * 1.1, len(groups) * 1.25))
    fig.suptitle("Nested-resampling split on 10 KMNIST images "
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
    save(fig, "resampling_split")
    plt.close(fig)
    print("figure written to", OUT)


if __name__ == "__main__":
    main()
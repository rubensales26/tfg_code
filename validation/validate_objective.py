"""Validation: the Objective wrapper produces EXACTLY the same result as a
plain, hand-written train/evaluate loop.

We train and evaluate one MLP on the standard KMNIST train/test split twice:

    PATH A : through hpo_lib.Objective  (the code under test)
    PATH B : an independent, from-scratch PyTorch loop that uses only the model
             class (same architecture), the same Adam optimizer, the same
             cross-entropy loss, the same data loaders, and the same seed --
             but NEITHER Objective NOR Trainer.

If the Objective adds no hidden behaviour, the two error rates must be
bit-identical. Everything runs on CPU so the comparison is exactly
reproducible (CUDA kernels can differ run-to-run even at a fixed seed).

Run:  python validation/validate_objective.py
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from hpo_lib.data import OuterFold
from hpo_lib.hp_config import HPConfig
from hpo_lib.models import KMnistMLP
from hpo_lib.objective import Objective
from torch.utils.data import TensorDataset

DATA = _ROOT / "datasets" / "KMNIST" / "raw"
DEVICE = torch.device("cpu")          # force determinism
SEED = 123
EPOCHS = 3
BATCH = 64
N_TRAIN, N_TEST = 3000, 1000

CONFIG = {
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "num_layers": 2,
    "hidden_units": 128,
    "dropout_rate": 0.2,              # a non-zero, RNG-consuming layer on purpose
}


def _load(npz_imgs, npz_labels, n):
    imgs = np.load(DATA / npz_imgs)["arr_0"][:n]
    labels = np.load(DATA / npz_labels)["arr_0"][:n]
    X = torch.tensor(imgs, dtype=torch.float32).unsqueeze(1) / 255.0
    y = torch.tensor(labels, dtype=torch.long)
    return X, y


def build_standard_split_fold() -> OuterFold:
    """One OuterFold whose single inner fold IS the canonical train/test split."""
    Xtr, ytr = _load("kmnist-train-imgs.npz", "kmnist-train-labels.npz", N_TRAIN)
    Xte, yte = _load("kmnist-test-imgs.npz", "kmnist-test-labels.npz", N_TEST)
    X = torch.cat([Xtr, Xte], dim=0)
    y = torch.cat([ytr, yte], dim=0)
    ds = TensorDataset(X, y)

    train_idx = np.arange(N_TRAIN)
    test_idx = np.arange(N_TRAIN, N_TRAIN + N_TEST)
    # a single inner fold = (train, test); OuterFold needs an outer test too,
    # which this validation never uses, so we point it at the same test block.
    fold = OuterFold(
        dataset=ds,
        train_idx=train_idx,
        test_idx=test_idx,
        inner_folds=[(train_idx, test_idx)],
        batch_size=BATCH,
        seed=SEED,
        index=0,
    )
    return fold


# ---------------------------------------------------------------------------
# PATH B: independent train/eval loop (mirrors trainer.py, uses no Trainer)
# ---------------------------------------------------------------------------

def manual_train_eval(fold: OuterFold, cfg: HPConfig) -> float:
    """Replicate one Objective inner-fold evaluation by hand (no Objective/Trainer)."""
    torch.manual_seed(SEED + 0)                       # same seeding as Objective (j=0)
    train_loader, test_loader = fold.get_inner_fold(0)

    model = KMnistMLP(cfg).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=cfg.learning_rate,
                                 weight_decay=cfg.weight_decay)
    # --- train ---
    for _ in range(EPOCHS):
        model.train()
        for batch in train_loader:
            X = batch[0].to(DEVICE)
            y = batch[1].to(DEVICE)
            optimizer.zero_grad()
            loss = F.cross_entropy(model(X), y)
            loss.backward()
            optimizer.step()
    # --- evaluate (pooled error rate, exactly like Trainer.evaluate) ---
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in test_loader:
            X = batch[0].to(DEVICE)
            y = batch[1].to(DEVICE)
            preds = torch.argmax(model(X), dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return 1.0 - correct / total


def main():
    fold = build_standard_split_fold()
    cfg = HPConfig.from_dict(CONFIG)

    # PATH A: through the Objective (dict input)
    obj = Objective(fold, default_epochs=EPOCHS, device=DEVICE, seed=SEED)
    err_objective = obj(CONFIG)

    # PATH B: independent manual loop
    err_manual = manual_train_eval(fold, cfg)

    # BONUS: Objective must give the same answer for dict vs HPConfig input
    obj2 = Objective(fold, default_epochs=EPOCHS, device=DEVICE, seed=SEED)
    err_objective_typed = obj2(cfg)

    print(f"PATH A  Objective (dict)          : {err_objective:.10f}")
    print(f"PATH B  manual loop (no Objective): {err_manual:.10f}")
    print(f"BONUS   Objective (HPConfig) : {err_objective_typed:.10f}")
    print(f"archive recorded {len(obj.archive)} evaluation(s); "
          f"mean_error={obj.archive[0]['mean_error']:.10f}")

    ok = True

    def check(cond, msg):
        nonlocal ok
        print(("PASS" if cond else "FAIL"), "-", msg)
        ok = ok and cond

    check(err_objective == err_manual,
          "Objective error == independent manual-loop error (bit-identical)")
    check(err_objective == err_objective_typed,
          "Objective gives identical result for dict and HPConfig input")
    check(obj.archive[0]["mean_error"] == err_objective,
          "archive mean_error matches the returned value")

    print("\nRESULT:", "VALIDATION PASSED" if ok else "VALIDATION FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
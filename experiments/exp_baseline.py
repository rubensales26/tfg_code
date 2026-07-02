"""
Baseline model: single fixed configuration, no hyperparameter search.

    Trains on the full official KMNIST training set (60 000 images) and
    evaluates once on the full official KMNIST test set (10 000 images).
    No nested resampling is used.

    The hyperparameters are chosen as a reasonable uninformed first guess:
        learning_rate = 1e-3   (Adam default from Kingma & Ba, 2014)
        weight_decay  = 1e-4
        num_layers    = 2
        hidden_units  = 256
        dropout_rate  = 0.1

Run (from the project root):
    python experiments/exp_baseline.py --plot
    python experiments/exp_baseline.py --quick
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from hpo_lib.data import KMnistDataset
from hpo_lib.hp_config import HPConfig
from hpo_lib.models import KMnistMLP
from hpo_lib.trainer import Trainer, pick_device
from hpo_lib.experiment import save_results, plot_convergence

_DEFAULT_DATA = str(_ROOT / "datasets" / "KMNIST" / "raw")

NAME = "baseline"

DEFAULT_CONFIG = HPConfig(
    learning_rate=1e-3,
    weight_decay=1e-4,
    num_layers=2,
    hidden_units=256,
    dropout_rate=0.1,
)


def main():
    parser = argparse.ArgumentParser(description="Baseline (no HPO) on KMNIST")
    parser.add_argument("--data-dir", default=_DEFAULT_DATA)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results-dir", default="./results")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--quick", action="store_true",
                        help="2 epochs for a fast smoke run")
    args = parser.parse_args()

    epochs = 2 if args.quick else args.epochs
    device = pick_device()

    ds = KMnistDataset(args.data_dir)
    train_loader = DataLoader(ds.load_train(), batch_size=args.batch_size,
                              shuffle=True,
                              generator=torch.Generator().manual_seed(args.seed))
    test_loader  = DataLoader(ds.load_test(),  batch_size=args.batch_size,
                              shuffle=False)

    print(f"Training set: {len(train_loader.dataset)} images")
    print(f"Test set:     {len(test_loader.dataset)} images")
    print(f"\nBaseline configuration (fixed, no search):")
    for k, v in DEFAULT_CONFIG.to_dict().items():
        print(f"    {k:<14} = {v}")

    torch.manual_seed(args.seed)
    trainer = Trainer(max_epochs=epochs, device=device)

    start = time.time()
    model = trainer.fit(KMnistMLP(DEFAULT_CONFIG), train_loader,
                        val_loader=test_loader)
    elapsed = time.time() - start

    result = trainer.evaluate(model, test_loader)
    print(f"\n  test error rate = {result['error_rate']:.4f}"
          f"    accuracy = {result['accuracy']:.4f}"
          f"    elapsed {elapsed:.1f}s")

    save_results(NAME, {
        "algorithm":      NAME,
        "config":         DEFAULT_CONFIG.to_dict(),
        "outer_error":    result["error_rate"],
        "outer_accuracy": result["accuracy"],
        "n_train":        len(train_loader.dataset),
        "n_test":         len(test_loader.dataset),
        "epochs":         epochs,
        "elapsed_s":      elapsed,
        "history":        trainer.history,
    }, args.results_dir)

    if args.plot:
        plot_convergence(NAME, trainer.history)


if __name__ == "__main__":
    main()

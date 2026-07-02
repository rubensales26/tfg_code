"""Validation: KMnistMLP (hpo_lib/models.py) builds the architecture its
configuration specifies and computes the two losses correctly.

    architecture : Flatten + num_layers x (Linear->ReLU->Dropout) + Linear(.,10);
                   correct widths, dropout probability, and input/output dims.
    forward      : maps (B,1,28,28) -> (B,10) logits.
    inner_loss   : equals F.cross_entropy (the ERM training loss).
    outer_loss   : equals 1 - accuracy (the generalization score).
    optimizer    : Adam with the configured learning_rate and weight_decay.

Run:  python validation/validate_models.py
"""
from __future__ import annotations

from pathlib import Path
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from hpo_lib.hp_config import HPConfig
from hpo_lib.models import KMnistMLP

_ok = True


def check(cond, msg):
    global _ok
    print(("PASS" if cond else "FAIL"), "-", msg)
    _ok = _ok and bool(cond)


def validate_architecture(num_layers, hidden_units, dropout):
    cfg = HPConfig(learning_rate=1e-3, weight_decay=1e-4,
                        num_layers=num_layers, hidden_units=hidden_units,
                        dropout_rate=dropout)
    model = KMnistMLP(cfg)
    mods = list(model.net)
    print(f"--- num_layers={num_layers}, hidden_units={hidden_units}, "
          f"dropout={dropout} ---")

    linears = [m for m in mods if isinstance(m, nn.Linear)]
    relus = [m for m in mods if isinstance(m, nn.ReLU)]
    drops = [m for m in mods if isinstance(m, nn.Dropout)]

    check(isinstance(mods[0], nn.Flatten), "first layer is Flatten")
    check(len(linears) == num_layers + 1,
          f"{num_layers} hidden + 1 output == {num_layers + 1} Linear layers")
    check(len(relus) == num_layers and len(drops) == num_layers,
          f"{num_layers} ReLU and {num_layers} Dropout layers")
    check(all(abs(d.p - dropout) < 1e-12 for d in drops),
          f"every Dropout uses p == {dropout}")
    check(linears[0].in_features == 28 * 28, "input Linear takes 784 features")
    check(all(lin.out_features == hidden_units for lin in linears[:-1])
          if num_layers > 0 else True, "hidden Linear widths == hidden_units")
    check(linears[-1].out_features == 10, "output Linear has 10 classes")

    out = model(torch.randn(5, 1, 28, 28))
    check(tuple(out.shape) == (5, 10), "forward maps (5,1,28,28) -> (5,10)")


def validate_losses():
    print("--- losses ---")
    cfg = HPConfig(1e-3, 1e-4, 2, 32, 0.0)
    model = KMnistMLP(cfg)
    torch.manual_seed(0)
    logits = torch.randn(8, 10)
    y = torch.randint(0, 10, (8,))

    check(torch.allclose(model.inner_loss(logits, y), F.cross_entropy(logits, y)),
          "inner_loss == F.cross_entropy")

    acc = (logits.argmax(dim=1) == y).float().mean()
    check(torch.allclose(model.outer_loss(logits, y), 1.0 - acc),
          "outer_loss == 1 - accuracy")


def validate_optimizer():
    print("--- optimizer ---")
    cfg = HPConfig(learning_rate=3e-3, weight_decay=5e-4,
                        num_layers=1, hidden_units=16, dropout_rate=0.0)
    model = KMnistMLP(cfg)
    opt = model.make_optimizer()
    check(isinstance(opt, torch.optim.Adam), "make_optimizer returns Adam")
    g = opt.param_groups[0]
    check(abs(g["lr"] - 3e-3) < 1e-12, "optimizer learning_rate matches config")
    check(abs(g["weight_decay"] - 5e-4) < 1e-12, "optimizer weight_decay matches config")


def main():
    print("=== KMnistMLP architecture ===")
    validate_architecture(num_layers=2, hidden_units=64, dropout=0.3)
    validate_architecture(num_layers=1, hidden_units=128, dropout=0.0)
    print("\n=== KMnistMLP losses / optimizer ===")
    validate_losses()
    validate_optimizer()
    print("\nRESULT:", "MODELS VALIDATED" if _ok else "VALIDATION FAILED")
    sys.exit(0 if _ok else 1)


if __name__ == "__main__":
    main()
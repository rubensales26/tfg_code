"""
Training and evaluation loop.

Fits a model for a fixed number of epochs with the inner (cross-entropy) loss
and scores it with the outer loss (1 - accuracy). It is loss-agnostic -- it just
calls the model's training_step and outer_loss -- so the same trainer drives both
the inner HPO evaluations and the final outer evaluation.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from .models import BaseModel


def pick_device() -> torch.device:
    """CUDA if available, else Apple MPS, else CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class Trainer:
    """Fits a model for ``max_epochs`` and evaluates it with the outer loss."""

    def __init__(
        self,
        max_epochs: int,
        device: torch.device | None = None,
        verbose: bool = False,
    ):
        self.max_epochs = max_epochs
        self.device = device if device is not None else pick_device()
        self.verbose = verbose
        # per-epoch trajectory, filled in by fit (for convergence plots)
        self.history = {"train_loss": [], "val_error": [], "val_accuracy": []}

    def fit(self, model: BaseModel, train_loader: DataLoader,
            val_loader: DataLoader | None = None) -> BaseModel:
        """
        Train the model for max_epochs on train_loader and return it.

        Records the per-epoch train loss in self.history; if val_loader is given,
        also records the validation error/accuracy each epoch. Recording does not
        affect training.
        """
        model = model.to(self.device)
        optimizer = model.make_optimizer()
        self.history = {"train_loss": [], "val_error": [], "val_accuracy": []}

        for epoch in range(self.max_epochs):
            model.train()
            running, n_batches = 0.0, 0
            for batch in train_loader:
                X = batch[0].to(self.device)
                y = batch[1].to(self.device)
                optimizer.zero_grad()
                loss = model.training_step((X, y))
                loss.backward()
                optimizer.step()
                running += loss.item()
                n_batches += 1

            avg = running / max(n_batches, 1)
            self.history["train_loss"].append(avg)

            # optional per-epoch validation curve
            if val_loader is not None:
                res = self.evaluate(model, val_loader)
                self.history["val_error"].append(res["error_rate"])
                self.history["val_accuracy"].append(res["accuracy"])

            if self.verbose:
                msg = (f"  epoch {epoch + 1:02d}/{self.max_epochs} "
                       f"train_loss={avg:.4f}")
                if val_loader is not None:
                    msg += f"  val_error={self.history['val_error'][-1]:.4f}"
                print(msg)
        return model

    @torch.no_grad()
    def evaluate(self, model: BaseModel, loader: DataLoader) -> dict:
        """
        Score the model with the outer loss (1 - accuracy).

        The error rate is pooled over all samples (not averaged per batch), so it
        is exact for any batch size.
        """
        model = model.to(self.device)
        model.eval()
        correct, total = 0, 0
        for batch in loader:
            X = batch[0].to(self.device)
            y = batch[1].to(self.device)
            preds = torch.argmax(model(X), dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
        accuracy = correct / max(total, 1)
        return {"error_rate": 1.0 - accuracy, "accuracy": accuracy, "n": total}

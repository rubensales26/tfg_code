"""
Models.

A trainable model knows three things: how to predict (forward), how predictions
are scored (the two losses), and how its parameters are updated (the optimizer).
BaseModel is that contract; KMnistMLP is the MLP used in the experiments.

Two losses:
    inner_loss  = cross-entropy   -> drives training (ERM)
    outer_loss  = 1 - accuracy    -> scores generalization (the objective)
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .hp_config import HPConfig


class BaseModel(nn.Module, ABC):
    """Abstract contract for every model: forward, two losses, an optimizer."""

    @abstractmethod
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Compute the logits h(X; theta)."""

    @abstractmethod
    def inner_loss(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """L_inn used for training (ERM)."""

    @abstractmethod
    def outer_loss(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """L_out used for evaluation (generalization error)."""

    def training_step(self, batch) -> torch.Tensor:
        X, y = batch[0], batch[1]
        return self.inner_loss(self(X), y)

    @abstractmethod
    def make_optimizer(self) -> optim.Optimizer:
        """Instantiate the gradient-based optimizer that updates theta."""


class KMnistMLP(BaseModel):
    """
    MLP for 28x28 KMNIST images (10 classes).

    The config fixes the architecture: num_layers hidden layers of width
    hidden_units, each followed by ReLU and dropout(dropout_rate). Training is
    governed by learning_rate and weight_decay via Adam.
    """

    INPUT_DIM = 28 * 28
    NUM_CLASSES = 10

    def __init__(self, config: HPConfig):
        super().__init__()
        self.config = config
        self.learning_rate = config.learning_rate
        self.weight_decay = config.weight_decay

        layers: list[nn.Module] = [nn.Flatten()]
        in_dim = self.INPUT_DIM
        for _ in range(config.num_layers):
            layers.append(nn.Linear(in_dim, config.hidden_units))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(config.dropout_rate))
            in_dim = config.hidden_units
        layers.append(nn.Linear(in_dim, self.NUM_CLASSES))
        self.net = nn.Sequential(*layers)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.net(X)

    def inner_loss(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Mean cross-entropy over the mini-batch."""
        return F.cross_entropy(y_hat, y)

    def outer_loss(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Error rate 1 - accuracy over the batch."""
        preds = torch.argmax(y_hat, dim=1)
        accuracy = (preds == y).float().mean()
        return 1.0 - accuracy

    def make_optimizer(self) -> optim.Optimizer:
        return optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

"""
The hyperparameter configuration lambda.
"""

from __future__ import annotations

from dataclasses import dataclass, replace, asdict


@dataclass(frozen=True)
class HPConfig:
    """
    One hyperparameter configuration (a single point lambda).

        learning_rate : Adam step size
        weight_decay  : L2 regularization
        num_layers    : number of hidden layers
        hidden_units  : units per hidden layer
        dropout_rate  : dropout after each hidden layer
    """

    learning_rate: float
    weight_decay:  float
    num_layers:    int
    hidden_units:  int
    dropout_rate:  float

    #  Helpers

    def to_dict(self) -> dict:
        """Return a plain dict of the hyperparameter values."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "HPConfig":
        """Build a config from a dict, coercing integer-valued fields."""
        return cls(
            learning_rate=float(d["learning_rate"]),
            weight_decay=float(d["weight_decay"]),
            num_layers=int(d["num_layers"]),
            hidden_units=int(d["hidden_units"]),
            dropout_rate=float(d["dropout_rate"]),
        )

    def replace(self, **changes) -> "HPConfig":
        """Return a copy with the given fields overridden (immutably)."""
        return replace(self, **changes)

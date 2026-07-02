"""
The bounded search space tilde-Lambda 

The search space is the Cartesian product of per-hyperparameter domains

        tilde-Lambda = tilde-Lambda_1 x ... x tilde-Lambda_l ,

each of which is continuous, log-scaled, or discrete/categorical. Every domain
knows how to do three things, which is all the search strategies need:

    sample(rng)            -> draw one value           (Random Search, GA init)
    grid(resolution)       -> list of values           (Grid Search)
    mutate(value, rng, s)  -> perturb one value        (Genetic Algorithm)

A ``SearchSpace`` is just an ordered mapping {name: Domain} that lifts these
operations to whole configurations (dicts).
"""

from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod
from itertools import product


class Domain(ABC):
    """A single bounded hyperparameter domain tilde-Lambda_i."""

    @abstractmethod
    def sample(self, rng: random.Random):
        """Draw one value uniformly from the domain."""

    @abstractmethod
    def grid(self, resolution: int) -> list:
        """Return ``resolution`` representative values spanning the domain."""

    @abstractmethod
    def mutate(self, value, rng: random.Random, scale: float = 0.1):
        """Return a perturbed copy of ``value`` (for GA mutation)."""


class Continuous(Domain):
    """
    A bounded real interval [low, high].

    If ``log`` is True the domain is sampled and gridded geometrically, which is
    the standard choice for scale hyperparameters such as the learning rate or
    the weight decay.
    """

    def __init__(self, low: float, high: float, log: bool = False):
        # Values must be consistent
        if low > high:
            raise ValueError(f"low={low} must not exceed high={high}")
        if log and low <= 0:
            raise ValueError("log domains require strictly positive bounds")
        
        self.low = float(low)
        self.high = float(high)
        self.log = bool(log)

    def sample(self, rng: random.Random) -> float:
        if self.log:
            # Convert to log space and sample from it
            u = rng.uniform(math.log(self.low), math.log(self.high))
            # Return the value in the original space
            return math.exp(u)

        return rng.uniform(self.low, self.high)

    def grid(self, resolution: int) -> list[float]:
        if resolution < 1:
            raise ValueError("resolution must be >= 1")
        
        if resolution == 1:
            # Get the mid point in the corresponding space
            if self.log:
                log_low, log_high = math.log(self.low), math.log(self.high)
                log_midpoint = (log_low + log_high) / 2
                return [math.exp(log_midpoint)]
            
            return [0.5 * (self.low + self.high)]
        
        if self.log:
            # Convert to log space
            a, b = math.log(self.low), math.log(self.high)
            # Get the equally spaced points in log space and return to the
            # original space. Clip into [low, high]: exp(log(.)) at the endpoints
            # can round a hair outside the bounds, and we want grid points to
            # stay strictly in-domain (consistent with mutate).
            pts = [math.exp(a + (b - a) * k / (resolution - 1))
                   for k in range(resolution)]
            return [min(max(v, self.low), self.high) for v in pts]

        # No log, resolution > 1 case
        pts = [self.low + (self.high - self.low) * k / (resolution - 1)
               for k in range(resolution)]
        return [min(max(v, self.low), self.high) for v in pts]

    def mutate(self, value: float, rng: random.Random, scale: float = 0.1) -> float:
        """
        Additive Gaussian perturbation, performed in log-space for log domains.
        The step std-dev is ``scale`` times the (log-)width of the interval, and
        the result is clipped back into the bounds.
        """
        if self.log:
            # Convert to log space
            a, b = math.log(self.low), math.log(self.high)
            # Add Gaussian perturbation
            v = math.log(value) + rng.gauss(0.0, scale * (b - a))
            # Clip value to the log domain
            v = min(max(v, a), b)

            # exp(log(.)) can round a hair outside [low, high] at the edges, so
            # clip once more in the original space to guarantee an in-bounds value
            return min(max(math.exp(v), self.low), self.high)

        # Add Gaussian perturbation
        v = value + rng.gauss(0.0, scale * (self.high - self.low))
        # Clip back to the domain
        v = min(max(v, self.low), self.high)

        return v


class Discrete(Domain):
    """
    A finite set of candidate values (ordinal or categorical).

    Used for the integer architectural hyperparameters (number of layers, width)
    and for any categorical hyperparameter once it has been numerically encoded.
    """

    def __init__(self, values: list):
        if len(values) == 0:
            raise ValueError("Discrete domain needs at least one value")
        self.values = list(values)

    def sample(self, rng: random.Random):
        return rng.choice(self.values)

    def grid(self, resolution: int | None = None) -> list:
        """All candidate values (resolution is ignored for discrete domains)."""
        return list(self.values)

    def mutate(self, value, rng: random.Random, scale: float = 0.1):
        """Switch to a different candidate value chosen uniformly at random."""
        if len(self.values) == 1:
            return self.values[0]
        
        others = [v for v in self.values if v != value]

        return rng.choice(others)


class SearchSpace:
    """
    An ordered product of named domains: tilde-Lambda = prod_i tilde-Lambda_i.

    The insertion order of ``domains`` fixes the gene order used by the genetic
    algorithm and the column order of the grid.
    """

    def __init__(self, domains: dict[str, Domain]):
        if len(domains) == 0:
            raise ValueError("SearchSpace needs at least one domain")
        self.domains = dict(domains)
        self.names = list(domains.keys())

    # whole-configuration operations

    def sample(self, rng: random.Random) -> dict:
        """Draw one configuration uniformly from tilde-Lambda."""
        return {name: dom.sample(rng) for name, dom in self.domains.items()}

    def grid(self, resolution: int | dict[str, int]) -> list[dict]:
        """
        Enumerate the full Cartesian grid.

        ``resolution`` is either a single int applied to every continuous
        domain, or a per-name dict. Discrete domains always yield all of
        their candidate values.
        """
        if isinstance(resolution, int):
            resolution = {name: resolution for name in self.names}
        
        # For each domain, access its grid passing the corresponding resolution,
        # taken from the resolution dictionary (default = 1)
        axes = [self.domains[name].grid(resolution.get(name, 1))
                for name in self.names]
        return [dict(zip(self.names, combo)) for combo in product(*axes)]

    def grid_size(self, resolution: int | dict[str, int]) -> int:
        """Number of points in the grid without materializing it."""
        # If resolution is an int instead of a dict, make it a dict
        if isinstance(resolution, int):
            resolution = {name: resolution for name in self.names}

        size = 1
        for name in self.names:
            dom = self.domains[name]
            if isinstance(dom, Discrete):
                size *= len(dom.values)
            else:
                size *= max(1, resolution.get(name, 1))
        return size

    def mutate(self, config: dict, rng: random.Random,
               mutation_rate: float = 0.2, scale: float = 0.1) -> dict:
        """Per-gene mutation: each coordinate is perturbed w.p. mutation_rate."""
        # Create a copu not to alter the original dict
        child = dict(config)

        for name, dom in self.domains.items():
            if rng.random() < mutation_rate:
                child[name] = dom.mutate(config[name], rng, scale)
        return child

    def crossover(self, parent_a: dict, parent_b: dict,
                  rng: random.Random) -> dict:
        """Uniform crossover: each gene inherited from either parent w.p. 0.5."""
        return {name: (parent_a[name] if rng.random() < 0.5 else parent_b[name])
                for name in self.names}


def default_search_space() -> SearchSpace:
    """
    The search space used in the numerical experiments.
    """
    return SearchSpace({
        "learning_rate": Continuous(1e-4, 1e-1, log=True),
        "weight_decay":  Continuous(1e-5, 1e-2, log=True),
        "num_layers":    Discrete([1, 2, 3]),
        "hidden_units":  Discrete([128, 256, 512]),
        "dropout_rate":  Continuous(0.0, 0.5, log=False),
    })

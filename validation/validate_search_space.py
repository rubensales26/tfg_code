"""Validation: the domain primitives in hpo_lib/search_space.py behave as
specified. These are the lowest-level building blocks (the grid-clamp bug lived
here), so they get direct unit checks rather than only indirect coverage through
the searchers.

    Continuous : in-bounds sampling; log sampling positive & spanning decades;
                 grid spacing (geometric for log, linear otherwise), endpoints
                 in-bounds; resolution-1 midpoint (geometric / arithmetic);
                 mutate clips into bounds; constructor validation.
    Discrete   : sampling and grid return candidates; mutate switches to a
                 DIFFERENT value; single-value edge case; constructor validation.
    SearchSpace: sample/grid keys; grid_size == len(grid); full Cartesian
                 product; mutate returns an in-bounds copy; crossover inherits
                 each gene from a parent.

Run:  python validation/validate_search_space.py
"""
from __future__ import annotations

import math
import random
from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from hpo_lib.search_space import Continuous, Discrete, SearchSpace

_ok = True


def check(cond, msg):
    global _ok
    print(("PASS" if cond else "FAIL"), "-", msg)
    _ok = _ok and bool(cond)


def approx(a, b, tol=1e-9):
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


# ---------------------------------------------------------------------------
# Continuous
# ---------------------------------------------------------------------------

def validate_continuous():
    print("=== Continuous ===")
    rng = random.Random(0)

    lin = Continuous(0.0, 0.5, log=False)
    samples = [lin.sample(rng) for _ in range(2000)]
    check(all(0.0 <= s <= 0.5 for s in samples), "linear sample stays in [low, high]")

    check(approx(lin.grid(1)[0], 0.25), "linear grid(1) is the arithmetic midpoint")
    g = lin.grid(5)
    check(len(g) == 5 and approx(g[0], 0.0) and approx(g[-1], 0.5),
          "linear grid(5) spans the closed interval")
    diffs = [g[i + 1] - g[i] for i in range(4)]
    check(all(approx(d, diffs[0]) for d in diffs), "linear grid is equally spaced")

    log = Continuous(1e-4, 1e-1, log=True)
    ls = [log.sample(rng) for _ in range(5000)]
    check(all(1e-4 <= s <= 1e-1 for s in ls), "log sample stays in [low, high]")
    check(all(s > 0 for s in ls), "log sample is strictly positive")
    check(min(ls) < 1e-3 and max(ls) > 1e-2, "log sample spans low and high decades")

    check(approx(log.grid(1)[0], math.sqrt(1e-4 * 1e-1)),
          "log grid(1) is the geometric mean")
    lg = log.grid(4)
    check(len(lg) == 4 and all(1e-4 <= v <= 1e-1 for v in lg),
          "log grid(4) endpoints are in-bounds (clamped)")
    check(approx(lg[0], 1e-4) and approx(lg[-1], 1e-1),
          "log grid(4) endpoints equal low and high (up to float tol)")
    ratios = [lg[i + 1] / lg[i] for i in range(3)]
    check(all(approx(r, ratios[0], 1e-6) for r in ratios),
          "log grid is geometrically spaced (constant ratio)")

    # mutate must clip back into bounds even with a large step from the edge
    clipped = [log.mutate(1e-1, rng, scale=2.0) for _ in range(2000)]
    check(all(1e-4 <= v <= 1e-1 for v in clipped),
          "log mutate clips results into [low, high]")
    clipped_lin = [lin.mutate(0.5, rng, scale=2.0) for _ in range(2000)]
    check(all(0.0 <= v <= 0.5 for v in clipped_lin),
          "linear mutate clips results into [low, high]")

    # constructor validation
    for bad, why in [(lambda: Continuous(1.0, 0.0), "low > high"),
                     (lambda: Continuous(0.0, 1.0, log=True), "log with low <= 0")]:
        raised = False
        try:
            bad()
        except ValueError:
            raised = True
        check(raised, f"Continuous rejects {why}")


# ---------------------------------------------------------------------------
# Discrete
# ---------------------------------------------------------------------------

def validate_discrete():
    print("\n=== Discrete ===")
    rng = random.Random(0)
    d = Discrete([1, 2, 3])

    check(all(d.sample(rng) in {1, 2, 3} for _ in range(1000)),
          "sample only returns candidate values")
    check(sorted(d.grid()) == [1, 2, 3], "grid returns all candidates")

    # mutate must switch to a DIFFERENT value
    switched = [d.mutate(2, rng) for _ in range(1000)]
    check(all(v in {1, 3} for v in switched),
          "mutate switches to a different candidate (never the input)")
    check(set(switched) == {1, 3}, "mutate can reach every other candidate")

    one = Discrete([7])
    check(one.sample(rng) == 7 and one.grid() == [7], "single-value sample/grid")
    check(one.mutate(7, rng) == 7, "single-value mutate returns the only value")

    raised = False
    try:
        Discrete([])
    except ValueError:
        raised = True
    check(raised, "Discrete rejects an empty value list")


# ---------------------------------------------------------------------------
# SearchSpace
# ---------------------------------------------------------------------------

def validate_search_space():
    print("\n=== SearchSpace ===")
    rng = random.Random(0)
    space = SearchSpace({
        "lr": Continuous(1e-3, 1e-1, log=True),
        "units": Discrete([16, 32, 64]),
        "drop": Continuous(0.0, 0.5),
    })
    names = space.names

    cfg = space.sample(rng)
    check(list(cfg.keys()) == names, "sample produces all domain keys in order")
    check(1e-3 <= cfg["lr"] <= 1e-1 and cfg["units"] in {16, 32, 64}
          and 0.0 <= cfg["drop"] <= 0.5, "sampled values are all in-domain")

    # grid_size matches the materialized grid, for int and dict resolution
    for res in (2, 3, {"lr": 4, "drop": 2}):
        check(space.grid_size(res) == len(space.grid(res)),
              f"grid_size == len(grid) for resolution {res}")

    # full Cartesian product on a tiny, fully-discrete space
    sp2 = SearchSpace({"a": Discrete([1, 2]), "b": Discrete([3, 4, 5])})
    grid = sp2.grid(1)
    combos = {(c["a"], c["b"]) for c in grid}
    check(len(grid) == 6 and combos == {(a, b) for a in (1, 2) for b in (3, 4, 5)},
          "grid enumerates the full Cartesian product without gaps/dupes")

    # mutate returns an in-bounds COPY, leaving the parent untouched
    parent = space.sample(rng)
    parent_snapshot = dict(parent)
    child = space.mutate(parent, rng, mutation_rate=1.0, scale=0.3)
    check(parent == parent_snapshot, "mutate does not modify the input config")
    check(list(child.keys()) == names, "mutated child keeps all keys")
    check(1e-3 <= child["lr"] <= 1e-1 and child["units"] in {16, 32, 64}
          and 0.0 <= child["drop"] <= 0.5, "mutated child stays in-domain")

    # crossover: every gene comes from one of the two parents
    pa, pb = space.sample(rng), space.sample(rng)
    inherited = True
    for _ in range(500):
        ch = space.crossover(pa, pb, rng)
        if set(ch.keys()) != set(names):
            inherited = False
            break
        for n in names:
            if ch[n] != pa[n] and ch[n] != pb[n]:
                inherited = False
        if not inherited:
            break
    check(inherited, "crossover inherits every gene from parent A or parent B")

    raised = False
    try:
        SearchSpace({})
    except ValueError:
        raised = True
    check(raised, "SearchSpace rejects an empty domain mapping")


def main():
    validate_continuous()
    validate_discrete()
    validate_search_space()
    print("\nRESULT:", "SEARCH SPACE VALIDATED" if _ok else "VALIDATION FAILED")
    sys.exit(0 if _ok else 1)


if __name__ == "__main__":
    main()
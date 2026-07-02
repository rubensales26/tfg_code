"""Validation: HPConfig (hpo_lib/hp_config.py) behaves as a correct,
immutable hyperparameter record.

    to_dict / from_dict round-trip; from_dict coerces integer-valued fields to
    int (so a searcher that emits 2.0 yields num_layers == 2, not 2.0); the
    dataclass is frozen (immutable); replace() returns a modified copy without
    mutating the original.

Run:  python validation/validate_config.py
"""
from __future__ import annotations

import dataclasses
from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from hpo_lib.hp_config import HPConfig

_ok = True


def check(cond, msg):
    global _ok
    print(("PASS" if cond else "FAIL"), "-", msg)
    _ok = _ok and bool(cond)


def main():
    print("=== HPConfig ===")
    c = HPConfig(learning_rate=1e-2, weight_decay=1e-3,
                      num_layers=2, hidden_units=128, dropout_rate=0.1)

    d = c.to_dict()
    check(set(d.keys()) == {"learning_rate", "weight_decay", "num_layers",
                            "hidden_units", "dropout_rate"},
          "to_dict exposes exactly the five hyperparameters")
    check(HPConfig.from_dict(d) == c, "from_dict(to_dict(c)) round-trips to c")

    # integer coercion: searchers/grids may produce float-valued ints
    coerced = HPConfig.from_dict({
        "learning_rate": 1e-2, "weight_decay": 1e-3,
        "num_layers": 2.0, "hidden_units": 128.0, "dropout_rate": 0.1,
    })
    check(coerced.num_layers == 2 and isinstance(coerced.num_layers, int),
          "from_dict coerces num_layers to int")
    check(coerced.hidden_units == 128 and isinstance(coerced.hidden_units, int),
          "from_dict coerces hidden_units to int")
    check(coerced == c, "coerced config equals the canonical one")

    # frozen / immutable
    raised = False
    try:
        c.learning_rate = 0.5            # type: ignore[misc]
    except dataclasses.FrozenInstanceError:
        raised = True
    check(raised, "config is frozen (attribute assignment raises)")

    # replace returns a modified copy, original untouched
    c2 = c.replace(num_layers=3)
    check(isinstance(c2, HPConfig) and c2.num_layers == 3,
          "replace returns a HPConfig with the field changed")
    check(c.num_layers == 2, "replace leaves the original config unchanged")
    check(c2.replace(num_layers=2) == c, "replace is the only difference (round-trip)")

    print("\nRESULT:", "CONFIG VALIDATED" if _ok else "VALIDATION FAILED")
    sys.exit(0 if _ok else 1)


if __name__ == "__main__":
    main()
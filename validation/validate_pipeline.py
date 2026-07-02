"""Validation (integration): each experiment script runs end-to-end and wires
the whole pipeline together correctly -- inner HPO over the disjoint resampling,
retraining the incumbent on the full outer-train set, a single held-out
outer-test evaluation, and a saved results JSON with a complete incumbent.

Each of the four scripts is launched as its own process (with --quick) and its
results JSON is checked. This actually trains tiny models, so it is the
integration test that ties every unit-validated component together.

Run:  python validation/validate_pipeline.py
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from hpo_lib.search_space import default_search_space

_ok = True


def check(cond, msg):
    global _ok
    print(("PASS" if cond else "FAIL"), "-", msg)
    _ok = _ok and bool(cond)


SCRIPTS = {
    "random": "exp_random_search.py",
    "grid": "exp_grid_search.py",
    "genetic": "exp_genetic.py",
    "sh": "exp_successive_halving.py",
}


def main():
    names = set(default_search_space().names)
    results_dir = tempfile.mkdtemp(prefix="hpo_validation_")

    for algo, script in SCRIPTS.items():
        print(f"\n=== {script} (--quick) ===")
        cmd = [sys.executable, str(_ROOT / "experiments" / script),
               "--quick", "--subset", "800", "--epochs", "1",
               "--results-dir", results_dir]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        check(proc.returncode == 0, f"[{algo}] script exits cleanly")
        if proc.returncode != 0:
            print(proc.stdout[-1500:])
            print(proc.stderr[-1500:])
            continue

        path = Path(results_dir) / f"{algo}.json"
        check(path.exists(), f"[{algo}] results JSON written")
        if not path.exists():
            continue
        r = json.loads(path.read_text())

        inc = r["incumbent"]
        check(isinstance(inc, dict) and set(inc.keys()) == names,
              f"[{algo}] incumbent is a complete config")
        check(0.0 <= r["outer_error"] <= 1.0,
              f"[{algo}] OUTER GE in [0,1] ({r['outer_error']:.4f})")
        check(0.0 <= r["inner_error"] <= 1.0,
              f"[{algo}] inner error in [0,1] ({r['inner_error']:.4f})")
        check(r["n_evaluations"] >= 1,
              f"[{algo}] at least one objective evaluation ({r['n_evaluations']})")
        check("Best configuration found by" in proc.stdout,
              f"[{algo}] prints the best configuration")

    print("\nRESULT:", "PIPELINE VALIDATED" if _ok else "VALIDATION FAILED")
    sys.exit(0 if _ok else 1)


if __name__ == "__main__":
    main()

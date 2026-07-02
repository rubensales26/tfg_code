"""Run every validation script in order and report a combined pass/fail.

Each script is a standalone process that prints its own PASS/FAIL lines and
exits 0 (passed) or 1 (failed). This driver runs them with the current Python
interpreter, captures their exit codes, and summarizes.

Run:  python validation/run_all.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# fast unit checks first, the model-training integration test last
SCRIPTS = [
    "validate_config.py",
    "validate_search_space.py",
    "validate_models.py",
    "validate_resampling.py",
    "validate_objective.py",
    "validate_pipeline.py",
]


def main():
    results = {}
    for script in SCRIPTS:
        print("=" * 70)
        print("RUNNING", script)
        print("=" * 70)
        proc = subprocess.run([sys.executable, str(HERE / script)])
        results[script] = proc.returncode == 0
        print()

    print("#" * 70)
    print("VALIDATION SUMMARY")
    print("#" * 70)
    for script, ok in results.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {script}")
    all_ok = all(results.values())
    print("#" * 70)
    print("OVERALL:", "ALL VALIDATIONS PASSED" if all_ok else "SOME VALIDATIONS FAILED")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
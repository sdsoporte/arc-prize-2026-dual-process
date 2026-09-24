#!/usr/bin/env python3
"""Leave-one-out ablation of the ARC-AGI-2 primitive DSL.

Answers a question the solver cannot answer about itself, and one the field is asking publicly: **which
primitive families actually produce correct answers?**

The public notebook `yusuketogashi/arc-baseline-rebuild` and the forum post `742790` ("A tiny, forkable ARC
solver: 8 symmetries + a global color map") both ask *"what primitive would you try first?"*. This measures
it: for each family, disable it and see how many of the 1,076 public training outputs stop being solved.

Usage
-----
    python3 experiments/arc2_ablation.py                 # full leave-one-out, writes the JSON
    python3 experiments/arc2_ablation.py --limit 200     # a quick subset
    python3 experiments/arc2_ablation.py --json /tmp/x.json

Measured on the **public training set**, not the leaderboard: it has ground truth AND the solver has
coverage there, which no other public split does (evaluation has ground truth but 0/120 coverage; the test
set has coverage but no solutions).

Read the result as **marginal** contribution: disabling one family measures what is lost when every other
family stays available, so redundancy hides effects. A delta of +/-1 output is noise-scale; exact zeros and
double-digit deltas are not.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

# isort: off
# These must come after the sys.path setup: eval_arc2 needs the repo root, the solver needs src/.
import arc2_dual_process_solver as solver  # noqa: E402
import eval_arc2  # noqa: E402
# isort: on

# Derived from the solver itself, so the list cannot drift from the code.
FAMILIES: list[str] = sorted({family for _, family in solver._FAMILY_PREFIXES} | {"twostage"})


def collect_failures(challenges: dict) -> dict[str, str]:
    """Candidates that raised, with the first exception text.

    A primitive that raises on every task is dead, and a dead primitive looks exactly like a useless one
    in every metric. That is how `rot270` stayed broken and invisible for an unknown period, so the
    ablation reports this rather than assuming it away.
    """
    seen: dict[str, str] = {}
    saved = solver.DISABLED_FAMILIES
    solver.DISABLED_FAMILIES = set()
    try:
        for task in challenges.values():
            trace: dict = {}
            solver.solve_arc_task(task, trace=trace)
            for name, err in (trace.get("failures") or {}).items():
                seen.setdefault(name, err)
    finally:
        solver.DISABLED_FAMILIES = saved
    return seen


def run_one(challenges: dict, solutions: dict, disabled: str | None, limit: int | None) -> dict:
    solver.DISABLED_FAMILIES = set() if disabled is None else {disabled}
    try:
        return eval_arc2.evaluate(challenges, solutions, limit=limit, solver=solver.solve_arc_task)
    finally:
        solver.DISABLED_FAMILIES = set()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=None, help="only the first N training tasks")
    ap.add_argument("--json", dest="json_out", default=str(ROOT / "experiments" / "arc2_ablation_results.json"))
    args = ap.parse_args(argv)

    challenges, solutions = eval_arc2.load_dataset("train")
    failures = collect_failures(challenges)

    base = run_one(challenges, solutions, None, args.limit)
    rows = []
    for family in FAMILIES:
        rep = run_one(challenges, solutions, family, args.limit)
        rows.append(
            {
                "disabled": family,
                "outputs": rep["correct_outputs"],
                "coverage": rep["coverage_tasks"],
                "solved": rep["tasks_fully_solved"],
                "delta_outputs": rep["correct_outputs"] - base["correct_outputs"],
            }
        )
    rows.sort(key=lambda r: r["delta_outputs"])

    print(f"\ntrain tasks: {base['tasks']}   test outputs: {base['test_outputs']}")
    print(f"{'config':<14}{'outputs':>9}{'coverage':>10}{'solved':>8}{'delta':>8}")
    print(f"{'COMPLETE':<14}{base['correct_outputs']:>9}{base['coverage_tasks']:>10}{base['tasks_fully_solved']:>8}{'-':>8}")
    for r in rows:
        print(f"{r['disabled']:<14}{r['outputs']:>9}{r['coverage']:>10}{r['solved']:>8}{r['delta_outputs']:>+8}")

    paying = [r for r in rows if r["delta_outputs"] < 0]
    dead = [r["disabled"] for r in rows if r["delta_outputs"] == 0]
    print("\npaying families : " + (", ".join(f"{r['disabled']} ({r['delta_outputs']:+d})" for r in paying) or "none"))
    print("redundant (0)   : " + (", ".join(dead) or "none"))
    if failures:
        print(f"\nCANDIDATES THAT RAISED ({len(failures)}) - each one may be a dead primitive:")
        for name, err in sorted(failures.items())[:15]:
            print(f"   {name:<28} {err}")
    else:
        print("\nno candidate raised on any task")

    out = {
        "dataset": "train",
        "tasks": base["tasks"],
        "test_outputs": base["test_outputs"],
        "baseline": {k: base[k] for k in ("correct_outputs", "coverage_tasks", "tasks_fully_solved")},
        "disable_one": rows,
        "paying_families": [r["disabled"] for r in paying],
        "redundant_families": dead,
        "candidates_that_raised": failures,
    }
    Path(args.json_out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

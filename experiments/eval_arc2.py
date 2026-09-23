#!/usr/bin/env python3
"""Canonical evaluation harness for the ARC-AGI-2 program-synthesis solver.

Why this exists
---------------
On 2026-09-23 the solver's ARC-AGI-2 leaderboard score of 0.00 was diagnosed. The cause was not the
search strategy but coverage: the candidate pool contained nothing that reproduced the training pairs
for ANY of the 120 public evaluation tasks, so every attempt fell through to a fallback that echoes the
input. Reporting only "0.00%" hides that. This harness reports the number that explains it.

Supersedes `benchmark_arc2_suite.py`, which reported a single stricter metric (task fully solved) and
whose output was never recorded. Both metrics are reported here.

Usage
-----
    python3 experiments/eval_arc2.py                    # evaluate + train
    python3 experiments/eval_arc2.py eval               # public evaluation set only
    python3 experiments/eval_arc2.py train --limit 200  # first N training tasks
    python3 experiments/eval_arc2.py all --json out.json
    python3 experiments/eval_arc2.py eval --timeout 20

Source of truth for the metric: the competition Evaluation page. For each test output, a task scores 1
if EITHER attempt matches the ground truth exactly, else 0; the final score averages over all test
outputs. A "task fully solved" is stricter and is reported separately.
"""

from __future__ import annotations

import argparse
import json
import signal
import statistics
import sys
import time
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from arc2_dual_process_solver import grids_equal, solve_arc_task  # noqa: E402

DATA = ROOT / "data" / "arc-agi-2"

DATASETS: dict[str, tuple[str, str]] = {
    "eval": ("arc-agi_evaluation_challenges.json", "arc-agi_evaluation_solutions.json"),
    "train": ("arc-agi_training_challenges.json", "arc-agi_training_solutions.json"),
}


class TaskTimeout(Exception):
    """Raised when a single task exceeds its time budget."""


def _on_alarm(signum, frame):  # noqa: ARG001
    raise TaskTimeout()


def _shape_matches(a: Any, b: Any) -> bool:
    if not a or not b or not isinstance(a, list) or not isinstance(b, list):
        return False
    if len(a) != len(b):
        return False
    return all(isinstance(r1, list) and isinstance(r2, list) and len(r1) == len(r2)
               for r1, r2 in zip(a, b, strict=True))


def _read_json(path: Path) -> dict:
    """Reads a JSON object, failing with an actionable message rather than a raw traceback."""
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError:
        raise SystemExit(
            f"error: missing dataset file {path}\n"
            f"       this harness needs the ARC-AGI-2 data under {DATA}"
        ) from None
    except json.JSONDecodeError as exc:
        raise SystemExit(f"error: {path} is not valid JSON: {exc}") from None


def load_dataset(name: str) -> tuple[dict, dict]:
    """Loads (challenges, solutions) for a dataset key of DATASETS."""
    challenges_file, solutions_file = DATASETS[name]
    return _read_json(DATA / challenges_file), _read_json(DATA / solutions_file)


def evaluate(
    challenges: dict,
    solutions: dict,
    *,
    limit: int | None = None,
    timeout: float | None = None,
    solver: Callable[..., list] = solve_arc_task,
) -> dict:
    """Runs the solver over a dataset and returns a report dict.

    `solver` is injectable so tests can drive this without the datasets or the real search.
    """
    task_ids = sorted(challenges)
    if limit:
        task_ids = task_ids[:limit]

    if timeout:
        signal.signal(signal.SIGALRM, _on_alarm)

    def _arm() -> None:
        # setitimer, not alarm(): alarm() takes whole seconds, so a fractional timeout would be
        # silently truncated to 0 and disable the guard altogether.
        if timeout:
            signal.setitimer(signal.ITIMER_REAL, timeout)

    def _disarm() -> None:
        signal.setitimer(signal.ITIMER_REAL, 0)

    total_outputs = 0
    correct_outputs = 0
    tasks_fully_solved = 0
    tasks_partly_solved = 0
    timeouts = 0
    errors = 0
    covered_tasks = 0
    matched_names: Counter[str] = Counter()
    candidate_counts: list[int] = []
    durations: list[float] = []
    misses = Counter()
    covered_ids: list[str] = []
    uncovered_ids: list[str] = []

    for task_id in task_ids:
        task = challenges[task_id]
        truth = solutions.get(task_id) or []
        trace: dict = {}
        t0 = time.perf_counter()
        try:
            _arm()
            attempts = solver(task, trace=trace)
            _disarm()
        except TaskTimeout:
            _disarm()
            timeouts += 1
            attempts = [{"attempt_1": t["input"], "attempt_2": t["input"]} for t in task["test"]]
            trace = {}
        except Exception as exc:  # noqa: BLE001
            _disarm()
            errors += 1
            attempts = [{"attempt_1": t["input"], "attempt_2": t["input"]} for t in task["test"]]
            trace = {}
            print(f"  ! {task_id}: {type(exc).__name__}: {exc}", file=sys.stderr)
        durations.append(time.perf_counter() - t0)

        if trace.get("n_candidates") is not None:
            candidate_counts.append(trace["n_candidates"])
        if trace.get("n_matching"):
            covered_tasks += 1
            covered_ids.append(task_id)
            for name in trace.get("matching", []):
                matched_names[name] += 1
        elif trace:
            uncovered_ids.append(task_id)

        task_hits = 0
        for i, ground_truth in enumerate(truth):
            total_outputs += 1
            got = attempts[i] if i < len(attempts) else {}
            a1, a2 = got.get("attempt_1"), got.get("attempt_2")
            if grids_equal(a1, ground_truth) or grids_equal(a2, ground_truth):
                correct_outputs += 1
                task_hits += 1
                continue
            # Only incorrect outputs reach here; classify why.
            inp = task["test"][i]["input"]
            if grids_equal(a1, inp) or grids_equal(a2, inp):
                misses["echoed the input"] += 1
            elif _shape_matches(a1, ground_truth) or _shape_matches(a2, ground_truth):
                misses["right shape, wrong cells"] += 1
            else:
                misses["wrong shape"] += 1

        if truth and task_hits == len(truth):
            tasks_fully_solved += 1
        elif task_hits:
            tasks_partly_solved += 1

    durations.sort()

    def pct(x: int, y: int) -> float:
        return round(100.0 * x / y, 4) if y else 0.0

    def percentile(values: list[float], q: float) -> float:
        if not values:
            return 0.0
        last = len(values) - 1
        rank = round(q * last)
        if rank < 0:
            rank = 0
        elif rank > last:
            rank = last
        return round(values[rank], 4)

    return {
        "tasks": len(task_ids),
        "test_outputs": total_outputs,
        "official_metric_pct": pct(correct_outputs, total_outputs),
        "correct_outputs": correct_outputs,
        "tasks_fully_solved": tasks_fully_solved,
        "tasks_fully_solved_pct": pct(tasks_fully_solved, len(task_ids)),
        "tasks_partly_solved": tasks_partly_solved,
        "tasks_failed": len(task_ids) - tasks_fully_solved - tasks_partly_solved,
        "coverage_tasks": covered_tasks,
        "coverage_pct": pct(covered_tasks, len(task_ids)),
        "timeouts": timeouts,
        "errors": errors,
        "candidates_per_task": {
            "min": min(candidate_counts) if candidate_counts else 0,
            "mean": round(statistics.fmean(candidate_counts), 1) if candidate_counts else 0.0,
            "max": max(candidate_counts) if candidate_counts else 0,
        },
        "latency_s": {
            "p50": percentile(durations, 0.50),
            "p95": percentile(durations, 0.95),
            "max": round(durations[-1], 4) if durations else 0.0,
            "total": round(sum(durations), 4),
        },
        "miss_reasons": dict(misses.most_common()),
        "matching_candidates": dict(matched_names.most_common(15)),
        "covered_task_ids": covered_ids[:50],
        "uncovered_task_ids_sample": uncovered_ids[:15],
    }


def format_report(name: str, report: dict) -> str:
    out = [f"\n=== {name} ==="]
    out.append(f"  official metric        {report['official_metric_pct']:>7.2f}%   "
               f"({report['correct_outputs']}/{report['test_outputs']} test outputs)")
    out.append(f"  task fully solved      {report['tasks_fully_solved_pct']:>7.2f}%   "
               f"({report['tasks_fully_solved']}/{report['tasks']})")
    out.append(f"  task partly solved     {report['tasks_partly_solved']:>7}     "
               f"failed {report['tasks_failed']}")
    out.append(f"  COVERAGE               {report['coverage_pct']:>7.2f}%   "
               f"({report['coverage_tasks']}/{report['tasks']} tasks have >=1 candidate that reproduces the train pairs)")
    c = report["candidates_per_task"]
    out.append(f"  candidates per task    min {c['min']}  mean {c['mean']}  max {c['max']}")
    lat = report["latency_s"]
    out.append(f"  latency                p50 {lat['p50']}s  p95 {lat['p95']}s  max {lat['max']}s  total {lat['total']}s")
    if report["timeouts"] or report["errors"]:
        out.append(f"  timeouts / errors      {report['timeouts']} / {report['errors']}")
    if report["miss_reasons"]:
        out.append("  why incorrect outputs missed:")
        for reason, n in report["miss_reasons"].items():
            out.append(f"      {reason:<26} {n}")
    if report["matching_candidates"]:
        out.append("  most frequent matching candidates:")
        for name_, n in list(report["matching_candidates"].items())[:8]:
            out.append(f"      {name_:<34} {n}")
    else:
        out.append("  most frequent matching candidates: none - the pool covered nothing")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", nargs="?", default="all", choices=["eval", "train", "all"])
    ap.add_argument("--limit", type=int, default=None, help="only the first N tasks of each dataset")
    ap.add_argument("--timeout", type=float, default=None, help="per-task seconds before falling back")
    ap.add_argument("--json", dest="json_out", default=None, help="also write the reports as JSON here")
    args = ap.parse_args(argv)

    names = ["eval", "train"] if args.dataset == "all" else [args.dataset]
    reports: dict[str, dict] = {}

    for name in names:
        challenges, solutions = load_dataset(name)
        print(f"running {name}: {len(challenges)} tasks"
              + (f" (limit {args.limit})" if args.limit else ""), flush=True)
        reports[name] = evaluate(challenges, solutions, limit=args.limit, timeout=args.timeout)
        print(format_report(name, reports[name]))

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(reports, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

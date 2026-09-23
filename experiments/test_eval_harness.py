#!/usr/bin/env python3
"""Tests for the ARC-AGI-2 evaluation harness.

Runs two ways:

    python3 experiments/test_eval_harness.py     # standalone, prints a summary
    pytest experiments/test_eval_harness.py      # as ordinary test functions

Everything here uses synthetic tasks and an injected fake solver, so the suite runs in milliseconds and
needs neither the ARC datasets nor the real search. Nothing in this file touches `data/`.

The point of these tests is that the harness's *arithmetic* is trustworthy: the metric definition, the
coverage count, and the classification of why an output missed. If the harness is wrong, every solver
conclusion drawn from it is wrong too.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_arc2 import _shape_matches, evaluate  # noqa: E402

GRID_A = [[1, 2], [3, 4]]
GRID_B = [[5, 6], [7, 8]]
GRID_WIDE = [[1, 2, 3], [4, 5, 6]]


def _task(n_test: int = 1) -> dict:
    return {"train": [{"input": GRID_A, "output": GRID_B}], "test": [{"input": GRID_A}] * n_test}


def _solver_returning(a1, a2, *, n_matching=1, n_candidates=10, delay=0.0):
    """Builds a fake solver with a fixed trace, so coverage paths can be driven deterministically."""

    def solver(task, trace=None):
        if delay:
            time.sleep(delay)
        if trace is not None:
            trace.update(
                n_candidates=n_candidates,
                n_matching=n_matching,
                matching=["fake"] * n_matching,
                used_fallback=not n_matching,
                reason="ok",
            )
        return [{"attempt_1": a1, "attempt_2": a2} for _ in task["test"]]

    return solver


# --- metric definition -------------------------------------------------------


def test_attempt_1_hit_scores_one():
    report = evaluate({"t": _task()}, {"t": [GRID_B]}, solver=_solver_returning(GRID_B, GRID_A))
    assert report["official_metric_pct"] == 100.0, report
    assert report["tasks_fully_solved"] == 1


def test_attempt_2_hit_also_scores_one():
    """The competition rule is 'either attempt matches', so attempt_2 alone must score."""
    report = evaluate({"t": _task()}, {"t": [GRID_B]}, solver=_solver_returning(GRID_A, GRID_B))
    assert report["official_metric_pct"] == 100.0, report


def test_both_attempts_wrong_scores_zero():
    report = evaluate({"t": _task()}, {"t": [GRID_B]}, solver=_solver_returning(GRID_A, GRID_WIDE))
    assert report["official_metric_pct"] == 0.0, report


def test_metric_averages_over_test_outputs_not_tasks():
    """A task with 2 test outputs and 1 hit must contribute 1/2, not 1/1."""
    challenges = {"t": {"train": [{"input": GRID_A, "output": GRID_B}],
                        "test": [{"input": GRID_A}, {"input": GRID_A}]}}
    solutions = {"t": [GRID_B, GRID_A]}
    report = evaluate(challenges, solutions, solver=_solver_returning(GRID_A, GRID_A))
    assert report["official_metric_pct"] == 50.0, report
    assert report["test_outputs"] == 2
    assert report["tasks_partly_solved"] == 1
    assert report["tasks_fully_solved"] == 0


# --- coverage ----------------------------------------------------------------


def test_coverage_counts_tasks_with_a_matching_candidate():
    report = evaluate({"t": _task()}, {"t": [GRID_B]}, solver=_solver_returning(GRID_A, GRID_A, n_matching=0))
    assert report["coverage_tasks"] == 0
    assert report["coverage_pct"] == 0.0
    assert report["uncovered_task_ids_sample"] == ["t"]


def test_coverage_is_independent_of_being_right():
    """A matching candidate that still gets the wrong answer is still coverage.

    This is the distinction that made the ARC-AGI-2 diagnosis legible: coverage explains the score,
    and here they deliberately disagree.
    """
    report = evaluate({"t": _task()}, {"t": [GRID_B]},
                      solver=_solver_returning(GRID_A, GRID_A, n_matching=3))
    assert report["coverage_tasks"] == 1
    assert report["official_metric_pct"] == 0.0
    assert report["matching_candidates"] == {"fake": 3}


def test_task_without_train_pairs_is_not_credited_as_coverage():
    challenges = {"t": {"train": [], "test": [{"input": GRID_A}]}}

    def solver(task, trace=None):
        if trace is not None:
            trace.update(n_candidates=0, n_matching=0, matching=[], used_fallback=True,
                         reason="no_train_pairs")
        return [{"attempt_1": GRID_A, "attempt_2": GRID_A}]

    report = evaluate(challenges, {"t": [GRID_A]}, solver=solver)
    assert report["coverage_tasks"] == 0
    assert report["official_metric_pct"] == 100.0


# --- miss classification -----------------------------------------------------


def test_miss_echoing_the_input_is_labelled():
    challenges = {"t": {"train": [{"input": GRID_A, "output": GRID_B}], "test": [{"input": GRID_A}]}}
    report = evaluate(challenges, {"t": [GRID_B]}, solver=_solver_returning(GRID_A, GRID_A))
    assert report["miss_reasons"] == {"echoed the input": 1}, report


def test_miss_with_right_shape_but_wrong_cells_is_labelled():
    near = [[1, 2], [3, 9]]
    report = evaluate({"t": _task()}, {"t": [GRID_B]}, solver=_solver_returning(near, near))
    assert report["miss_reasons"] == {"right shape, wrong cells": 1}, report


def test_miss_with_wrong_shape_is_labelled():
    report = evaluate({"t": _task()}, {"t": [GRID_B]}, solver=_solver_returning(GRID_WIDE, GRID_WIDE))
    assert report["miss_reasons"] == {"wrong shape": 1}, report


# --- robustness --------------------------------------------------------------


def test_solver_exception_is_counted_not_raised():
    def exploding(task, trace=None):
        raise ValueError("boom")

    report = evaluate({"t": _task()}, {"t": [GRID_B]}, solver=exploding)
    assert report["errors"] == 1
    assert report["official_metric_pct"] == 0.0


def test_timeout_falls_back_and_is_counted():
    report = evaluate({"t": _task()}, {"t": [GRID_B]},
                      timeout=0.2, solver=_solver_returning(GRID_B, GRID_B, delay=0.6))
    assert report["timeouts"] == 1
    assert report["official_metric_pct"] == 0.0


def test_limit_takes_the_first_n_sorted_tasks():
    challenges = {tid: {"train": [{"input": GRID_A, "output": GRID_B}], "test": [{"input": GRID_A}]}
                  for tid in ("ccc", "aaa", "bbb")}
    solutions = {tid: [GRID_B] for tid in challenges}
    report = evaluate(challenges, solutions, limit=2, solver=_solver_returning(GRID_B, GRID_B))
    assert report["tasks"] == 2, report


def test_shape_matches_helper():
    assert _shape_matches(GRID_A, GRID_B) is True
    assert _shape_matches(GRID_A, GRID_WIDE) is False
    assert _shape_matches(None, GRID_A) is False
    assert _shape_matches([], []) is False


def test_empty_dataset_does_not_divide_by_zero():
    report = evaluate({}, {}, solver=_solver_returning(GRID_A, GRID_A))
    assert report["official_metric_pct"] == 0.0
    assert report["coverage_pct"] == 0.0
    assert report["tasks"] == 0


def _main() -> int:
    tests = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_") and callable(fn)]
    failures = []
    for name, fn in tests:
        try:
            fn()
        except AssertionError as exc:
            failures.append((name, exc))
            print(f"  FAIL  {name}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failures.append((name, exc))
            print(f"  ERROR {name}: {type(exc).__name__}: {exc}")
        else:
            print(f"  ok    {name}")
    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())

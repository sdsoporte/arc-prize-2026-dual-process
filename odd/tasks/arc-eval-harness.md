# ODD Feature — `arc-eval-harness`

> **Status:** in progress
> **Branch:** `feat/arc-eval-harness`
> **Base commit:** `37d6100` (docs(odd): close kaggle-ops T5 and mark the feature complete)
> **Created:** 2026-09-23
> **Engram mirror topic key:** `arc-paper-track/odd/arc-eval-harness/tasks`

---

## 1. Goal

Give this repository **one trustworthy, tested way to measure solver quality**, so every future solver
change is measured instead of assumed. The harness must report not just the score but the number that
*explains* it: how many tasks the candidate pool actually covers.

## 2. Why now

A diagnostic on 2026-09-23 established the root cause of the ARC-AGI-2 leaderboard score of `0.00`:

| Measure | EVAL (120 public tasks) | TRAIN (200 of 1000) |
|---|---|---|
| Official metric (per test output) | **0.00% (0/172)** | 6.94% (15/216) |
| Tasks with >=1 candidate reproducing the train pairs | **0/120 (0.0%)** | 13/200 (6.5%) |
| Candidates built per task | 54-258 (avg 174) | 54-259 |

`solve_arc_task` builds a fixed pool of ~174 whole-grid operations and keeps only those reproducing
**100%** of the training pairs. On the public evaluation set **nothing survives, for every one of the 120
tasks**, so both attempts fall through to a fallback that echoes the input. Median latency is `0.01 s`.

Three consequences that make a harness the mandatory next step:

1. The writeup's claim *"synthesizing rules for 17.9% of benchmark tasks"* is **not reproducible**. The
   measured value is 6.94% on train and 0.0% on the public evaluation set, and **no source for `17.9`
   exists anywhere in this repository** (`grep -rn 17.9` returns only vendored noise).
2. `experiments/benchmark_arc2_suite.py` already existed and independently produced the same numbers
   (`0/120`, `13/200`) - but **its output was never recorded anywhere**, and it measures a different,
   stricter metric (task fully solved) than the official one.
3. The never-deployed 666-line DSL improvement was previously flagged as a risk. It is now known to be
   **irrelevant to the score**: that solver has 0% coverage on eval. Work in that direction was about to
   be wasted.

Without a canonical instrument, the next solver change repeats the same cycle of guessing.

## 3. Non-goals

- Not a solver **improvement**. One primitive was repaired because it could never execute at all (see §8);
  no primitive was added and no search strategy changed. The repair's effect is measured, not assumed.
- Not an ARC-AGI-3 harness. Scope is the ARC-AGI-2 program-synthesis path; a future feature may generalise.
- Not a leaderboard predictor. The public evaluation set is a proxy; the hidden test set is unavailable.
- Does not fix the false `17.9%` claim in the paper. That is recorded here as a risk for the paper feature.

## 4. Constraints

| Constraint | Value | Source |
|---|---|---|
| Public eval set | 120 tasks, 172 test outputs, with solutions | `data/arc-agi-2/` |
| Training set | 1000 tasks, with solutions | `data/arc-agi-2/` |
| Local interpreter | Python 3.14.4, `numpy` 2.5.3, `pytest` 9.1.1 available | local probe |
| Test convention | no pytest config exists; `experiments/test_*.py` are standalone scripts. New tests should run both ways | existing tree |
| Solver is inlined into the submission kernel | `src/arc2_dual_process_solver.py` is copied into the ARC-AGI-2 kernel notebook, so changes must stay backward compatible | `notebooks/arc2_submission_kernel/` |
| Runtime | full 120-task eval ran in ~3.4 s | measured |

## 5. Allowed edit surfaces

- `experiments/eval_arc2.py`
- `experiments/benchmark_arc2_suite.py`
- `experiments/test_eval_harness.py`
- `src/arc2_dual_process_solver.py`
- `odd/tasks/arc-eval-harness.md`

## 6. Tasks

| # | Task | Status | Evidence |
|---|---|---|---|
| T1 | Scaffold ODD tracking: branch, feature doc, Engram mirror, todo | done | this file; branch `feat/arc-eval-harness` |
| T2 | Add a non-breaking trace hook to `solve_arc_task` | done | regression unchanged; `007bbfb7` -> `n_matching=1` (`kronecker_self`), `0934a4d8` -> `n_matching=0` |
| T3 | Implement `experiments/eval_arc2.py` and supersede `benchmark_arc2_suite.py` | done | run output; `experiments/arc2_baseline.json` recorded |
| T4 | Add `experiments/test_eval_harness.py` covering the scoring and coverage logic | done | 15/15 standalone and under `pytest` |
| T5 | Independent verification pass | pending | verifier report |

## 7. Acceptance criteria

1. One command reports: official metric, task-level solve rate, coverage (tasks with >=1 matching solver),
   candidate-pool size distribution, latency percentiles, and a failure breakdown.
2. Coverage is measured from the solver's real internals, not by reimplementing its candidate logic.
3. The harness reproduces the recorded baseline in `experiments/arc2_baseline.json`: eval `0/120`, train
   `0.00%` and `42/1000` fully solved (post-repair values; the pre-repair train figure was `41/1000`).
4. The existing callers of `solve_arc_task` keep working unchanged.
5. The scoring and coverage logic are covered by a test that runs both as a plain script and under `pytest`.

## 8. Evidence log

- 2026-09-23 - diagnostic run (instrumented copy in `/tmp`, repo untouched): `0.00%` on eval, `6.94%` on
  200 train tasks, coverage `0/120` and `13/200`.
- 2026-09-23 - **cross-validation**: `experiments/benchmark_arc2_suite.py` independently reported
  `Solved 0/120 (0.00%)` and `Solved 13/200 (6.50%)`, matching the diagnostic. Two independent
  measurements agree, so the zero is real and not an instrument artefact.
- 2026-09-23 - positive control: on the 200 train tasks the harness scores `6.94%`, and the covered task
  ids are ARC-AGI-1 classics (`007bbfb7`, `0d3d703e`, `1cf80156`, `1e0a9b12`), confirming the metric
  computes rather than always returning zero.
- 2026-09-23 - `grep -rn "17.9"` finds no source for the writeup's coverage claim.
- 2026-09-23 - **Defect found while adding the trace hook: `rot270` could never execute.**
  `get_d4_ops()` defined it as `zip(*g)[::-1]`, which raises
  `TypeError: 'zip' object is not subscriptable` because a `zip` object is not subscriptable. The solver
  swallows every candidate exception with `except Exception: continue`, so the primitive was silently
  absent from both the direct pool and the two-stage spatial pool. Fixed to `[list(x) for x in zip(*g)][::-1]`
  and verified algebraically (`rot270 == rot90` applied three times).
- 2026-09-23 - **The repair's effect, measured rather than assumed** (same harness, solver swapped in):

  | dataset | variant | official % | fully solved | coverage |
  |---|---|---|---|---|
  | eval | pre-repair | 0.0000 | 0 | 0/120 |
  | eval | post-repair | 0.0000 | 0 | 0/120 |
  | train | pre-repair | 4.0892 | 41 | 42/1000 |
  | train | post-repair | 4.1822 | 42 | 43/1000 |

  So the repair recovers exactly **one task in 1000** (`+0.093 pp`) and nothing on eval, and `rot270`
  never appears among the winning candidates. A real bug with a near-zero effect - which is itself
  evidence that the primitive pool is not the binding constraint.
- 2026-09-23 - **Defect found by the new tests: fractional timeouts were silently disabled.** The harness
  used `signal.alarm(int(timeout))`, so any timeout below one second became `alarm(0)`, which cancels the
  alarm. `test_timeout_falls_back_and_is_counted` failed and exposed it; switched to
  `signal.setitimer(signal.ITIMER_REAL, timeout)`, which accepts fractional seconds.
- 2026-09-23 - **Full-set measurement overturns the 200-task subsample.** Earlier reconnaissance reported
  `6.94%` and `13/200` on the first 200 training tasks. Over all 1000 training tasks the metrics are
  `4.18%` official and `42/1000` fully solved: the first 200 are an easier-than-average prefix and
  overstate coverage by about 50%.
- 2026-09-23 - `experiments/benchmark_arc2_suite.py` removed. It had independently produced the matching
  `0/120` and `13/200` figures used as cross-validation, but its output was never recorded and it reported
  only the stricter task-level metric, which the harness now also reports.
- 2026-09-23 - `experiments/arc2_baseline.json` committed, so this measurement can never again be lost.
- 2026-09-23 - the winning candidate names show the only productive structure in the pool is
  `<spatial op> + color_map` (`largest_object+color_map` x7, `color_map` x5, `identity+color_map` x5,
  `crop_nonzero+color_map` x5, `gravity_*+color_map` x5 each). Single whole-grid ops contribute little.

## 9. Known risks carried (not fixed by this feature)

| Risk | Impact | Owner feature |
|---|---|---|
| Writeup claims 17.9% rule synthesis; measured 0.0% on eval | Accuracy / Completeness rubric | paper refresh |
| Solver coverage is 0% on ARC-AGI-2 eval; ARC-AGI-2 from 0.00 is a research problem | Accuracy rubric | solver strategy (undecided) |
| No submission kernel runs the Laya System 1 | Theory / Completeness rubric | solver integration |
| No `pytest` configuration exists in the repository | tooling | environment bootstrap |

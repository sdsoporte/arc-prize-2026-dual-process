#!/usr/bin/env python3
"""Paired strategy probe for the ARC-AGI-3 agent.

The question
------------
Two strategy changes are suggested by recorded *human* play
(``docs/ARC3_REPLAYS_FINDINGS.md``), not by intuition:

* **H1 - click locality.** Humans use ACTION6 29.7% of the time (the most of any
  action) and their click targets are spatially concentrated; the top 8
  destinations are ~18% of all clicks. Our agent clicks uniformly at random over
  the whole 64x64 grid and weights ACTION6 down to 0.5 against a movement weight
  of 4.0.
* **H2 - repetition.** Human play is strongly autocorrelated (``repPrev`` 36-65%
  in 15 of 25 games, ``maxRun`` up to 18) and they almost never undo, while our
  agent multiplies the weight of the last action by 0.1 while oscillating.

This probe asks a deliberately small question: **does either change move the
score by more than the agent's own seed noise?** It is an instrument, not a
strategy decision. It measures; it does not pick.

Why the design is paired
------------------------
The agent's score is dominated by its RNG. With the seed fixed, 13 sweeps of
identical code spread over **0.173 to 1.090** (6.3x). Comparing two variants on
one seed says nothing, and comparing means of independent runs needs far more
samples than are affordable. Comparing variants **at the same seed** cancels the
seed effect: the paired delta is the informative quantity, and a consistent sign
across seeds is evidence where an absolute number is not.

So the probe runs a factorial - several seeds x five arms - and reports the
per-seed paired delta of every non-baseline arm against ``baseline`` rather than
only the means.

Five arms
---------
==================  ==========================================================
``baseline``        the unmodified :class:`MyAgent`
``click_local``     ACTION6 targets near the agent's tracked position
``click_reuse``     ACTION6 mostly re-uses coordinates already clicked
``click_weight``    ``CLICK_WEIGHT = 1.5`` instead of 0.5
``no_rep_penalty``  ``REPEAT_PENALTY = 1.0`` (no repetition penalty)
==================  ==========================================================

The three seams the arms override (``CLICK_WEIGHT``, ``REPEAT_PENALTY`` and
``_choose_click_coords``) are behaviour-preserving additions to ``MyAgent``: the
baseline arm runs the real class, and the shipped default seed must still score
0.2017 / 3-of-183 (see the module docstring of ``arc3_local_eval``).

HARD SAFETY RULE - OFFLINE, and never against the tracked environment files
---------------------------------------------------------------------------
The engine is constructed in ``OperationMode.OFFLINE`` only, with
``environments_dir`` pointed at a copy under ``/tmp``. An ``arc_agi`` run in
``OperationMode.NORMAL`` silently re-downloads the game sources and **overwrites**
the tracked files under ``data/arc-agi-3/environment_files``; that already
happened once. The copy is resolved through
``arc3_calibration.resolve_env_dir``, which refuses any directory inside the
repository's tracked ``environment_files``, and this module additionally refuses
anything that does not resolve under ``/tmp``. Verify with::

    git -C <repo> status --short -- data/arc-agi-3/environment_files/

Reuse, not reimplementation
---------------------------
The sweep and the scorer come from ``experiments/arc3_local_eval.py`` verbatim
(``build_arcade``, ``collect_plans``, ``run_sweep``, ``aggregate``,
``load_agent_class``). This module only wires the arms, the seed loop, and the
paired-delta analysis around them.

Usage
-----
    data/arc-agi-3-agents/.venv/bin/python experiments/arc3_strategy_probe.py
    data/arc-agi-3-agents/.venv/bin/python experiments/arc3_strategy_probe.py \\
        --seeds 1,2,3,4,5,6,7,8 --json experiments/arc3_probe_results.json --verbose

The interpreter matters: ``arc_agi`` / ``arcengine`` are installed in
``data/arc-agi-3-agents/.venv`` (Python 3.12); a system Python 3.14 cannot import
them.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import statistics
import sys
import tempfile
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

EXPERIMENTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENTS_DIR.parent
# Same sibling-module import pattern the test suite uses: put ``experiments/`` on
# the path so ``arc3_local_eval`` resolves without turning the directory into a
# package.
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

import arc3_calibration as calibration  # noqa: E402
import arc3_local_eval as local_eval  # noqa: E402

SCHEMA = "arc3-strategy-probe/1"
AGENT_MODULE = "src/arc3_spatial_memory_agent.py"
DEFAULT_SEEDS = (1, 2, 3, 4, 5, 6)
DEFAULT_JSON = EXPERIMENTS_DIR / "arc3_probe_results.json"
# The reviewed /tmp copy the calibration benchmark populates from the repository
# source; populated here too if missing, never the repository directory itself.
DEFAULT_ENV_DIR = "/tmp/arc3-env"  # noqa: S108

BASELINE_ARM = "baseline"

ARM_NAMES: tuple[str, ...] = (
    "baseline",
    "click_local",
    "click_reuse",
    "click_weight",
    "no_rep_penalty",
)

ARM_DESCRIPTIONS: dict[str, str] = {
    "baseline": "the unmodified MyAgent",
    "click_local": "ACTION6 targets near the agent's tracked (pos_x, pos_y), clamped to 0..63",
    "click_reuse": "ACTION6 mostly re-uses coordinates already clicked, explores a new one otherwise",
    "click_weight": "CLICK_WEIGHT = 1.5 (default 0.5)",
    "no_rep_penalty": "REPEAT_PENALTY = 1.0 (default 0.1, i.e. no repetition penalty)",
}


# --------------------------------------------------------------------------- #
# Small coercions (kept in a try so the ast-grep "unchecked throwing call" rule is
# satisfied, matching arc3_local_eval.py and arc3_calibration.py)
# --------------------------------------------------------------------------- #


def _as_int(value: Any, default: int = 0) -> int:
    """Coerce an aggregate value to ``int``, falling back instead of raising."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value: Any, default: float = 0.0) -> float:
    """Coerce an aggregate value to ``float``, falling back instead of raising."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _fmt(value: Any, width: int = 9) -> str:
    """Render a possibly-missing delta as a fixed-width number (``nan`` if absent)."""
    return f"{_as_float(value, math.nan):{width}.4f}"


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #


@dataclass
class RunResult:
    """Aggregate outcome of one arm at one seed (the whole 25-game sweep)."""

    arm: str
    seed: int
    score: float
    levels_completed: int
    levels_available: int
    games_errored: int
    failed_games: list[str] = field(default_factory=list)
    wall_time_s: float = 0.0
    error: str | None = None


# --------------------------------------------------------------------------- #
# Arm definitions (subclasses of the real MyAgent)
# --------------------------------------------------------------------------- #


def build_arm_classes(agent_class: type) -> dict[str, type]:
    """Build the five arms as subclasses of the loaded ``MyAgent``.

    Subclassing keeps the arms measuring the *real* ``choose_action`` /
    ``_record_action``; each arm overrides only the one seam it names.
    """

    class ClickLocalAgent(agent_class):  # type: ignore[misc, valid-type]
        """ACTION6 near the tracked agent position instead of uniform over the grid."""

        LOCAL_RADIUS = 4

        def _choose_click_coords(self) -> tuple[int, int]:
            dx = self.rng.randint(-self.LOCAL_RADIUS, self.LOCAL_RADIUS)
            dy = self.rng.randint(-self.LOCAL_RADIUS, self.LOCAL_RADIUS)
            return (
                min(63, max(0, self.pos_x + dx)),
                min(63, max(0, self.pos_y + dy)),
            )

    class ClickReuseAgent(agent_class):  # type: ignore[misc, valid-type]
        """ACTION6 mostly re-uses coordinates already clicked, occasionally explores."""

        REUSE_PROBABILITY = 0.8

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self._clicked: list[tuple[int, int]] = []
            self._clicked_set: set[tuple[int, int]] = set()

        def _choose_click_coords(self) -> tuple[int, int]:
            if self._clicked and self.rng.random() < self.REUSE_PROBABILITY:
                return self.rng.choice(self._clicked)
            coords = (self.rng.randint(0, 63), self.rng.randint(0, 63))
            if coords not in self._clicked_set:
                self._clicked_set.add(coords)
                self._clicked.append(coords)
            return coords

    class ClickWeightAgent(agent_class):  # type: ignore[misc, valid-type]
        """ACTION6 weighted 1.5 in the heuristic pool instead of 0.5."""

        CLICK_WEIGHT = 1.5

    class NoRepPenaltyAgent(agent_class):  # type: ignore[misc, valid-type]
        """No repetition penalty while oscillating."""

        REPEAT_PENALTY = 1.0

    return {
        "baseline": agent_class,
        "click_local": ClickLocalAgent,
        "click_reuse": ClickReuseAgent,
        "click_weight": ClickWeightAgent,
        "no_rep_penalty": NoRepPenaltyAgent,
    }


# --------------------------------------------------------------------------- #
# Seed plumbing
# --------------------------------------------------------------------------- #


@contextlib.contextmanager
def seeded(seed: int) -> Iterator[None]:
    """Set ``ARC3_AGENT_SEED`` for one run and restore the previous value after.

    ``MyAgent.__init__`` reads the variable through ``_resolve_seed``, so setting
    it around the sweep pins the agent's RNG without touching global state.
    """
    previous = os.environ.get("ARC3_AGENT_SEED")
    os.environ["ARC3_AGENT_SEED"] = str(seed)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("ARC3_AGENT_SEED", None)
        else:
            os.environ["ARC3_AGENT_SEED"] = previous


def parse_seeds(text: str) -> list[int]:
    """Parse a comma-separated seed list, rejecting empty input."""
    seeds: list[int] = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            seeds.append(int(chunk))
        except ValueError:
            raise ValueError(f"seed {chunk!r} is not an integer") from None
    if not seeds:
        raise ValueError("no seeds given (expected e.g. 1,2,3,4,5,6)")
    return seeds


# --------------------------------------------------------------------------- #
# One arm x one seed
# --------------------------------------------------------------------------- #


def run_one(
    arm: str,
    agent_class: type,
    plans: Sequence[Any],
    seed: int,
) -> RunResult:
    """Run one full sweep for ``arm`` at ``seed``; never raise.

    ``run_sweep`` already turns every per-game failure into a report row, so the
    only way to reach the ``except`` here is an arm-level failure (arcade setup,
    agent construction). That is returned as a named, errored result so the
    factorial keeps going.
    """
    started = time.monotonic()
    try:
        arcade = local_eval.build_arcade()

        def make_env(plan: Any) -> Any:
            return arcade.make(plan.game_id)

        def make_agent(plan: Any, env: Any) -> Any:
            return agent_class(
                card_id="strategy-probe",
                game_id=plan.game_id,
                agent_name=f"{agent_class.__name__}.{plan.game_id}",
                ROOT_URL="http://localhost",
                record=False,
                arc_env=env,
                tags=["strategy-probe"],
            )

        with seeded(seed):
            games = local_eval.run_sweep(
                plans,
                make_env=make_env,
                make_agent=make_agent,
                time_budget_s=local_eval.DEFAULT_TIME_BUDGET_S,
                clock=time.monotonic,
            )
    except Exception as exc:  # noqa: BLE001 - one bad arm/seed must not abort the factorial
        return RunResult(
            arm=arm,
            seed=seed,
            score=0.0,
            levels_completed=0,
            levels_available=0,
            games_errored=0,
            wall_time_s=round(time.monotonic() - started, 3),
            error=f"{type(exc).__name__}: {exc}",
        )

    aggregate = local_eval.aggregate(games)
    failed_games = [game["game_id"] for game in games if game["status"] == "error"]
    return RunResult(
        arm=arm,
        seed=seed,
        score=_as_float(aggregate.get("mean_score")),
        levels_completed=_as_int(aggregate.get("total_levels_completed")),
        levels_available=_as_int(aggregate.get("total_levels_available")),
        games_errored=_as_int(aggregate.get("games_errored")),
        failed_games=failed_games,
        wall_time_s=round(time.monotonic() - started, 3),
    )


# --------------------------------------------------------------------------- #
# Paired-delta analysis
# --------------------------------------------------------------------------- #


def summarize_deltas(values: Sequence[float]) -> dict[str, Any]:
    """Sign counts, mean/median, and range for a list of paired deltas."""
    if not values:
        return {
            "n": 0,
            "positive": 0,
            "negative": 0,
            "zero": 0,
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
        }
    return {
        "n": len(values),
        "positive": sum(1 for value in values if value > 0),
        "negative": sum(1 for value in values if value < 0),
        "zero": sum(1 for value in values if value == 0),
        "mean": round(statistics.fmean(values), 6),
        "median": round(statistics.median(values), 6),
        "min": round(min(values), 6),
        "max": round(max(values), 6),
    }


def build_deltas(
    results: dict[str, dict[int, RunResult]],
    seeds: Sequence[int],
    arms: Sequence[str],
) -> dict[str, dict[str, Any]]:
    """Per-seed paired delta of every non-baseline arm against baseline.

    A pair whose arm or baseline run errored is reported as ``None`` with the
    reason, and contributes to no sign count: a failed run is not a zero.
    """
    deltas: dict[str, dict[str, Any]] = {}
    for arm in arms:
        if arm == BASELINE_ARM:
            continue
        per_seed: dict[str, Any] = {}
        values: list[float] = []
        for seed in seeds:
            arm_run = results[arm][seed]
            base_run = results[BASELINE_ARM][seed]
            if arm_run.error or base_run.error:
                per_seed[str(seed)] = {
                    "delta": None,
                    "reason": arm_run.error or base_run.error,
                }
                continue
            delta = round(arm_run.score - base_run.score, 6)
            per_seed[str(seed)] = {"delta": delta, "reason": None}
            values.append(delta)
        deltas[arm] = {"per_seed": per_seed, **summarize_deltas(values)}
    return deltas


def seed_noise(baseline_scores: dict[int, float]) -> dict[str, Any]:
    """The baseline arm's own spread across the seeds: the scale to judge against."""
    values = list(baseline_scores.values())
    spread = max(values) - min(values) if values else 0.0
    return {
        "min": round(min(values), 6) if values else None,
        "max": round(max(values), 6) if values else None,
        "range": round(spread, 6),
        "mean": round(statistics.fmean(values), 6) if values else None,
        "stdev": round(statistics.pstdev(values), 6) if len(values) > 1 else 0.0,
    }


def assess_effects(
    deltas: dict[str, dict[str, Any]], noise: dict[str, Any]
) -> dict[str, Any]:
    """Conservative, transparent verdict per arm and overall.

    An arm is called a *candidate signal* only when **every** informative seed moves
    in the same direction, at least half of all seeds are informative, and the mean
    delta exceeds the baseline's own seed-to-seed standard deviation. Otherwise it
    is reported as not distinguishable from seed noise at this sample size. The raw
    numbers are always printed alongside so the reader can disagree with the rule.
    """
    threshold = _as_float(noise.get("stdev"), 0.0)
    per_arm: dict[str, str] = {}
    candidates: list[str] = []
    for arm, data in deltas.items():
        n = data["n"]
        positive = data["positive"]
        negative = data["negative"]
        informative = positive + negative
        # At least three seeds must be paired, every informative seed must point the
        # same way, and at least half of them must be informative at all. Fewer than
        # three seeds cannot separate a change from noise.
        consistent = (
            n >= 3
            and informative > 0
            and (positive == informative or negative == informative)
            and informative >= max(2, n // 2)
        )
        magnitude = abs(_as_float(data.get("mean"), 0.0))
        if consistent and magnitude > threshold:
            per_arm[arm] = "candidate signal"
            candidates.append(arm)
        else:
            per_arm[arm] = "not distinguishable from seed noise"
    if candidates:
        overall = (
            "At least one arm shows a sign-consistent paired delta larger than the "
            f"baseline seed spread ({threshold:.4f}): {', '.join(candidates)}. Treat "
            "it as a candidate, not a result, at this sample size."
        )
    else:
        overall = (
            "No arm's paired delta is distinguishable from seed noise at this sample "
            f"size (baseline seed-to-seed stdev {threshold:.4f}). This is a null "
            "result, and an honest one."
        )
    return {"threshold_stdev": threshold, "per_arm": per_arm, "overall": overall}


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #


def format_scores_table(seeds: Sequence[int], arms: Sequence[str], per_arm: dict[str, dict[int, float]]) -> str:
    header = f"{'arm':16} | " + " ".join(f"{'s' + str(seed):>10}" for seed in seeds)
    lines = [header, "-" * len(header)]
    for arm in arms:
        cells = " ".join(f"{per_arm[arm][seed]:10.4f}" for seed in seeds)
        lines.append(f"{arm:16} | {cells}")
    return "\n".join(lines)


def format_delta_table(seeds: Sequence[int], deltas: dict[str, dict[str, Any]]) -> str:
    header = (
        f"{'arm':16} | "
        + " ".join(f"{'s' + str(seed):>10}" for seed in seeds)
        + f" | {'+ve':>3} {'-ve':>3} {'0':>3} | {'mean':>9} {'median':>9} "
        f"{'min':>9} {'max':>9}"
    )
    lines = [header, "-" * len(header)]
    for arm, data in deltas.items():
        cells = []
        for seed in seeds:
            value = data["per_seed"][str(seed)]["delta"]
            cells.append(f"{value:10.4f}" if value is not None else f"{'n/a':>10}")
        lines.append(
            f"{arm:16} | "
            + " ".join(cells)
            + f" | {data['positive']:>3} {data['negative']:>3} {data['zero']:>3} | "
            f"{_fmt(data['mean'])} {_fmt(data['median'])} {_fmt(data['min'])} {_fmt(data['max'])}"
        )
    return "\n".join(lines)


def format_arm_table(arms: Sequence[str], results: dict[str, dict[int, RunResult]]) -> str:
    header = f"{'arm':16} | {'mean_score':>10} | {'levels':>9} | {'errored':>7}"
    lines = [header, "-" * len(header)]
    for arm in arms:
        runs = list(results[arm].values())
        mean_score = statistics.fmean(run.score for run in runs) if runs else 0.0
        levels = sum(run.levels_completed for run in runs)
        available = sum(run.levels_available for run in runs)
        errored = sum(run.games_errored for run in runs) + sum(
            1 for run in runs if run.error
        )
        lines.append(
            f"{arm:16} | {mean_score:10.4f} | {levels:>4}/{available:<4} | {errored:>7}"
        )
    return "\n".join(lines)


def build_report(
    *,
    seeds: Sequence[int],
    arms: Sequence[str],
    results: dict[str, dict[int, RunResult]],
    deltas: dict[str, dict[str, Any]],
    noise: dict[str, Any],
    assessment: dict[str, Any],
    env_dir: str,
    wall_time_s: float,
    verbose: bool,
) -> dict[str, Any]:
    runs = []
    for arm in arms:
        for seed in seeds:
            run = results[arm][seed]
            runs.append(
                {
                    "arm": arm,
                    "seed": seed,
                    "score": round(run.score, 6),
                    "levels_completed": run.levels_completed,
                    "levels_available": run.levels_available,
                    "games_errored": run.games_errored,
                    "failed_games": run.failed_games,
                    "wall_time_s": run.wall_time_s,
                    "error": run.error,
                }
            )
    per_arm = {}
    for arm in arms:
        scores = {seed: round(results[arm][seed].score, 6) for seed in seeds}
        per_arm[arm] = {
            "description": ARM_DESCRIPTIONS[arm],
            "per_seed_scores": {str(seed): scores[seed] for seed in seeds},
            "mean_score": round(statistics.fmean(scores.values()), 6),
            "total_levels_completed": sum(
                results[arm][seed].levels_completed for seed in seeds
            ),
            "total_levels_available": sum(
                results[arm][seed].levels_available for seed in seeds
            ),
        }
    return {
        "schema": SCHEMA,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "operation_mode": "OFFLINE",
        "interpreter": sys.executable,
        "agent_module": AGENT_MODULE,
        "environments_dir": env_dir,
        "seeds": list(seeds),
        "arms": list(arms),
        "arm_descriptions": ARM_DESCRIPTIONS,
        "verbose": verbose,
        "wall_time_s": round(wall_time_s, 3),
        "per_arm": per_arm,
        "deltas": deltas,
        "seed_noise": noise,
        "assessment": assessment,
        "runs": runs,
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Paired strategy probe for the ARC-AGI-3 agent (OFFLINE only).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--seeds",
        default=",".join(str(seed) for seed in DEFAULT_SEEDS),
        help=(
            "Comma-separated integer seeds; every arm runs the whole list "
            f"(default: {','.join(str(seed) for seed in DEFAULT_SEEDS)})."
        ),
    )
    parser.add_argument(
        "--json",
        default=str(DEFAULT_JSON),
        help=f"Write the full report to PATH (default: {DEFAULT_JSON}).",
    )
    parser.add_argument(
        "--env-dir",
        default=DEFAULT_ENV_DIR,
        help=(
            "OFFLINE environments_dir to load the local builds from; must be a copy "
            f"under /tmp (default: {DEFAULT_ENV_DIR})."
        ),
    )
    parser.add_argument(
        "--refresh-env-dir",
        action="store_true",
        help="Re-copy the repository source into --env-dir before running (no deletion).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show the framework's per-action INFO logs (suppressed by default).",
    )
    return parser.parse_args(argv)


def _refuse_unsafe_env_dir(env_dir: Path) -> str | None:
    """Return a refusal message if ``env_dir`` is not a /tmp copy, else ``None``.

    Belt and braces on top of ``arc3_calibration.resolve_env_dir``: the engine must
    never be handed a directory the calibration copy logic considers tracked.
    """
    resolved = env_dir.resolve()
    for forbidden in calibration.FORBIDDEN_ENV_DIRS:
        if resolved == forbidden or forbidden in resolved.parents:
            return (
                f"refusing environments_dir={resolved}: inside the repository's tracked "
                "environment_files. Use a copy under /tmp."
            )
    temp_root = Path(tempfile.gettempdir()).resolve()
    if resolved != temp_root and temp_root not in resolved.parents:
        return (
            f"refusing environments_dir={resolved}: not under {temp_root}. The engine "
            "must read from a /tmp copy."
        )
    return None


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        seeds = parse_seeds(args.seeds)
    except ValueError as exc:
        print(f"setup error: {exc}", file=sys.stderr)
        return 2

    local_eval.configure_logging(args.verbose)

    if not local_eval.ARC_AGI_AVAILABLE:
        print(
            "arc_agi is not importable with this interpreter "
            f"({local_eval.ARC_AGI_IMPORT_ERROR or 'unknown reason'}). Run with "
            "data/arc-agi-3-agents/.venv/bin/python.",
            file=sys.stderr,
        )
        return 2

    env_dir, env_error = calibration.resolve_env_dir(
        Path(args.env_dir), calibration.ENV_SOURCE, args.refresh_env_dir
    )
    if env_dir is None:
        print(f"setup error: {env_error}", file=sys.stderr)
        return 2
    refusal = _refuse_unsafe_env_dir(env_dir)
    if refusal is not None:
        print(f"setup error: {refusal}", file=sys.stderr)
        return 2
    # Point the reused harness at the /tmp copy. The operation mode stays OFFLINE
    # because ``build_arcade`` hard-codes it and never exposes a switch.
    local_eval.ENVIRONMENTS_DIR = env_dir

    try:
        agent_class = local_eval.load_agent_class()
    except Exception as exc:  # noqa: BLE001 - setup failure must not traceback
        print(f"setup error: could not load the agent: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    arms = list(ARM_NAMES)
    arm_classes = build_arm_classes(agent_class)

    arcade = local_eval.build_arcade()
    plans = local_eval.collect_plans(arcade)
    if not plans:
        print(f"setup error: no games found under {env_dir}", file=sys.stderr)
        return 2

    print("ARC-AGI-3 paired strategy probe (OFFLINE)")
    print(f"env_dir: {env_dir}   games: {len(plans)}   seeds: {seeds}")
    print(f"arms: {', '.join(arms)}\n")

    started = time.monotonic()
    results: dict[str, dict[int, RunResult]] = {arm: {} for arm in arms}
    for arm in arms:
        for seed in seeds:
            run = run_one(arm, arm_classes[arm], plans, seed)
            results[arm][seed] = run
            detail = f"error={run.error}" if run.error else f"score={run.score:.4f}"
            failed = f" failed_games={run.failed_games}" if run.failed_games else ""
            print(
                f"  [{arm:14}] seed={seed:<3} {detail} "
                f"levels={run.levels_completed}/{run.levels_available} "
                f"({run.wall_time_s:.2f}s){failed}",
                flush=True,
            )

    per_arm_scores = {
        arm: {seed: results[arm][seed].score for seed in seeds} for arm in arms
    }
    deltas = build_deltas(results, seeds, arms)
    noise = seed_noise(per_arm_scores[BASELINE_ARM])
    assessment = assess_effects(deltas, noise)

    print("\nPer-seed scores by arm")
    print(format_scores_table(seeds, arms, per_arm_scores))

    print("\nPaired delta vs baseline at the same seed")
    print(format_delta_table(seeds, deltas))

    print("\nPer-arm own result (summed over seeds)")
    print(format_arm_table(arms, results))

    print(
        "\nSeed noise (baseline arm across seeds): "
        f"min={noise['min']:.4f} max={noise['max']:.4f} range={noise['range']:.4f} "
        f"mean={noise['mean']:.4f} stdev={noise['stdev']:.4f}"
    )
    print("\nVerdict per arm")
    for arm, verdict in assessment["per_arm"].items():
        print(f"  {arm:16} {verdict}")
    print(f"\nConclusion: {assessment['overall']}")

    report = build_report(
        seeds=seeds,
        arms=arms,
        results=results,
        deltas=deltas,
        noise=noise,
        assessment=assessment,
        env_dir=str(env_dir),
        wall_time_s=time.monotonic() - started,
        verbose=args.verbose,
    )
    json_path = Path(args.json)
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

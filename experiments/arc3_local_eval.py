#!/usr/bin/env python3
"""Local ARC-AGI-3 scoring harness.

Why this exists
---------------
ARC-AGI-2/3 allow a single leaderboard submission per day and the code freezes on
2026-11-02, so the agent needs a *local* number to iterate against. The official
starter's ``scripts/play_local.py`` already plays all 25 games in-process, but its
final score is unusable offline: ``arc.get_scorecard()`` reports
``scorecard_id: None`` and ``Aggregate scorecard score: 0.0`` because scorecards
are API-backed. ``arc_agi.EnvironmentScoreCalculator`` is the official scorer and
*does* work locally. This module wires that scorer into the same episode loop and
prints per-game and aggregate numbers.

Provenance of the episode loop
------------------------------
The driver below mirrors, line for line, the two upstream sources that
``scripts/play_local.py`` relies on:

* ``data/arc-agi-3-kaggle-starter/scripts/play_local.py`` -- how each game is
  built (``Arcade(...).make(game_id)``), how the agent is constructed, how all
  games are swept by default, the ``--game`` comma filter, and the ``--max-steps``
  action cap. That file lives in a **gitignored** directory, so it is deliberately
  *not* imported: depending on it would reproduce the build-path drift this feature
  exists to document, and the loop is only a few lines.
* ``data/arc-agi-3/ARC-AGI-3-Agents/agents/agent.py::Agent.main()`` -- the loop
  body (``is_done`` / ``choose_action`` / ``take_action`` / ``append_frame`` /
  ``action_counter`` / ``cleanup``) is a verbatim copy, with exactly one addition:
  a per-action deadline check, because ``Agent.main()`` is a blocking call with no
  hook for a per-game time budget. Without it a pathological game stalls the whole
  sweep.

HARD SAFETY RULE: OFFLINE mode only
-----------------------------------
The ARC-AGI-3 engine in ``OperationMode.NORMAL`` silently re-downloads the game
sources and **overwrites** the tracked files under
``data/arc-agi-3/environment_files/``. That is why this harness hard-codes
``OperationMode.OFFLINE`` and never exposes a mode switch: a measurement tool that
mutates the thing it measures is not a measurement tool. Verify the directory is
still clean after a run with::

    git -C <repo> status --short -- data/arc-agi-3/environment_files/

Scoring model
-------------
The official scorer is used, never reimplemented. For every level of every game we
call ``EnvironmentScoreCalculator.add_level(completed, actions_taken,
baseline_actions, game_id)`` and read the final ``.score`` back. A level reports
``(baseline_actions / actions_taken) * 100`` capped at 100 when completed and
``0.0`` otherwise; a game's score is the mean over **all** of its levels, so
levels that were never reached must still be reported (as not completed) or the
denominator shrinks and the mean is inflated.

``actions_taken`` for level *m* is derived from the frames only: the frame index
at which ``levels_completed`` first reached *m*, minus the frame index at which it
first reached *m - 1*. ``agent.frames[k]`` is the frame after *k* actions
(``frames[0]`` is the synthetic pre-episode frame), so level 1's ``actions_taken``
is simply its first transition index.

Usage
-----
    data/arc-agi-3-agents/.venv/bin/python experiments/arc3_local_eval.py \
        --json experiments/arc3_baseline.json

    # one game, smaller budget, no agent chatter
    data/arc-agi-3-agents/.venv/bin/python experiments/arc3_local_eval.py \
        --game ar25 --max-steps 200 --quiet

The interpreter matters: ``arc_agi`` / ``arcengine`` are installed in
``data/arc-agi-3-agents/.venv`` (Python 3.12) with ``cp312`` wheels; a system
Python 3.14 cannot import them and the module only supports importing for the
pure logic (tests).
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib
import importlib.util
import json
import logging
import sys
import tempfile
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
ENVIRONMENTS_DIR = REPO_ROOT / "data" / "arc-agi-3" / "environment_files"
FRAMEWORK_DIR = REPO_ROOT / "data" / "arc-agi-3" / "ARC-AGI-3-Agents"
AGENT_PATH = REPO_ROOT / "src" / "arc3_spatial_memory_agent.py"
# The verified API uses recordings_dir="/tmp/arc3rec"; expressed through tempfile so
# the resolved path is identical without hard-coding a world-writable location.
RECORDINGS_DIR = str(Path(tempfile.gettempdir()) / "arc3rec")

SCHEMA = "arc3-local-eval/1"
DEFAULT_MAX_STEPS = 500
DEFAULT_TIME_BUDGET_S = 60.0

# ``arc_agi`` is imported lazily so that the pure scoring/aggregation logic can be
# imported by the test suite on an interpreter that does not have the engine
# installed (only the CLI / engine wiring actually needs it at runtime).
_ARC_AGI: Any = None
ARC_AGI_AVAILABLE = False
ARC_AGI_IMPORT_ERROR: str | None = None
try:  # pragma: no cover - depends on which interpreter runs the suite
    _ARC_AGI = importlib.import_module("arc_agi")
    ARC_AGI_AVAILABLE = True
except ImportError as exc:  # pragma: no cover
    ARC_AGI_IMPORT_ERROR = str(exc)

logger = logging.getLogger("arc3_local_eval")


def _arc_agi_module() -> Any:
    """Return the ``arc_agi`` module or explain which interpreter is required."""
    if _ARC_AGI is None:
        raise RuntimeError(
            "arc_agi is not importable with this interpreter "
            f"({ARC_AGI_IMPORT_ERROR or 'unknown reason'}). Use "
            "data/arc-agi-3-agents/.venv/bin/python."
        )
    return _ARC_AGI


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #


@dataclasses.dataclass(frozen=True)
class GamePlan:
    """One game in the sweep: what to play and the metadata needed to score it."""

    game_id: str  # short id, e.g. "ar25"
    env_game_id: str  # versioned id from EnvironmentInfo, e.g. "ar25-0c556536"
    baseline_actions: tuple[int, ...] = ()  # per-level human baseline, from metadata


@dataclasses.dataclass
class LevelOutcome:
    """Per-level result derived from the episode frames + environment metadata."""

    level: int  # 1-indexed
    completed: bool
    actions_taken: int
    baseline_actions: int | None = None


# --------------------------------------------------------------------------- #
# Level-transition detection and scoring (pure / no engine required)
# --------------------------------------------------------------------------- #


def _as_int(value: Any, default: int = 0) -> int:
    """Coerce a frame/metadata value to ``int``, falling back instead of raising."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value: Any, default: float = 0.0) -> float:
    """Coerce a frame/score value to ``float``, falling back instead of raising."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def level_action_counts(
    levels_completed: Sequence[int], n_levels: int
) -> list[LevelOutcome]:
    """Derive per-level ``actions_taken`` from the frame-by-frame level counter.

    ``levels_completed[k]`` is the number of levels cleared after *k* actions
    (``levels_completed[0]`` is the pre-episode frame). Level *m* was cleared at
    the first index whose counter reached *m*, so its cost is that index minus the
    index at which level *m - 1* was cleared (0 for level 1). Levels that never
    appear in the series are returned as not completed with ``actions_taken = 0``:
    they still have to be handed to the official scorer, or the mean divisor is
    wrong, but no actions were ever spent reaching them.
    """
    first_reached: dict[int, int] = {}
    highest = 0
    for index, value in enumerate(levels_completed):
        if value > highest:
            for level in range(highest + 1, value + 1):
                first_reached.setdefault(level, index)
            highest = value

    outcomes: list[LevelOutcome] = []
    previous_index = 0
    for level in range(1, n_levels + 1):
        index = first_reached.get(level)
        if index is None:
            outcomes.append(LevelOutcome(level=level, completed=False, actions_taken=0))
        else:
            outcomes.append(
                LevelOutcome(
                    level=level,
                    completed=True,
                    actions_taken=index - previous_index,
                )
            )
            previous_index = index
    return outcomes


def build_levels(
    levels_completed: Sequence[int],
    baseline_actions: Sequence[int],
    win_levels: int | None = None,
) -> tuple[list[LevelOutcome], list[str]]:
    """Resolve the level set to score and attach per-level baselines.

    The denominator follows the official scorer, which iterates
    ``range(len(env_info.baseline_actions))``; when metadata has no baselines we
    fall back to the ``win_levels`` reported by the environment. Any mismatch
    between the two is recorded as an anomaly instead of aborting the sweep, since
    a bad metadata row must not cost the other 24 games.
    """
    baselines = list(baseline_actions)
    anomalies: list[str] = []

    if baselines:
        n_levels = len(baselines)
    elif win_levels:
        n_levels = _as_int(win_levels)
        anomalies.append(
            f"no baseline_actions in metadata; scored {n_levels} levels from "
            f"win_levels with a zero baseline"
        )
    else:
        n_levels = 0
        anomalies.append("no baseline_actions and no win_levels; game not scoreable")

    if baselines and win_levels is not None and len(baselines) != _as_int(win_levels):
        anomalies.append(
            f"baseline_actions has {len(baselines)} entries but win_levels is "
            f"{win_levels}; scoring uses the {len(baselines)} baseline entries"
        )

    outcomes = level_action_counts(levels_completed, n_levels)
    for outcome in outcomes:
        index = outcome.level - 1
        outcome.baseline_actions = baselines[index] if index < len(baselines) else 0

    series = list(levels_completed)
    observed = max(series, default=0)
    if observed > n_levels:
        anomalies.append(
            f"agent reported {observed} completed levels but only {n_levels} are "
            f"scoreable; extras ignored"
        )
    if any(later < earlier for earlier, later in zip(series, series[1:], strict=False)):
        anomalies.append("levels_completed is not monotonic across frames")

    for outcome in outcomes:
        if outcome.completed and outcome.actions_taken <= 0:
            anomalies.append(
                f"level {outcome.level} transitioned with {outcome.actions_taken} "
                f"actions (level counter jumped)"
            )

    return outcomes, anomalies


def _new_score_calculator(game_id: str) -> Any:
    """Build the official per-environment scorer (the only scorer used here)."""
    return _arc_agi_module().EnvironmentScoreCalculator(id=game_id)


def score_levels(
    levels: Sequence[LevelOutcome],
    game_id: str,
    calculator_factory: Callable[[str], Any] | None = None,
) -> Any:
    """Feed the derived levels to the official scorer and return its score object."""
    factory = calculator_factory or _new_score_calculator
    calculator = factory(game_id)
    for outcome in levels:
        calculator.add_level(
            completed=outcome.completed,
            actions_taken=outcome.actions_taken,
            baseline_actions=outcome.baseline_actions if outcome.baseline_actions else 0,
            game_id=game_id,
        )
    return calculator.to_score()


def aggregate(games: Sequence[dict]) -> dict:
    """Aggregate per-game reports: mean score, level totals, total wall time.

    The mean is taken over every game in the sweep, not only the successful ones:
    a game that errored or timed out scored nothing, and dropping it would inflate
    the headline number.
    """
    count = len(games)
    mean_score = (
        sum(_as_float(game["score"]) for game in games) / count if count else 0.0
    )
    return {
        "games_scored": sum(1 for game in games if game["status"] != "error"),
        "games_errored": sum(1 for game in games if game["status"] == "error"),
        "games_timed_out": sum(1 for game in games if game["timed_out"]),
        "mean_score": round(mean_score, 6),
        "total_levels_completed": sum(_as_int(game["levels_completed"]) for game in games),
        "total_levels_available": sum(_as_int(game["levels_available"]) for game in games),
        "total_wall_time_s": round(
            sum(_as_float(game["wall_time_s"]) for game in games), 3
        ),
    }


# --------------------------------------------------------------------------- #
# Episode driver
# --------------------------------------------------------------------------- #


def play_episode(
    agent: Any,
    time_budget_s: float | None,
    clock: Callable[[], float] = time.monotonic,
) -> bool:
    """Drive one episode to completion or to a budget. Returns ``True`` if timed out.

    The loop body is a verbatim copy of ``Agent.main()`` (see the module
    docstring) with one addition: the deadline check at the top of each iteration.
    The per-action INFO log line is kept as well, but is only rendered when INFO is
    enabled, so the framework's one-line-per-action flood is suppressed by default
    and reappears under ``--verbose``. The deadline check is per action, so a single
    pathological ``choose_action`` call can still overrun the budget once; that is
    the finest granularity available without running each game in a subprocess.
    ``clock`` is injectable so the budget can be tested deterministically.
    """
    started = clock()
    agent.timer = time.time()
    timed_out = False
    while (
        not agent.is_done(agent.frames, agent.frames[-1])
        and agent.action_counter <= agent.MAX_ACTIONS
    ):
        if time_budget_s is not None and (clock() - started) >= time_budget_s:
            timed_out = True
            break
        action = agent.choose_action(
            agent.frames,
            agent._convert_raw_frame_data(
                agent.arc_env.observation_space if agent.arc_env else None
            ),
        )
        if frame := agent.take_action(action):
            agent.append_frame(frame)
            if logger.isEnabledFor(logging.INFO):
                logger.info(
                    "%s - %s: count %s, levels completed %s, avg fps %s",
                    agent.game_id,
                    getattr(action, "name", action),
                    agent.action_counter,
                    frame.levels_completed,
                    agent.fps,
                )
        agent.action_counter += 1

    agent.cleanup()
    return timed_out


def build_record(
    plan: GamePlan,
    agent: Any,
    timed_out: bool,
    wall_time_s: float,
    calculator_factory: Callable[[str], Any] | None = None,
) -> dict:
    """Turn a finished (or failed) agent into the JSON report row for one game."""
    frames = list(getattr(agent, "frames", []))
    series = [_as_int(frame.levels_completed) for frame in frames]
    win_levels = getattr(frames[-1], "win_levels", None) if frames else None

    levels, anomalies = build_levels(series, plan.baseline_actions, win_levels)
    score_object = score_levels(levels, plan.game_id, calculator_factory)
    level_scores = list(getattr(score_object, "level_scores", None) or [])

    actions_used = _as_int(getattr(agent, "action_counter", 0))
    if agent is not None and actions_used != len(frames) - 1:
        anomalies.append(
            f"action_counter ({actions_used}) != frames appended ({len(frames) - 1}); "
            "some actions produced no frame"
        )

    return {
        "game_id": plan.game_id,
        "env_game_id": plan.env_game_id,
        "status": "timeout" if timed_out else "ok",
        "score": round(_as_float(getattr(score_object, "score", 0.0)), 6),
        "levels_completed": _as_int(getattr(score_object, "levels_completed", 0)),
        "levels_available": len(levels),
        "win_levels": _as_int(win_levels) if win_levels is not None else None,
        "actions_used": actions_used,
        "wall_time_s": round(_as_float(wall_time_s), 3),
        "timed_out": bool(timed_out),
        "error": None,
        "anomalies": anomalies,
        "levels": [
            {
                "level": outcome.level,
                "completed": outcome.completed,
                "actions_taken": outcome.actions_taken,
                "baseline_actions": outcome.baseline_actions,
                "level_score": level_scores[index]
                if index < len(level_scores)
                else 0.0,
            }
            for index, outcome in enumerate(levels)
        ],
    }


def _empty_record(plan: GamePlan, error: str) -> dict:
    return {
        "game_id": plan.game_id,
        "env_game_id": plan.env_game_id,
        "status": "error",
        "score": 0.0,
        "levels_completed": 0,
        "levels_available": len(plan.baseline_actions),
        "win_levels": None,
        "actions_used": 0,
        "wall_time_s": 0.0,
        "timed_out": False,
        "error": error,
        "anomalies": [],
        "levels": [],
    }


def _recover_record(
    plan: GamePlan,
    agent: Any,
    message: str,
    wall_time_s: float,
    calculator_factory: Callable[[str], Any] | None,
) -> dict:
    """Preserve whatever a failed game already measured, then record the error.

    Kept out of ``run_sweep``'s ``except`` block on purpose: it holds the only
    boolean expression in the recovery path, and an ``except`` suite containing
    ``and``/``or`` trips pi-lens's ``no-boolean-in-except`` rule (a false positive
    here, since the expression is in the handler body, not the exception tuple).
    """
    has_frames = agent is not None and bool(getattr(agent, "frames", None))
    if has_frames:
        try:
            record = build_record(plan, agent, False, wall_time_s, calculator_factory)
        except Exception:  # noqa: BLE001 - fall back to the empty record below
            return _empty_record(plan, message)
        record["status"] = "error"
        record["error"] = message
        return record
    return _empty_record(plan, message)


def run_sweep(
    plans: Sequence[GamePlan],
    *,
    make_env: Callable[[GamePlan], Any],
    make_agent: Callable[[GamePlan, Any], Any],
    time_budget_s: float | None = DEFAULT_TIME_BUDGET_S,
    clock: Callable[[], float] = time.monotonic,
    calculator_factory: Callable[[str], Any] | None = None,
    on_game: Callable[[dict], None] | None = None,
) -> list[dict]:
    """Play every plan, converting any failure into a report row instead of a crash.

    A game that cannot build its environment, whose agent raises, or that blows its
    time budget is recorded with ``status`` / ``error`` / ``timed_out`` and the
    sweep continues. Partial results already collected from the frames are kept.
    """
    records: list[dict] = []
    for index, plan in enumerate(plans, 1):
        started = clock()
        agent: Any = None
        logger.info("[%d/%d] %s", index, len(plans), plan.game_id)
        try:
            env = make_env(plan)
            if env is None:
                raise RuntimeError("arc.make() returned None (game not found locally)")
            agent = make_agent(plan, env)
            timed_out = play_episode(agent, time_budget_s, clock)
            record = build_record(
                plan, agent, timed_out, clock() - started, calculator_factory
            )
        except Exception as exc:  # noqa: BLE001 - one bad game must not abort the sweep
            message = f"{type(exc).__name__}: {exc}"
            logger.warning("%s failed: %s", plan.game_id, message)
            record = _recover_record(
                plan, agent, message, clock() - started, calculator_factory
            )
        records.append(record)
        if on_game is not None:
            on_game(record)
    return records


# --------------------------------------------------------------------------- #
# Repository / engine wiring
# --------------------------------------------------------------------------- #


def load_agent_class(agent_path: Path = AGENT_PATH) -> Any:
    """Import ``MyAgent`` from ``src/arc3_spatial_memory_agent.py`` via importlib.

    The agent module does ``from agents.agent import Agent``, so the framework
    directory has to be on ``sys.path`` (this is the same importlib pattern
    ``play_local.py`` uses).
    """
    if not agent_path.exists():
        raise FileNotFoundError(f"agent module not found: {agent_path}")
    if str(FRAMEWORK_DIR) not in sys.path:
        sys.path.insert(0, str(FRAMEWORK_DIR))
    if str(REPO_ROOT / "src") not in sys.path:
        sys.path.insert(0, str(REPO_ROOT / "src"))

    spec = importlib.util.spec_from_file_location("arc3_local_eval_user_agent", agent_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load agent module from {agent_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "MyAgent"):
        raise RuntimeError(f"{agent_path} must define a class named `MyAgent`")
    return module.MyAgent


def build_arcade(logger_override: logging.Logger | None = None) -> Any:
    """Create the OFFLINE Arcade. Handing it a logger suppresses its stdout spam."""
    arc_agi = _arc_agi_module()
    return arc_agi.Arcade(
        operation_mode=arc_agi.OperationMode.OFFLINE,
        environments_dir=str(ENVIRONMENTS_DIR),
        recordings_dir=RECORDINGS_DIR,
        logger=logger_override or logging.getLogger("arc3_local_eval.arcade"),
    )


def collect_plans(arc: Any, wanted: set[str] | None = None) -> list[GamePlan]:
    """List the local environments and build one plan per game (all by default)."""
    plans: list[GamePlan] = []
    for info in arc.get_environments():
        short_id = info.game_id.split("-")[0]
        if wanted is not None and short_id not in wanted:
            continue
        plans.append(
            GamePlan(
                game_id=short_id,
                env_game_id=info.game_id,
                baseline_actions=tuple(info.baseline_actions or ()),
            )
        )
    return plans


def configure_logging(verbose: bool) -> None:
    """Silence the framework's one-line-per-action INFO log unless asked for it.

    ``agents.agent.Agent.main`` logs at INFO through the root logger, so raising
    the root level to WARNING is what removes the per-step flood.
    """
    root = logging.getLogger()
    root.setLevel(logging.INFO if verbose else logging.WARNING)
    if not root.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
        root.addHandler(handler)


def format_summary(report: dict) -> str:
    """Human-readable table for the console (the JSON is the machine artifact)."""
    lines = [
        f"{'game':6} {'score':>8} {'levels':>9} {'actions':>8} {'time_s':>8}  status",
        "-" * 60,
    ]
    for game in report["games"]:
        levels = f"{game['levels_completed']}/{game['levels_available']}"
        status = game["status"]
        if game["error"]:
            status = f"error: {game['error']}"
        lines.append(
            f"{game['game_id']:6} {game['score']:8.4f} {levels:>9} "
            f"{game['actions_used']:8} {game['wall_time_s']:8.3f}  {status}"
        )
    aggregate_block = report["aggregate"]
    lines += [
        "-" * 60,
        f"games={len(report['games'])} errored={aggregate_block['games_errored']} "
        f"timed_out={aggregate_block['games_timed_out']}",
        f"mean score: {aggregate_block['mean_score']:.4f}",
        f"levels completed: {aggregate_block['total_levels_completed']}"
        f"/{aggregate_block['total_levels_available']}",
        f"total wall time: {aggregate_block['total_wall_time_s']:.2f}s",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Local ARC-AGI-3 scoring harness (OFFLINE only).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--game",
        default=None,
        help="Comma-separated short game ids (e.g. ar25,vc33). Default: all games.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=DEFAULT_MAX_STEPS,
        help=f"Per-game action budget (default: {DEFAULT_MAX_STEPS}).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIME_BUDGET_S,
        help=(
            "Per-game wall-clock budget in seconds; a game exceeding it is reported "
            f"as timed out and the sweep continues (default: {DEFAULT_TIME_BUDGET_S})."
        ),
    )
    parser.add_argument("--json", default=None, help="Write the full report to PATH.")
    parser.add_argument(
        "--list", action="store_true", help="List the available games and exit."
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show the framework's per-action INFO logs (suppressed by default).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Print nothing but the JSON path (for scripted runs).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(args.verbose)

    if not ARC_AGI_AVAILABLE:
        print(
            "arc_agi is not importable with this interpreter "
            f"({ARC_AGI_IMPORT_ERROR or 'unknown reason'}).\n"
            "Run this module with data/arc-agi-3-agents/.venv/bin/python.",
            file=sys.stderr,
        )
        return 2

    if args.game:
        wanted = {part.strip().split("-")[0] for part in args.game.split(",") if part.strip()}
    else:
        wanted = None

    arc = None
    try:
        arc = build_arcade()
        plans = collect_plans(arc, wanted)
    except Exception as exc:  # noqa: BLE001 - report setup failure without a traceback
        print(
            f"could not initialise the OFFLINE arcade: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1

    if args.list:
        for plan in plans:
            print(f"{plan.game_id}: {plan.env_game_id} ({len(plan.baseline_actions)} levels)")
        return 0

    if not plans:
        print(f"no games matched --game {args.game!r}", file=sys.stderr)
        return 1
    if wanted:
        missing = wanted - {plan.game_id for plan in plans}
        if missing:
            print(f"unknown game id(s): {sorted(missing)}", file=sys.stderr)
            return 1

    try:
        agent_class = load_agent_class()
    except Exception as exc:  # noqa: BLE001 - report setup failure without a traceback
        print(f"could not load the agent: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    agent_class.MAX_ACTIONS = args.max_steps

    def make_env(plan: GamePlan) -> Any:
        return arc.make(plan.game_id)

    def make_agent(plan: GamePlan, env: Any) -> Any:
        return agent_class(
            card_id="local-eval",
            game_id=plan.game_id,
            agent_name=f"MyAgent.local.{plan.game_id}",
            ROOT_URL="http://localhost",
            record=False,
            arc_env=env,
            tags=["local-eval"],
        )

    def show(record: dict) -> None:
        if args.quiet:
            return
        status = record["status"]
        if record["error"]:
            status = f"error: {record['error']}"
        print(
            f"  {record['game_id']:6} score={record['score']:.4f} "
            f"levels={record['levels_completed']}/{record['levels_available']} "
            f"actions={record['actions_used']} time={record['wall_time_s']:.3f}s "
            f"({status})",
            flush=True,
        )

    if not args.quiet:
        print(
            f"Playing {len(plans)} games OFFLINE "
            f"(max_steps={args.max_steps}, timeout={args.timeout}s)\n"
        )

    started = time.monotonic()
    games = run_sweep(
        plans,
        make_env=make_env,
        make_agent=make_agent,
        time_budget_s=args.timeout,
        clock=time.monotonic,
        on_game=show,
    )
    report = {
        "schema": SCHEMA,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "operation_mode": "OFFLINE",
        "interpreter": sys.executable,
        "agent_module": str(AGENT_PATH.relative_to(REPO_ROOT)),
        "max_steps": args.max_steps,
        "time_budget_s": args.timeout,
        "games": games,
        "aggregate": aggregate(games),
    }
    report["aggregate"]["total_wall_time_s"] = round(time.monotonic() - started, 3)

    if args.json:
        output = Path(args.json)
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    if not args.quiet:
        print("\n" + format_summary(report))
        if args.json:
            print(f"\nwrote {args.json}")
    elif args.json:
        print(f"wrote {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

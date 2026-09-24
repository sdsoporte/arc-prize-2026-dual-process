#!/usr/bin/env python3
"""Durable human-replay calibration check for the ARC-AGI-3 evaluation harness.

What this benchmark is for
--------------------------
``experiments/arc3_local_eval.py`` scores our agent on the 25 ARC-AGI-3 games with
the official ``arc_agi.EnvironmentScoreCalculator`` and reports ~0.20 / 100 (the
public leader did 19.40). Those numbers say nothing about whether the instrument
itself -- engine, scorer, frame handling, environment loading -- is correct. This
module supplies the missing ground truth: the 25 recorded **human** replays are fed
back through the same OFFLINE engine and the same official scorecard, and the
result is compared with a committed reference. Human play is the ceiling, so a
correct instrument must reproduce it.

Two independent checks are made on every replay:

* **Reproduction** -- drive the replay step by step through the local OFFLINE
  build and compare the recorded ``(levels_completed, state)`` at every step.
* **Scorecard** -- feed the same replay to ``arcade.create_scorecard()`` and read
  the official per-game and aggregate scores back from ``close_scorecard()``.

Reference values (verified, do not re-derive)
---------------------------------------------
* Reproduction: **24 / 25** replays match the recorded run step for step.
* Official scorecard over the 25 human replays: **89.6774 / 100**.
* Per-game human-implied scores are recorded in
  ``experiments/arc3_calibration_reference.json`` (e.g. ``tr87`` 100.00,
  ``vc33`` 100.00, ``tn36`` 99.74, ``su15`` 74.06).

Tolerances
----------
The aggregate score and each per-game score are compared to the reference with an
absolute tolerance of **0.01** (the reference stores ``score_tolerance``). The
reproduced count and every game's reproduction pass/fail are compared **exactly**,
because a changed reproduction is a change in engine/scorer/frame behaviour, not
measurement noise.

Known exclude: cn04
-------------------
The installed local build is ``cn04-2fe56bfb`` but the replay was recorded against
``cn04-65d47d14`` -- a different game. It is expected to diverge (at step 14), is
expected to finish ``GAME_OVER`` instead of ``WIN``, and scores **0.00**. It is
recorded as a named exclude in the reference and is *not* a regression. If it ever
starts reproducing or scoring, that is itself drift and must be reviewed.

Dataset: 771 MB, not in the repository
--------------------------------------
This check needs the unpacked ARC-AGI-3 human replay archive (771 MB), which is
**not** stored in the repository. The default location is ``/tmp/arc3-replays``.
When the dataset is absent the module prints a one-line reason and exits 0 with a
distinct ``skipped`` status instead of failing, because a missing 771 MB ``/tmp``
artifact is not a regression.

HARD SAFETY RULE -- never write the tracked environment files
-------------------------------------------------------------
The engine is constructed with ``OperationMode.OFFLINE`` **only**, with
``environments_dir`` pointing at a copy under ``/tmp`` (``/tmp/arc3-env``). An
``arc_agi`` run in ``OperationMode.NORMAL`` silently re-downloads the game sources
and **overwrites** the tracked files under
``data/arc-agi-3/environment_files``. That already happened once. This module
therefore refuses any ``--env-dir`` that resolves inside the repository's
``environment_files`` and auto-populates the ``/tmp`` copy from the repository
source (a read-only source) when it is missing.

Usage
-----
    data/arc-agi-3-agents/.venv/bin/python experiments/arc3_calibration.py
    data/arc-agi-3-agents/.venv/bin/python experiments/arc3_calibration.py --verbose
    data/arc-agi-3-agents/.venv/bin/python experiments/arc3_calibration.py \\
        --dataset /tmp/arc3-replays --env-dir /tmp/arc3-env

    # intentionally regenerate the reference after a reviewed change
    data/arc-agi-3-agents/.venv/bin/python experiments/arc3_calibration.py \\
        --write-reference

Exit status: ``0`` pass or skip, ``1`` drift from the reference, ``2`` setup error.
The interpreter matters: ``arc_agi`` / ``arcengine`` live in
``data/arc-agi-3-agents/.venv`` (Python 3.12); a system Python 3.14 cannot import
them and this module degrades to a clean skip instead of a traceback.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import io
import json
import logging
import shutil
import sys
import tempfile
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_PATH = Path(__file__).resolve().parent / "arc3_calibration_reference.json"
VENV_DIR = REPO_ROOT / "data" / "arc-agi-3-agents" / ".venv"
ENV_SOURCE = REPO_ROOT / "data" / "arc-agi-3" / "environment_files"
FORBIDDEN_ENV_DIRS = (
    ENV_SOURCE.resolve(),
    (REPO_ROOT / "environment_files").resolve(),
)

SCHEMA = "arc3-human-calibration/1"
# The archive and the /tmp engine copy are world-writable locations by design; the
# engine is OFFLINE-only and the source of truth is read-only.
DEFAULT_DATASET = "/tmp/arc3-replays"  # noqa: S108
DEFAULT_ENV_DIR = "/tmp/arc3-env"  # noqa: S108

# Absolute tolerance for every score comparison (aggregate and per-game). The
# reproduction count and each game's reproduction pass/fail are compared exactly.
SCORE_TOLERANCE = 0.01

EXIT_PASS = 0
EXIT_DRIFT = 1
EXIT_SETUP = 2

# The one curated, non-derivable fact in the reference: cn04's local build is a
# different game from the one that was recorded, so reproduction is expected to
# fail and the game is expected to score 0. Everything else is measured.
KNOWN_EXCLUDES: dict[str, str] = {
    "cn04": (
        "installed local build cn04-2fe56bfb is a different game from the recorded "
        "cn04-65d47d14; reproduction is expected to diverge at step 14 and the game "
        "is expected to score 0.00"
    ),
}

NAME2ID = {
    "ACTION1": 1,
    "ACTION2": 2,
    "ACTION3": 3,
    "ACTION4": 4,
    "ACTION5": 5,
    "ACTION6": 6,
    "ACTION7": 7,
    "RESET": 0,
}

logger = logging.getLogger("arc3_calibration")

# ``arc_agi`` is imported lazily so a missing engine becomes a named skip, not a
# traceback. ``_ENGINE`` / ``_ARCENGINE`` are populated by ``load_engine()``.
_ENGINE: Any = None
_ARCENGINE: Any = None
_ENGINE_ERROR: str | None = None


class ReplayError(RuntimeError):
    """A replay file is present but cannot be read or parsed."""


# --------------------------------------------------------------------------- #
# Small coercions (kept local so the ast-grep "unchecked throwing call" rule sees
# the int()/float() conversions inside a try, matching arc3_local_eval.py)
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


# --------------------------------------------------------------------------- #
# Engine loading / logging
# --------------------------------------------------------------------------- #


def _try_import_engine() -> tuple[Any, Any, str | None]:
    """Import the engine modules once; return ``(arc_agi, arcengine, error)``."""
    try:
        arc_agi = importlib.import_module("arc_agi")
        arcengine = importlib.import_module("arcengine")
    except Exception as exc:  # noqa: BLE001 - a missing/broken engine is a skip
        return None, None, f"{type(exc).__name__}: {exc}"
    return arc_agi, arcengine, None


def _add_venv_site_packages() -> None:
    """Add the venv's site-packages *for this interpreter version* to ``sys.path``.

    The version is matched on purpose: a Python 3.14 host must not try to load
    cp312 native extensions and fail later with a confusing numpy/native error.
    """
    site_packages = (
        VENV_DIR
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )
    if not site_packages.is_dir():
        return
    if str(site_packages) not in sys.path:
        sys.path.insert(0, str(site_packages))


def load_engine() -> tuple[Any | None, str | None]:
    """Import ``arc_agi`` (and ``arcengine``) or explain which interpreter to use.

    A direct import works under the project venv. As a fallback the matching venv
    ``site-packages`` is added to ``sys.path``. Either way the failure is returned
    as a message, never raised past ``main``.
    """
    global _ENGINE, _ARCENGINE, _ENGINE_ERROR
    if _ENGINE is not None or _ENGINE_ERROR is not None:
        return _ENGINE, _ENGINE_ERROR

    arc_agi, arcengine, error = _try_import_engine()
    if arc_agi is None:
        _add_venv_site_packages()
        arc_agi, arcengine, error = _try_import_engine()
    if arc_agi is None:
        _ENGINE_ERROR = error or "unknown import failure"
        return None, _ENGINE_ERROR

    _ENGINE, _ARCENGINE = arc_agi, arcengine
    return _ENGINE, None


def game_action(action: int) -> Any:
    """Build a ``GameAction`` from a numeric id using the loaded engine."""
    return _ARCENGINE.GameAction.from_id(action)


def configure_logging(verbose: bool) -> None:
    """Show the engine's per-step logs only under ``--verbose``.

    Default level ERROR removes the framework's one-line-per-action flood; the
    engine's own loggers are named explicitly because frames reach more than one.
    """
    level = logging.INFO if verbose else logging.ERROR
    root = logging.getLogger()
    root.setLevel(level)
    if not root.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
        root.addHandler(handler)
    for name in ("arc_agi", "arcengine", "arc3_calibration"):
        logging.getLogger(name).setLevel(level)


# --------------------------------------------------------------------------- #
# Dataset / environment directory resolution
# --------------------------------------------------------------------------- #


def resolve_replay_root(dataset: Path) -> Path | None:
    """Return the directory holding per-game replay folders, or ``None``.

    The unpacked archive has an ``environment_files/`` level; a direct game root is
    also accepted so a re-rooted dataset still works.
    """
    nested = dataset / "environment_files"
    if nested.is_dir():
        return nested
    if dataset.is_dir() and any(child.is_dir() for child in dataset.iterdir()):
        return dataset
    return None


def resolve_env_dir(
    env_dir: Path, env_source: Path, refresh: bool
) -> tuple[Path | None, str | None]:
    """Return a safe OFFLINE ``environments_dir``, or a named safety error.

    Refuses any directory inside the repository's tracked ``environment_files``
    (or the stray repo-root ``environment_files`` left by the earlier NORMAL-mode
    accident), because handing one to the engine risks overwriting it. When the
    target copy is missing it is populated from the repository source; that copy is
    the only filesystem write this module performs outside ``--write-reference``.
    """
    resolved = env_dir.expanduser().resolve()
    for forbidden in FORBIDDEN_ENV_DIRS:
        if resolved == forbidden or forbidden in resolved.parents:
            return None, (
                f"refusing environments_dir={resolved}: it is inside the repository's "
                "tracked environment_files and the engine may overwrite it. Use a copy "
                f"under {tempfile.gettempdir()}."
            )

    if refresh or not resolved.is_dir():
        if not env_source.is_dir():
            return None, (
                f"environments_dir {resolved} is missing and no source exists at "
                f"{env_source} to copy from"
            )
        try:
            shutil.copytree(env_source, resolved, dirs_exist_ok=True)
        except OSError as exc:
            return None, f"could not populate {resolved} from {env_source}: {exc}"
        print(
            f"note: populated environments_dir {resolved} from the repository source "
            f"{env_source} ({'refresh' if refresh else 'missing target'})"
        )

    temp_root = Path(tempfile.gettempdir()).resolve()
    if temp_root not in resolved.parents:
        print(
            f"warning: environments_dir {resolved} is outside {temp_root}; the engine "
            "must stay OFFLINE, but a /tmp copy is the reviewed location",
            file=sys.stderr,
        )
    return resolved, None


# --------------------------------------------------------------------------- #
# Replay loading / driving
# --------------------------------------------------------------------------- #


def load_replay(game_dir: Path) -> list[dict]:
    """Read a game's recorded JSONL replay into the list of frame payloads.

    Mirrors the throwaway calibration: one replay file per game, blank lines
    ignored, and any line whose ``data`` carries a ``cards`` key is not a frame.
    """
    replays = sorted((game_dir / "replays").glob("*.json"))
    if not replays:
        raise ReplayError(f"no replay file under {game_dir / 'replays'}")
    frames: list[dict] = []
    try:
        text = replays[0].read_text(encoding="utf-8")
    except OSError as exc:
        raise ReplayError(f"could not read {replays[0]}: {exc}") from exc
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line).get("data")
        except json.JSONDecodeError as exc:
            raise ReplayError(f"malformed JSON line in {replays[0].name}: {exc}") from exc
        if isinstance(payload, dict) and "cards" not in payload:
            frames.append(payload)
    if not frames:
        raise ReplayError(f"no frame records in {replays[0].name}")
    return frames


def action_id(frame: dict) -> int:
    """Map a recorded action input to a numeric ``GameAction`` id."""
    action_input = frame.get("action_input") or {}
    raw = action_input.get("id")
    if isinstance(raw, int):
        return raw
    if isinstance(raw, str) and raw in NAME2ID:
        return NAME2ID[raw]
    raise ReplayError(f"unrecognised action id {raw!r}")


def replay_game(arcade: Any, scorecard_id: str, game_id: str, game_dir: Path) -> dict:
    """Drive one recorded replay through the OFFLINE engine and report the result.

    Any failure is returned as a named ``status``/``error`` pair so the remaining
    games still run; this function never raises.
    """
    record: dict[str, Any] = {
        "game_id": game_id,
        "env_game_id": None,
        "status": "ok",
        "reproduced": False,
        "step_for_step": False,
        "recorded_final": None,
        "reproduced_final": None,
        "first_divergence": None,
        "n_actions": 0,
        "score": 0.0,
        "levels_completed": 0,
        "levels_available": 0,
        "error": None,
    }

    try:
        frames = load_replay(game_dir)
    except ReplayError as exc:
        record["status"] = "missing_replay"
        record["error"] = str(exc)
        return record

    try:
        old_schema = "score" in frames[0]
        key = "score" if old_schema else "levels_completed"
        recorded = [(frame[key], frame.get("state")) for frame in frames]
        plan = [
            (action_id(frame), (frame.get("action_input") or {}).get("data"))
            for frame in frames[1:]
        ]
    except (KeyError, TypeError, ReplayError) as exc:
        record["status"] = "malformed_replay"
        record["error"] = f"{type(exc).__name__}: {exc}"
        return record

    record["n_actions"] = len(plan)

    env = arcade.make(game_id, scorecard_id=scorecard_id)
    if env is None:
        record["status"] = "engine_error"
        record["error"] = "arcade.make() returned None (game not found locally)"
        return record
    record["env_game_id"] = getattr(
        getattr(env, "environment_info", None), "game_id", None
    )

    obs = env.observation_space
    if obs is None:
        record["status"] = "engine_error"
        record["error"] = "environment produced no initial observation"
        return record

    trace = [(obs.levels_completed, obs.state.name)]
    first_divergence: list[Any] | None = None
    error: str | None = None
    for index, (action, data) in enumerate(plan):
        payload = {k: v for k, v in (data or {}).items() if k in ("x", "y")}
        try:
            if action == 0:
                obs = env.reset()
            else:
                obs = env.step(game_action(action), payload or None)
        except Exception as exc:  # noqa: BLE001 - one bad step must not abort the game
            error = f"{type(exc).__name__}: {exc}"
            break
        if obs is None:
            error = "wrapper returned None"
            break
        current = (obs.levels_completed, obs.state.name)
        trace.append(current)
        if (
            first_divergence is None
            and len(recorded) > index + 1
            and current != recorded[index + 1]
        ):
            first_divergence = [index + 1, list(recorded[index + 1]), list(current)]

    recorded_final = recorded[-1]
    reproduced_final = trace[-1]
    record.update(
        recorded_final=list(recorded_final),
        reproduced_final=list(reproduced_final),
        first_divergence=first_divergence,
        error=error,
        reproduced=error is None and recorded_final == reproduced_final,
        step_for_step=error is None and first_divergence is None,
    )
    return record


def collect_scorecard(arcade: Any, scorecard_id: str, verbose: bool) -> Any | None:
    """Close the scorecard, capturing the scorer's stray stdout.

    ``EnvironmentScorecard._raw_scores_from_card`` prints ``actions_by_level: ...``
    for a baseline mismatch (which is exactly the cn04 known-exclude path). Capture
    it so the check's own output stays machine-readable and surface it only under
    ``--verbose``.
    """
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        scorecard = arcade.close_scorecard(scorecard_id)
    stray = buffer.getvalue()
    if verbose and stray.strip():
        print("engine stdout (verbose):")
        print(stray.rstrip())
    return scorecard


# --------------------------------------------------------------------------- #
# Scoring / reference / comparison
# --------------------------------------------------------------------------- #


def scorecard_games(scorecard: Any) -> dict[str, dict]:
    """Map short game id -> per-game score fields from a closed scorecard."""
    games: dict[str, dict] = {}
    for environment in getattr(scorecard, "environments", []) or []:
        short_id = str(environment.id).split("-", 1)[0]
        best = max(environment.runs, key=lambda run: run.levels_completed)
        games[short_id] = {
            "env_game_id": str(environment.id),
            "score": round(_as_float(environment.score), 6),
            "levels_completed": _as_int(best.levels_completed),
            "levels_available": len(best.level_scores or []),
        }
    return games


def scorecard_aggregate(scorecard: Any) -> dict:
    return {
        "official_score": round(_as_float(scorecard.score), 6),
        "total_levels_completed": _as_int(scorecard.total_levels_completed),
        "total_levels": _as_int(scorecard.total_levels),
        "total_environments_completed": _as_int(scorecard.total_environments_completed),
        "total_environments": _as_int(scorecard.total_environments),
        "total_actions": _as_int(scorecard.total_actions),
    }


def build_reference(report: dict) -> dict:
    """Turn a live run into the committed reference document."""
    games = []
    for game in report["games"]:
        game_id = game["game_id"]
        exclude_reason = KNOWN_EXCLUDES.get(game_id)
        games.append(
            {
                "game_id": game_id,
                "env_game_id": game["env_game_id"],
                "status": game["status"],
                "known_exclude": bool(exclude_reason),
                "exclude_reason": exclude_reason,
                "reproduced": bool(game["reproduced"]),
                "step_for_step": bool(game["step_for_step"]),
                "expected_score": round(_as_float(game["score"]), 2),
                "levels_completed": _as_int(game["levels_completed"]),
                "levels_available": _as_int(game["levels_available"]),
            }
        )
    games.sort(key=lambda game: game["game_id"])
    return {
        "schema": SCHEMA,
        "generated_at": report["generated_at"],
        "operation_mode": "OFFLINE",
        "interpreter": report["interpreter"],
        "dataset": report["dataset"],
        "dataset_note": (
            "unpacked ARC-AGI-3 human replay archive (771 MB); not stored in the "
            "repository, default /tmp/arc3-replays"
        ),
        "score_tolerance": SCORE_TOLERANCE,
        "replay_count": report["replay_count"],
        "reproduced_count": report["reproduced_count"],
        "known_excludes": [
            {"game_id": game_id, "reason": reason}
            for game_id, reason in sorted(KNOWN_EXCLUDES.items())
        ],
        "aggregate": report["aggregate"],
        "games": games,
    }


def compare_report(report: dict, reference: dict, tolerance: float) -> list[str]:
    """Return one human-readable drift line per mismatch (empty means pass)."""
    drifts: list[str] = []

    def check_exact(label: str, actual: Any, expected: Any) -> None:
        if actual != expected:
            drifts.append(f"{label}: {actual!r} != reference {expected!r}")

    check_exact(
        "replay_count", report["replay_count"], reference.get("replay_count")
    )
    check_exact(
        "reproduced_count",
        report["reproduced_count"],
        reference.get("reproduced_count"),
    )

    actual_aggregate = report["aggregate"]
    expected_aggregate = reference.get("aggregate", {})
    actual_score = _as_float(actual_aggregate.get("official_score"))
    expected_score = _as_float(expected_aggregate.get("official_score"))
    if abs(actual_score - expected_score) > tolerance:
        drifts.append(
            f"aggregate official_score: {actual_score:.4f} != reference "
            f"{expected_score:.4f} (tolerance {tolerance})"
        )
    for field in (
        "total_levels_completed",
        "total_levels",
        "total_environments_completed",
        "total_environments",
        "total_actions",
    ):
        check_exact(
            f"aggregate {field}",
            actual_aggregate.get(field),
            expected_aggregate.get(field),
        )

    expected_games = {game["game_id"]: game for game in reference.get("games", [])}
    seen: set[str] = set()
    for game in report["games"]:
        game_id = game["game_id"]
        seen.add(game_id)
        expected = expected_games.get(game_id)
        if expected is None:
            drifts.append(f"{game_id}: present in run but missing from reference")
            continue
        check_exact(
            f"{game_id} env_game_id", game["env_game_id"], expected.get("env_game_id")
        )
        check_exact(f"{game_id} status", game["status"], expected.get("status"))
        check_exact(
            f"{game_id} reproduced", bool(game["reproduced"]), expected.get("reproduced")
        )
        check_exact(
            f"{game_id} step_for_step",
            bool(game["step_for_step"]),
            expected.get("step_for_step"),
        )
        check_exact(
            f"{game_id} levels_completed",
            _as_int(game["levels_completed"]),
            expected.get("levels_completed"),
        )
        check_exact(
            f"{game_id} levels_available",
            _as_int(game["levels_available"]),
            expected.get("levels_available"),
        )
        actual_game_score = _as_float(game["score"])
        expected_game_score = _as_float(expected.get("expected_score"))
        if abs(actual_game_score - expected_game_score) > tolerance:
            drifts.append(
                f"{game_id} score: {actual_game_score:.4f} != reference "
                f"{expected_game_score:.4f} (tolerance {tolerance})"
            )
    for game_id in sorted(set(expected_games) - seen):
        drifts.append(f"{game_id}: in reference but missing from run")
    return drifts


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #


def format_table(report: dict) -> str:
    """Human-readable per-game table (explicit named results, not prose)."""
    lines = [
        f"{'game':6} {'env_game_id':16} {'repro':5} {'steps':5} {'score':>8} "
        f"{'levels':>7}  status",
        "-" * 68,
    ]
    for game in report["games"]:
        status = game["status"]
        if game["status"] == "ok" and game["game_id"] in KNOWN_EXCLUDES:
            status = "ok [KNOWN EXCLUDE]"
        if game["error"]:
            status = f"{game['status']}: {game['error']}"
        lines.append(
            f"{game['game_id']:6} {str(game['env_game_id']):16} "
            f"{'yes' if game['reproduced'] else 'no':5} "
            f"{'yes' if game['step_for_step'] else 'no':5} "
            f"{game['score']:8.2f} "
            f"{game['levels_completed']}/{game['levels_available']:<5} {status}"
        )
    aggregate = report["aggregate"]
    lines += [
        "-" * 68,
        f"reproduced: {report['reproduced_count']}/{report['replay_count']}",
        f"official score: {aggregate['official_score']:.4f}",
        f"levels: {aggregate['total_levels_completed']}/{aggregate['total_levels']}  "
        f"environments: {aggregate['total_environments_completed']}/"
        f"{aggregate['total_environments']}  actions: {aggregate['total_actions']}",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Human-replay calibration check for the ARC-AGI-3 harness (OFFLINE only).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help=(
            "Unpacked ARC-AGI-3 human replay archive root (default: "
            f"{DEFAULT_DATASET}). It is 771 MB and is not in the repository; when it "
            "is absent the check skips with exit 0."
        ),
    )
    parser.add_argument(
        "--env-dir",
        default=DEFAULT_ENV_DIR,
        help=(
            "OFFLINE environments_dir to load the local builds from "
            f"(default: {DEFAULT_ENV_DIR}). Must not be inside the repository's "
            "tracked environment_files."
        ),
    )
    parser.add_argument(
        "--env-source",
        default=str(ENV_SOURCE),
        help=(
            "Repository source copied into --env-dir when the copy is missing "
            f"(default: {ENV_SOURCE})."
        ),
    )
    parser.add_argument(
        "--refresh-env-dir",
        action="store_true",
        help="Re-copy --env-source into --env-dir before running (no deletion).",
    )
    parser.add_argument(
        "--reference",
        default=str(REFERENCE_PATH),
        help=f"Reference JSON to compare against (default: {REFERENCE_PATH}).",
    )
    parser.add_argument(
        "--write-reference",
        action="store_true",
        help=(
            "Overwrite --reference with the observed run and exit 0. Only for a "
            "reviewed change to the expectations; the normal check never writes."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show the engine's per-step logs and captured scorer stdout.",
    )
    return parser.parse_args(argv)


def _write_reference_file(report: dict, reference_path: Path) -> int:
    reference = build_reference(report)
    reference_path.write_text(json.dumps(reference, indent=2) + "\n", encoding="utf-8")
    print(f"wrote reference {reference_path} from this run")
    for game_id, reason in sorted(KNOWN_EXCLUDES.items()):
        game = next((item for item in report["games"] if item["game_id"] == game_id), None)
        if game is None:
            continue
        if game["reproduced"] or _as_float(game["score"]) > SCORE_TOLERANCE:
            print(
                f"WARNING: known exclude {game_id} now reproduces={game['reproduced']} "
                f"score={game['score']:.2f}; the exclude reason ({reason}) no longer "
                "holds and the reference needs review.",
                file=sys.stderr,
            )
    print("RESULT: reference_written")
    return EXIT_PASS


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    dataset = Path(args.dataset).expanduser()

    if not dataset.exists():
        print(
            f"RESULT: skipped (dataset_absent) -- {dataset} does not exist. The 771 MB "
            "ARC-AGI-3 human replay archive is not stored in the repository."
        )
        return EXIT_PASS

    replay_root = resolve_replay_root(dataset)
    if replay_root is None:
        print(
            f"RESULT: skipped (dataset_empty) -- {dataset} exists but holds no "
            "environment_files/ replay folders."
        )
        return EXIT_PASS

    game_dirs = sorted(child for child in replay_root.iterdir() if child.is_dir())
    if not game_dirs:
        print(
            f"RESULT: skipped (dataset_empty) -- {replay_root} holds no per-game "
            "replay folders."
        )
        return EXIT_PASS

    configure_logging(args.verbose)

    arc_agi, engine_error = load_engine()
    if arc_agi is None:
        print(
            "RESULT: skipped (arc_agi_unavailable) -- arc_agi is not importable with "
            f"this interpreter ({sys.version.split()[0]}): {engine_error}. Run with "
            "data/arc-agi-3-agents/.venv/bin/python."
        )
        return EXIT_PASS

    env_dir, env_error = resolve_env_dir(
        Path(args.env_dir), Path(args.env_source), args.refresh_env_dir
    )
    if env_dir is None:
        print(f"RESULT: setup_error -- {env_error}", file=sys.stderr)
        return EXIT_SETUP

    recordings_dir = str(Path(tempfile.gettempdir()) / "arc3_calibration_recordings")
    try:
        arcade = arc_agi.Arcade(
            operation_mode=arc_agi.OperationMode.OFFLINE,
            environments_dir=str(env_dir),
            recordings_dir=recordings_dir,
            logger=logging.getLogger("arc3_calibration.arcade"),
        )
        scorecard_id = arcade.create_scorecard()
    except Exception as exc:  # noqa: BLE001 - setup failure must not traceback
        print(
            "RESULT: setup_error -- could not initialise the OFFLINE arcade: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return EXIT_SETUP

    print(
        f"ARC-AGI-3 human-replay calibration (OFFLINE)\n"
        f"dataset: {replay_root}\nenv_dir: {env_dir}\ngames:   {len(game_dirs)}\n"
    )

    started = time.monotonic()
    games: list[dict] = []
    for game_dir in game_dirs:
        game_id = game_dir.name
        record = replay_game(arcade, scorecard_id, game_id, game_dir)
        games.append(record)
        if args.verbose:
            logger.info(
                "%s reproduced=%s error=%s",
                game_id,
                record["reproduced"],
                record["error"],
            )

    scorecard = collect_scorecard(arcade, scorecard_id, args.verbose)
    if scorecard is None:
        print("RESULT: setup_error -- close_scorecard() returned None", file=sys.stderr)
        return EXIT_SETUP

    scores = scorecard_games(scorecard)
    for game in games:
        game_score = scores.get(game["game_id"])
        if game_score is None:
            if game["status"] == "ok":
                game["status"] = "unscored"
                game["error"] = game["error"] or "game missing from the closed scorecard"
            continue
        game["score"] = game_score["score"]
        game["levels_completed"] = game_score["levels_completed"]
        game["levels_available"] = game_score["levels_available"]

    report = {
        "schema": SCHEMA,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "operation_mode": "OFFLINE",
        "interpreter": sys.executable,
        "dataset": str(dataset),
        "env_dir": str(env_dir),
        "wall_time_s": round(time.monotonic() - started, 3),
        "replay_count": len(games),
        "reproduced_count": sum(1 for game in games if game["reproduced"]),
        "games": games,
        "aggregate": scorecard_aggregate(scorecard),
    }

    print(format_table(report))
    print()

    if args.write_reference:
        return _write_reference_file(report, Path(args.reference))

    reference_path = Path(args.reference)
    if not reference_path.exists():
        print(
            f"RESULT: setup_error -- reference {reference_path} not found; run with "
            "--write-reference only after reviewing the expectations",
            file=sys.stderr,
        )
        return EXIT_SETUP
    try:
        reference = json.loads(reference_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(
            f"RESULT: setup_error -- could not read reference {reference_path}: {exc}",
            file=sys.stderr,
        )
        return EXIT_SETUP

    tolerance = _as_float(reference.get("score_tolerance"), SCORE_TOLERANCE)
    drifts = compare_report(report, reference, tolerance)

    print(
        f"reference: {reference_path}\n"
        f"expected reproduced {reference.get('reproduced_count')}/"
        f"{reference.get('replay_count')}, aggregate "
        f"{reference.get('aggregate', {}).get('official_score')} "
        f"(tolerance {tolerance})"
    )
    if drifts:
        print()
        for line in drifts:
            print(f"DRIFT: {line}")
        print(f"\nRESULT: drift (exit {EXIT_DRIFT}) -- {len(drifts)} difference(s)")
        return EXIT_DRIFT
    print("\nRESULT: pass")
    return EXIT_PASS


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Tests for the local ARC-AGI-3 scoring harness.

Runs two ways:

    python3 experiments/test_arc3_local_eval.py
    data/arc-agi-3-agents/.venv/bin/python -m pytest experiments/test_arc3_local_eval.py

Everything here drives synthetic frames and a fake agent/environment, so the whole
suite runs in milliseconds and **plays no real game** and never reads or writes
``data/``. The point is that the harness's arithmetic is trustworthy: which frame
index a level was cleared at, whether unreached levels still occupy the mean's
divisor, and what the aggregate is. If the harness is wrong, every conclusion drawn
from a sweep is wrong too.

The module under test delegates scoring to the official
``arc_agi.EnvironmentScoreCalculator``. To keep the suite independent of the engine
install it is exercised through an injectable calculator factory: the real scorer
when ``arc_agi`` is importable (the venv interpreter), otherwise a local
re-statement of the official arithmetic. One test compares the two directly.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import arc3_local_eval  # noqa: E402
from arc3_local_eval import (  # noqa: E402
    GamePlan,
    aggregate,
    build_levels,
    level_action_counts,
    play_episode,
    run_sweep,
    score_levels,
)

logging.getLogger("arc3_local_eval").setLevel(logging.CRITICAL)


class StandaloneSkip(Exception):
    """Skip signal for the standalone runner (pytest has its own)."""


def _skip(reason: str) -> None:
    """Skip a test under pytest, or raise a skip under the standalone runner."""
    if "pytest" in sys.modules:  # pragma: no cover - depends on how the suite is run
        import pytest

        pytest.skip(reason)
    raise StandaloneSkip(reason)


# --------------------------------------------------------------------------- #
# An independent statement of the official arithmetic
# --------------------------------------------------------------------------- #
#
# Written here, not imported from the module under test: `arc3_local_eval` never
# implements this formula, it delegates to `EnvironmentScoreCalculator`. Reproducing
# it locally is what lets the wiring tests run without the engine installed.


class ReferenceScoreCalculator:
    """`EnvironmentScoreCalculator`'s contract, re-stated for interpreters without arc_agi."""

    def __init__(self, id: str | None = None) -> None:
        self.id = id
        self.level_scores: list[float] = []
        self.levels_completed = 0

    def add_level(
        self,
        completed: bool,
        actions_taken: int,
        baseline_actions: int,
        game_id: str | None = None,
    ) -> None:
        if completed and actions_taken > 0:
            self.level_scores.append(min((baseline_actions / actions_taken) * 100, 100.0))
            self.levels_completed += 1
        else:
            self.level_scores.append(0.0)

    def to_score(self):
        score = (
            sum(self.level_scores) / len(self.level_scores) if self.level_scores else 0.0
        )
        return _FakeScore(
            score=score,
            levels_completed=self.levels_completed,
            level_scores=list(self.level_scores),
        )


class _FakeScore:
    def __init__(self, score, levels_completed, level_scores):
        self.score = score
        self.levels_completed = levels_completed
        self.level_scores = level_scores


def _calculator_factory(game_id: str):
    """The real scorer when the engine is importable, else the local re-statement."""
    if arc3_local_eval.ARC_AGI_AVAILABLE:
        official = arc3_local_eval._arc_agi_module().EnvironmentScoreCalculator
        return official(id=game_id)
    return ReferenceScoreCalculator(id=game_id)


# --------------------------------------------------------------------------- #
# Fakes: no real game, no real clock
# --------------------------------------------------------------------------- #


class FakeFrame:
    def __init__(self, levels_completed: int, win_levels: int, state: str = "NOT_FINISHED"):
        self.levels_completed = levels_completed
        self.win_levels = win_levels
        self.state = state


class FakeEnv:
    observation_space = None


class FakeAgent:
    """Deterministic stand-in for MyAgent, replaying a scripted level counter.

    ``series[k]`` is the ``levels_completed`` value the k-th action produces, so a
    transition index is just an index into the list.
    """

    def __init__(
        self,
        series,
        win_levels: int,
        max_actions: int = 10,
        raise_at: int | None = None,
    ) -> None:
        self.game_id = "fake"
        self.frames = [FakeFrame(0, win_levels)]
        self.action_counter = 0
        self.MAX_ACTIONS = max_actions
        self.arc_env = FakeEnv()
        self.timer = 0.0
        self.cleaned_up = False
        self._series = list(series)
        self._raise_at = raise_at

    @property
    def fps(self) -> float:
        """Present for parity with Agent; the harness logs it when INFO is on."""
        return 0.0

    def is_done(self, frames, latest_frame) -> bool:
        return latest_frame.state == "WIN"

    def _convert_raw_frame_data(self, raw):
        return None

    def choose_action(self, frames, latest_frame):
        return None

    def take_action(self, action):
        if self._raise_at is not None and self.action_counter == self._raise_at:
            raise RuntimeError("boom")
        if self.action_counter < len(self._series):
            level = self._series[self.action_counter]
        else:
            level = self._series[-1] if self._series else 0
        win_levels = self.frames[-1].win_levels
        state = "WIN" if level >= win_levels else "NOT_FINISHED"
        return FakeFrame(level, win_levels, state)

    def append_frame(self, frame) -> None:
        self.frames.append(frame)

    def cleanup(self) -> None:
        self.cleaned_up = True


class CountingClock:
    """A monotonic clock that advances by ``step`` on every read."""

    def __init__(self, step: float = 1.0) -> None:
        self._now = 0.0
        self._step = step

    def __call__(self) -> float:
        value = self._now
        self._now += self._step
        return value


def _record(
    game_id: str,
    *,
    score: float = 0.0,
    levels_completed: int = 0,
    levels_available: int = 1,
    wall_time_s: float = 0.0,
    status: str = "ok",
    timed_out: bool = False,
    error: str | None = None,
) -> dict:
    return {
        "game_id": game_id,
        "status": status,
        "score": score,
        "levels_completed": levels_completed,
        "levels_available": levels_available,
        "wall_time_s": wall_time_s,
        "timed_out": timed_out,
        "error": error,
    }


# --------------------------------------------------------------------------- #
# Level-transition detection and actions_taken derivation
# --------------------------------------------------------------------------- #


def test_level_transition_indices_become_actions_taken():
    levels = level_action_counts([0, 0, 1, 1, 1, 2], 3)
    assert [outcome.completed for outcome in levels] == [True, True, False]
    assert [outcome.actions_taken for outcome in levels] == [2, 3, 0]


def test_first_level_cost_is_its_transition_index():
    """frames[0] is the pre-episode frame, so level 1 costs exactly its index."""
    assert level_action_counts([0, 0, 0, 1], 1)[0].actions_taken == 3


def test_actions_taken_is_the_gap_between_consecutive_transitions():
    levels = level_action_counts([0, 1, 0, 0, 0, 0, 2], 2)
    assert levels[0].actions_taken == 1
    assert levels[1].actions_taken == 5


def test_unreached_levels_are_reported_not_completed():
    levels, anomalies = build_levels([0, 1], [10, 20, 30], 3)
    assert len(levels) == 3
    assert [outcome.completed for outcome in levels] == [True, False, False]
    assert levels[0].actions_taken == 1
    assert levels[0].baseline_actions == 10
    assert levels[2].baseline_actions == 30
    assert anomalies == []


def test_level_counter_jump_is_flagged():
    levels, anomalies = build_levels([0, 0, 2], [10, 10], 2)
    assert [outcome.completed for outcome in levels] == [True, True]
    assert levels[1].actions_taken == 0
    assert any("jumped" in anomaly for anomaly in anomalies), anomalies


def test_non_monotonic_level_counter_is_flagged():
    _, anomalies = build_levels([0, 1, 0], [10, 10], 2)
    assert any("monotonic" in anomaly for anomaly in anomalies), anomalies


# --------------------------------------------------------------------------- #
# Metadata robustness: bad baselines must be reported, never crash
# --------------------------------------------------------------------------- #


def test_baseline_shorter_than_win_levels_is_reported():
    """The official scorer iterates baseline_actions, so that is the denominator."""
    levels, anomalies = build_levels([0, 1], [10], win_levels=5)
    assert len(levels) == 1
    assert any("win_levels is 5" in anomaly for anomaly in anomalies), anomalies


def test_missing_baselines_fall_back_to_win_levels():
    levels, anomalies = build_levels([0, 1], [], win_levels=2)
    assert len(levels) == 2
    assert all(outcome.baseline_actions == 0 for outcome in levels)
    assert any("no baseline_actions" in anomaly for anomaly in anomalies), anomalies


def test_game_without_metadata_is_not_scoreable():
    levels, anomalies = build_levels([0], [], win_levels=None)
    assert levels == []
    assert any("not scoreable" in anomaly for anomaly in anomalies), anomalies


# --------------------------------------------------------------------------- #
# Scoring wiring: the official scorer, fed the full level set
# --------------------------------------------------------------------------- #


def test_score_uses_all_levels_as_denominator():
    """One level solved against three baselines must score 1/3 of the level, not 1/1."""
    levels, _ = build_levels([0, 1], [10, 10, 10], 3)
    score = score_levels(levels, "g", _calculator_factory)
    assert score.levels_completed == 1
    assert len(score.level_scores) == 3
    assert round(score.score, 4) == round(100 / 3, 4)


def test_level_score_is_capped_at_100():
    """Beating the human baseline (1 action vs baseline 10) caps the level at 100."""
    levels, _ = build_levels([0, 1], [10, 10], 2)
    score = score_levels(levels, "g", _calculator_factory)
    assert score.level_scores[0] == 100.0
    assert round(score.score, 6) == 50.0


def test_reference_formula_matches_the_official_scorer():
    """The local test formula must agree with the scorer the harness delegates to."""
    if not arc3_local_eval.ARC_AGI_AVAILABLE:
        _skip("arc_agi is not importable with this interpreter")
    official = arc3_local_eval._arc_agi_module().EnvironmentScoreCalculator
    levels, _ = build_levels([0, 0, 1, 0, 0, 2], [40, 30, 20, 10], 4)
    from_official = score_levels(levels, "g", lambda gid: official(id=gid))
    from_reference = score_levels(levels, "g", lambda gid: ReferenceScoreCalculator(id=gid))
    assert round(from_official.score, 9) == round(from_reference.score, 9)
    assert from_official.levels_completed == from_reference.levels_completed
    assert len(from_official.level_scores) == len(from_reference.level_scores)


# --------------------------------------------------------------------------- #
# Logging suppression
# --------------------------------------------------------------------------- #


def test_configure_logging_suppresses_info_unless_verbose():
    """The framework logs one line per action at INFO; it must be off by default."""
    root = logging.getLogger()
    original_level = root.level
    original_handlers = list(root.handlers)
    try:
        arc3_local_eval.configure_logging(verbose=False)
        assert root.level >= logging.WARNING
        arc3_local_eval.configure_logging(verbose=True)
        assert root.level == logging.INFO
    finally:
        root.setLevel(original_level)
        root.handlers[:] = original_handlers


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #


def test_aggregate_mean_and_totals():
    games = [
        _record("a", score=1.5, levels_completed=2, levels_available=8, wall_time_s=0.5),
        _record("b", score=0.0, levels_completed=0, levels_available=6, wall_time_s=0.25),
    ]
    result = aggregate(games)
    assert result["mean_score"] == 0.75
    assert result["total_levels_completed"] == 2
    assert result["total_levels_available"] == 14
    assert result["total_wall_time_s"] == 0.75


def test_aggregate_counts_failures_as_zero_in_the_mean():
    games = [
        _record("boom", status="error", error="RuntimeError: no env", levels_available=6),
        _record("slow", status="timeout", timed_out=True, levels_available=7),
        _record("good", score=3.0, levels_completed=1, levels_available=6),
    ]
    result = aggregate(games)
    assert result["mean_score"] == 1.0
    assert result["games_errored"] == 1
    assert result["games_timed_out"] == 1
    assert result["total_levels_available"] == 19


def test_aggregate_of_no_games_is_zero():
    result = aggregate([])
    assert result["mean_score"] == 0.0
    assert result["total_levels_available"] == 0


# --------------------------------------------------------------------------- #
# Budgets
# --------------------------------------------------------------------------- #


def test_time_budget_ends_the_episode_and_is_reported():
    agent = FakeAgent([0], win_levels=8, max_actions=100_000)
    timed_out = play_episode(agent, time_budget_s=5.0, clock=CountingClock(step=1.0))
    assert timed_out is True
    assert agent.cleaned_up is True
    assert 0 < agent.action_counter < agent.MAX_ACTIONS


def test_no_time_budget_means_the_action_budget_is_the_only_limit():
    """The upstream loop is ``action_counter <= MAX_ACTIONS``, i.e. budget + 1 actions."""
    agent = FakeAgent([0], win_levels=8, max_actions=25)
    timed_out = play_episode(agent, time_budget_s=None, clock=time.monotonic)
    assert timed_out is False
    assert agent.action_counter == 26


# --------------------------------------------------------------------------- #
# The sweep never aborts on a bad game
# --------------------------------------------------------------------------- #


def test_sweep_continues_after_environment_and_agent_failures():
    plans = [
        GamePlan("bad-env", "bad-env-1", (5,)),
        GamePlan("missing-env", "missing-env-1", (5,)),
        GamePlan("bad-agent", "bad-agent-1", (5,)),
        GamePlan("good", "good-1", (5,)),
    ]

    def make_env(plan):
        if plan.game_id == "bad-env":
            raise RuntimeError("no environment")
        if plan.game_id == "missing-env":
            return None
        return FakeEnv()

    def make_agent(plan, env):
        if plan.game_id == "bad-agent":
            raise RuntimeError("agent exploded")
        return FakeAgent([1], win_levels=1, max_actions=3)

    games = run_sweep(
        plans,
        make_env=make_env,
        make_agent=make_agent,
        time_budget_s=None,
        calculator_factory=_calculator_factory,
    )
    assert [game["status"] for game in games] == ["error", "error", "error", "ok"]
    assert games[0]["error"].startswith("RuntimeError")
    assert "returned None" in games[1]["error"]
    assert games[3]["score"] == 100.0
    assert games[3]["levels_completed"] == 1


def test_sweep_keeps_partial_results_when_the_agent_raises_mid_episode():
    plan = GamePlan("g", "g-1", (10, 10))
    agent = FakeAgent([0, 0, 1, 1], win_levels=2, max_actions=50, raise_at=4)
    games = run_sweep(
        [plan],
        make_env=lambda plan: FakeEnv(),
        make_agent=lambda plan, env: agent,
        time_budget_s=None,
        calculator_factory=_calculator_factory,
    )
    assert games[0]["status"] == "error"
    assert "boom" in games[0]["error"]
    assert games[0]["actions_used"] == 4
    assert games[0]["levels_completed"] == 1
    assert games[0]["levels_available"] == 2


def test_sweep_continues_after_a_timeout():
    plans = [GamePlan("slow", "slow-1", (10,)), GamePlan("fast", "fast-1", (10,))]

    def make_agent(plan, env):
        if plan.game_id == "slow":
            return FakeAgent([0], win_levels=1, max_actions=100_000)
        return FakeAgent([1], win_levels=1, max_actions=10)

    games = run_sweep(
        plans,
        make_env=lambda plan: FakeEnv(),
        make_agent=make_agent,
        time_budget_s=3.0,
        clock=CountingClock(step=1.0),
        calculator_factory=_calculator_factory,
    )
    assert games[0]["timed_out"] is True
    assert games[0]["status"] == "timeout"
    assert games[1]["status"] == "ok"
    assert games[1]["score"] == 100.0


# --------------------------------------------------------------------------- #
# Reference numbers: the whole pipeline, deterministically
# --------------------------------------------------------------------------- #
#
# A verified sweep of all 25 games at a 500-action budget produced a mean score of
# 0.6084 with 3 of 183 levels completed (ar25 1/8, m0r0 1/6, r11l 1/6). The live
# agent is stochastically seeded (MyAgent.__init__ calls random.seed(time)), so a
# re-run lands on a different sample; these constants pin the arithmetic by
# replaying the reference sweep's *level-transition indices* through the real
# pipeline. If the harness is correct this must come out at exactly 0.6084 / 3 of 183.

# len(baseline_actions) per game = the mean's divisor, read from each environment's
# metadata.json. Only the count matters for the 22 games that complete no level.
REFERENCE_LEVEL_COUNTS = {
    "ar25": 8,
    "bp35": 9,
    "cd82": 6,
    "cn04": 6,
    "dc22": 6,
    "ft09": 6,
    "g50t": 7,
    "ka59": 7,
    "lf52": 10,
    "lp85": 8,
    "ls20": 7,
    "m0r0": 6,
    "r11l": 6,
    "re86": 8,
    "s5i5": 8,
    "sb26": 8,
    "sc25": 6,
    "sk48": 8,
    "sp80": 6,
    "su15": 9,
    "tn36": 7,
    "tr87": 6,
    "tu93": 9,
    "vc33": 7,
    "wa30": 9,
}
# First-level human baseline for the only three games the reference sweep completes.
REFERENCE_FIRST_BASELINE = {"ar25": 32, "m0r0": 30, "r11l": 22}
# Frame index at which level 1 was first observed in the reference sweep.
REFERENCE_FIRST_LEVEL_INDEX = {"ar25": 266, "m0r0": 142, "r11l": 36}
REFERENCE_MEAN_SCORE = 0.6084
REFERENCE_LEVELS_COMPLETED = 3
REFERENCE_LEVELS_AVAILABLE = 183


def _reference_plans() -> list[GamePlan]:
    plans = []
    for game_id, n_levels in REFERENCE_LEVEL_COUNTS.items():
        first_baseline = REFERENCE_FIRST_BASELINE.get(game_id, 1)
        # Unreached levels never contribute to the score, so only the first baseline
        # needs to be the real one.
        baselines = (first_baseline,) + (1,) * (n_levels - 1)
        plans.append(GamePlan(game_id, f"{game_id}-ref", baselines))
    return plans


def _reference_agent(plan: GamePlan) -> FakeAgent:
    index = REFERENCE_FIRST_LEVEL_INDEX.get(plan.game_id)
    n_levels = len(plan.baseline_actions)
    if index is None:
        return FakeAgent([0], win_levels=n_levels, max_actions=5)
    # `FakeAgent.take_action(c)` produces the frame that lands at `frames[c + 1]`, so a
    # transition at frame index `index` needs its 1 at `series[index - 1]`.
    return FakeAgent([0] * (index - 1) + [1], win_levels=n_levels, max_actions=index + 5)


def test_reference_sweep_numbers_reproduce():
    games = run_sweep(
        _reference_plans(),
        make_env=lambda plan: FakeEnv(),
        make_agent=lambda plan, env: _reference_agent(plan),
        time_budget_s=None,
        calculator_factory=_calculator_factory,
    )
    result = aggregate(games)
    assert result["total_levels_completed"] == REFERENCE_LEVELS_COMPLETED
    assert result["total_levels_available"] == REFERENCE_LEVELS_AVAILABLE
    assert round(result["mean_score"], 4) == REFERENCE_MEAN_SCORE
    scoring_games = {game["game_id"]: game for game in games if game["levels_completed"]}
    assert set(scoring_games) == set(REFERENCE_FIRST_LEVEL_INDEX)


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #


def _main() -> int:
    tests = [
        (name, fn)
        for name, fn in sorted(globals().items())
        if name.startswith("test_") and callable(fn)
    ]
    failures = []
    skipped = 0
    for name, fn in tests:
        try:
            fn()
        except StandaloneSkip as exc:
            skipped += 1
            print(f"  skip  {name}: {exc}")
        except AssertionError as exc:
            failures.append((name, exc))
            print(f"  FAIL  {name}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failures.append((name, exc))
            print(f"  ERROR {name}: {type(exc).__name__}: {exc}")
        else:
            print(f"  ok    {name}")
    passed = len(tests) - len(failures) - skipped
    print(f"\n{passed}/{len(tests)} passed, {skipped} skipped")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())

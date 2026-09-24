#!/usr/bin/env python3
"""Tests for ``scripts/build_arc2_notebook.py``.

Run with::

    python3 -m pytest scripts/test_build_arc2_notebook.py -q

These tests are the anti-drift guarantee for the ARC-AGI-2 submission notebook. The
notebook used to be hand-maintained, so it drifted 111 lines behind
``src/arc2_dual_process_solver.py`` and kept two dead paths in the artifact that was
actually submitted (a ``rot270`` primitive that raised on every call, and a dispatch that
called ``(name, fn)`` tuples). A drift detector that cannot detect drift is worse than
none, so the hand-edit case is exercised end to end through the CLI.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent
sys.path.insert(0, str(SCRIPTS_DIR))

import build_arc2_notebook as builder  # noqa: E402  (path inserted above)

SOLVER_SOURCE_PATH = REPO_ROOT / "src" / "arc2_dual_process_solver.py"
COMMITTED_NOTEBOOK_PATH = (
    REPO_ROOT / "notebooks" / "arc2_submission_kernel" / "kaggle_arc2_submission.ipynb"
)
BUILDER_PATH = SCRIPTS_DIR / "build_arc2_notebook.py"

HEADER_CELL_INDEX = 0
SOLVER_CELL_INDEX = 2
SUBMISSION_CELL_INDEX = 3

#: The dead primitive as it was deployed: ``zip(*g)`` is already an iterator, so
#: subscripting it raises ``TypeError: 'zip' object is not subscriptable``.
BROKEN_ROT270 = "lambda g: [list(x) for x in zip(*g)[::-1]]"


def _cell_source(notebook: dict, index: int) -> str:
    source = notebook["cells"][index]["source"]
    if isinstance(source, list):
        source = "".join(source)
    return source


def _run_cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(BUILDER_PATH), *args],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )


@pytest.fixture(scope="module")
def solver_source() -> str:
    return SOLVER_SOURCE_PATH.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def generated_notebook() -> dict:
    return builder.build_notebook(
        SOLVER_SOURCE_PATH.read_text(encoding="utf-8")
    )


def test_solver_cell_body_is_byte_identical_to_source(solver_source, generated_notebook):
    cell = _cell_source(generated_notebook, SOLVER_CELL_INDEX)
    # Byte equality is the assertion, not similarity: this is what removes the drift
    # that hid the two dead paths.
    assert cell == solver_source, "embedded solver body drifted from src/"
    assert cell.encode("utf-8") == solver_source.encode("utf-8")


def test_generation_is_idempotent(solver_source):
    first = builder.generate(SOLVER_SOURCE_PATH)
    second = builder.generate(SOLVER_SOURCE_PATH)
    assert first == second, "generating twice must produce identical bytes"
    assert first.encode("utf-8") == second.encode("utf-8")


def test_check_exits_zero_on_freshly_generated_notebook(tmp_path):
    notebook_path = tmp_path / "kaggle_arc2_submission.ipynb"

    write = _run_cli("--notebook", str(notebook_path))
    assert write.returncode == 0, write.stderr
    assert notebook_path.exists()

    check = _run_cli("--check", "--notebook", str(notebook_path))
    assert check.returncode == 0, check.stderr
    assert "OK" in check.stdout


def test_check_fails_when_notebook_is_edited_by_hand(tmp_path):
    notebook_path = tmp_path / "kaggle_arc2_submission.ipynb"

    write = _run_cli("--notebook", str(notebook_path))
    assert write.returncode == 0, write.stderr

    original = notebook_path.read_text(encoding="utf-8")
    # A genuine hand edit: reintroduce the dead primitive in the solver cell. The header
    # quotes the broken form as documentation, so mutate only the working form (which
    # appears in cell 2 alone).
    working_rot270 = "lambda g: [list(x) for x in zip(*g)][::-1]"
    assert working_rot270 in original
    notebook_path.write_text(
        original.replace(working_rot270, BROKEN_ROT270, 1),
        encoding="utf-8",
    )

    check = _run_cli("--check", "--notebook", str(notebook_path))
    assert check.returncode != 0, "hand edit was not detected"
    assert "DRIFT" in check.stderr
    assert "cell 2" in check.stderr


def _load_generated_solver_cell(generated_notebook, tmp_path):
    """Import the generated solver cell as a module, so it is exercised as written."""
    source = _cell_source(generated_notebook, SOLVER_CELL_INDEX)
    module_path = tmp_path / "generated_solver_cell.py"
    module_path.write_text(source, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("generated_solver_cell", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_rot270_in_notebook_is_the_working_form(generated_notebook, tmp_path):
    source = _cell_source(generated_notebook, SOLVER_CELL_INDEX)
    assert BROKEN_ROT270 not in source, "the dead rot270 form must be gone"

    module = _load_generated_solver_cell(generated_notebook, tmp_path)
    ops = dict(module.get_d4_ops())
    assert "rot270" in ops
    grid = [[1, 2], [3, 4]]
    result = ops["rot270"](grid)
    # It must return a grid, not raise TypeError: 'zip' object is not subscriptable.
    assert result == [[2, 4], [1, 3]]


def test_dispatch_is_exercised_on_a_matching_task(generated_notebook, tmp_path):
    """The fixed dispatch must actually run when a candidate matches."""
    module = _load_generated_solver_cell(generated_notebook, tmp_path)
    # identity on a task with two identical-shape pairs: the matching branch is taken.
    task = {
        "train": [{"input": [[1, 2], [3, 4]], "output": [[1, 2], [3, 4]]}],
        "test": [{"input": [[5, 6], [7, 8]]}],
    }
    trace: dict = {}
    attempts = module.solve_arc_task(task, trace)
    assert trace["n_matching"] > 0
    assert attempts[0]["attempt_1"] == [[5, 6], [7, 8]]


def test_dispatch_calls_the_function_not_the_tuple(generated_notebook):
    source = _cell_source(generated_notebook, SOLVER_CELL_INDEX)
    assert "matching_solvers[0][1](inp)" in source
    assert "matching_solvers[0](inp)" not in source, "dead dispatch form is back"


def test_submission_cell_reports_coverage_not_a_key_count(generated_notebook):
    source = _cell_source(generated_notebook, SUBMISSION_CELL_INDEX)
    # It must not present a submission-key count as evidence of work...
    assert "with {len(submission)} tasks" not in source
    assert "successfully" not in source
    assert "\u2705" not in source
    # ...and it must report coverage instead.
    assert "solve_arc_task(task, trace)" in source
    assert "n_matching" in source
    assert "tasks_with_coverage" in source
    assert "tasks_fell_back" in source
    assert "echo_input" in source
    assert "zero_grid" in source
    # It must not print a success line when coverage is zero.
    assert "tasks_with_coverage == 0" in source
    assert "WARNING" in source


def test_header_states_provenance_and_does_not_claim_the_kernel_is_current(
    generated_notebook,
):
    header = _cell_source(generated_notebook, HEADER_CELL_INDEX)
    assert "src/arc2_dual_process_solver.py" in header
    assert "scripts/build_arc2_notebook.py" in header
    # The deployed revision carried the two dead paths; the header says so.
    assert "2026-09-23" in header
    assert "rot270" in header
    assert "matching_solvers" in header
    # It must not claim the live kernel implements what was scored.
    assert "Official code submission" not in header
    assert "coverage" in header.lower()


def test_committed_notebook_matches_the_generator():
    """The on-disk artifact must be exactly what the builder produces (invariant 1)."""
    check = _run_cli("--check")
    assert check.returncode == 0, check.stderr

#!/usr/bin/env python3
"""Tests for ``scripts/build_arc3_notebook.py``.

Run with::

    python3 -m pytest scripts/test_build_arc3_notebook.py -q

The tests are the anti-drift guarantee for the ARC-AGI-3 submission notebook: the agent
cell must be byte-identical to ``src/``, generation must be idempotent, and ``--check``
must actually detect a hand edit. A drift detector that cannot detect drift is worse
than none, so the hand-edit case is exercised end to end through the CLI.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent
sys.path.insert(0, str(SCRIPTS_DIR))

import build_arc3_notebook as builder  # noqa: E402  (path inserted above)

AGENT_SOURCE_PATH = REPO_ROOT / "src" / "arc3_spatial_memory_agent.py"
COMMITTED_NOTEBOOK_PATH = (
    REPO_ROOT / "notebooks" / "arc3_submission_kernel" / "submission.ipynb"
)
BUILDER_PATH = SCRIPTS_DIR / "build_arc3_notebook.py"
WRITEFILE_LINE = "%%writefile /tmp/my_agent.py"

AGENT_CELL_INDEX = 2
RUNNER_CELL_INDEX = 3
HEADER_CELL_INDEX = 0


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
def agent_source() -> str:
    return AGENT_SOURCE_PATH.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def generated_notebook() -> dict:
    return builder.build_notebook(
        AGENT_SOURCE_PATH.read_text(encoding="utf-8")
    )


def test_agent_cell_body_is_byte_identical_to_source(agent_source, generated_notebook):
    cell = _cell_source(generated_notebook, AGENT_CELL_INDEX)
    assert cell.startswith(WRITEFILE_LINE + "\n"), "agent cell must start with %%writefile"

    body = cell.split("\n", 1)[1]
    # Byte equality is the assertion, not similarity: this is what removes the
    # random-seeding drift the notebook used to ship.
    assert body == agent_source, "embedded agent body drifted from src/"
    assert body.encode("utf-8") == agent_source.encode("utf-8")


def test_generation_is_idempotent():
    first = builder.generate(AGENT_SOURCE_PATH)
    second = builder.generate(AGENT_SOURCE_PATH)
    assert first == second, "generating twice must produce identical bytes"
    assert first.encode("utf-8") == second.encode("utf-8")


def test_check_exits_zero_on_freshly_generated_notebook(tmp_path):
    notebook_path = tmp_path / "submission.ipynb"

    write = _run_cli("--notebook", str(notebook_path))
    assert write.returncode == 0, write.stderr
    assert notebook_path.exists()

    check = _run_cli("--check", "--notebook", str(notebook_path))
    assert check.returncode == 0, check.stderr
    assert "OK" in check.stdout


def test_check_fails_when_notebook_is_edited_by_hand(tmp_path):
    notebook_path = tmp_path / "submission.ipynb"

    write = _run_cli("--notebook", str(notebook_path))
    assert write.returncode == 0, write.stderr

    original = notebook_path.read_text(encoding="utf-8")
    # A genuine hand edit: revert the agent cell's MAX_ACTIONS to the old value.
    assert "MAX_ACTIONS = 500" in original
    notebook_path.write_text(
        original.replace("MAX_ACTIONS = 500", "MAX_ACTIONS = 80", 1),
        encoding="utf-8",
    )

    check = _run_cli("--check", "--notebook", str(notebook_path))
    assert check.returncode != 0, "hand edit was not detected"
    assert "DRIFT" in check.stderr
    assert "cell 2" in check.stderr


def test_runner_cell_declares_the_deployed_seed(generated_notebook):
    runner = _cell_source(generated_notebook, RUNNER_CELL_INDEX)
    assert "ARC3_AGENT_SEED=16" in runner
    assert runner.index("ARC3_AGENT_SEED=16") < runner.index("python main.py --agent myagent")

    # One source of truth only: the override is *set* in exactly one place, as an
    # inline assignment on the agent launch command (not in the .env block and not
    # as an os.environ assignment).
    assignment_cells = [
        i
        for i in range(len(generated_notebook["cells"]))
        if any(
            line.strip().startswith("ARC3_AGENT_SEED=")
            for line in _cell_source(generated_notebook, i).splitlines()
        )
    ]
    assert assignment_cells == [RUNNER_CELL_INDEX]
    assert "os.environ" not in runner


def test_header_drops_false_provenance(generated_notebook):
    header = _cell_source(generated_notebook, HEADER_CELL_INDEX)
    assert "agent/my_agent.py" not in header
    assert "scripts/build_notebook.py" not in header
    assert "make submit" not in header

    # ... and states the truth instead.
    assert "src/arc3_spatial_memory_agent.py" in header
    assert "scripts/build_arc3_notebook.py" in header
    assert "ARC3_AGENT_SEED=16" in header

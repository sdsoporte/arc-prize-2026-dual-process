"""Generates official submission.json for ARC-AGI-2 benchmark using Dual-Process Program Synthesis."""

import json
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.arc2_dual_process_solver import solve_arc_task


def generate_submission():
    project_root = Path(__file__).resolve().parent.parent
    test_challenges_path = project_root / "data" / "arc-agi-2" / "arc-agi_test_challenges.json"
    sample_sub_path = project_root / "data" / "arc-agi-2" / "sample_submission.json"
    output_path = project_root / "submissions" / "arc2_submission.json"

    with open(test_challenges_path, "r", encoding="utf-8") as f:
        test_tasks = json.load(f)

    with open(sample_sub_path, "r", encoding="utf-8") as f:
        sample_sub = json.load(f)

    print(f"Loaded {len(test_tasks)} test tasks to predict.")
    submission = {}

    for idx, (task_id, task) in enumerate(test_tasks.items()):
        submission[task_id] = solve_arc_task(task)

    # Verification against sample_submission keys
    missing_keys = set(sample_sub.keys()) - set(submission.keys())
    assert len(missing_keys) == 0, f"Missing keys in submission: {missing_keys}"

    # Verify formatting of each task output
    for task_id, attempts in submission.items():
        assert isinstance(attempts, list), f"Expected list of attempts for task {task_id}"
        for attempt in attempts:
            assert "attempt_1" in attempt and "attempt_2" in attempt, f"Malformed attempt keys in {task_id}"
            assert isinstance(attempt["attempt_1"], list), f"attempt_1 must be list of lists in {task_id}"
            assert isinstance(attempt["attempt_2"], list), f"attempt_2 must be list of lists in {task_id}"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(submission, f)

    file_size_kb = output_path.stat().st_size / 1024
    print(f"✅ Submission successfully generated at {output_path} ({file_size_kb:.1f} KB)")
    print(f"   Total tasks submitted: {len(submission)}")


if __name__ == "__main__":
    generate_submission()

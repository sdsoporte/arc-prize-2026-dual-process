"""Benchmark suite for testing ARC-AGI-2 System 2 solver on real tasks."""

import json
import time
from pathlib import Path
import sys

src_dir = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(src_dir))

from arc2_dual_process_solver import solve_arc_task, grids_equal

def benchmark_dataset(challenges_path: Path, solutions_path: Path, max_tasks: int = 200):
    if not challenges_path.exists() or not solutions_path.exists():
        print(f"Skipping {challenges_path.name}: file not found")
        return

    with open(challenges_path, "r") as f:
        challenges = json.load(f)
    with open(solutions_path, "r") as f:
        solutions = json.load(f)

    task_ids = list(challenges.keys())[:max_tasks]
    solved_count = 0
    total = len(task_ids)
    t0 = time.perf_counter()

    for tid in task_ids:
        task = challenges[tid]
        ground_truth = solutions.get(tid, [])
        if not ground_truth:
            continue

        preds = solve_arc_task(task)
        # Check if attempt_1 or attempt_2 matches ground truth for all test pairs
        all_test_solved = True
        for idx, true_out in enumerate(ground_truth):
            p1 = preds[idx]["attempt_1"] if idx < len(preds) else None
            p2 = preds[idx]["attempt_2"] if idx < len(preds) else None
            if not (grids_equal(p1, true_out) or grids_equal(p2, true_out)):
                all_test_solved = False
                break
        if all_test_solved:
            solved_count += 1

    elapsed = time.perf_counter() - t0
    rate = (solved_count / total) * 100 if total > 0 else 0
    print(f"[{challenges_path.stem}] Solved {solved_count}/{total} ({rate:.2f}%) in {elapsed:.2f}s")
    return solved_count, total

if __name__ == "__main__":
    base = Path(__file__).resolve().parent.parent / "data" / "arc-agi-2"
    print("=== BASELINE SOLVER BENCHMARK ===")
    benchmark_dataset(
        base / "arc-agi_evaluation_challenges.json",
        base / "arc-agi_evaluation_solutions.json",
        max_tasks=120
    )
    benchmark_dataset(
        base / "arc-agi_training_challenges.json",
        base / "arc-agi_training_solutions.json",
        max_tasks=200
    )

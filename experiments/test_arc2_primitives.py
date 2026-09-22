"""Test suite and benchmark for ARC-AGI-2 transformation primitives."""

import json
from pathlib import Path
from typing import Callable, Any, Dict, List, Tuple


def grids_equal(g1: Any, g2: Any) -> bool:
    if len(g1) != len(g2):
        return False
    if len(g1) == 0:
        return True
    if len(g1[0]) != len(g2[0]):
        return False
    for r1, r2 in zip(g1, g2):
        if r1 != r2:
            return False
    return True


# --- 1. Color Map Synthesizer ---
def check_color_map(train_pairs: list[dict]) -> Callable[[list[list[int]]], list[list[int]]] | None:
    """Checks if output is a deterministic 1-to-1 or many-to-1 color mapping of input with same dimensions."""
    mapping = {}
    for pair in train_pairs:
        inp, out = pair["input"], pair["output"]
        if len(inp) != len(out) or len(inp[0]) != len(out[0]):
            return None
        for r_in, r_out in zip(inp, out):
            for c_in, c_out in zip(r_in, r_out):
                if c_in in mapping and mapping[c_in] != c_out:
                    return None
                mapping[c_in] = c_out

    def fn(g: list[list[int]]) -> list[list[int]]:
        return [[mapping.get(c, c) for c in row] for row in g]
    return fn


# --- 2. Dihedral D4 Symmetries ---
def get_d4_ops() -> list[tuple[str, Callable[[list[list[int]]], list[list[int]]]]]:
    return [
        ("rot90", lambda g: [list(x) for x in zip(*g[::-1])]),
        ("rot180", lambda g: [r[::-1] for r in g[::-1]]),
        ("rot270", lambda g: [list(x) for x in zip(*g)[::-1]]),
        ("flip_h", lambda g: [r[::-1] for r in g]),
        ("flip_v", lambda g: g[::-1]),
        ("transpose", lambda g: [list(x) for x in zip(*g)]),
        ("anti_transpose", lambda g: [list(x) for x in zip(*[r[::-1] for r in g[::-1]])]),
    ]


# --- 3. Bounding Box & Crop Heuristics ---
def crop_box(g: list[list[int]], r1: int, r2: int, c1: int, c2: int) -> list[list[int]]:
    return [row[c1:c2 + 1] for row in g[r1:r2 + 1]]


def get_crop_ops(train_pairs: list[dict]) -> list[tuple[str, Callable[[list[list[int]]], list[list[int]]]]]:
    ops = []
    # Crop all non-zero
    def crop_nonzero(g: list[list[int]]) -> list[list[int]]:
        pts = [(r, c) for r, row in enumerate(g) for c, val in enumerate(row) if val != 0]
        if not pts:
            return g
        return crop_box(g, min(r for r, c in pts), max(r for r, c in pts), min(c for r, c in pts), max(c for r, c in pts))
    ops.append(("crop_nonzero", crop_nonzero))

    # Crop by specific color
    for color in range(10):
        def make_crop_color(col=color):
            def fn(g: list[list[int]]) -> list[list[int]]:
                pts = [(r, c) for r, row in enumerate(g) for c, val in enumerate(row) if val == col]
                if not pts:
                    return g
                return crop_box(g, min(r for r, c in pts), max(r for r, c in pts), min(c for r, c in pts), max(c for r, c in pts))
            return fn
        ops.append((f"crop_color_{color}", make_crop_color()))

    return ops


# --- 4. Tiling & Mirroring ---
def get_mirror_ops() -> list[tuple[str, Callable[[list[list[int]]], list[list[int]]]]]:
    return [
        ("mirror_h", lambda g: [r + r[::-1] for r in g]),
        ("mirror_v", lambda g: g + g[::-1]),
        ("mirror_hv", lambda g: [r + r[::-1] for r in g] + [r + r[::-1] for r in g[::-1]]),
        ("tile_2x2", lambda g: [r + r for r in g] + [r + r for r in g]),
        ("tile_3x3", lambda g: [r * 3 for r in g] * 3),
    ]


# --- 5. Connected Components ---
def get_connected_components(g: list[list[int]], bg: int = 0) -> list[list[tuple[int, int]]]:
    h = len(g)
    w = len(g[0]) if h > 0 else 0
    visited = [[False] * w for _ in range(h)]
    components = []

    for r in range(h):
        for c in range(w):
            if g[r][c] != bg and not visited[r][c]:
                comp = []
                queue = [(r, c)]
                visited[r][c] = True
                while queue:
                    curr_r, curr_c = queue.pop(0)
                    comp.append((curr_r, curr_c))
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = curr_r + dr, curr_c + dc
                        if 0 <= nr < h and 0 <= nc < w and not visited[nr][nc] and g[nr][nc] != bg:
                            visited[nr][nc] = True
                            queue.append((nr, nc))
                components.append(comp)
    return components


def extract_component_grid(g: list[list[int]], comp: list[tuple[int, int]], bg: int = 0) -> list[list[int]]:
    min_r = min(r for r, c in comp)
    max_r = max(r for r, c in comp)
    min_c = min(c for r, c in comp)
    max_c = max(c for r, c in comp)
    pts_set = set(comp)
    return [
        [g[r][c] if (r, c) in pts_set else bg for c in range(min_c, max_c + 1)]
        for r in range(min_r, max_r + 1)
    ]


def get_object_ops() -> list[tuple[str, Callable[[list[list[int]]], list[list[int]]]]]:
    def largest_object(g: list[list[int]]) -> list[list[int]]:
        comps = get_connected_components(g)
        if not comps:
            return g
        largest = max(comps, key=len)
        return extract_component_grid(g, largest)

    def smallest_object(g: list[list[int]]) -> list[list[int]]:
        comps = get_connected_components(g)
        if not comps:
            return g
        smallest = min(comps, key=len)
        return extract_component_grid(g, smallest)

    return [
        ("largest_object", largest_object),
        ("smallest_object", smallest_object),
    ]


# --- 6. Two-Stage Synthesizer: Spatial / Extraction -> Color Mapping ---
def find_spatial_color_composition(train_pairs: list[dict], spatial_ops: list[tuple[str, Callable]]) -> list[tuple[str, Callable]]:
    valid_programs = []

    for s_name, s_fn in spatial_ops:
        try:
            # Check if dimensions match for all pairs after spatial transform
            intermediate_pairs = []
            dim_mismatch = False
            for p in train_pairs:
                inter = s_fn(p["input"])
                out = p["output"]
                if len(inter) != len(out) or len(inter[0]) != len(out[0]):
                    dim_mismatch = True
                    break
                intermediate_pairs.append({"input": inter, "output": out})

            if dim_mismatch:
                continue

            # Check if there is a consistent color mapping
            cmap_fn = check_color_map(intermediate_pairs)
            if cmap_fn is not None:
                def make_prog(s=s_fn, c=cmap_fn):
                    return lambda g: c(s(g))
                valid_programs.append((f"{s_name}+color_map", make_prog()))
        except Exception:
            continue

    return valid_programs


# --- Benchmark Suite ---
def benchmark_on_dataset(challenges_path: str, solutions_path: str, max_tasks: int = 120):
    with open(challenges_path, "r", encoding="utf-8") as f:
        challenges = json.load(f)
    with open(solutions_path, "r", encoding="utf-8") as f:
        solutions = json.load(f)

    solved_count = 0
    solved_by_type = {}
    total = min(len(challenges), max_tasks)

    for task_id in list(challenges.keys())[:total]:
        task = challenges[task_id]
        train_pairs = task.get("train", [])
        test_inputs = task.get("test", [])
        if task_id not in solutions:
            continue
        expected = solutions[task_id]

        candidate_fns: list[tuple[str, Callable]] = []

        # 1. Direct Color mapping
        cmap_fn = check_color_map(train_pairs)
        if cmap_fn:
            candidate_fns.append(("color_map", cmap_fn))

        # Spatial ops list
        spatial_ops = []
        spatial_ops.extend(get_d4_ops())
        spatial_ops.extend(get_crop_ops(train_pairs))
        spatial_ops.extend(get_mirror_ops())
        spatial_ops.extend(get_object_ops())

        # 2. Pure spatial ops
        candidate_fns.extend(spatial_ops)

        # 3. Two-Stage Composition: Spatial -> Color Map
        candidate_fns.extend(find_spatial_color_composition(train_pairs, spatial_ops))

        # Check candidate functions against 100% train pairs
        valid_solvers = []
        for name, fn in candidate_fns:
            try:
                matches_train = True
                for p in train_pairs:
                    pred = fn(p["input"])
                    if not grids_equal(pred, p["output"]):
                        matches_train = False
                        break
                if matches_train:
                    valid_solvers.append((name, fn))
            except Exception:
                continue

        # Evaluate on test set
        if valid_solvers:
            test_correct = False
            for name, solver in valid_solvers:
                all_tests_match = True
                for t_idx, true_out in enumerate(expected):
                    try:
                        pred = solver(test_inputs[t_idx]["input"])
                        if not grids_equal(pred, true_out):
                            all_tests_match = False
                            break
                    except Exception:
                        all_tests_match = False
                        break
                if all_tests_match:
                    test_correct = True
                    solved_by_type[name] = solved_by_type.get(name, 0) + 1
                    break
            if test_correct:
                solved_count += 1

    print(f"Total benchmarked: {total}")
    print(f"Tasks solved exactly: {solved_count} ({solved_count / total * 100:.2f}%)")
    print(f"Breakdown by primitive: {solved_by_type}")


if __name__ == "__main__":
    import sys
    print("--- EVALUATION DATASET ---")
    benchmark_on_dataset(
        "data/arc-agi-2/arc-agi_evaluation_challenges.json",
        "data/arc-agi-2/arc-agi_evaluation_solutions.json",
        max_tasks=120
    )
    print("\n--- TRAINING DATASET (First 200 tasks) ---")
    benchmark_on_dataset(
        "data/arc-agi-2/arc-agi_training_challenges.json",
        "data/arc-agi-2/arc-agi_training_solutions.json",
        max_tasks=200
    )

"""Bounded Symbolic DSL Synthesizer (System 2) for ARC-AGI-2."""

from __future__ import annotations
from typing import Callable, Any
import numpy as np


def to_list(grid: Any) -> list[list[int]]:
    if isinstance(grid, np.ndarray):
        return grid.tolist()
    return [list(r) for r in grid]


# 1. Isometries / Geometries
def op_identity(g: list[list[int]]) -> list[list[int]]:
    return [r[:] for r in g]


def op_rot90(g: list[list[int]]) -> list[list[int]]:
    return [list(x) for x in zip(*g[::-1])]


def op_rot180(g: list[list[int]]) -> list[list[int]]:
    return [r[::-1] for r in g[::-1]]


def op_rot270(g: list[list[int]]) -> list[list[int]]:
    return [list(x) for x in zip(*g)[::-1]]


def op_flip_h(g: list[list[int]]) -> list[list[int]]:
    return [r[::-1] for r in g]


def op_flip_v(g: list[list[int]]) -> list[list[int]]:
    return g[::-1]


def op_transpose(g: list[list[int]]) -> list[list[int]]:
    return [list(x) for x in zip(*g)]


# 2. Bounding Box & Cropping
def op_crop_nonzero(g: list[list[int]]) -> list[list[int]]:
    rows = len(g)
    cols = len(g[0]) if rows > 0 else 0
    non_zeros = [(r, c) for r in range(rows) for c in range(cols) if g[r][c] != 0]
    if not non_zeros:
        return g
    min_r = min(r for r, c in non_zeros)
    max_r = max(r for r, c in non_zeros)
    min_c = min(c for r, c in non_zeros)
    max_c = max(c for r, c in non_zeros)
    return [row[min_c:max_c + 1] for row in g[min_r:max_r + 1]]


def op_tile_2x2(g: list[list[int]]) -> list[list[int]]:
    return [r + r for r in g] + [r + r for r in g]


def op_tile_3x3(g: list[list[int]]) -> list[list[int]]:
    return [r * 3 for r in g] * 3


# 3. Gravity primitives
def op_gravity_down(g: list[list[int]]) -> list[list[int]]:
    h = len(g)
    w = len(g[0]) if h > 0 else 0
    res = [[0 for _ in range(w)] for _ in range(h)]
    for c in range(w):
        col_vals = [g[r][c] for r in range(h) if g[r][c] != 0]
        zeros = [0] * (h - len(col_vals))
        new_col = zeros + col_vals
        for r in range(h):
            res[r][c] = new_col[r]
    return res


def op_gravity_up(g: list[list[int]]) -> list[list[int]]:
    h = len(g)
    w = len(g[0]) if h > 0 else 0
    res = [[0 for _ in range(w)] for _ in range(h)]
    for c in range(w):
        col_vals = [g[r][c] for r in range(h) if g[r][c] != 0]
        zeros = [0] * (h - len(col_vals))
        new_col = col_vals + zeros
        for r in range(h):
            res[r][c] = new_col[r]
    return res


# 4. Color transformations
def make_color_substitutor(c_src: int, c_dst: int) -> Callable[[list[list[int]]], list[list[int]]]:
    def fn(g: list[list[int]]) -> list[list[int]]:
        return [[c_dst if cell == c_src else cell for cell in r] for r in g]
    return fn


def make_color_swap(c1: int, c2: int) -> Callable[[list[list[int]]], list[list[int]]]:
    def fn(g: list[list[int]]) -> list[list[int]]:
        res = []
        for r in g:
            new_r = []
            for cell in r:
                if cell == c1:
                    new_r.append(c2)
                elif cell == c2:
                    new_r.append(c1)
                else:
                    new_r.append(cell)
            res.append(new_r)
        return res
    return fn


BASE_OPERATORS: list[tuple[str, Callable[[list[list[int]]], list[list[int]]]]] = [
    ("identity", op_identity),
    ("rot90", op_rot90),
    ("rot180", op_rot180),
    ("rot270", op_rot270),
    ("flip_h", op_flip_h),
    ("flip_v", op_flip_v),
    ("transpose", op_transpose),
    ("crop_nonzero", op_crop_nonzero),
    ("tile_2x2", op_tile_2x2),
    ("tile_3x3", op_tile_3x3),
    ("gravity_down", op_gravity_down),
    ("gravity_up", op_gravity_up),
]


def grids_equal(g1: list[list[int]], g2: list[list[int]]) -> bool:
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


def solve_task_symbolic(train_pairs: list[dict], test_inputs: list[dict]) -> list[dict[str, list[list[int]]]] | None:
    """Bounded System 2 program synthesis over candidate DSL library."""
    if not train_pairs:
        return None

    # Determine colors present across training pairs
    train_colors = set()
    for p in train_pairs:
        for r in p["input"]:
            train_colors.update(r)
        for r in p["output"]:
            train_colors.update(r)

    candidate_ops = list(BASE_OPERATORS)

    # Add dynamic color substitutions observed in train
    for c1 in train_colors:
        for c2 in train_colors:
            if c1 != c2:
                candidate_ops.append((f"sub_{c1}_{c2}", make_color_substitutor(c1, c2)))
                candidate_ops.append((f"swap_{c1}_{c2}", make_color_swap(c1, c2)))

    # Phase 1: Search exact single operator consistency on ALL training demonstrations
    matching_ops = []
    for name, op in candidate_ops:
        try:
            solved_all = True
            for pair in train_pairs:
                pred = op(pair["input"])
                if not grids_equal(pred, pair["output"]):
                    solved_all = False
                    break
            if solved_all:
                matching_ops.append((name, op))
        except Exception:
            continue

    # Phase 2: If no single operator matches, search 2-step compositions
    if not matching_ops:
        geo_ops = BASE_OPERATORS[:7]
        for name1, op1 in geo_ops:
            for name2, op2 in candidate_ops[:20]:
                try:
                    solved_all = True
                    for pair in train_pairs:
                        pred = op2(op1(pair["input"]))
                        if not grids_equal(pred, pair["output"]):
                            solved_all = False
                            break
                    if solved_all:
                        matching_ops.append((f"{name1}+{name2}", lambda g, o1=op1, o2=op2: o2(o1(g))))
                        if len(matching_ops) >= 2:
                            break
                except Exception:
                    continue
            if len(matching_ops) >= 2:
                break

    if not matching_ops:
        return None

    # Apply best matched operator to test inputs
    best_op = matching_ops[0][1]
    second_op = matching_ops[1][1] if len(matching_ops) > 1 else best_op

    results = []
    for test_item in test_inputs:
        inp = test_item["input"]
        attempt_1 = best_op(inp)
        attempt_2 = second_op(inp)
        results.append({
            "attempt_1": attempt_1,
            "attempt_2": attempt_2,
        })

    return results

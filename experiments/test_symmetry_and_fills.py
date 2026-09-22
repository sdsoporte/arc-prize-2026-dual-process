"""Test symmetry completion, overlay, and flood fill primitives on ARC training tasks."""

import json
from typing import Callable, Any


def grids_equal(g1: Any, g2: Any) -> bool:
    if len(g1) != len(g2) or (g1 and len(g1[0]) != len(g2[0])):
        return False
    return all(r1 == r2 for r1, r2 in zip(g1, g2))


# 1. Symmetry Overlays (Pixelwise non-zero OR / union)
def overlay_grids(g1: list[list[int]], g2: list[list[int]]) -> list[list[int]]:
    return [
        [c2 if c1 == 0 else c1 for c1, c2 in zip(r1, r2)]
        for r1, r2 in zip(g1, g2)
    ]


def get_symmetry_overlays() -> list[tuple[str, Callable]]:
    rot90 = lambda g: [list(x) for x in zip(*g[::-1])]
    rot180 = lambda g: [r[::-1] for r in g[::-1]]
    flip_h = lambda g: [r[::-1] for r in g]
    flip_v = lambda g: g[::-1]

    return [
        ("overlay_flip_h", lambda g: overlay_grids(g, flip_h(g))),
        ("overlay_flip_v", lambda g: overlay_grids(g, flip_v(g))),
        ("overlay_rot180", lambda g: overlay_grids(g, rot180(g))),
        ("overlay_all_sym", lambda g: overlay_grids(overlay_grids(g, flip_h(g)), flip_v(g))),
    ]


# 2. Enclosed Hole Filling
def fill_enclosed_holes(g: list[list[int]], fill_color: int) -> list[list[int]]:
    h = len(g)
    w = len(g[0]) if h > 0 else 0
    # Find all cells reachable from grid boundary without crossing non-zero cells
    visited = [[False] * w for _ in range(h)]
    queue = []
    for r in range(h):
        for c in range(w):
            if (r == 0 or r == h - 1 or c == 0 or c == w - 1) and g[r][c] == 0:
                visited[r][c] = True
                queue.append((r, c))

    while queue:
        r, c = queue.pop(0)
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and not visited[nr][nc] and g[nr][nc] == 0:
                visited[nr][nc] = True
                queue.append((nr, nc))

    # All unvisited cells with value 0 are enclosed!
    return [
        [fill_color if (g[r][c] == 0 and not visited[r][c]) else g[r][c] for c in range(w)]
        for r in range(h)
    ]


def test_extra_primitives():
    with open("data/arc-agi-2/arc-agi_training_challenges.json") as f:
        train_c = json.load(f)
    with open("data/arc-agi-2/arc-agi_training_solutions.json") as f:
        train_s = json.load(f)

    solved_overlay = 0
    solved_fill = 0
    solved_tasks = []

    for tid, t in train_c.items():
        pairs = t.get("train", [])
        tests = t.get("test", [])
        if tid not in train_s:
            continue
        exp = train_s[tid]

        # Test symmetry overlays
        for name, fn in get_symmetry_overlays():
            try:
                if all(grids_equal(fn(p["input"]), p["output"]) for p in pairs):
                    if all(grids_equal(fn(tests[i]["input"]), exp[i]) for i in range(len(tests))):
                        solved_overlay += 1
                        solved_tasks.append((tid, name))
                        break
            except Exception:
                continue

        # Test hole filling for colors 1..9
        for fill_c in range(1, 10):
            try:
                fn = lambda g, c=fill_c: fill_enclosed_holes(g, c)
                if all(grids_equal(fn(p["input"]), p["output"]) for p in pairs):
                    if all(grids_equal(fn(tests[i]["input"]), exp[i]) for i in range(len(tests))):
                        solved_fill += 1
                        solved_tasks.append((tid, f"fill_holes_{fill_c}"))
                        break
            except Exception:
                continue

    print(f"Symmetry Overlay Solved: {solved_overlay}")
    print(f"Hole Filling Solved: {solved_fill}")
    print(f"Sample solved tasks: {solved_tasks[:10]}")


if __name__ == "__main__":
    test_extra_primitives()

"""Dual-Process Solver for ARC-AGI-2: High-speed heuristic screening + symbolic program synthesis."""

from __future__ import annotations
from typing import Callable, Any, Dict, List, Tuple
import json
from pathlib import Path


def grids_equal(g1: Any, g2: Any) -> bool:
    if len(g1) != len(g2) or (g1 and len(g1[0]) != len(g2[0])):
        return False
    return all(r1 == r2 for r1, r2 in zip(g1, g2))


# --- 1. Symmetries & Dihedrals (D4) ---
def get_d4_ops() -> list[tuple[str, Callable[[list[list[int]]], list[list[int]]]]]:
    return [
        ("identity", lambda g: [r[:] for r in g]),
        ("rot90", lambda g: [list(x) for x in zip(*g[::-1])]),
        ("rot180", lambda g: [r[::-1] for r in g[::-1]]),
        ("rot270", lambda g: [list(x) for x in zip(*g)[::-1]]),
        ("flip_h", lambda g: [r[::-1] for r in g]),
        ("flip_v", lambda g: g[::-1]),
        ("transpose", lambda g: [list(x) for x in zip(*g)]),
        ("anti_transpose", lambda g: [list(x) for x in zip(*[r[::-1] for r in g[::-1]])]),
    ]


# --- 2. Color Mapping ---
def check_color_map(train_pairs: list[dict]) -> Callable[[list[list[int]]], list[list[int]]] | None:
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


# --- 3. Bounding Boxes & Cropping ---
def crop_box(g: list[list[int]], r1: int, r2: int, c1: int, c2: int) -> list[list[int]]:
    return [row[c1:c2 + 1] for row in g[r1:r2 + 1]]


def get_crop_ops() -> list[tuple[str, Callable[[list[list[int]]], list[list[int]]]]]:
    ops = []
    def crop_nonzero(g: list[list[int]]) -> list[list[int]]:
        pts = [(r, c) for r, row in enumerate(g) for c, val in enumerate(row) if val != 0]
        if not pts:
            return g
        return crop_box(g, min(r for r, c in pts), max(r for r, c in pts), min(c for r, c in pts), max(c for r, c in pts))
    ops.append(("crop_nonzero", crop_nonzero))

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


# --- 4. Connected Components & Object Extraction ---
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
        return extract_component_grid(g, max(comps, key=len)) if comps else g

    def smallest_object(g: list[list[int]]) -> list[list[int]]:
        comps = get_connected_components(g)
        return extract_component_grid(g, min(comps, key=len)) if comps else g

    return [
        ("largest_object", largest_object),
        ("smallest_object", smallest_object),
    ]


# --- 5. Symmetry Overlays ---
def overlay_grids(g1: list[list[int]], g2: list[list[int]]) -> list[list[int]]:
    return [
        [c2 if c1 == 0 else c1 for c1, c2 in zip(r1, r2)]
        for r1, r2 in zip(g1, g2)
    ]


def get_overlay_ops() -> list[tuple[str, Callable[[list[list[int]]], list[list[int]]]]]:
    rot180 = lambda g: [r[::-1] for r in g[::-1]]
    flip_h = lambda g: [r[::-1] for r in g]
    flip_v = lambda g: g[::-1]

    return [
        ("overlay_flip_h", lambda g: overlay_grids(g, flip_h(g))),
        ("overlay_flip_v", lambda g: overlay_grids(g, flip_v(g))),
        ("overlay_rot180", lambda g: overlay_grids(g, rot180(g))),
        ("overlay_hv", lambda g: overlay_grids(overlay_grids(g, flip_h(g)), flip_v(g))),
    ]


# --- 6. Enclosed Hole Filling ---
def fill_enclosed_holes(g: list[list[int]], fill_color: int) -> list[list[int]]:
    h = len(g)
    w = len(g[0]) if h > 0 else 0
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

    return [
        [fill_color if (g[r][c] == 0 and not visited[r][c]) else g[r][c] for c in range(w)]
        for r in range(h)
    ]


# --- 7. Rescaling (Upsample & Downsample) ---
def upsample_block(g: list[list[int]], k: int) -> list[list[int]]:
    res = []
    for row in g:
        expanded_row = []
        for cell in row:
            expanded_row.extend([cell] * k)
        for _ in range(k):
            res.append(expanded_row[:])
    return res


def downsample_block(g: list[list[int]], k: int, mode: str = "nonzero") -> list[list[int]]:
    h, w = len(g), len(g[0]) if g else 0
    out_h, out_w = h // k, w // k
    res = [[0] * out_w for _ in range(out_h)]
    for r in range(out_h):
        for c in range(out_w):
            block = [g[r * k + dr][c * k + dc] for dr in range(k) for dc in range(k)]
            if mode == "nonzero":
                nz = [v for v in block if v != 0]
                res[r][c] = max(set(nz), key=nz.count) if nz else 0
            elif mode == "mode":
                res[r][c] = max(set(block), key=block.count)
    return res


# --- Master Task Solver ---
def solve_arc_task(task: dict) -> list[dict[str, list[list[int]]]]:
    train_pairs = task.get("train", [])
    test_inputs = task.get("test", [])

    if not train_pairs:
        return [{"attempt_1": t["input"], "attempt_2": t["input"]} for t in test_inputs]

    # Collect all candidate transformation functions
    candidates: list[tuple[str, Callable[[list[list[int]]], list[list[int]]]]] = []

    # 1. Direct Color Map
    cmap_fn = check_color_map(train_pairs)
    if cmap_fn:
        candidates.append(("color_map", cmap_fn))

    # 2. D4 Symmetries
    d4_ops = get_d4_ops()
    candidates.extend(d4_ops)

    # 3. Crops & Objects
    crop_ops = get_crop_ops()
    candidates.extend(crop_ops)

    obj_ops = get_object_ops()
    candidates.extend(obj_ops)

    # 4. Overlays
    candidates.extend(get_overlay_ops())

    # 5. Hole Filling (colors 1..9)
    for c in range(1, 10):
        candidates.append((f"fill_holes_{c}", lambda g, col=c: fill_enclosed_holes(g, col)))

    # 6. Scaling (upsample / downsample)
    h_in, h_out = len(train_pairs[0]["input"]), len(train_pairs[0]["output"])
    w_in, w_out = len(train_pairs[0]["input"][0]), len(train_pairs[0]["output"][0])
    if h_in > 0 and w_in > 0 and h_out > 0 and w_out > 0:
        if h_out % h_in == 0 and w_out % w_in == 0 and h_out // h_in == w_out // w_in:
            k = h_out // h_in
            if k > 1:
                candidates.append((f"upsample_{k}", lambda g, scale=k: upsample_block(g, scale)))
        elif h_in % h_out == 0 and w_in % w_out == 0 and h_in // h_out == w_in // w_out:
            k = h_in // h_out
            if k > 1:
                candidates.append((f"downsample_{k}_nz", lambda g, scale=k: downsample_block(g, scale, "nonzero")))
                candidates.append((f"downsample_{k}_mode", lambda g, scale=k: downsample_block(g, scale, "mode")))

    # 7. Two-Stage Composition: Spatial -> Color Map
    spatial_pool = d4_ops + crop_ops + obj_ops
    for s_name, s_fn in spatial_pool:
        try:
            intermediate_pairs = []
            dim_match = True
            for p in train_pairs:
                inter = s_fn(p["input"])
                out = p["output"]
                if len(inter) != len(out) or len(inter[0]) != len(out[0]):
                    dim_match = False
                    break
                intermediate_pairs.append({"input": inter, "output": out})
            if dim_match:
                cm = check_color_map(intermediate_pairs)
                if cm:
                    candidates.append((f"{s_name}+color_map", lambda g, s=s_fn, c=cm: c(s(g))))
        except Exception:
            continue

    # Evaluate candidates against 100% of training demonstrations
    matching_solvers: list[Callable] = []
    for name, fn in candidates:
        try:
            if all(grids_equal(fn(p["input"]), p["output"]) for p in train_pairs):
                matching_solvers.append(fn)
        except Exception:
            continue

    # Fallback generators for attempt_2 or when no exact solver found
    # Check if dimensions are constant across train outputs
    const_out_h = len(train_pairs[0]["output"])
    const_out_w = len(train_pairs[0]["output"][0])
    is_const_dim = all(len(p["output"]) == const_out_h and len(p["output"][0]) == const_out_w for p in train_pairs)

    same_dim_as_in = all(len(p["input"]) == len(p["output"]) and len(p["input"][0]) == len(p["output"][0]) for p in train_pairs)

    # Dominant color
    fg_colors = [c for p in train_pairs for r in p["output"] for c in r if c != 0]
    dominant_color = max(set(fg_colors), key=fg_colors.count) if fg_colors else 1

    def fallback_1(inp: list[list[int]]) -> list[list[int]]:
        if same_dim_as_in:
            return [r[:] for r in inp]
        elif is_const_dim:
            return [[0] * const_out_w for _ in range(const_out_h)]
        return [r[:] for r in inp]

    def fallback_2(inp: list[list[int]]) -> list[list[int]]:
        base = fallback_1(inp)
        return [[dominant_color if cell != 0 else 0 for cell in r] for r in base]

    results = []
    for t_item in test_inputs:
        inp = t_item["input"]
        if len(matching_solvers) >= 2:
            attempt_1 = matching_solvers[0](inp)
            attempt_2 = matching_solvers[1](inp)
        elif len(matching_solvers) == 1:
            attempt_1 = matching_solvers[0](inp)
            attempt_2 = fallback_2(inp)
        else:
            attempt_1 = fallback_1(inp)
            attempt_2 = fallback_2(inp)

        results.append({
            "attempt_1": attempt_1,
            "attempt_2": attempt_2,
        })

    return results

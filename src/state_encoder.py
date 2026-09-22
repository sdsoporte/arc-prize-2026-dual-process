"""Encodes ARC-AGI-3 environment state into concise structured representations for System 1."""

from typing import Any, Dict, List, Optional
import numpy as np
from arcengine import FrameData, GameState


def summarize_grid(grid: List[List[int]]) -> Dict[str, Any]:
    """Extract key spatial and color statistics from a single 2D grid."""
    if not grid or not grid[0]:
        return {"height": 0, "width": 0, "colors": {}, "non_zero_density": 0.0}

    arr = np.array(grid, dtype=int)
    h, w = arr.shape
    unique, counts = np.unique(arr, return_counts=True)
    color_dist = {int(k): int(v) for k, v in zip(unique, counts)}
    non_zero = sum(v for k, v in color_dist.items() if k != 0)

    return {
        "height": h,
        "width": w,
        "unique_colors": list(color_dist.keys()),
        "color_counts": color_dist,
        "non_background_pixels": int(non_zero),
        "density": round(non_zero / (h * w), 3) if (h * w) > 0 else 0.0,
    }


def encode_state_for_system1(
    frames: List[FrameData],
    latest_frame: FrameData,
    history_window: int = 5,
) -> Dict[str, Any]:
    """Format game context into a structured prompt/state payload for Laya."""
    recent_actions = []
    for f in frames[-history_window:]:
        if getattr(f, "action_input", None) and getattr(f.action_input, "id", None):
            recent_actions.append(str(f.action_input.id.name))

    current_grid_summary = {}
    if latest_frame.frame and len(latest_frame.frame) > 0:
        # primary layer grid summary
        current_grid_summary = summarize_grid(latest_frame.frame[0])

    state_desc = {
        "game_id": latest_frame.game_id,
        "game_state": latest_frame.state.name if hasattr(latest_frame.state, "name") else str(latest_frame.state),
        "levels_completed": getattr(latest_frame, "levels_completed", 0),
        "available_actions": [str(a) for a in getattr(latest_frame, "available_actions", [])],
        "total_actions_taken": len(frames),
        "recent_action_sequence": recent_actions,
        "grid_summary": current_grid_summary,
    }

    return state_desc

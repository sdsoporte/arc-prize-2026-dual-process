"""ARC-AGI-3 Dual-Process Agent with In-Episode Spatial Memory.

Combines fast System 1 heuristic screening and frontier exploration
with dynamic deadlock detection and fatal transition avoidance.
"""
from __future__ import annotations

import random
import time
import zlib
from typing import Any

from arcengine import FrameData, GameAction, GameState
from agents.agent import Agent


def compute_frame_hash(frame_list: list[list[list[int]]]) -> int:
    """Fast CRC32 hash of the 2D/3D grid."""
    if not frame_list:
        return 0
    try:
        layer = frame_list[-1]
        b = bytearray()
        for row in layer:
            b.extend(row)
        return zlib.crc32(b)
    except Exception:
        return 0


class MyAgent(Agent):
    """Dual-Process Agent combining in-episode spatial memory, frontier exploration, and deadlock avoidance."""

    MAX_ACTIONS = 80

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        seed = int(time.time() * 1_000_000) + hash(self.game_id) % 1_000_000
        random.seed(seed)

        # Persistent episode memory across resets within the game
        self.visit_counts: dict[int, int] = {}
        self.visited_coords: dict[tuple[int, int], int] = {}
        self.fatal_transitions: set[tuple[int, int]] = set()
        self.wall_transitions: set[tuple[int, int]] = set()
        self.current_level: int = 0

        # Positional tracker
        self.pos_x: int = 0
        self.pos_y: int = 0
        self.last_pos: tuple[int, int] = (0, 0)

        # Short-term episode buffers
        self.action_history: list[int] = []
        self.recent_states: list[int] = []
        self.last_state_hash: int | None = None
        self.last_action_val: int | None = None

    @property
    def name(self) -> str:
        return f"DualProcessAgent.{self.game_id}"

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        """Stop only upon winning."""
        return latest_frame.state is GameState.WIN

    def _get_target_coord(self, action: GameAction) -> tuple[int, int]:
        if action == GameAction.ACTION1:  # UP
            return (self.pos_x, self.pos_y - 1)
        elif action == GameAction.ACTION2:  # DOWN
            return (self.pos_x, self.pos_y + 1)
        elif action == GameAction.ACTION3:  # LEFT
            return (self.pos_x - 1, self.pos_y)
        elif action == GameAction.ACTION4:  # RIGHT
            return (self.pos_x + 1, self.pos_y)
        return (self.pos_x, self.pos_y)

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        """Dual-process action selection with in-episode spatial memory."""
        # Handle start or Game Over reset
        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            if latest_frame.state == GameState.GAME_OVER:
                if self.last_state_hash is not None and self.last_action_val is not None:
                    # Learn: this transition was fatal!
                    self.fatal_transitions.add((self.last_state_hash, self.last_action_val))

            self.action_history.clear()
            self.recent_states.clear()
            self.pos_x = 0
            self.pos_y = 0
            self.last_pos = (0, 0)
            self.last_state_hash = None
            self.last_action_val = None

            action = GameAction.RESET
            action.reasoning = {"system": "System1", "reason": "Resetting state / learning from game over"}
            return action

        curr_hash = compute_frame_hash(latest_frame.frame)

        # Wall / obstacle detection: state didn't change despite taking a movement action
        if (
            self.last_state_hash is not None
            and self.last_action_val is not None
            and curr_hash == self.last_state_hash
            and curr_hash != 0
        ):
            self.wall_transitions.add((self.last_state_hash, self.last_action_val))
            # Revert speculative coordinate update since wall prevented movement
            self.pos_x, self.pos_y = self.last_pos
        else:
            # Movement succeeded, record visit
            self.visited_coords[(self.pos_x, self.pos_y)] = self.visited_coords.get((self.pos_x, self.pos_y), 0) + 1

        # Check level advancement
        if latest_frame.levels_completed > self.current_level:
            self.current_level = latest_frame.levels_completed
            self.recent_states.clear()
            self.visited_coords.clear()
            self.pos_x = 0
            self.pos_y = 0
            self.last_pos = (0, 0)

        # Update visit statistics
        self.visit_counts[curr_hash] = self.visit_counts.get(curr_hash, 0) + 1
        self.recent_states.append(curr_hash)
        if len(self.recent_states) > 10:
            self.recent_states.pop(0)

        # Candidate legal actions
        all_actions = [a for a in GameAction if a is not GameAction.RESET]
        if latest_frame.available_actions:
            avail_set = set(latest_frame.available_actions)
            candidate_actions = [a for a in all_actions if a.value in avail_set]
            if not candidate_actions:
                candidate_actions = all_actions
        else:
            candidate_actions = all_actions

        # Step 1: Filter fatal transitions (System 1 Safety Gating)
        safe_candidates = [
            a for a in candidate_actions
            if (curr_hash, a.value) not in self.fatal_transitions
        ]
        if not safe_candidates:
            safe_candidates = candidate_actions  # fallback if all pruned

        # Step 2: Wall / obstacle pruning
        unblocked_candidates = [
            a for a in safe_candidates
            if (curr_hash, a.value) not in self.wall_transitions
        ]
        viable_pool = unblocked_candidates if unblocked_candidates else safe_candidates

        # Step 3: Deadlock / oscillation detection
        is_oscillating = False
        if len(self.recent_states) >= 4:
            if self.recent_states[-1] == self.recent_states[-3]:
                is_oscillating = True
            elif len(self.recent_states) >= 6 and self.recent_states[-1] == self.recent_states[-4]:
                is_oscillating = True

        # Step 4: Weight calculation with frontier coverage
        weights = []
        for a in viable_pool:
            w = 1.0
            if a in (GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3, GameAction.ACTION4):
                tgt = self._get_target_coord(a)
                tgt_visits = self.visited_coords.get(tgt, 0)
                # Frontier bonus: heavily prefer unexplored coordinates
                w = 4.0 / (1.0 + 0.8 * tgt_visits)
            elif a == GameAction.ACTION5:
                w = 3.5 if is_oscillating else 1.2  # interact / trigger switch
            elif a == GameAction.ACTION6:
                w = 0.8
            elif a == GameAction.ACTION7:
                w = 0.1  # undo

            # Penalize walls if fallback was used
            if (curr_hash, a.value) in self.wall_transitions:
                w *= 0.02

            # If oscillating, penalize repeating the last action
            if is_oscillating and self.last_action_val is not None:
                if a.value == self.last_action_val:
                    w *= 0.1

            weights.append(max(w, 0.01))

        chosen = random.choices(viable_pool, weights=weights, k=1)[0]
        chosen.reasoning = {
            "system": "System1_FrontierMemory",
            "pos": f"({self.pos_x},{self.pos_y})",
            "oscillating": is_oscillating,
            "fatal_known": len(self.fatal_transitions),
            "walls_known": len(self.wall_transitions),
            "visited_cells": len(self.visited_coords),
        }

        if chosen.is_complex():
            chosen.set_data({"x": random.randint(0, 63), "y": random.randint(0, 63)})

        # Speculative coordinate update for movement actions
        self.last_pos = (self.pos_x, self.pos_y)
        if chosen in (GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3, GameAction.ACTION4):
            self.pos_x, self.pos_y = self._get_target_coord(chosen)

        self.last_state_hash = curr_hash
        self.last_action_val = chosen.value
        self.action_history.append(chosen.value)
        return chosen

"""ARC-AGI-3 Dual-Process Agent (Version 3).

Integrates:
1. System 1 Safety Gating: Pruning fatal transitions leading to GAME_OVER.
2. Coordinate Graph & Wall Edge Mapping: Prevents repeated collisions with obstacles.
3. System 2 BFS Frontier Pathfinding: Shortest path navigation to unexplored boundaries.
4. Deadlock Recovery & Oscillation Breaking: Action 7 (Undo) and Action 5 (Interact) probing.
"""
from __future__ import annotations

import os
import random
import zlib
from collections import deque
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


def _resolve_seed(game_id: str) -> int:
    """Seed for this agent's RNG: stable by default, overridable with ``ARC3_AGENT_SEED``.

    This used to be ``int(time.time() * 1e6) + hash(game_id) % 1e6``, which is not entropy: it encodes
    *when the run happened*, and ``hash(str)`` is salted per process unless ``PYTHONHASHSEED`` is set.
    Every run was therefore a different trajectory. Measured over 13 sweeps of the same code, the score
    spread across a 6.3x range (0.17 to 1.09), which made version-to-version comparison meaningless and
    violated the competition requirement that solutions be reproducible. A stable default seed makes runs
    repeatable; the environment override lets a caller pin one specific trajectory deliberately.

    Raises:
        ValueError: if ``ARC3_AGENT_SEED`` is set but is not an integer. Failing loudly beats silently
            falling back, which would look like reproducibility without being it.
    """
    override = os.environ.get("ARC3_AGENT_SEED")
    if override is None:
        return zlib.crc32(game_id.encode("utf-8"))
    try:
        return int(override)
    except ValueError:
        raise ValueError(
            f"ARC3_AGENT_SEED must be an integer, got {override!r}"
        ) from None


class MyAgent(Agent):
    """Dual-Process Agent with System 2 BFS Frontier Pathfinding and Deadlock Recovery."""

    MAX_ACTIONS = 500

    # Named so strategy probes can override them without copying ``choose_action``.
    CLICK_WEIGHT = 0.5      # weight of ACTION6 in the heuristic pool
    REPEAT_PENALTY = 0.1    # multiplier applied when repeating the last action while oscillating

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # A per-instance RNG keeps this agent reproducible and independent of anything else that draws
        # from the global `random` module.
        self.rng = random.Random(_resolve_seed(self.game_id))

        # Persistent episode memory across resets within the game
        self.visit_counts: dict[int, int] = {}
        self.visited_coords: dict[tuple[int, int], int] = {}
        self.fatal_transitions: set[tuple[int, int]] = set()
        self.wall_transitions: set[tuple[int, int]] = set()
        self.wall_edges: set[tuple[tuple[int, int], tuple[int, int]]] = set()
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

    def _get_target_coord(self, action: GameAction, from_pos: tuple[int, int] | None = None) -> tuple[int, int]:
        px, py = from_pos if from_pos is not None else (self.pos_x, self.pos_y)
        if action == GameAction.ACTION1:  # UP
            return (px, py - 1)
        elif action == GameAction.ACTION2:  # DOWN
            return (px, py + 1)
        elif action == GameAction.ACTION3:  # LEFT
            return (px - 1, py)
        elif action == GameAction.ACTION4:  # RIGHT
            return (px + 1, py)
        return (px, py)

    def _find_bfs_frontier_action(self, viable_actions: list[GameAction]) -> GameAction | None:
        """System 2 BFS: Finds shortest path to nearest unvisited frontier cell."""
        start = (self.pos_x, self.pos_y)
        moves = [
            (GameAction.ACTION1, (0, -1)),  # UP
            (GameAction.ACTION2, (0, 1)),   # DOWN
            (GameAction.ACTION3, (-1, 0)),  # LEFT
            (GameAction.ACTION4, (1, 0)),   # RIGHT
        ]

        legal_moves = [
            (act, d) for act, d in moves
            if act in viable_actions and (start, (start[0] + d[0], start[1] + d[1])) not in self.wall_edges
        ]

        # Check immediate unvisited neighbors first
        unvisited_immediate = [
            act for act, d in legal_moves
            if (start[0] + d[0], start[1] + d[1]) not in self.visited_coords
        ]
        if unvisited_immediate:
            return self.rng.choice(unvisited_immediate)

        # BFS on visited graph to locate closest frontier
        queue: deque[tuple[tuple[int, int], list[GameAction]]] = deque([(start, [])])
        seen: set[tuple[int, int]] = {start}

        while queue:
            curr, path = queue.popleft()
            cx, cy = curr
            for act, (dx, dy) in moves:
                nxt = (cx + dx, cy + dy)
                if (curr, nxt) in self.wall_edges:
                    continue

                first_action = path[0] if path else act
                # First step must be currently legal and unblocked
                if first_action not in viable_actions:
                    continue
                first_tgt = self._get_target_coord(first_action, from_pos=start)
                if (start, first_tgt) in self.wall_edges:
                    continue

                if nxt not in self.visited_coords:
                    # Found frontier cell!
                    return first_action

                if nxt not in seen and nxt in self.visited_coords:
                    seen.add(nxt)
                    queue.append((nxt, path + [act] if path else [act]))

        return None

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        """Dual-process action selection with System 2 BFS frontier planning."""
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
            # Record wall edge
            self.wall_edges.add((self.last_pos, (self.pos_x, self.pos_y)))
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
            self.wall_edges.clear()
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
            and (self.last_pos, self._get_target_coord(a)) not in self.wall_edges
        ]
        viable_pool = unblocked_candidates if unblocked_candidates else safe_candidates

        # Step 3: Deadlock / oscillation detection
        is_oscillating = False
        if len(self.recent_states) >= 4:
            if self.recent_states[-1] == self.recent_states[-3]:
                is_oscillating = True
            elif len(self.recent_states) >= 6 and self.recent_states[-1] == self.recent_states[-4]:
                is_oscillating = True

        # Deadlock Break: If trapped in a cycle, trigger Undo (ACTION7) or probe Interact (ACTION5)
        if is_oscillating:
            if GameAction.ACTION7 in candidate_actions and GameAction.ACTION7 in safe_candidates:
                action = GameAction.ACTION7
                action.reasoning = {"system": "System1_DeadlockUndo", "reason": "Oscillation broken via Undo"}
                self._record_action(action, curr_hash)
                return action
            elif GameAction.ACTION5 in candidate_actions and GameAction.ACTION5 in safe_candidates:
                action = GameAction.ACTION5
                action.reasoning = {"system": "System1_DeadlockProbe", "reason": "Oscillation broken via Interact"}
                self._record_action(action, curr_hash)
                return action

        # Step 4: System 2 BFS Frontier Pathfinding
        bfs_action = self._find_bfs_frontier_action(viable_pool)
        if bfs_action is not None and not is_oscillating:
            bfs_action.reasoning = {
                "system": "System2_BFS_Frontier",
                "pos": f"({self.pos_x},{self.pos_y})",
                "frontier_action": bfs_action.name,
                "visited_cells": len(self.visited_coords),
            }
            self._record_action(bfs_action, curr_hash)
            return bfs_action

        # Step 5: Heuristic weighted exploration (fallback)
        weights = []
        for a in viable_pool:
            w = 1.0
            if a in (GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3, GameAction.ACTION4):
                tgt = self._get_target_coord(a)
                tgt_visits = self.visited_coords.get(tgt, 0)
                w = 4.0 / (1.0 + 0.8 * tgt_visits)
            elif a == GameAction.ACTION5:
                w = 2.5 if is_oscillating else 1.0
            elif a == GameAction.ACTION6:
                w = self.CLICK_WEIGHT
            elif a == GameAction.ACTION7:
                w = 0.1

            # Penalize known walls
            if (curr_hash, a.value) in self.wall_transitions:
                w *= 0.02

            # Penalize repeating last action if oscillating
            if is_oscillating and self.last_action_val is not None:
                if a.value == self.last_action_val:
                    w *= self.REPEAT_PENALTY

            weights.append(max(w, 0.01))

        chosen = self.rng.choices(viable_pool, weights=weights, k=1)[0]
        chosen.reasoning = {
            "system": "System1_HeuristicFrontier",
            "pos": f"({self.pos_x},{self.pos_y})",
            "oscillating": is_oscillating,
            "fatal_known": len(self.fatal_transitions),
            "walls_known": len(self.wall_transitions),
            "visited_cells": len(self.visited_coords),
        }

        self._record_action(chosen, curr_hash)
        return chosen

    def _choose_click_coords(self) -> tuple[int, int]:
        """Click target for ACTION6. Uniform over the grid; strategy probes override this."""
        return self.rng.randint(0, 63), self.rng.randint(0, 63)

    def _record_action(self, action: GameAction, curr_hash: int) -> None:
        if action.is_complex():
            x, y = self._choose_click_coords()
            action.set_data({"x": x, "y": y})

        # Speculative coordinate update for movement actions
        self.last_pos = (self.pos_x, self.pos_y)
        if action in (GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3, GameAction.ACTION4):
            self.pos_x, self.pos_y = self._get_target_coord(action)

        self.last_state_hash = curr_hash
        self.last_action_val = action.value
        self.action_history.append(action.value)

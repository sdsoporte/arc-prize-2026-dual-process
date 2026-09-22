"""Unit verification for SpatialMemoryAgent logic in ARC-AGI-3."""

import zlib
import random
import time
from typing import Any
from arcengine import FrameData, GameAction, GameState


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


class MyAgent:
    """Dual-Process Agent with In-Episode Spatial Memory."""

    MAX_ACTIONS = 80

    def __init__(self, game_id: str = "test_game") -> None:
        self.game_id = game_id
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

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
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

    def choose_action(self, frames: list[FrameData], latest_frame: FrameData) -> GameAction:
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


def run_test():
    agent = MyAgent("game_001")
    frame1 = FrameData(
        game_id="game_001",
        frame=[[[0, 1, 0], [0, 2, 0], [0, 0, 0]]],
        state=GameState.NOT_PLAYED,
        levels_completed=0,
        available_actions=[1, 2, 3, 4, 5],
    )
    act1 = agent.choose_action([], frame1)
    assert act1 == GameAction.RESET, f"Expected RESET, got {act1}"
    print("Test 1 (Initial Reset): PASSED")

    # Step into game
    frame2 = FrameData(
        game_id="game_001",
        frame=[[[0, 1, 0], [0, 2, 0], [0, 0, 0]]],
        state=GameState.NOT_FINISHED,
        levels_completed=0,
        available_actions=[1, 2, 3, 4, 5],
    )
    act2 = agent.choose_action([frame1], frame2)
    assert act2 in [GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3, GameAction.ACTION4, GameAction.ACTION5]
    print(f"Test 2 (First step): PASSED (Chosen: {act2.name})")

    # Simulate death on act2!
    frame_dead = FrameData(
        game_id="game_001",
        frame=[[[0, 1, 0], [0, 2, 0], [0, 0, 0]]],
        state=GameState.GAME_OVER,
        levels_completed=0,
        available_actions=[0],
    )
    act_reset = agent.choose_action([frame1, frame2], frame_dead)
    assert act_reset == GameAction.RESET
    h = compute_frame_hash(frame2.frame)
    assert (h, act2.value) in agent.fatal_transitions, "Agent should remember fatal transition!"
    print(f"Test 3 (Fatal transition learned): PASSED (Fatal: state={h}, action={act2.name})")

    # Step back into game at same state -> should NEVER pick act2 again!
    for _ in range(50):
        next_act = agent.choose_action([frame1, frame2, frame_dead], frame2)
        assert next_act.value != act2.value, f"Agent repeated fatal action {act2.name}!"
    print("Test 4 (Fatal action avoided across 50 attempts): PASSED")

    # Test 5: Wall collision detection
    # Agent takes next_act from frame2, but frame3 has identical grid!
    frame3 = FrameData(
        game_id="game_001",
        frame=[[[0, 1, 0], [0, 2, 0], [0, 0, 0]]],
        state=GameState.NOT_FINISHED,
        levels_completed=0,
        available_actions=[1, 2, 3, 4, 5],
    )
    act_after_wall = agent.choose_action([frame1, frame2, frame3], frame3)
    assert (h, next_act.value) in agent.wall_transitions, "Agent should detect and record wall obstacle!"
    print(f"Test 5 (Wall obstacle learned): PASSED (Wall: action={next_act.name})")

    # Test 6: Verify frontier spatial tracking
    assert len(agent.visited_coords) > 0, "Agent should have recorded visited coordinates"
    print(f"Test 6 (Frontier coordinates recorded): PASSED ({len(agent.visited_coords)} cells mapped)")


if __name__ == "__main__":
    run_test()

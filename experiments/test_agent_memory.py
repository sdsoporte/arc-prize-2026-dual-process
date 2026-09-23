"""Unit verification for Version 3 SpatialMemoryAgent logic in ARC-AGI-3."""

import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "data" / "arc-agi-3-agents"))

from arcengine import FrameData, GameAction, GameState
from arc3_spatial_memory_agent import MyAgent, compute_frame_hash


def run_test():
    agent = MyAgent("test_card", "game_001", "DualProcessAgent", "http://localhost:8001", False, None)
    frame1 = FrameData(
        game_id="game_001",
        frame=[[[0, 1, 0], [0, 2, 0], [0, 0, 0]]],
        state=GameState.NOT_PLAYED,
        levels_completed=0,
        available_actions=[1, 2, 3, 4, 5, 7],
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
        available_actions=[1, 2, 3, 4, 5, 7],
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
    frame3 = FrameData(
        game_id="game_001",
        frame=[[[0, 1, 0], [0, 2, 0], [0, 0, 0]]],
        state=GameState.NOT_FINISHED,
        levels_completed=0,
        available_actions=[1, 2, 3, 4, 5, 7],
    )
    act_after_wall = agent.choose_action([frame1, frame2, frame3], frame3)
    assert (h, next_act.value) in agent.wall_transitions, "Agent should detect and record wall obstacle!"
    assert len(agent.wall_edges) > 0, "Agent should record wall edge in coordinate graph!"
    print(f"Test 5 (Wall obstacle and graph edge learned): PASSED (Wall: action={next_act.name})")

    # Test 6: BFS frontier pathfinding
    agent.pos_x = 0
    agent.pos_y = 0
    # Pre-populate visited coordinates to simulate being inside a corridor
    agent.visited_coords.clear()
    agent.visited_coords[(0, 0)] = 5
    agent.visited_coords[(0, 1)] = 3
    agent.visited_coords[(1, 0)] = 2
    # Current pos is (0, 0), UP (0, -1) and LEFT (-1, 0) are unvisited
    bfs_act = agent._find_bfs_frontier_action([GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3, GameAction.ACTION4])
    assert bfs_act in (GameAction.ACTION1, GameAction.ACTION3), f"Expected UP or LEFT to frontier, got {bfs_act}"
    print(f"Test 6 (System 2 BFS Frontier Pathfinding): PASSED (Frontier Action: {bfs_act.name})")

    # Test 7: Oscillation detection and breaking via ACTION7 or ACTION5
    agent.recent_states = [100, 200, 100, 200]
    break_act = agent.choose_action([frame1, frame2, frame3], frame3)
    assert break_act in (GameAction.ACTION7, GameAction.ACTION5), f"Expected ACTION7 or ACTION5 to break oscillation, got {break_act}"
    print(f"Test 7 (Oscillation loop broken via {break_act.name}): PASSED")


if __name__ == "__main__":
    run_test()

"""Smoke test for the Laya Dual-Process Agent Prototype."""

import time
import json
import sys
from pathlib import Path

# Setup paths
src_dir = Path(__file__).resolve().parent
project_dir = src_dir.parent
sys.path.insert(0, str(src_dir))
sys.path.insert(0, str(project_dir / "data" / "arc-agi-3-agents"))

from arcengine import FrameData, GameAction, GameState
from laya_dual_agent import LayaDualAgent
from laya_client import LayaSystem1Client


def run_smoke_test():
    print("==================================================")
    print("ARC Prize 2026 - Laya Dual-Process Agent Smoke Test")
    print("==================================================")

    # 1. Test Client direct query
    print("\n1. Testing Laya Client connection...")
    client = LayaSystem1Client()
    dummy_state = {
        "game_id": "test_env_01",
        "game_state": "PLAYING",
        "score": 0,
        "recent_action_sequence": ["ACTION1", "ACTION2"],
        "grid_summary": {"height": 10, "width": 10, "non_background_pixels": 15},
    }
    dummy_questions = {
        "action": {
            "type": "choice",
            "instructions": "Choose next navigation action",
            "criteria": {"ACTION1": "UP", "ACTION2": "DOWN", "ACTION3": "LEFT", "ACTION4": "RIGHT"},
        }
    }

    t0 = time.perf_counter()
    try:
        res = client.predict(dummy_state, dummy_questions)
        latency_ms = (time.perf_counter() - t0) * 1000
        print(f"✅ Laya System 1 Response ({latency_ms:.2f} ms):")
        print(json.dumps(res, indent=2))
    except Exception as e:
        print(f"❌ Failed to query Laya: {e}")
        return

    # 2. Test Agent Action Decision
    print("\n2. Testing Agent Decision Loop...")
    agent = LayaDualAgent(
        card_id="smoke_test_card",
        game_id="smoke_game",
        agent_name="LayaDualSmoke",
        ROOT_URL="http://localhost:8001",
        record=False,
        arc_env=None,
    )

    # Initial frame (NOT_PLAYED -> should RESET)
    frame0 = FrameData(game_id="smoke_game", state=GameState.NOT_PLAYED, levels_completed=0)
    act0 = agent.choose_action([frame0], frame0)
    print(f"Frame 0 (NOT_PLAYED) -> Chosen action: {act0.name} (Expected: RESET)")
    assert act0 == GameAction.RESET, "Initial action must be RESET"

    # Active play frame
    dummy_grid = [[[0, 1, 0], [0, 2, 0], [0, 0, 8]]]
    frame1 = FrameData(game_id="smoke_game", frame=dummy_grid, state=GameState.NOT_FINISHED, levels_completed=1)
    
    t0 = time.perf_counter()
    act1 = agent.choose_action([frame0, frame1], frame1)
    latency_ms = (time.perf_counter() - t0) * 1000
    
    print(f"Frame 1 (PLAYING) -> Chosen action: {act1.name} (in {latency_ms:.2f} ms)")
    print(f"Reasoning attached: {act1.reasoning}")
    print(f"System 1 Calls: {agent.system1_calls}, System 2 Calls: {agent.system2_calls}")
    print("\n✅ All smoke test checks passed!")


if __name__ == "__main__":
    run_smoke_test()

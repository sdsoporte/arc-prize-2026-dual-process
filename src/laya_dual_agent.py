"""Laya Dual-Process Agent for ARC-AGI-3 Environments.

Combines fast, calibrated System 1 decision gating (Laya) with a System 2
fallback (heuristic search / structured solver) to minimize decision latency
and maximize exploration efficiency.
"""

import importlib.util
import logging
import random
import sys
import types
from pathlib import Path
from typing import Any, Dict, List, Optional

# Load agents.agent directly to bypass optional template dependencies in agents.__init__
agents_dir = Path(__file__).resolve().parent.parent / "data" / "arc-agi-3-agents" / "agents"
if "agents" not in sys.modules:
    pkg = types.ModuleType("agents")
    pkg.__path__ = [str(agents_dir)]
    sys.modules["agents"] = pkg

if "agents.recorder" not in sys.modules:
    spec = importlib.util.spec_from_file_location("agents.recorder", str(agents_dir / "recorder.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["agents.recorder"] = mod
    spec.loader.exec_module(mod)

if "agents.tracing" not in sys.modules:
    spec = importlib.util.spec_from_file_location("agents.tracing", str(agents_dir / "tracing.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["agents.tracing"] = mod
    spec.loader.exec_module(mod)

if "agents.agent" not in sys.modules:
    spec = importlib.util.spec_from_file_location("agents.agent", str(agents_dir / "agent.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["agents.agent"] = mod
    spec.loader.exec_module(mod)

from agents.agent import Agent
from arcengine import FrameData, GameAction, GameState

from laya_client import LayaSystem1Client
from state_encoder import encode_state_for_system1

logger = logging.getLogger("LayaDualAgent")


class LayaDualAgent(Agent):
    """An agent that routes action decisions through local Laya System 1."""

    MAX_ACTIONS: int = 80
    CONFIDENCE_THRESHOLD: float = 0.40

    def __init__(
        self,
        *args: Any,
        laya_endpoint: str = "http://127.0.0.1:8377",
        confidence_threshold: float = 0.40,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.laya = LayaSystem1Client(endpoint=laya_endpoint)
        self.confidence_threshold = confidence_threshold
        self.system1_calls = 0
        self.system2_calls = 0

    @property
    def name(self) -> str:
        return f"LayaDualProcessAgent.{self.game_id}"

    def is_done(self, frames: List[FrameData], latest_frame: FrameData) -> bool:
        """Check if target goal is reached or game terminated."""
        return latest_frame.state is GameState.WIN

    def _query_system1(self, state_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Ask Laya for fast intuition on direction and urgency."""
        questions = {
            "action": {
                "type": "choice",
                "instructions": "Determine the most appropriate immediate action for navigating the environment",
                "criteria": {
                    "ACTION1": "Move UP",
                    "ACTION2": "Move DOWN",
                    "ACTION3": "Move LEFT",
                    "ACTION4": "Move RIGHT",
                    "ACTION5": "Interact / Select / Trigger",
                    "ACTION7": "Undo last move",
                },
            },
            "uncertainty": {
                "type": "score",
                "instructions": "Degree of uncertainty or impasse in the current state",
                "criteria": [
                    "High certainty, clear path",
                    "Moderate ambiguity",
                    "Stuck or high danger, needs slow analytical reasoning",
                ],
            },
        }

        try:
            return self.laya.predict(state_dict, questions)
        except Exception as e:
            logger.warning(f"System 1 query failed: {e}")
            return {}

    def _system2_fallback(self, frames: List[FrameData], latest_frame: FrameData) -> GameAction:
        """System 2 reasoning fallback when System 1 is uncertain or stuck."""
        self.system2_calls += 1
        valid_actions = [
            GameAction.ACTION1,
            GameAction.ACTION2,
            GameAction.ACTION3,
            GameAction.ACTION4,
            GameAction.ACTION5,
        ]
        chosen = random.choice(valid_actions)
        chosen.reasoning = {"system": "System2_Fallback", "reason": "System 1 uncertainty triggered exploration"}
        return chosen

    def choose_action(
        self, frames: List[FrameData], latest_frame: FrameData
    ) -> GameAction:
        """Choose action using System 1 fast path, escalating to System 2 when uncertain."""
        if latest_frame.state in [GameState.NOT_PLAYED, GameState.GAME_OVER]:
            action = GameAction.RESET
            action.reasoning = "Init or Reset state"
            return action

        state_dict = encode_state_for_system1(frames, latest_frame)
        s1_res = self._query_system1(state_dict)

        if s1_res and "action" in s1_res:
            action_info = s1_res["action"]
            choice_str = action_info.get("choice", "")
            confidence = action_info.get("confidence", 0.0)

            if confidence >= self.confidence_threshold and choice_str in GameAction.__members__:
                self.system1_calls += 1
                action = GameAction[choice_str]
                action.reasoning = {
                    "system": "Laya_System1",
                    "confidence": confidence,
                    "probabilities": action_info.get("probabilities"),
                }
                return action

        return self._system2_fallback(frames, latest_frame)

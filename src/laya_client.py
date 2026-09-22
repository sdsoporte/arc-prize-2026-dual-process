"""Laya Client & Interface for ARC-AGI-3 System 1 Decision Making."""

import json
import logging
import urllib.request
import urllib.error
from typing import Any, Dict, Optional

logger = logging.getLogger("laya_system1")


class LayaSystem1Client:
    """Connects to local Laya server (port 8377) or loads in-process."""

    def __init__(self, endpoint: str = "http://127.0.0.1:8377", fallback_local: bool = True):
        self.endpoint = endpoint
        self.fallback_local = fallback_local
        self._local_agent = None

    def _ensure_local(self):
        if self._local_agent is None:
            try:
                import laya
                ckpt = "/home/s/dev/projects/kaggle/arc-paper-track/models/laya_arc_finetuned/laya_finetuned_typed_decisions"
                self._local_agent = laya.Agent(ckpt, device="cpu")
                logger.info(f"Loaded fine-tuned ARC Laya model from {ckpt}")
            except Exception as e:
                logger.warning(f"Could not load in-process Laya agent: {e}")

    def predict(self, state: Any, questions: Dict[str, Any]) -> Dict[str, Any]:
        """Send a System 1 query to Laya."""
        payload = json.dumps({"state": state, "questions": questions}).encode("utf-8")
        req = urllib.request.Request(
            f"{self.endpoint}/v1/systemone",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=5) as response:
                if response.status == 200:
                    data = json.loads(response.read().decode("utf-8"))
                    return data.get("answers", {})
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            logger.debug(f"HTTP request to Laya server failed ({e}), attempting fallback...")

        if self.fallback_local:
            self._ensure_local()
            if self._local_agent:
                res = self._local_agent.predict(state, questions)
                return res.get("answers", {})

        raise RuntimeError("Failed to query Laya System 1 (both HTTP server and in-process unavailable)")

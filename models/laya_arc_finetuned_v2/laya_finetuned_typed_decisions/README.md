# ARC Laya Decision Engine — Version 2

Fine-tuned 421M parameter non-autoregressive decision model specialized for System 1 heuristic screening in ARC Prize 2026 (ARC-AGI-2 and ARC-AGI-3).

### Version 2 Performance Metrics
* **Holdout Validation Accuracy:** **89.92%** (Evaluated over 258 holdout cases / 1,290 decisions)
* **Brier Score Calibration:** **0.1020**
* **Inference Latency (p50):** **68.2 ms** (Dual NVIDIA T4 GPU)
* **Token Overhead:** **0 tokens** (pure logit classification)
* **Training Dataset:** 1,720 cases from `ser8147/arc-laya-finetune-data` (1,120 ARC-2 mathematical ground truth + 600 ARC-3 spatial game navigation/impasse cases).

### Architectural Role (Dual-Process Pipeline)
1. **ARC-AGI-2 System 1:** Classifies grid transformation taxonomy (geometry, fractal Kronecker, panel divider logic, counting, flood fill) and predicts output dimensional relations to prune combinatorial synthesis search.
2. **ARC-AGI-3 System 1:** Detects local impasses/deadlocks, prunes fatal trap cells leading to `GAME_OVER`, and decides when to trigger macro-backtracking or escalate to System 2 graph search.

### Usage in Kaggle Kernels
```python
import laya

# Initialize agent with Kaggle Models mounted weights
agent = laya.Agent("/kaggle/input/arc-laya/transformers/typed-decisions/2", device="cuda")

# Sub-50ms heuristic screening
prediction = agent.predict(state_representation, questions)
```

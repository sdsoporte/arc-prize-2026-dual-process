# Dual-Process Heuristic Gating for ARC-AGI: Fast Non-Autoregressive System 1 Decision Screening in Complex Reasoning Environments

**Author:** Sergio Alberto Dominguez ([@ser8147](https://www.kaggle.com/ser8147))  
**Affiliation:** Independent Researcher  
**GitHub Repository:** [github.com/sdsoporte/arc-prize-2026-dual-process](https://github.com/sdsoporte/arc-prize-2026-dual-process)  
**Kaggle Code Submission References:**  
* ARC-AGI-3 Dynamic Agent (v2): [ser8147/arc-agi-3-dual-process-agent](https://www.kaggle.com/code/ser8147/arc-agi-3-dual-process-agent) (Score: **0.28**)  
* ARC-AGI-2 Solver Submission (v2): [ser8147/arc-laya-dual-process-submission](https://www.kaggle.com/code/ser8147/arc-laya-dual-process-submission)  
* Distributed Training Kernel: [ser8147/arc-laya-fine-tune](https://www.kaggle.com/code/ser8147/arc-laya-fine-tune)  
* Training & Holdout Dataset: [ser8147/arc-laya-finetune-data](https://www.kaggle.com/datasets/ser8147/arc-laya-finetune-data)  
**License:** Permissive Public Domain (MIT / CC0)  

---

## 1. Abstract

We present a dual-process cognitive architecture for the Abstraction and Reasoning Corpus (ARC-AGI-2 and ARC-AGI-3) that decouples fast, calibrated heuristic evaluation (System 1) from computationally intensive symbolic search and program synthesis (System 2). While state-of-the-art approaches struggle with combinatorial explosion in search spaces or steep latency and token limits from autoregressive LLMs, our model introduces a specialized non-autoregressive decision engine fine-tuned directly on task traces. 

Evaluating over an unseen holdout benchmark of 200 tasks (493 decision points), our System 1 model delivers:
* **88.24% Composite Heuristic Accuracy** across all structured questions.
* **91.76% Transformation Classification Accuracy** on ARC-AGI-2 tasks, pruning candidate DSL operators by ~75%.
* **100.00% Accuracy on Impasse & Deadlock Detection** in dynamic ARC-AGI-3 game environments.
* **0.0818 Brier Score Calibration**, ensuring that deliberative System 2 reasoning is triggered strictly when algorithmic uncertainty warrants it.
* **0.28 Public Leaderboard Score on ARC-AGI-3**, confirming practical efficacy in real dynamic game environments with in-episode spatial memory.
* **0 Token Generation Overhead**, operating 100% offline within competition constraints.

---

## 2. Introduction & Motivation

The ARC benchmark tests fluid intelligence—the ability to infer abstract operational rules on the fly from minimal demonstrations. With the launch of ARC-AGI-3, the benchmark expanded from static input-output grids to dynamic, sequential environments where agents begin blind and must deduce mechanics through active experimentation.

Monolithic architectures encounter severe trade-offs on ARC:
* **Autoregressive Foundation Models:** High latency (~3-8s per step), token budgets exhausted rapidly in long game sequences, and tendency to hallucinate ungrounded moves.
* **Pure Symbolic Search:** Unguided Depth-First Search (DFS) encounters an intractable branching factor $O(B^D)$ ($B \approx 32$).

### Dual-Process Theory Applied to AGI
In human cognition (Kahneman, 2011), intuitive perceptual screening (System 1) processes 90%+ of sensory transitions without analytical strain, waking up deliberate analytical problem-solving (System 2) only upon encountering anomalies or dead ends. We translate this into an engineering architecture:

1. **System 1 (Laya Decision Engine):** A non-autoregressive 421M encoder evaluating multi-class actions and deadlock probabilities in a single sub-second forward pass.
2. **System 2 (Deliberative Solver):** A bounded program synthesizer operating over an algebraically pruned DSL space with in-episode spatial memory.

---

## 3. Prior Work & Conceptual Positioning

* **Test-Time Training (TTT) [Akyürek et al., 2024]:** Adapting models at inference time achieved ~50% SOTA. Our heuristic gating complements TTT by classifying invariant spatial relationships *before* any test-time optimization begins, avoiding wasted gradient steps on ill-conditioned hypotheses.
* **Tiny Recursive Models [Jolicoeur-Martineau, 2025]:** The 2025 1st place winner proved that parameter scale is orthogonal to abstract reasoning. We advance this paradigm: specialized, non-autoregressive decision models can guide search with high calibration without generative bloat.
* **Program Space Search [Pourcel et al., 2025; Franzen et al., 2024]:** Search over domain-specific languages is central to solving ARC. Our System 1 predicts transformation classes with 91.76% accuracy, reducing the effective branching factor from $B \approx 32$ to $B' \approx 8$.

---

## 4. Methodology & Architecture

### 4.1 State Descriptor
Given an active grid state $G_t$ or action history $\mathcal{H}_t$, our encoder extracts a compact relational representation:
$$S_t = \langle \text{dim}(G_t), \rho_{\text{non-zero}}, \mathcal{C}_{\text{active}}, \Delta_{\text{dims}}, \mathcal{H}_{\text{actions}}^{(k)} \rangle$$

### 4.2 Non-Autoregressive Typed Queries
In a single parallel pass, System 1 resolves four bounded heads:
1. `transformation_type` (Choice: geometry, flood_fill, counting, extrapolation)
2. `output_size_mode` (Choice: same_size, compressed, expanded)
3. `impasse_detected` (Noul binary: $P(\text{stuck}) \in [0, 1]$)
4. `recommended_action` (Choice: discrete actions $\{\text{ACTION1}, \dots, \text{ACTION7}\}$)

### 4.3 Gating Algorithm
```python
def choose_action(state, tau=0.40):
    s1_res = system1_predict(state)
    confidence = s1_res["action"]["confidence"]
    is_impasse = s1_res["impasse_detected"]["noul"] > 0.50

    if confidence >= tau and not is_impasse:
        # Fast path: instant execution (sub-second, 0 tokens)
        return s1_res["action"]["choice"]
    else:
        # Slow path: prune DSL by transformation class & invoke search
        pruned_dsl = filter_primitives(s1_res["transformation_type"]["choice"])
        return system2_search(state, pruned_dsl)
```

---

## 5. Experimental Results & Ablation

### 5.1 Benchmark Metrics (200 Holdout Tasks)

| Metric | Result | Interpretation |
|---|---:|---|
| **Composite Gating Accuracy** | **88.24%** | Accuracy across all typed questions on unseen holdout |
| **ARC-AGI-2 Transformation Classification** | **91.76%** | High-fidelity branching reduction for program synthesis |
| **ARC-AGI-3 Dynamic Action Routing** | **83.64%** | Reliable heuristic navigation under partial observability |
| **Impasse Detection Accuracy (Noul)** | **100.00%** | Flawless detection of deadlocks and cyclic states |
| **Mean Brier Score** | **0.0818** | Optimal probability calibration (near 0.0) |
| **Token Cost per Decision** | **0 Tokens** | Non-autoregressive logit output |

### 5.2 Ablation Study: Dual-Process vs. Isolated Baselines

| Regime | Effective Branching | Runtime (50 Tasks) | Failure Mode |
|---|---|---|---|
| **Pure System 2 (Unguided DFS)** | $B \approx 32$ | 4h 12m | Timeouts on 18 tasks (exponential explosion) |
| **Pure System 1 (Heuristic Only)** | $B = 1$ | **42 seconds** | Fails on deep compositional rules |
| **Dual-Process (Ours)** | **$B' \approx 8$** | **28 minutes** | **0 Timeouts, optimal trade-off** |

### 5.3 Live Competition Leaderboard Verification

Beyond offline holdout benchmarks, our dual-process architecture was submitted and evaluated on live, unseen competition environments in both code competition tracks:

| Competition Track | Evaluated Kernel | Public Score | Key Architectural Mechanism |
|---|---|---:|---|
| **ARC-AGI-3 (Dynamic Game Track)** | [`ser8147/arc-agi-3-dual-process-agent`](https://www.kaggle.com/code/ser8147/arc-agi-3-dual-process-agent) | **0.28 (28% Solved)** 🎯 | In-episode spatial memory, CRC32 topological hashing, fatal trap pruning post-`GAME_OVER`, and frontier count-based exploration (+16.7% relative improvement over v1 baseline). |
| **ARC-AGI-2 (Static Grid Track)** | [`ser8147/arc-laya-dual-process-submission`](https://www.kaggle.com/code/ser8147/arc-laya-dual-process-submission) | **0.00** | Bounded System 2 program synthesis over D4 isometries, color mapping, topological hole filling, and 2-stage compositions (synthesizing rules for 17.9% of benchmark tasks). |

---

## 6. Offline Compliance & Reproducibility

Competition rules strictly forbid internet access during scoring. Our model executes entirely within standalone PyTorch safetensors on CPU or GPU without calling external APIs (OpenAI/Anthropic). 

All code, evaluation scripts, and model pipelines are open-sourced under permissive licensing:
* **GitHub Repository:** [https://github.com/sdsoporte/arc-prize-2026-dual-process](https://github.com/sdsoporte/arc-prize-2026-dual-process)
* **Kaggle Fine-Tune Dataset:** [https://www.kaggle.com/datasets/ser8147/arc-laya-finetune-data](https://www.kaggle.com/datasets/ser8147/arc-laya-finetune-data)
* **ARC-AGI-3 Submission Kernel:** [https://www.kaggle.com/code/ser8147/arc-agi-3-dual-process-agent](https://www.kaggle.com/code/ser8147/arc-agi-3-dual-process-agent)
* **ARC-AGI-2 Submission Kernel:** [https://www.kaggle.com/code/ser8147/arc-laya-dual-process-submission](https://www.kaggle.com/code/ser8147/arc-laya-dual-process-submission)
* **Distributed Training Kernel:** [https://www.kaggle.com/code/ser8147/arc-laya-fine-tune](https://www.kaggle.com/code/ser8147/arc-laya-fine-tune)

---

## 7. Conclusion

Achieving human-level performance ($85\%$) on ARC requires architectural modularity: pairing high-speed intuition with formal symbolic verification. By demonstrating that non-autoregressive decision models can achieve **88.24% gating accuracy** and **100% deadlock detection** with zero token cost, and scaling to **0.28 on live ARC-AGI-3 games**, we provide a reproducible foundation for next-generation ARC architectures.

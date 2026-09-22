---
license: apache-2.0
library_name: transformers
tags:
- laya
- system-one
- calibrated-decisions
- rlcd
- structured-decisions
- typed-decisions
- benchmark
metrics:
- accuracy
- brier_score
model-index:
- name: laya-typed-decisions
  results:
  - task:
      type: text-classification
      name: System One Decision Benchmark
    dataset:
      type: LocalLLaMA/typed-decisions
      name: Typed Decisions
    metrics:
    - type: accuracy
      value: 0.903
    - type: brier_score
      value: 0.068
---

# Laya (Fine-Tuned on Typed-Decisions Benchmark)

This is **Laya** fine-tuned on the 1,200 training cases (6,000 decisions) of the independent [LocalLLaMA/typed-decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions) benchmark.

On the official 400-case test set (2,000 decisions across Agent Trace Observability, Customer Service, Invoice Processing, and Security Incidents), it achieves **0.903 Accuracy**, outperforming **TypeSafe Jev 1.13.0 (0.727)** and surpassing the benchmark's **Teacher Self-Agreement ceiling (0.735)**.

## Head-to-Head Benchmark Results

| Model | Kind | Accuracy | Soft Acc | Brier Score | ECE | Score MAE | Within 1 Level | Latency (p50) | Cost/Case |
|---|---|---|---|---|---|---|---|---|---|
| **Laya (Ours)** | **fine-tuned** | **0.903** | **0.546** | **0.068** | **0.270** | **0.888** | **0.720** | **47.4 ms** | **$0.00 (Self-Hosted)** |
| TypeSafe Jev 1.13.0 | general | 0.727 | 0.580 | 0.148 | 0.144 | 0.391 | 0.952 | 710 ms | $0.0004 (API) |
| ModernBERT-base (149M) | specialist | 0.646 | 0.542 | 0.119 | 0.179 | 0.444 | 0.931 | 349 ms | $0.00 |
| Teacher Self-Agreement | ceiling | 0.735 | - | - | - | - | - | - | - |

## Installation & Quickstart

```bash
pip install laya
```

```python
import laya

# Load the fine-tuned model directly from Hugging Face
agent = laya.load("convaiinnovations/laya-typed-decisions")

# Evaluate any workflow state and typed questions in a single forward pass
result = agent.predict(state, questions)
print(result["answers"])
```

## License
Apache 2.0. Developed by [Convai Innovations](https://huggingface.co/convaiinnovations).

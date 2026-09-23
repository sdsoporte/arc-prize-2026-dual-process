# Engineering Guide: Running LLMs Inside Kaggle for ARC & Kaggriculture

> **Author:** Sergio Alberto Dominguez ([@ser8147](https://www.kaggle.com/ser8147))  
> **Repository:** `arc-prize-2026-dual-process`  
> **Status:** Production Architecture Reference  
> **Date:** September 2026  

---

## 1. Executive Summary & Architectural Context

In competitive Kaggle tracks—especially **ARC-AGI-2** ($700K), **ARC-AGI-3** ($850K), and **Kaggriculture**—leveraging Large Language Models (LLMs) requires navigating strict sandbox constraints: **zero internet access during evaluation**, strict execution time caps (9 hours max for submissions, 1-2s latency per simulation turn), and a 30-hour weekly GPU quota.

Naively dropping an autoregressive LLM into every decision point causes immediate timeouts or crashes. Under our **Dual-Process Cognitive Architecture**, LLMs are **NOT** the primary decision loop; they serve as **System 2 Deliberative Engines**, invoked conditionally when fast System 1 heuristic screening detects high uncertainty ($\tau < 0.40$) or deadlocks.

```
                      [ Environment State / Puzzle Demonstration ]
                                          │
                                          ▼
                         [ System 1: Fast Heuristic Filter ]
                                 (< 50ms, 0 Tokens)
                                          │
                        Is Confidence High & No Deadlock?
                                   ├─── YES ───► Fast Path (Execute heuristic/DSL)
                                   │
                                   └─── NO  ───► [ System 2: Kaggle Offline LLM ]
                                                 • ARC-2: Propose Python DSL programs
                                                 • ARC-3: Synthesize macro-plans
                                                 • Kaggriculture: Strategic crop allocation
```

---

## 2. Kaggle Infrastructure & Constraint Matrix

### Hardware & Environment Limits

| Resource | Kaggle Specification | Implication for LLM Deployments |
|---|---|---|
| **GPU Accelerator** | **2x NVIDIA Tesla T4** (16 GB VRAM each = 32 GB total) OR **1x Nvidia P100** (16 GB) | T4 has Tensor Cores; supports FP16, INT8, INT4. P100 lacks INT4 Tensor cores (T4 is preferred for quantization). |
| **CPU / RAM** | 4 vCPUs, 30 GB System RAM | Can run quantized 7B GGUF on CPU if GPU is busy; 30 GB RAM prevents OOM when loading models. |
| **Disk Space** | 20 GB in `/kaggle/working`, read-only `/kaggle/input` | Model weights must reside in `/kaggle/input/` (do NOT copy to `/kaggle/working` to avoid filling disk). |
| **Internet Access** | **Strictly DISABLED** during competition re-runs | **No OpenAI, Anthropic, or Hugging Face Hub downloads.** All weights, tokenizers, and dependencies must be mounted. |
| **Timeout Limits** | 9 hours for competition scoring; 12 hours interactive session | 7B models taking 3s/step can only take ~10,800 steps total. Selective gating is mandatory. |
| **Weekly Quota** | 30 hours GPU compute per week (resets weekly) | Offline development on local machines; use Kaggle GPU quota exclusively for submissions and fine-tuning. |

---

## 3. How to Attach Models in Kaggle (No Internet)

Kaggle provides the **Kaggle Models Hub**, mounting open-weight models directly into `/kaggle/input/` as read-only volumes.

### 3.1 Declarative Configuration via `kernel-metadata.json`

Add model references directly into your `kernel-metadata.json` so Kaggle pre-mounts them:

```json
{
  "id": "ser8147/arc-laya-dual-process-submission",
  "title": "arc-laya-dual-process-submission",
  "code_file": "kaggle_arc2_submission.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": false,
  "enable_gpu": true,
  "enable_internet": false,
  "competition_sources": [
    "arc-prize-2026-arc-agi-2"
  ],
  "model_sources": [
    "google/gemma-2/transformers/2b-it/2",
    "qwen/qwen-2.5-coder-7b-instruct/transformers/default/1"
  ]
}
```

### 3.2 Finding Model Mount Paths

When mounted, weights appear under:
```bash
/kaggle/input/<model-slug>/<framework>/<variation>/<version>/
# Examples:
# /kaggle/input/gemma-2/transformers/2b-it/2/
# /kaggle/input/qwen-2.5-coder-7b-instruct/transformers/default/1/
```

To dynamically locate mounted models in Python without hardcoding:
```python
import os, glob

def locate_model_dir(pattern: str) -> str:
    matches = glob.glob(f"/kaggle/input/**/{pattern}/**/config.json", recursive=True)
    if not matches:
        raise FileNotFoundError(f"Model matching '{pattern}' not found in /kaggle/input")
    return os.path.dirname(matches[0])
```

---

## 4. Production Implementation Recipes

### Recipe 1: 4-Bit Quantized 7B/9B Models (DeepSeek-R1-Distill / Qwen2.5-Coder)
*Best for: ARC-AGI-2 Complex Code Induction & Symbolic Synthesis*

Kaggle's default PyTorch environment includes `bitsandbytes` and `accelerate`. Using 4-bit NF4 quantization allows a 7B or 9B model to fit into **~5.5 GB VRAM** (fitting easily on one T4 GPU, leaving the second GPU free for validation or parallel batching).

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

model_path = locate_model_dir("qwen-2.5-coder-7b-instruct")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=bnb_config,
    device_map="auto",
    local_files_only=True,
    torch_dtype=torch.float16,
)

def generate_program_hypothesis(prompt: str) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.2,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
```

---

### Recipe 2: Sub-Second 1.5B/2B SLM (Gemma-2-2B / Qwen2.5-Coder-1.5B)
*Best for: ARC-AGI-3 & Kaggriculture Real-Time Interactive Turns*

Small Language Models (SLMs) run in native FP16/BF16 without quantization overhead, achieving **>60 tokens/sec** on an NVIDIA T4.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = locate_model_dir("gemma-2-2b-it")

tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="cuda:0",
    torch_dtype=torch.bfloat16,
    local_files_only=True,
)

@torch.inference_mode()
def quick_action_policy(state_summary: str) -> str:
    prompt = f"<bos><start_of_turn>user\nState: {state_summary}\nChoose ACTION (1-5):<end_of_turn>\n<start_of_turn>model\n"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
    outputs = model.generate(**inputs, max_new_tokens=16, temperature=0.1)
    return tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
```

---

### Recipe 3: GGUF Inference via `llama-cpp-python` (Zero-VRAM / Hybrid Execution)
*Best for: Running LLMs on CPU without consuming GPU memory reserved for neural vision/encoders*

1. Create a Kaggle Dataset containing the pre-compiled `llama_cpp_python` wheel and `.gguf` weight file.
2. Install offline and execute:

```python
# Install from offline wheel
# !pip install --no-index --find-links /kaggle/input/llama-cpp-wheels llama-cpp-python

from llama_cpp import Llama

llm = Llama(
    model_path="/kaggle/input/deepseek-r1-q4-gguf/deepseek-r1-distill-qwen-7b-q4_k_m.gguf",
    n_ctx=2048,
    n_gpu_layers=24,  # Offload 24 layers to T4, rest to CPU RAM
    n_threads=4,      # 4 vCPUs
    verbose=False,
)

def query_gguf(prompt: str) -> str:
    response = llm(
        prompt,
        max_tokens=256,
        stop=["<|im_end|>", "\n\n\n"],
        temperature=0.3,
    )
    return response["choices"][0]["text"].strip()
```

---

### Recipe 4: Offline Dependency Management (`pip install` without internet)

Kaggle kernels in competition rerun mode cannot access PyPI. To use third-party libraries (`vllm`, `tiktoken`, `outlines`, `lark`):

1. **On your local machine**, download the exact binary wheels for Linux x86_64, Python 3.12:
   ```bash
   pip download -d ./offline_wheels \
       --platform manylinux2014_x86_64 \
       --only-binary=:all: \
       --python-version 312 \
       outlines lark
   ```
2. **Upload `offline_wheels/` as a Kaggle Dataset:**
   ```bash
   kaggle datasets create -p ./offline_wheels
   ```
3. **In the submission notebook**, install before any imports:
   ```python
   !pip install --no-index --find-links /kaggle/input/my-offline-wheels outlines lark
   ```

---

## 5. Domain-Specific Integration Strategies

### 5.1 ARC-AGI-2: Program Induction & DSL Code Generation
* **Problem:** 68% of ARC tasks preserve dimensions; 16% have constant dimensions. Pure symbolic search hits exponential bottlenecks on compositional rules with depth $D \ge 3$.
* **LLM Role:** Act as a **Candidate Program Proposer**.
* **Workflow:**
  1. System 1 classifies the transformation family (`geometry`, `topology`, `color_permutation`, `extrapolation`).
  2. If System 1 confidence < 0.40, prompt a 7B coding model with the grid representations of the 3-5 training pairs.
  3. LLM emits a small Python function `def transform(grid: list[list[int]]) -> list[list[int]]:`.
  4. System 2 executes the generated function in a restricted sandboxed process against the training demonstrations.
  5. If the function passes 100% of training pairs, execute it on test inputs (`attempt_1`).

### 5.2 ARC-AGI-3: Dynamic Interactive Agent
* **Problem:** Scoring games run in real-time loops. Calling an LLM every frame causes timeout.
* **LLM Role:** **Hierarchical Macro-Planner**.
* **Workflow:**
  1. System 1 runs at 60 FPS: CRC32 state hashing, obstacle avoidance, frontier exploration.
  2. When System 1 detects an **Impasse** (visited loop $S_t \in \mathcal{H}_{\text{recent}}$ for $> 6$ steps) or a **Level Reset** post-`GAME_OVER`:
  3. Invoke a fast 1.5B/2B LLM once to analyze the sequence of state transitions and output a **Macro-Plan** (e.g. `[ACTION3, ACTION3, ACTION5, ACTION1]`).
  4. System 1 consumes and executes the queue of macro-actions.

### 5.3 Kaggriculture: Turn-Based Farm Management
* **Problem:** Each turn allows evaluating economic decisions (market prices, weather, crop decay, labour allocation).
* **LLM Role:** **Strategic Regret & Portfolio Optimization**.
* **Workflow:**
  1. Compress the game observation into an economic summary (liquid cash, seed costs, weather forecast, expected yield).
  2. Prompt an LLM at turn 0 and turn milestones (every 20 turns) to set high-level investment targets (e.g., "maximize high-yield rice, hedge with corn against drought").
  3. Rule-based executor manages the deterministic per-tick harvesting and planting.

---

## 6. Recommended Action Plan for Our Stack

1. **Immediate (ARC-AGI-2):**
   - Package `qwen/qwen-2.5-coder-7b-instruct` or `google/gemma-2/transformers/2b-it` into `kernel-metadata.json`.
   - Build a Python DSL execution sandbox that tests LLM-generated transformation code on training demonstrations before predicting test items.
2. **ARC-AGI-3:**
   - Keep System 1 (our current 0.28 agent) as the primary execution spine.
   - Attach a 1.5B model strictly for impasse breakthrough when deadlock probability $P(\text{stuck}) > 0.80$.
3. **Paper Track Synergy:**
   - Documenting this offline LLM System 2 integration with verified Brier calibration provides the exact empirical bridge between neural intuition and symbolic verification required for top prize consideration.

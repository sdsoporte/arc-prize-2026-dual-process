# Local and Kaggle Environment

> **Verified:** 2026-09-23 on this machine and against the live `ser8147` Kaggle account.
> **Companion:** [`docs/KAGGLE_OPS.md`](KAGGLE_OPS.md) — the Kaggle command/API runbook.

This document answers two questions:

1. What can this machine actually run today, and how do I bootstrap it for this repository?
2. How do I consume Kaggle's compute **deliberately** instead of by accident?

Everything below was probed, not recalled. Where a fact could not be determined, the document says so
and gives the probe command that would resolve it.

---

## 1. Local environment — observed state

### The probe

```
$ python3 -V
Python 3.14.4
```

A single probe that reports interpreter, package manager, and whether each package this repository
touches is importable:

```bash
python3 - <<'PY'
import importlib
for m in ["numpy","torch","transformers","arcengine","arc_agi","pandas","pyarrow",
          "safetensors","datasets","peft","trl","accelerate","kaggle","kagglesdk"]:
    try:
        mod = importlib.import_module(m)
        print(f"PRESENT  {m}  {getattr(mod, '__version__', '?')}")
    except Exception as e:
        print(f"MISSING  {m}  ({type(e).__name__})")
PY
```

### Observed result, 2026-09-23

```
PRESENT  numpy  2.5.3
MISSING  torch  (ModuleNotFoundError)
MISSING  transformers  (ModuleNotFoundError)
MISSING  arcengine  (ModuleNotFoundError)
MISSING  arc_agi  (ModuleNotFoundError)
MISSING  pandas  (ModuleNotFoundError)
MISSING  pyarrow  (ModuleNotFoundError)
PRESENT  kaggle  2.2.4
PRESENT  kagglesdk  0.1.37
MISSING  safetensors  (ModuleNotFoundError)
MISSING  datasets  (ModuleNotFoundError)
MISSING  peft  (ModuleNotFoundError)
MISSING  trl  (ModuleNotFoundError)
MISSING  accelerate  (ModuleNotFoundError)
```

### Interpreter and package-manager layout

```
$ which python3 pip
/usr/bin/python3
/usr/bin/pip

$ python3 -m pip --version
pip 25.1.1 from /usr/lib/python3/dist-packages/pip (python 3.14)

$ echo "VIRTUAL_ENV=${VIRTUAL_ENV:-unset}"
VIRTUAL_ENV=unset
$ ls -d .venv venv        # neither exists
```

- The interpreter is the **system** Python (`/usr/bin/python3`), version **3.14.4**.
- There is **no virtual environment** activated and none checked into the repo.
- `pip` is the system pip (25.1.1) resolving under `/usr/lib/python3/dist-packages`.
- The Kaggle tooling lives in the **user site-packages**:
  `/home/s/.local/lib/python3.14/site-packages/{kaggle,kagglesdk}`.
- The user site-packages directory holds only 51 entries (packages and their `.dist-info` metadata
  combined) — this is a lean environment, not a data-science one.

---

## 2. What is missing, and why it matters

| Missing package | Who needs it | Consequence today |
|---|---|---|
| `torch` | `models/laya_arc_finetuned*/train_ddp.py`, the Laya clients, the training kernel | No local training or inference |
| `transformers` | every path that loads the Laya encoder/tokenizer | Cannot load the model locally |
| `arcengine` | `src/arc3_spatial_memory_agent.py`, `src/state_encoder.py`, `src/laya_dual_agent.py`, `src/test_prototype.py` | Those modules fail at import |
| `arc_agi` | the ARC-AGI-3 agent harness | Cannot run an agent locally |
| `pandas` / `pyarrow` | dataset JSONL tooling, notebooks | No tabular/JSONL convenience layer beyond stdlib |
| `safetensors` | loading `model.safetensors` | Cannot deserialize the published checkpoint |
| `datasets`, `accelerate`, `peft`, `trl` | the training notebook (`notebooks/kaggle_laya_arc_train.ipynb`) | Training deps only present inside Kaggle |

Everything the Kaggle CLI and the raw HTTP API need (`kaggle`, `kagglesdk`, `curl`, `python3`) **is**
present, which is why the read-only runbook in `docs/KAGGLE_OPS.md` works on this machine unchanged.

---

## 3. Bootstrapping the local environment

### 3.1 The Python-version problem (read this first)

The interpreter is **3.14.4**. The repository's vendored offline wheels were built for **CPython 3.12**:

```
$ ls data/arc-agi-3/arc_agi_3_wheels/ | head
arc_agi-0.9.8-py3-none-any.whl
arcengine-0.9.3-py3-none-any.whl
numpy-2.4.4-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl
pillow-12.2.0-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl
...
```

- **Pure-Python wheels** (`py3-none-any`) — `arc_agi-0.9.8`, `arcengine-0.9.3` — can be installed on
  3.14.
- **Compiled wheels** (`cp312`) — `numpy-2.4.4`, `pillow-12.2.0`, `matplotlib-3.10.8`, etc. — **cannot**
  be installed on 3.14. They must be rebuilt for 3.14 or obtained from a 3.12 environment.

Practical consequences:

- On Python 3.14 you can reasonably install the ARC-AGI-3 *harness* packages from the vendored wheels,
  but not the compiled scientific stack from that wheel directory.
- `torch` and `transformers` are not present in the vendored wheels at all; they must come from PyPI
  (network) or from Kaggle.
- Do not silently assume a 3.14 environment can host the training stack. If local training is required,
  use a **Python 3.12 virtual environment** (matching the wheel tags) or run on Kaggle.

### 3.2 Offline ARC-AGI-3 install (pure-Python wheels)

The wheels are checked into the repo at `data/arc-agi-3/arc_agi_3_wheels/`. The ARC-AGI-3 submission
notebook installs them offline exactly like this:

```
!pip install --no-index --find-links \
    /kaggle/input/competitions/arc-prize-2026-arc-agi-3/arc_agi_3_wheels \
    arc-agi python-dotenv
```

The local equivalent, into a **dedicated 3.12 virtualenv**, would be:

```bash
python3.12 -m venv .venv
. .venv/bin/activate
pip install --no-index --find-links data/arc-agi-3/arc_agi_3_wheels arc-agi python-dotenv
```

> **Not executed.** Creating a virtualenv and installing packages mutates the environment and is outside
> the read-only policy of this task. The command shape is taken from the existing notebook (offline
> `--no-index --find-links` recipe) and from the vendored wheel names.

The `arc-agi` wheel provides the `arc_agi` package; `arcengine-0.9.3` provides `arcengine`, which
`src/arc3_spatial_memory_agent.py` imports. The vendored `arc-agi` requires Python `>=3.12`.

### 3.3 The Laya training / inference stack

The repository's own notebook records the dependency set it used:

```
!pip install -q -U "laya>=0.1.6" "transformers>=4.48.0" "datasets>=3.0.0" \
    safetensors huggingface_hub pyarrow pandas scipy accelerate tabulate
```

and the training script consumes `torch`, `torch.distributed` (NCCL), `safetensors.torch`, and
`transformers.AutoTokenizer`:

```python
# models/laya_arc_finetuned_v2/train_ddp.py
import torch
import torch.distributed as dist
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer
from laya.common import build_model, proper_reward, QTYPES
```

The **published checkpoint records the library that saved it**. From
`models/laya_arc_finetuned_v2/laya_finetuned_typed_decisions/encoder/config.json`:

| Field | Value |
|---|---|
| `transformers_version` | **`5.17.0`** |
| `model_type` | `modernbert` |
| `architectures` | `["ModernBertForMaskedLM"]` |

and from the same model's `rl_agent_config.json`:

| Field | Value |
|---|---|
| `encoder` | `answerdotai/ModernBERT-large` |
| `head_layers` | 2 |
| `amp_dtype` | `bf16` |
| `training.world_size` | 1 |
| `training.hours` | 1.96 |

So the version that actually matters for loading the produced encoder is **transformers ≥ 5.x line**
(the artifact says 5.17.0), which is **newer** than the `>=4.48.0` floor the notebook pins. A 4.x
install can import the notebook deps but may not reconstruct the saved `ModernBertForMaskedLM` config
identically.

### 3.4 What version string to trust — and what I could not determine

**The honest position:**

- The artifact records `transformers_version: 5.17.0`. That is a string written by the library at save
  time; it is strong evidence of the saving environment, not proof of the minimum loadable version.
- The notebook pins `transformers>=4.48.0` and `laya>=0.1.6`. Those are floors, not a tested lock.
- There is **no `requirements.txt`, `pyproject.toml`, lockfile, or `environment.yml`** at the repository
  root. The only dependency manifest in the tree is
  `data/arc-agi-3-agents/pyproject.toml`, which describes the vendored ARC-AGI-3 *agents* framework, not
  the Laya model stack.
- **I could not determine the exact `torch` build** (version and CUDA variant) that produced
  `model.safetensors`. Nothing in the repo pins it.

Probe commands that would resolve the unresolved items (all **not executed** here — they query or mutate
the package environment):

```bash
python3 -m pip index versions torch                 # available torch versions for this interpreter
python3 -m pip index versions transformers          # confirm a 5.17.0 release exists on the index
python3 -m pip download --no-deps torch -d /tmp/x   # inspect wheel tags without installing (writes outside repo)
kaggle kernels logs ser8147/arc-laya-finetune       # the training kernel's own pip install line
python3 -c "import json;print(json.load(open('models/laya_arc_finetuned_v2/laya_finetuned_typed_decisions/encoder/config.json'))['transformers_version'])"
```

If a local environment must exactly reproduce the published model, prefer reading the training kernel's
log (source of truth for what ran) over inferring from the notebook cell.

### 3.5 Suggested bootstrap, by goal

| Goal | Environment | Packages |
|---|---|---|
| Inspect Kaggle state / writeups | current Python 3.14 | already complete (`kaggle`, `kagglesdk`, `curl`) |
| Run the ARC-AGI-3 agent harness locally | Python 3.12 venv | `arc-agi`, `arcengine` (+ pure-Python wheel deps) from `data/arc-agi-3/arc_agi_3_wheels/` |
| Generate the ARC-AGI-2 submission JSON | current Python 3.14 | `numpy` (present); `src/arc2_symbolic_dsl.py` is numpy-only |
| Load / fine-tune the Laya model | Python 3.12 venv **or Kaggle** | `torch` (CUDA build), `transformers` (5.x), `laya>=0.1.6`, `safetensors`, `datasets`, `accelerate` |
| Full offline competition-style run | Kaggle notebook | notebook's own `pip install` cell, no internet |

---

## 4. Consuming Kaggle compute deliberately

### 4.1 The weekly quota

```
$ kaggle quota
resource  used   remaining  total   refreshAt
--------  -----  ---------  ------  -------------------
GPU       1.37h  28.63h     30.00h  2026-09-26T00:00:00
TPU       0.00h  20.00h     20.00h  2026-09-26T00:00:00
```

- GPU: **30 h/week**, 1.37 h used, **28.63 h remaining**, refresh `2026-09-26T00:00:00`.
- TPU: **20 h/week**, 0 used.
- Quota is consumed by **kernel executions**. The reading commands in `docs/KAGGLE_OPS.md` (list, status,
  files, quota, writeup access) are read-only and do not run a kernel. A `kaggle kernels push` triggers a
  run and therefore spends quota.
- `kaggle quota` is the only trustworthy reading; check it before and after any run.

### 4.2 How an accelerator is selected

Two mechanisms, in order of precedence:

1. **`kernel-metadata.json` booleans** — `enable_gpu: true` / `enable_tpu: true`.
2. **`kaggle kernels push --accelerator <value>`** — overrides the booleans. In the CLI source this is
   sent as the request's **`machine_shape`**.

The `kagglesdk` request docstring lists the supported `machine_shape` values:

```
The machine shape to use for this session. Currently supported options:
 * NvidiaTeslaT4
 * NvidiaTeslaP100
 * Tpu1VmV38
```

> **Documented from the SDK docstring, not exercised.** Selecting an accelerator requires a real push,
> which runs a kernel and consumes quota; that was not performed.

Both `enable_gpu` and `enable_tpu` are marked **`DEPRECATED: use machine_shape instead`** in the SDK.

Observed in this repo's metadata: `notebooks/kernel-metadata.json` uses the deprecated booleans
(`enable_gpu: true`, `enable_tpu: false`); `notebooks/arc3_submission_kernel/kernel-metadata.json` sets
`enable_gpu: true`, `enable_internet: false`.

### 4.3 Local vs Kaggle — the practical decision

| Situation | Where to run | Why |
|---|---|---|
| Read Kaggle state, inspect writeups, list artifacts | **Local** | No quota cost; the CLI/API work here |
| Pull a kernel's log or file list | **Local** | Read-only, no quota |
| Edit a notebook and try one cell | **Local** for syntax, **Kaggle** for anything importing `arcengine`/`torch` | Those packages are absent locally |
| Fine-tune Laya / run DDP | **Kaggle** (or a 3.12 env with a CUDA torch) | `torch` missing; 30 h/week GPU is available and offline by rule |
| Generate the ARC-AGI-2 submission JSON | **Either** | `src/arc2_symbolic_dsl.py` is numpy-only and numpy is present |
| Run the ARC-AGI-3 agent end-to-end | **Kaggle** | Needs `arc_agi` + `arcengine` + competition harness |
| Iterate on a competition submission | **Kaggle, deliberately** | Each `competitions submit` on ARC-AGI-2/3 burns 1/day |

Budget rule of thumb: with 28.63 GPU-hours left and a weekly refresh, a single fine-tune in this repo
recorded `training.hours: 1.96` (1.96 h for 1 epoch, `world_size: 1`). That is ~7% of the weekly quota
per run. Treat each push as a spend.

### 4.4 Offline execution is a competition constraint

The ARC-AGI-3 submission kernel declares `enable_internet: false` and installs dependencies only from
the competition's own wheel directory. Any local bootstrap for parity should therefore use
`--no-index --find-links`, not PyPI. `docs/KAGGLE_LLM_INTEGRATION_GUIDE.md` documents the offline
`pip install` recipe for Kaggle notebooks.

---

## 5. Unresolved / explicitly unverified

- **Exact `torch` version and CUDA build** that produced `model.safetensors` — not determinable from the
  repository. Probe: the training kernel log / a fresh environment resolve.
- **Whether `transformers` 5.17.0 is loadable on Python 3.14** — not tested. The safest path is a Python
  3.12 environment or Kaggle itself.
- **Whether the cp312 wheels in `data/arc-agi-3/arc_agi_3_wheels/` are sufficient to build a full 3.12
  environment** — not tested; only their filenames and the notebook's `--find-links` recipe were read.
- **Any accelerator value beyond the three SDK-documented ones** — not tested; requires a real push.
- **Local quota accounting** — `kaggle quota` reads the account's weekly totals; how individual runs are
  metered was not measured.
- **Virtualenv creation and package installation** — deliberately **not executed** in this task, because
  it mutates the environment and lies outside the repository's read-only policy.

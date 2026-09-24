# ODD Feature — `system1-wiring`

> **Status:** in progress
> **Branch:** `feat/system1-wiring` (stacked on `feat/arc3-strategy-probe`, 10 commits ahead of `main`)
> **Base:** `02a12e1` · `main` == `origin/main` == `bf576e2`
> **Created:** 2026-09-24
> **Serves:** `odd/OBJECTIVE.md` §4 criterion **Completeness** (primary) and **Accuracy** (secondary)
> **Engram mirror topic key:** `arc-paper-track/odd/system1-wiring/tasks`

---

## 1. Goal

Make the paper's central claim **true**: put System 1 into the submission path, and **measure whether it
helps before spending a competition submission on it.**

Today `model_sources: []` on all six kernels — the model is published and consumed by nothing, while the
writeup describes it as the mechanism that gates the submission. That gap is the admissibility problem, not
a rubric detail.

> **REVISION, 2026-09-24 — the measurement answered before the build (see §11).** The direct measurement of
> the underlying question returned **zero headroom: 0 of 43 covered tasks**. The first candidate that
> reproduces the training pairs is *always* the correct one, because the DSL's candidates are functionally
> redundant wherever coverage exists. So **T3, T4 and T7 are cancelled, not deferred.**
>
> The goal is therefore unchanged in wording and changed in method: the paper's claim gets made true by
> **declaring the System 1 separate and reporting the measured null**, rather than by wiring it in. That is
> what §11 measured, and it is a stronger position than either wiring it blindly or hand-waving it.

## 2. Why ARC-AGI-2, and why a ranker rather than a pruner

**Two design decisions, both forced by measurement.**

**2.1 The paper's architecture belongs to ARC-AGI-2.** The writeup says System 1 classifies the
transformation family and prunes the DSL before System 2 searches. That *is* the ARC-AGI-2 path. ARC-AGI-3
uses the other role (impasse detection, action routing). So ARC-AGI-2 is the track where wiring makes the
described architecture real.

**2.2 A gate that PRUNES makes the score worse.** Measured coverage — a candidate that reproduces every
training pair — is:

| | DEPLOYED (317 lines) | LOCAL (666 lines) |
|---|---|---|
| TEST (240) | 5/240 = **2.1%** | 15/240 = **6.2%** |
| EVAL (120) | 0/120 = 0.0% | 0/120 = 0.0% |

Pruning candidates can only *reduce* `n_matching > 0`, so a pruning gate lowers coverage. A **ranking** gate
leaves coverage untouched and changes **which two attempts get submitted** — and that is where the official
metric can actually move. It is also falsifiable: on the tasks where coverage exists, does the gate place
the correct candidate in the top two?

That question is the feature's decision gate (§8).

## 3. Feasibility — verified, not assumed

| Fact | Value | Source |
|---|---|---|
| `laya` wheel is tiny and pure Python | `laya-0.3.11-py3-none-any.whl` = **106 KB** | `pip download` |
| Offline install pattern already used in this repo | the ARC-3 kernel runs `pip install --no-index --find-links /kaggle/input/.../arc_agi_3_wheels` | kernel cell 1 |
| ARC-2 kernel internet | `enable_internet: False` → wheel must arrive as a dataset source | live metadata |
| ARC-2 kernel GPU | `enable_gpu: False` → CPU it is | live metadata |
| CPU cost | 240 tasks x ~2 s = **~8 min** against a ~9 h notebook budget | v1 benchmark p50 1581 ms |
| Model mount path | `/kaggle/input/arc-laya/transformers/typed-decisions/<version>` | model card |
| `torch` / `transformers` | already in the Kaggle image, no install needed | Kaggle runtime |
| **`kernels push` costs a submission?** | **No.** push runs the notebook in commit mode; only `competitions submit` spends quota | `kaggle kernels push --help` |
| Submission quota | **1/day per track**, resets 00:00 UTC. Both tracks showed `Remaining today: 1` at 03:15 UTC | `submission-limits` |

The last two are what make this feature affordable: **everything can be wired, pushed, and measured before
the single daily submission is spent.**

## 4. Non-goals

- Does not re-argue the architecture in prose. It makes it true or declares it separate.
- Does not touch ARC-AGI-3's agent policy. That track gets a **deploy of the already-built v3**, nothing more.
- Does not chase the ARC-AGI-2 score. Even a perfect ranker is capped by 2.1–6.2% coverage.
- Does not fix the paper. That is the follow-on feature, after these numbers exist.

## 5. Allowed edit surfaces

- `odd/OBJECTIVE.md`
- `odd/tasks/system1-wiring.md`
- `src/arc2_dual_process_solver.py`
- `experiments/arc2_ranker_eval.py`
- `experiments/test_arc2_ranker_eval.py`
- `notebooks/arc2_submission_kernel/kaggle_arc2_submission.ipynb`
- `notebooks/arc2_submission_kernel/kernel-metadata.json`
- `notebooks/arc3_submission_kernel/submission.ipynb`
- `notebooks/arc3_submission_kernel/kernel-metadata.json`
- `scripts/kaggle/laya_wheel_dataset.sh`
- `docs/SYSTEM1_WIRING.md`

## 6. Tasks

| # | Task | Status | Evidence |
|---|---|---|---|
| T1 | Write the governing objective and scaffold this feature | done | `odd/OBJECTIVE.md`; this file |
| T2 | Publish the `laya` wheel as a Kaggle dataset (offline install source) | done | `ser8147/laya-wheel`, **public**, `ready`; reproducible via `scripts/kaggle/laya_wheel_dataset.sh` |
| T3 | Add a **ranker seam** to the ARC-2 solver and an offline evaluation harness | **cancelled** | T4's measurement returned 0 headroom; building it would have produced a null at the cost of days |
| T4 | **The decision gate**: measure whether the ranker puts the correct candidate in the top 2 | **done (early)** | ran one level earlier than designed — see §11 |
| T5 | Fix the artifact wiring: pin the HF revision, drop the spurious dataset edge, link the HuggingFace repo, retire/document the legacy lineage. **`add model_sources` is CANCELLED** — under §11's pivot the model is a declared offline component, so it must stay uncabled | pending | wiring audit + kernel metadata |
| T6 | Deploy ARC-AGI-3 **v3** (regenerate the notebook from the current seeded agent, push, submit ~0.35) | pending | kernel version + leaderboard |

> **Scope after the pivot (2026-09-24).** With T3/T4/T7 cancelled, this feature is no longer about making the
> paper's architecture true. It is now about making the **artifacts honest** (T5) and landing one bounded
> deploy (T6). The paper work moved to its own feature, and the ARC-AGI-2 number to `arc2-reference`.
> See `odd/OBJECTIVE.md` §10 and §11.
| T7 | Submit ARC-AGI-2 — **only if T4 justifies it** | **cancelled** | T4 returned zero; there is no reason to spend a submission |
| T8 | Independent verification | pending | verifier report |

## 7. Acceptance criteria

1. The model has **at least one declared consumer** (`model_sources` non-empty) and that kernel **runs
   offline** (`enable_internet: False`).
2. The ranker's effect is measured **before** any submission is spent, on the tasks where coverage exists,
   against the no-ranker baseline.
3. If T4 shows no improvement, **the submission is not spent** and the paper records the null result. A
   clean negative is an acceptable outcome of this feature.
4. No kernel metadata change is made that would break the rerun path without a measured justification.
5. `data/arc-agi-3/environment_files/` is untouched; the local harness is re-run only in OFFLINE mode.
6. Every claim in `docs/SYSTEM1_WIRING.md` is accompanied by the command that produced it.

## 8. The decision gate (T4)

The number that decides whether T7 happens:

```
for each task where >= 1 candidate reproduces every training pair:
    does the correct candidate appear in the ranker's top 2?

report: top-2 hit rate  WITH ranker  vs  WITHOUT ranker (candidate order as built)
```

- If the ranker's hit rate exceeds the baseline **and** the difference is larger than seed noise, submit.
- If not, do not submit, and the paper says so with the measured number.
- The ARC-AGI-2 official metric is measured on the public eval set too, but note **coverage there is 0/120**,
  so no ranker can move it. That is already a known ceiling and must be stated, not hidden.

## 9. Known risks carried

| Risk | Impact | Owner |
|---|---|---|
| Coverage on eval is 0/120, so ARC-AGI-2's score cannot move regardless of the ranker | Accuracy stays 0.00 on ARC-AGI-2 | inherent; state it |
| Corpus is capped at 2.1–6.2% coverage | The ranker can only act on a handful of tasks, so its measured effect will be small and noisy | this feature |
| `enable_gpu: False` and CPU latency | ~8 min, acceptable | none |
| A model-source change alters the kernel metadata, and only a real submission exercises the rerun | A config mistake costs the day's submission | verify the commit-mode run first |
| v3 has never been scored; predicted ~0.35 from a reproducible local 0.3508 | It may land anywhere in the 0.17–1.09 seed range | T6 |

## 11. The decision gate, run early — and it returned zero

T4 was specified to decide whether to spend a submission. Measuring first showed it was **unmeasurable as
designed**, because coverage and ground truth do not coincide:

| set | coverage | ground truth? |
|---|---|---|
| EVAL (120) | **0/120** | yes — but nothing to rank |
| TEST (240) | 5/240 deployed, 15/240 local | **no** — unmeasurable |
| TRAIN (1000) | 43/1000 (4.3%) | yes — the only set with both |

So the gate was run on the train set instead, against the underlying question rather than the proxy:

```
tasks with coverage                                       43
  with MORE THAN ONE candidate reproducing the train      18
  where those candidates DISAGREE on the output            1   <-- the only place a ranker could act
  the correct output exists among the candidates           42
  ...and is already in the current top-2 (no ranker)       42
  position of the correct candidate in build order         [0]

MAXIMUM ranker headroom: 0 tasks of 43
```

**The first matching candidate is always the correct one.** The candidates are **functionally redundant**
wherever coverage exists — `identity+color_map` and `color_map` yield identical output — so ordering a list of
equivalents cannot change anything. (Measured on `test[0]`; tasks with several test inputs give the official
metric more outputs, but the conclusion is unambiguous.)

**Consequences:**
1. **T3 and T4 cancelled** — the measurement that justified them returned zero.
2. **T7 cancelled** — no reason to spend a submission.
3. **The System 1 has no measured role in either track.** ARC-AGI-2: pruning can only reduce coverage, and
   ranking has zero headroom. ARC-AGI-3: the heuristic layer is already inert (92.8% single-candidate
   pools), so a learned gate faces the same collapsed choice set, and the real decisions come from the BFS
   frontier and deadlock branches.
4. **This resolves `odd/OBJECTIVE.md` §9's open decision.** System 1 is **offline-measured and declared
   separate** — not as the cheap path, but because wiring it was *measured* to have no effect. That is a
   stronger claim in the paper than either wiring it or asserting it.

## 12. Evidence log

- 2026-09-24 — created. `main` == `origin/main` == `bf576e2`; this branch stacks 11 commits of prior
  verified work (`feat/arc3-eval-harness`, `feat/arc3-strategy-probe`) that are not yet on `main`.
- 2026-09-24 — **T2 done.** `ser8147/laya-wheel` published **public**, status `ready`, containing
  `laya-0.3.11-py3-none-any.whl` (106,531 bytes) and a provenance README. Verified by public search
  (`kaggle datasets list -s "laya wheel"` returns it), not just by the owner-scoped listing.
  - The wheel's **required** deps (`torch>=2.0.0`, `transformers>=4.48.0`, `safetensors>=0.4.0`,
    `huggingface_hub>=0.20.0`, `numpy>=1.20.0`) are all present in the Kaggle Python image, so
    `pip install --no-index` resolves them locally. Everything else sits behind optional extras
    (`serve`, `fast`, `mcp`, `onnx`, `langchain`, `langgraph`) and is not needed for inference.
  - **`kaggle datasets create` defaults to PRIVATE**; the public flag is `-u/--public`. The first attempt
    was private and had to be deleted and recreated.
  - The dataset carries the **upstream Apache-2.0 licence**, not the CLI's CC0 default, because it
    redistributes someone else's work. SHA-256 `1ee717dd05a742135869383af9b66b20e...` is pinned in the
    script, so a new upstream release is reviewed rather than silently absorbed.
  - `scripts/kaggle/laya_wheel_dataset.sh` is the recipe: fetch, verify the hash, assemble, and publish
    only with an explicit `--create` (dry run by default, verified exit 0 and nothing uploaded).
- 2026-09-24 — wiring audit findings that motivate the feature: `model_sources: []` on all six kernels; the
  ARC-2 kernel declares `ser8147/arc-laya-finetune-data` and **never reads it** (its only `/kaggle/input`
  access is the competition's own test challenges); the training kernel calls
  `snapshot_download('convaiinnovations/laya')` with **no pinned revision**; the legacy lineage
  (`laya-gate-finetune`, `fork-of-laya-gate-finetune` → `laya-gate-finetune-data`) is disconnected from the
  model entirely.
- 2026-09-24 — quota verified: `Remaining today: 1` on both tracks at 03:15 UTC; `kernels push` does not
  consume it.

# ODD Feature — `arc2-reference`

> **Status:** in progress
> **Branch:** `feat/system1-wiring` (stacked; none of the session's branches are on `main` yet)
> **Created:** 2026-09-24
> **Serves:** `odd/OBJECTIVE.md` §4 criterion **Accuracy** (ARC-AGI-2), and §6a.3 by giving the ablation a
> measured comparison point. Feeds the ARC-AGI-2 Grand Prize's Accuracy criterion.
> **Engram mirror topic key:** `arc-paper-track/odd/arc2-reference/tasks`

---

## 1. Goal

Get a **real, non-zero ARC-AGI-2 leaderboard number** by running the field's leading public pipeline as a
**declared reference implementation**, so that (a) Accuracy stops being 0.00 on that track, and (b) the
primitive ablation has something measured to be compared against.

## 2. This is not our work, and the paper must say so

The pipeline is public and permissively licensed, and the field replicates it en masse. Using it is legal
(external data and tools must be publicly available and free, which this is) and it is the norm — but:

- It must be presented as a **reference implementation**, never as our contribution.
- The ablation remains our work; the reference is the ruler it is measured against.
- The honest framing is a **measured map of ARC-AGI-2 approaches**, three points on one axis:

| approach | public training outputs (pass@2) | leaderboard |
|---|---|---|
| minimal public solver (symmetries + colour map) | 9/1076 (0.84%) | — |
| **our ablation** — only the families that pay | **45/1076 (4.18%)** | `0.00` |
| the field's full neural pipeline (this feature) | — | **~30, to be measured** |

## 3. What the evidence says about the payoff, and its limit

```
ARC-AGI-2 field (2,172 teams, measured)
  >= 40 :    4   ( 0.2%)
  >= 30 :  745   (34.3%)   <-- where a fork lands
  >=  1 : 1413   (65.1%)
    0.00 :  722   (33.2%)  <-- us today
```

So this takes Accuracy from the **bottom third to the upper-middle** — a large rubric gain — while making us
one of **745 teams in the same 30–40 band**, with only 4 above 40. **It buys Accuracy, not distinction**, and
the paper must say that plainly. That concentration is itself a finding: it is the signature of a single
public pipeline being replicated at scale.

## 4. Feasibility — verified

| Fact | Value |
|---|---|
| Public pipeline | `mikelou1/arc-agi2-lb33-89-minimal-perfpatch` (113 votes) — **now pulls successfully** (the 403 recorded on 2026-09-24 is stale); it is the ~33.89 original |
| Pullable forks | `manderson240/arc-agi-2-fork-lb33-89-20260903` (chosen: 1174 lines, a minimal perf-patch of the canonical, scored **31.81**, rank 211), `qiuqiuh/arc-highscore-lb3389-replica` (50 votes), `rokaiyasomapti/reproduce-nvarc-2025-results`, `luxluxshan/arc2-nvarc-v1` |
| Base model, offline | **Not `qwen-lm/qwen-3`.** The pipeline loads the already-SFT'd `sorokin/qwen3_4b_grids15_sft139` (`transformers/bfloat16/1`, 7.27 GB, Apache 2.0). It mounts offline and resolves correctly as a `model_sources` entry — the grids15 SFT is part of the recipe, so raw Qwen3-4B would not reproduce it |
| Offline dependencies | **NOT SATISFIED — see the gate verdict below.** `unsloth`, `trl` and `bitsandbytes` are absent from the offline image, and no fork installs them |
| Runtime budget | ARC-AGI-2 allows **12 h** CPU/GPU; L4x4 machines (96 GB) available |
| Internet | **disabled** at rerun → no `pip install`; the base model must arrive via `model_sources` |
| Submissions | **1/day**, resets 00:00 UTC; `kernels push` does **not** consume one |
| GPU quota | 30 h/week, 28.63 h remaining, refreshes 2026-09-26 |
| Deadline | **2026-11-02** |

### Gate verdict — measured 2026-09-24, in the target environment

A commit-mode run of `notebooks/arc2_reference_kernel/` on `NvidiaL4` (4x L4, sm_89, 22.0 GiB each,
`bf16_supported=True`) with `enable_internet: False` printed this from the offline image, and the kernel
stopped on its own hard gate:

```text
torch          True         2.10.0+cu128
transformers   True         5.0.0          <-- unsloth's API surface here targets transformers 4.x
unsloth        False        NOT INSTALLED (PackageNotFoundError)   <-- BLOCKS THE PIPELINE
peft           True         0.19.1
trl            False        NOT INSTALLED (PackageNotFoundError)   <-- BLOCKS (UnslothTrainer subclasses trl)
bitsandbytes   False        NOT INSTALLED (PackageNotFoundError)
datasets       True         5.0.0
accelerate     True         1.13.0
flash_attn     False        NOT INSTALLED
xformers       False        NOT INSTALLED
```

**Verdict: FAILED on dependency availability offline.** (**SUPERSEDED — see the resolution below.**)
`arc_solver.py` imports
`from unsloth import FastLanguageModel, UnslothTrainingArguments, UnslothTrainer` unconditionally, and
**none of the four pullable forks contains a single `pip install`, `wget`, `git clone` or
`snapshot_download`** — they all assume an image that ships `unsloth`. The current Kaggle image does not,
and `enable_internet: False` forbids installing it.

What *did* resolve offline, and therefore is not the problem: the competition data mounted correctly, the
base model mounted at `/kaggle/input/models/sorokin/qwen3_4b_grids15_sft139/transformers/bfloat16/1`
(6.77 GiB, 10 files including both safetensors shards), and four L4 GPUs were allocated.

Why the fork's own metadata still says the pipeline once ran offline: every pulled fork carries
`isInternetEnabled: false`, `dockerImageVersionId: 31090`, `exception: null`, `duration: 1548.2s`, and one
shared `kernelVersion` dataSource — the provenance of a single real run of the *canonical* notebook.
That run reached `benchmark_selection_algos()`, which calls `np.max()` on a list populated only from real
decoded results and raises `ValueError` on an empty one; `exception: null` therefore implies `unsloth`
imported successfully in image 31090. The dependency was present then and is absent now. This is an
inference from numpy semantics plus the shape of that cell, not a captured log.

Image pinning cannot recover it: the CLI accepts `docker_image_pinning_type` only in
`["original", "latest"]`, and a brand-new kernel has no older "original" environment to return to.

**What would be needed** (item 1 below is now DONE — see the resolution above; kept for the record):

1. A declared offline wheel source attached as `dataset_sources` supplying `unsloth`, `trl`,
   `bitsandbytes` and their build dependencies, installed with `pip install --no-index --find-links`.
   A public candidate exists — `asmaaalgiers/unsloth-offline-wheels` (4.39 GB, updated 2026-05-30) — but it
   must be verified against this image's `transformers 5.0.0` / `torch 2.10.0`, and it does not obviously
   cover `trl` or `bitsandbytes`.
2. Or a cooperative change to the pinned image (the `docker_image_pinning_type` mechanism does not allow a
   third party to select an arbitrary historical image).
3. Inference cost is a **second, independent** limit: a faithful competition rerun runs 240 test tasks
   through the upstream `12 h - 10 min` budget on 4 GPUs. Measured billing: a 69 s L4x4 commit session
   moved the weekly quota `1.37h -> 1.39h` (0.02 h), which is consistent with 1x wall-clock billing and
   inconsistent with 4x (which would have been ~0.08 h) — so a full rerun costs roughly 11.5 h of the
   28.6 h remaining, not 46 h. Affordable; the dependency is the binding blocker.

### Resolution — 2026-09-24 (R7): the wheelhouse unblocks the install

`ser8147/arc2-unsloth-wheelhouse` is a **public 15-wheel dataset (191.6 MiB)** built by
`scripts/kaggle/unsloth_wheelhouse.sh`: pinned versions, a per-wheel SHA-256 manifest
(`WHEELS.sha256`), a generated per-package licence table, a provenance README, and a
**dependency-closure check that refuses to publish a half-vendored set** (it evaluates every
`Requires-Dist` marker for the target interpreter and requires each one to be satisfied by a vendored
wheel or by a distribution the image already ships). The kernel declares it under `dataset_sources` and
installs it with `pip install --no-index --find-links /kaggle/input/arc2-unsloth-wheelhouse <15 pins>`, then
prints the same package table again and raises on a missing wheelhouse, on a wheelhouse that disagrees with
its own `PINNED.txt`, on a non-zero `pip` exit, on a pin that did not take effect, or on a failed import.

The **before/after table from one commit-mode run** (kernel version 3) — same rows, same order:

```text
package         before                        after
---             ---                           ---
torch           True   2.10.0+cu128            True   2.10.0+cu128   (deliberately not vendored)
transformers    True   5.0.0                   True   4.57.6         DOWNGRADED
unsloth         False  NOT INSTALLED           True   2026.9.11      vendored
unsloth_zoo     False  NOT INSTALLED           True   2026.9.7       vendored
peft            True   0.19.1                  True   0.19.1
trl             False  NOT INSTALLED           True   0.24.0         vendored
bitsandbytes    False  NOT INSTALLED           True   0.50.2         vendored
xformers        False  NOT INSTALLED           True   0.0.34         vendored
datasets        True   5.0.0                   True   4.3.0          DOWNGRADED
accelerate      True   1.13.0                  True   1.13.0
tokenizers      True   0.22.2                  True   0.22.2         left alone (already satisfies)
huggingface_hub True   1.11.0                  True   0.36.2         DOWNGRADED (transformers <1.0)
safetensors     True   0.7.0                   True   0.7.0          left alone
sentencepiece   True   0.2.1                   True   0.2.1          left alone
numpy           True   2.0.2                   True   2.0.2
tqdm            True   4.67.3                  True   4.67.3
flash_attn      False  NOT INSTALLED           False  NOT INSTALLED  (not vendorable offline)
tensorflow      True   2.20.0                  True   2.20.0         (uninstalled by the kernel after)
```

`pip exit code 0 after 34s`; the install plus the import gate reported `OFFLINE INSTALL OK in 98s`. The
`pip check` run afterwards lists **only pre-existing image conflicts** (bigframes/google-adk missing
`google-cloud-bigquery-storage`, google-colab's `jupyter-server`/`pandas` pins, dopamine-rl, moviepy) and
**none caused by this wheelhouse** — an earlier 30-wheel cut of it pinned `fsspec` and
`importlib-metadata` needlessly and did trip `gcsfs` (wants exactly `fsspec==2025.3.0`) and
`opentelemetry-api` (wants `importlib-metadata<8.8.0`); those pins were removed once the image inventory
was measured. This is why the shipped set is 15 wheels and not "everything unlisted".

**Transformers version: 4.57.6.** Inside unsloth 2026.9.11's `>=4.51.3,<=5.5.0` minus its exclusions
(5.0.0, 5.1.0), inside `unsloth_zoo` 2026.9.7's `>=4.52.4` minus the same exclusions, and the **newest 4.x
that also satisfies `trl==0.24.0`'s `transformers>=4.56.1`**. The image ships exactly 5.0.0, which unsloth
excludes, so the downgrade is forced rather than cosmetic. Two more downgrades are forced by the same graph:
`huggingface_hub` 1.11.0 -> 0.36.2 and `dill` 0.4.1 -> 0.4.0.

### What the unblock exposed: CUDA OOM at 21.65 / 22.03 GiB

The four runs moved the kernel from "cannot start" to "runs, then runs out of GPU memory", to a
bounded, clean completion:

| run | kernel version | terminal | what happened |
| --- | --- | --- | --- |
| 1 | v1 | ERROR after 31 s | pre-existing run; stopped on the preflight's own `unsloth` gate |
| 2 | v2 | ERROR after ~2 min | wheelhouse installed successfully (30-wheel cut), stopped on an over-strict pin assert: `torchvision pinned to 0.25.0 but installed 0.25.0+cu128` — the image's own CUDA build, i.e. a false alarm, since fixed |
| 3 | v3 | ERROR after ~11 min | install 34 s, gate passed, solver started, 240 tasks queued, **8 recorded task completions** (the "10 tasks solved" recorded at the time is not reproducible from the v3 log — 8 `finished` records), then CUDA OOM |
| 4 | v4 | **`COMPLETE`** after ~37 min | R8 applied (`use_gradient_checkpointing=True`). Install 34 s, `OFFLINE INSTALL OK in 96s`, 240 tasks queued, **24 tasks completed**, **no OOM**, `starter.py exit code 0 after 1788s`, submission-contract check **PASS** (21/240 tasks decoded). Measured in the R8 section below |

Run 3's fatal error, from one rank of `arc_solver.py`:

```text
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 104.00 MiB.
GPU 0 has a total capacity of 22.03 GiB of which 69.06 MiB is free.
Including non-PyTorch memory, this process has 21.96 GiB memory in use.
Of the allocated memory 21.65 GiB is allocated by PyTorch, and 66.83 MiB is reserved by PyTorch
but unallocated.
```

`starter.py` exited 1 after 456 s, and the checked subprocess call made that fatal, exactly as designed.
Tasks completed before the OOM: `00d62c1b` 291.8 s, `0520fde7` 59.3 s, `00576224` 84.8 s, `00dbd492` 248.9 s,
`009d5c81` 272.1 s, `007bbfb7` 104.6 s, `017c7c7b` 70.4 s, `025d127b` 96.2 s (4 ranks, 1 process per L4).
Peak training allocations per rank ranged 13.1-18.7 GiB.

This is **not** a dependency failure and **not** fragmentation: only 66.8 MiB was "reserved but
unallocated", so there was almost nothing for the allocator to reclaim — it is a genuine peak of 21.65 GiB
on a 22.03 GiB device. `PYTORCH_ALLOC_CONF=expandable_segments:True`, the remedy the message itself
suggests, therefore cannot fix it. The binding constraint is the upstream memory profile:
`use_gradient_checkpointing=False`, `max_seq_length=8192`, `load_in_4bit=False`, bf16, 128 task-time
training augmentations, and no `flash_attn` in the image. The attention backend is not the lever either:
the OOM raised in `apply_lora_mlp_swiglu` -> `matmul_lora`, i.e. in MLP activations, which FA2 would not
shrink.

**Planning consequence:** a full competition rerun walks the same 240-task path on the same machine, so the
dependency blocker is behind us but a **memory blocker now stands in front of a submission**.

**Correction to the previous extrapolation (2026-09-24).** The row above claiming that 240 tasks "cannot
finish in the upstream budget at that rate" was wrong on its own numbers: 10 tasks in 456 s is 45.6 s/task,
and 240 tasks at that rate is **~3.0 h**, which fits the 12 h budget with room to spare. (The "10 tasks" is
itself not reproducible — the v3 log holds 8 `finished` records — but the arithmetic point stands.) The counter-argument
is that the 10 completed tasks may be the fastest ones, so the rate over a long run is not known from run 3.
Both readings are plausible and **neither is measured**; the measurement is the point of R8's run below.

### R8 decision — one declared deviation: gradient checkpointing ON

**`use_gradient_checkpointing` is `True`, not upstream's `False`** — the only upstream logic change this
notebook makes, applied at both declaration sites: the `FastLanguageModel.from_pretrained` call and the
`peft_params` dict passed to `FastLanguageModel.get_peft_model` (the later call, which would otherwise
re-disable what the first enables). It is a pure memory-for-compute trade: the model sees exactly the same
data, the same sequence length (`max_seq_length=8192`) and the same 128 task-time training augmentations.
That is why it was chosen over the alternatives — `max_seq_length` would truncate the input, and reducing
the augmentation count would change how candidates are scored.

**Consequence, recorded rather than hidden: this notebook is no longer an unmodified reference.** The
deviation is declared in the notebook's own first cell under "The one declared deviation from the
reference", so the artifact carries its own caveat. `gradient_checkpointing=False` in `train_args` is a
*different* `TrainingArguments` flag and is deliberately left at its upstream value.

Cost expectation: gradient checkpointing typically costs 20-30% of throughput, so the post-change
seconds/task is the rate that matters; the pre-change 45.6 s/task figure is not the number to extrapolate.

### R8 measurement — one bounded commit run, kernel v4, terminal `COMPLETE`

Run 4 = kernel **v4**, `ARC_REFERENCE_COMMIT_BUDGET=1800` (commit-mode default raised from 1500), the real
240-task test set, **one run, no retries**. Pushed 2026-09-24 06:43:12Z, terminal `COMPLETE` 07:20:38Z
(~37 min of wall clock, the 1800 s budget spent inside it).

**The OOM did not recur.** Zero `OutOfMemoryError` records in 2 157 log entries. Memory, from the kernel's
own `torch.cuda.max_memory_allocated()` prints (`reset_peak_memory_stats()` before each phase):

| phase | v3 (GC off) | v4 (GC on) |
| --- | --- | --- |
| training peak, per rank | 13.1-18.7 GiB (max 18 686 MB) | **12.8 GiB, flat: 13 123/13 124 MB on all 4 ranks and every task** |
| inference peak, per rank | max 12 049 MB | max 17 201 MB (code path unchanged by this deviation) |

The training peak collapsing to a task-independent constant is the signature checkpointing should produce
(activations freed and recomputed), and it leaves roughly 9 GiB of headroom where run 3 died at 21.65 GiB
in `apply_lora_mlp_swiglu` -> `matmul_lora`.

**Throughput.** **24 tasks completed** (24 unique task ids, all from the first 24 ids of the sorted 240-task
queue) inside a parallel window of **1 688.4 s** — each rank's first completion minus its own printed
elapsed puts all four task loops at ~253 s of log time, and the last completion is at 1 941.7 s. That is
**70.4 s per task of 4-rank wall clock**, or **279.1 s of rank time per task**. Per-task seconds in
completion order, so the tail is visible:

| rank | per-task seconds (completion order) |
| --- | --- |
| 0 | 334.4, 776.5, 136.9, 168.6, 251.5 |
| 1 | 98.6, 272.9, 107.6, 502.7, 318.6, 368.1 |
| 2 | 349.7, 72.3, 1109.2, 157.8 |
| 3 | 122.1, 83.7, 110.0, 151.9, 127.9, 101.7, 567.1, 134.9, 274.6 |

mean **279.1 s**, median **163.2 s**, min 72.3 s, max **1 109.2 s** — the mean is 1.7x the median, so the
distribution is heavy-tailed and the tail is the whole question.

**What checkpointing cost, measured on identical tasks.** Eight tasks ran in both v3 and v4, so the ratio is
not confounded by task choice: `00576224` 84.8 -> 98.6, `007bbfb7` 104.6 -> 122.1, `017c7c7b` 70.4 -> 83.7,
`025d127b` 96.2 -> 110.0, `00d62c1b` 291.8 -> 334.4, `009d5c81` 272.1 -> 349.7, `00dbd492` 248.9 -> 272.9,
`0520fde7` 59.3 -> 72.3 — **mean ratio x1.176 (+17.6%)**, at the low end of the quoted 20-30%. It is a
constant per-task cost, not a degradation over the run.

**Extrapolation to 240 tasks.** 240 x 70.4 s = **16 884 s = 4.69 h** of 4-rank wall clock, against the
upstream `12 h - 10 min` budget — it fits, with ~2.5x headroom. Sensitivity, stated because 24 tasks is a
small sample: at the second half's mean per-task time (397.2 s of rank time) it is **6.6 h** (still fits);
at the single worst observed task (1 109.2 s) it is 18.5 h (would not fit). Conclusion: **fits, with the tail
as the only threat** — not "fits comfortably".

**Rate stability.** Completions per 400 s window over the 1 688 s run: **7, 5, 2, 6**, then 4 in the final
88 s. Bursty, not degrading: the slow window (1 053-1 453 s) is exactly where the 1 109 s / 776 s / 567 s
tasks overlap, and the run's last segment is its fastest. The second half's mean per-task time (397.2 s) is
2.5x the first half's (161.1 s), which tracks task difficulty rather than any pipeline slowdown. With 24 of
240 tasks measured, the remaining 216 are an unmeasured population.

**The previous claim was wrong, and so was its counter-argument.** "240 tasks cannot finish in the upstream
budget" was never measured, and the 45.6 s/task it rested on is not a rate: it divides *starter.py's whole
wall time* (which includes ~130 s of model load and the rank stagger) by tasks. The comparable measured
aggregate for v3's own window is 391.5 s / 8 recorded completions = 48.9 s/task — and those 8 are the
fastest tasks in the set. Neither number is the post-change rate; 70.4 s/task (wall) and 279.1 s/task (rank
time) are.

**The submission-contract check was reached, and passed** — v3 never got there, so this requirement was
"unverified", and it is now verified:

```text
test task ids                : 240
test outputs (task x input)  : 259
tasks with decoded results   : 21/240
decoded candidate outputs    : 355
outputs with a non-placeholder attempt : 21
problems                     : 0
VERDICT                      : PASS
```

`starter.py exit code 0 after 1788s` (v3: exit 1 after 456 s), so the fail-loud runner let the notebook
through to the submission cell. The 3-task gap between 24 completions and 21 decoded tasks is work still in
flight when the budget expired.

**Not directly logged, stated as an inference:** no unsloth or HF line says "gradient checkpointing
enabled". That GC was active rests on three independent observations — the flat training peak, the +17.6%
per-task cost on identical tasks, and the absence of the 21.65 GiB allocation that killed run 3 in the same
code path.

**Quota.** `kaggle quota` moved **1.78 h -> 2.87 h used** (remaining 28.22 h -> 27.13 h) for 37 min of wall
clock: this shape bills ~1.7-2x wall clock (run 3: ~11 min wall -> +0.39 h). The "~0.5 h" authorization was
a wall-clock estimate; the actual cost was ~1.1 h, visible only after the run.

## 5. Non-goals

- Does not claim novelty for a fork, and does not hide that it is one.
- Does not attempt 85% (the Bonus Prize is out of reach; only 4 teams exceed 40).
- Does not touch ARC-AGI-2's own score prediction beyond this reference.
- Does not spend a submission until the commit-mode run is verified.

## 6. Allowed edit surfaces

- `odd/tasks/arc2-reference.md`
- `notebooks/arc2_submission_kernel/**`
- `scripts/kaggle/**`
- `experiments/arc2_ablation.py`
- `experiments/arc2_ablation_results.json`

## 7. Tasks

| # | Task | Status | Evidence |
|---|---|---|---|
| R1 | Scaffold this feature and update the governing objective | in progress | `odd/OBJECTIVE.md` revised; this file |
| R2 | Land the ablation as a repo tool (`experiments/arc2_ablation.py`) | done | reproduces COMPLETE 45/43/42, `panel` −11, `scale` −5, `collinear` −4, and six families at exactly 0; also reports zero raising candidates |
| R3 | Adapt the reference pipeline to run **offline**: `model_sources` for the base model, a declared wheel source for the packages the image omits, `enable_internet: False` | **done** | Adaptation is complete: `model_sources` resolves offline, the wheelhouse installs, and the pipeline starts. See R7 and §4 |
| R4 | Push and verify in commit mode (free — does not consume a submission) | **done (bounded)** | **v4 → terminal `COMPLETE`.** Install 34 s, gate passed, 240 tasks queued, **24 tasks completed on 4 ranks, no OOM**, `starter.py exit code 0 after 1788s`, and the submission-contract check ran and reported `problems: 0 / VERDICT: PASS` with 21/240 tasks decoded. The bounded budget (1 800 s) is what stopped it, not a failure — the full 240-task rerun is still unmeasured. No submission spent |
| R7 | Supply the missing dependencies offline as a declared `dataset_sources` wheel source, then re-run the gate | **done** | `ser8147/arc2-unsloth-wheelhouse` — public, 15 wheels, 191.6 MiB, SHA-256 manifest, per-package licence table, dependency-closure check; built by `scripts/kaggle/unsloth_wheelhouse.sh`. Before/after table in §4 |
| R8 | Resolve the per-rank CUDA OOM before any submission | **done (bounded)** | Owner decision 2026-09-24: `use_gradient_checkpointing` `False` -> `True`, one parameter, at both declaration sites; nothing else. Declared as the notebook's one deviation from the reference (§4, and the notebook's own first cell). Verified in v4: **no OOM**, training peak flat at 12.8 GiB per rank (was 13.1-18.7 GiB), +17.6% per-task cost on 8 identical tasks, 24 tasks completed, contract check PASS. Full details and per-task times in §4 |
| R5 | Spend **one** submission and record the score, with the 50%-of-test-data caveat | pending | leaderboard |
| R6 | Independent verification | pending | verifier report |

## 8. Acceptance criteria

1. The ablation is reproducible from the repository in one command, and its numbers match the ones already
   recorded (45/1076 full; `panel` −11, `scale` −5, `collinear` −4; D4/color/gravity/crop/counting/twostage
   exactly 0).
2. The reference kernel runs with `enable_internet: False` and a declared `model_sources` entry.
3. No submission is spent before a successful commit-mode run.
4. The recorded score is quoted **with** both variance caveats: the leaderboard uses **50% of the test data**,
   and a nondeterministic notebook's posted score is a rerun maximum (per forum 742027).
5. The paper writes it up as a **declared reference implementation**, in the same table as our ablation.

## 9. Known risks carried

| Risk | Impact | Owner |
|---|---|---|
| ~~The fork's dependencies (`unsloth`, `transformers` versions) may not exist in the offline image~~ **FIRED 2026-09-24, RESOLVED 2026-09-24** | The rerun could not start. Resolved by `ser8147/arc2-unsloth-wheelhouse`; cost of the whole session 0.39 h of quota, no submission spent | this feature — R7 |
| ~~The package supplying `unsloth` also needs a `transformers < 5` downgrade to match the API surface the pipeline uses~~ **FIRED and handled** | Confirmed: the image ships exactly `transformers` 5.0.0, which unsloth excludes. `transformers` 4.57.6, `huggingface_hub` 0.36.2 and `dill` 0.4.0 are all downgrades, not choices | R7 |
| ~~**NEW: a full competition rerun runs the same 240-task path and now hits CUDA OOM at 21.65/22.03 GiB per rank**~~ **FIRED 2026-09-24, RESOLVED 2026-09-24 by R8** | A submission spent on the old configuration would have burned a rerun or produced a partial artifact. With `use_gradient_checkpointing=True` the training peak is flat at 12.8 GiB per rank and 24 tasks ran without OOM | this feature — R8 |
| Per-task LoRA TTT costs real GPU hours | Quota pressure (28.22 h -> 27.13 h remaining after the R8 run, which billed ~1.1 h for 37 min of wall clock) | this feature |
| ~~12 h runtime cap: "a full 240-task rerun cannot finish in the upstream budget at that rate"~~ **CORRECTED AND MEASURED 2026-09-24** | The v3 claim was never measured and its 45.6 s/task divisor was invalid. Measured post-change rate: **70.4 s/task of 4-rank wall clock → 240 tasks = 4.69 h**, fitting the upstream `12 h - 10 min` budget with ~2.5x headroom. Sensitivity: 6.6 h at the second half's mean, 18.5 h at the worst single task measured — so the heavy tail, not the mean, is the real exposure, and 216 of 240 tasks remain unmeasured | measured in run 4 (§4) |
| **Inference peak is now the binding memory constraint, and R8 did not touch it** | Per-rank inference peak reached 17 201 MB of 22.03 GiB in v4 (v3 max 12 049 MB) — only ~5 GiB of headroom, on a phase GC does not cover. Larger test grids in the unmeasured 216 tasks could still OOM the inference path | this feature — R5 |
| **The bounded run is not the competition run** | 24 of 240 tasks measured; `global_end_time` bounds a commit run to 1800 s by our own edit, while a rerun uses the upstream `12 h - 10 min`. The 240-task path end to end is still unexercised on this machine | this feature — R5 |
| The score lands anywhere in the field's 28–34 band | No distinction, and it is nondeterministic | stated in the paper |
| A fork declared honestly can still be read as padding | Reviewer perception | the paper keeps it to one comparison row |

## 10. Evidence log

- 2026-09-24 — **R8 applied and measured: the OOM is gone and the run completes.** One declared
  adaptation: `use_gradient_checkpointing` `False` -> `True` at both declaration sites (the
  `FastLanguageModel.from_pretrained` call and the `peft_params` dict passed to
  `FastLanguageModel.get_peft_model`), nothing else — a pure memory-for-compute trade that changes no input.
  It **ends this notebook's claim to be an unmodified reference**, declared in the notebook's own first cell
  ("The one declared deviation from the reference") and in §4 here. Commit-mode run **v4** with
  `ARC_REFERENCE_COMMIT_BUDGET=1800` (the commit-mode default raised from 1500 in that same cell) on the real
  240-task set: terminal **`COMPLETE`**, pushed 06:43:12Z -> complete 07:20:38Z (~37 min wall), `pip exit
  code 0 in 34 s`, `OFFLINE INSTALL OK in 96s`, 240 tasks queued, **24 tasks completed**, **zero
  `OutOfMemoryError` records** in 2 157 log entries, and `starter.py exit code 0 after 1788s` (v3: exit 1
  after 456 s). The **submission-contract check ran and passed**: `problems: 0`, `VERDICT: PASS`, 240 test
  ids, 259 outputs, 21/240 tasks decoded, 355 candidate outputs — the requirement v3 never reached.
  `torch.cuda.max_memory_allocated()` peaks: training **13 123/13 124 MB on every rank and every task**
  (v3: 13.1-18.7 GiB, and a fatal 21.65 GiB allocation), inference up to **17 201 MB** (v3: 12 049 MB) — GC
  does not cover inference, so that is now the binding memory constraint. Rate: 24 tasks in a 1 688.4 s
  parallel window = **70.4 s/task of 4-rank wall clock, 279.1 s of rank time per task** (mean 279.1 s,
  median 163.2 s, max 1 109.2 s), extrapolating 240 tasks to **4.69 h** — inside the upstream
  `12 h - 10 min` budget, with the heavy tail as the only threat (6.6 h at the second half's mean, 18.5 h at
  the worst single task). Checkpointing's cost, from 8 tasks measured in both v3 and v4: **x1.176 (+17.6%)**,
  constant per task. Completions per 400 s window 7/5/2/6 then 4 in the final 88 s: bursty, not degrading.
  Quota `1.78h -> 2.87h` (+1.09 h) for 0.62 h of wall clock — this shape bills ~1.7-2x wall clock, so the run
  cost more than the ~0.5 h it was authorized at, which is only visible after the fact. No submission spent;
  `kaggle competitions submit` was never invoked.
- 2026-09-24 — **The dependency blocker is RESOLVED, and a new memory blocker is exposed.** Built and
  published `ser8147/arc2-unsloth-wheelhouse` (public, 15 wheels, 191.6 MiB, `WHEELS.sha256`, generated
  licence table, dependency-closure check) with `scripts/kaggle/unsloth_wheelhouse.sh`; wired it into
  `kernel-metadata.json` as the only `dataset_sources` entry and into a new install cell that runs
  `pip install --no-index --find-links` and then **fails loudly** on a missing wheelhouse, a wheelhouse that
  disagrees with its own `PINNED.txt`, a non-zero `pip` exit, a pin that did not take effect, or a failed
  import. Commit-mode runs: **v2** `ERROR` after ~2 min (wheelhouse installed; over-strict pin assert on
  `torchvision 0.25.0+cu128`, fixed); **v3** `ERROR` after ~11 min with `pip exit code 0 in 34 s`, 15/15 pins
  in effect, `OFFLINE INSTALL OK in 98s`, all imports true, **10 solver tasks completed** across 4 L4 ranks
  (`00d62c1b` 291.8 s, `0520fde7` 59.3 s, `00576224` 84.8 s, `00dbd492` 248.9 s, `009d5c81` 272.1 s,
  `007bbfb7` 104.6 s, `017c7c7b` 70.4 s, `025d127b` 96.2 s), then
  `torch.OutOfMemoryError ... 21.65 GiB allocated by PyTorch, 66.83 MiB reserved but unallocated` on a
  22.03 GiB L4 and `starter.py exit code 1 after 456s`. `pip check` afterwards reports only pre-existing
  image conflicts, none caused by the wheelhouse. No submission spent. Quota `1.39h -> 1.78h` (0.39 h).
  Blocked on R8, an owner decision.
- 2026-09-24 — `transformers` version decided: **4.57.6**, the newest 4.x inside unsloth 2026.9.11's
  `>=4.51.3,<=5.5.0` minus its exclusions AND inside `unsloth_zoo` 2026.9.7's range AND satisfying
  `trl==0.24.0`'s `transformers>=4.56.1`. Also forced by the same graph: `huggingface_hub` 1.11.0 -> 0.36.2
  and `dill` 0.4.1 -> 0.4.0. `torch` is deliberately not vendored.
- 2026-09-24 — the first cut of the wheelhouse had 30 wheels. It was wrong: it pinned `fsspec`,
  `importlib-metadata`, `zipp`, `tokenizers`, `safetensors`, `diffusers`, `typer`, `shellingham`,
  `annotated-doc`, `multiprocess`, `sentencepiece`, `nest-asyncio`, `torchvision`, `docstring-parser` and
  `typeguard`, all of which the image ALREADY satisfies. That cut downgraded `fsspec` from the image's
  2025.3.0 and tripped `gcsfs` (which wants exactly 2025.3.0), and upgraded `importlib-metadata` past the
  `<8.8.0` `opentelemetry-api` requires. The kernel's own preflight `pip list --format=freeze` is what made
  the unnecessary pins visible; the shipped set is 15 wheels and `pip check` is clean of new conflicts.
- 2026-09-24 — **Gate FAILED, measured in the target environment.** `notebooks/arc2_reference_kernel/`
  built (12 cells, nbformat 4.4) and pushed as `ser8147/arc2-reference-nvarc` (`is_private: true`,
  `enable_internet: false`, `machine_shape: NvidiaL4`, `model_sources:
  ["sorokin/qwen3_4b_grids15_sft139/transformers/bfloat16/1"]`). Terminal status `ERROR` after ~31 s,
  on the new preflight cell's own hard gate — the honest outcome, because the alternative was a silent
  placeholder submission. Preflight evidence: 4x `NVIDIA L4` sm_89 22.0 GiB `bf16_supported=True`;
  competition data and the 6.77 GiB base model both mounted correctly; `unsloth`/`trl`/`bitsandbytes`
  **not importable**. Fork fidelity verified mechanically: `arc_loader.py`, `arc_decoder.py` and the
  `tensorflow` cell are byte-identical to `manderson240`'s fork, and `arc_solver.py` / `starter.py` differ
  only by the challenge-set selection switch (`ARC_REFERENCE_TEST_SET`) that was added to let the commit
  artifact exercise the real competition path. No submission spent. Quota `1.37h -> 1.39h`.
- 2026-09-24 — the *canonical* notebook now pulls: `kaggle kernels pull
  mikelou1/arc-agi2-lb33-89-minimal-perfpatch` succeeds (9 cells, 1003 lines). It has no `pip install`
  either, the same `local_files_only=True` model load, the same `kgmon` selection and the same
  `12 * 3600` budget. The 403 recorded the previous day no longer reproduces.
- 2026-09-24 — model source corrected: the pipeline needs `sorokin/qwen3_4b_grids15_sft139`
  (`kaggle models get` → instance slug `bfloat16`, framework TRANSFORMERS, version 1, 7 267 271 302 bytes,
  Apache 2.0, 42 votes), not `qwen-lm/qwen-3`. All four forks reference it by that path.

- 2026-09-24 — **R2 done.** `experiments/arc2_ablation.py` reproduces the measurement in one command and
  writes `experiments/arc2_ablation_results.json`. Verified output: COMPLETE 45 outputs / 43 coverage / 42
  solved; **`panel` −11, `scale` −5, `collinear` −4**, `holes`/`kronecker`/`overlay` −2, `object`/`tiling`
  −1; and **D4 (8 symmetry ops), `color`, `counting`, `crop`, `gravity`, `twostage` exactly 0**.
  - A behaviour-preserving seam was added to the solver to make this measurable: `DISABLED_FAMILIES` (empty
    by default, so shipped behaviour is unchanged — the baseline still reads 45/1076) plus `family_of()`.
  - **The two silent `except Exception: continue` blocks in the candidate loops now RECORD the failure**
    instead of swallowing it, and the trace carries `failures`. That matters for this tool specifically: a
    primitive that raises on every task is dead, and a dead primitive is indistinguishable from a useless
    one in every metric — which is exactly how `rot270` stayed broken and invisible. The tool prints a
    **CANDIDATES THAT RAISED** section, and on this run it reports **none**, independently reconfirming the
    verifier's finding that no other dead primitive exists.
- 2026-09-24 — created. Governed by the revised `odd/OBJECTIVE.md`, whose §4 now ranks Theory first and marks
  Novelty **narrowed**, §6 splits the evidence into *ours* (unoccupied) and *already published* (cite only),
  §7 gains the ablation-versus-taxonomy contradiction, and §10 states the pivot.
- 2026-09-24 — ablation measured (before this feature existed), 1000 train tasks / 1076 outputs, leave-one-out:
  COMPLETE 45 outputs / 43 coverage / 42 solved; `panel` 34 (−11), `scale` 40 (−5), `collinear` 41 (−4),
  `overlay`/`holes`/`kronecker` 43 (−2), `object`/`tiling` 44 (−1), and **D4 (8 ops), `color`, `gravity`,
  `crop`, `counting`, `twostage` all exactly 45 (+0)**.
- 2026-09-24 — route verified: four forks pull successfully, the canonical notebook 403s, and Qwen3-4B is
  published on Kaggle Models.

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

The three runs moved the kernel from "cannot start" to "runs, then runs out of GPU memory":

| run | kernel version | terminal | what happened |
| --- | --- | --- | --- |
| 1 | v1 | ERROR after 31 s | pre-existing run; stopped on the preflight's own `unsloth` gate |
| 2 | v2 | ERROR after ~2 min | wheelhouse installed successfully (30-wheel cut), stopped on an over-strict pin assert: `torchvision pinned to 0.25.0 but installed 0.25.0+cu128` — the image's own CUDA build, i.e. a false alarm, since fixed |
| 3 | v3 | ERROR after ~11 min | install 34 s, gate passed, solver started, 240 tasks queued, **10 tasks solved**, then CUDA OOM |

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
| R4 | Push and verify in commit mode (free — does not consume a submission) | partial | v3 → terminal `ERROR` after ~11 min. The install is proven (`pip exit code 0` in 34 s, 15/15 pins in effect, every import true) and the solver ran **10 tasks** on 4 ranks before a CUDA OOM at 21.65/22.03 GiB. No submission spent |
| R7 | Supply the missing dependencies offline as a declared `dataset_sources` wheel source, then re-run the gate | **done** | `ser8147/arc2-unsloth-wheelhouse` — public, 15 wheels, 191.6 MiB, SHA-256 manifest, per-package licence table, dependency-closure check; built by `scripts/kaggle/unsloth_wheelhouse.sh`. Before/after table in §4 |
| R8 | *(new)* Resolve the per-rank CUDA OOM before any submission | proposed | Needs an owner decision. The levers are upstream memory hyperparameters (`use_gradient_checkpointing`, `max_seq_length`, augmentation count), and changing them ends this notebook's claim to be an unmodified reference. `expandable_segments` cannot help (only 66.8 MiB was reserved-but-unallocated). See §4 |
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
| **NEW: a full competition rerun runs the same 240-task path and now hits CUDA OOM at 21.65/22.03 GiB per rank** | A submission spent on the current configuration is likely to burn a rerun or produce a partial artifact. Needs an owner decision before R5 | this feature — R8 |
| Per-task LoRA TTT costs real GPU hours | Quota pressure (28.22 h/week after this session) | this feature |
| 12 h runtime cap | 10 tasks took ~7.6 min of a 25 min bounded budget; a full 240-task rerun cannot finish in the upstream budget at that rate | measured in run 3 |
| The score lands anywhere in the field's 28–34 band | No distinction, and it is nondeterministic | stated in the paper |
| A fork declared honestly can still be read as padding | Reviewer perception | the paper keeps it to one comparison row |

## 10. Evidence log

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

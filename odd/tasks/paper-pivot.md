# ODD Feature — `paper-pivot`

> **Status:** in progress
> **Branch:** to be created (`feat/paper-pivot`) off `main` when the measurements it needs are final
> **Created:** 2026-09-24
> **Serves:** `odd/OBJECTIVE.md` §4 criterion **Completeness** (primary), then Theory, Progress, Novelty
> **Engram mirror topic key:** `arc-paper-track/odd/paper-pivot/tasks`

---

## 1. Goal

Make the writeup **true, complete and theory-forward**, and publish every public artifact **consistently in
one coordinated step** so that no two of them disagree while the rewrite is in flight.

## 2. The owner's decision, recorded

> **2026-09-24 — publish together, not piecemeal.** The model card's stale numbers (v1's `Brier 0.0818`
> while the model serves v2) were found and a correction was prepared. The owner chose to **withhold it**
> until the writeup rewrite is ready, because publishing a card that says "not wired into any submission"
> while the live writeup still claims the System 1 gates the submission would make the two artifacts
> contradict each other — a second, self-inflicted Completeness failure.
>
> **Therefore: no public-facing edit to any artifact until the writeup rewrite is drafted.** Then the
> writeup, the model card, `paper/draft.md` and the README all change in one pass.

## 3. What the paper becomes

Not "our architecture works", but:

> *An offline human-play ruler for ARC-AGI-3, an audit of the benchmark's own baseline, and a measured map of
> which primitive families actually pay — plus the negative results of our own architecture, which the
> instrument made visible.*

### 3a. The three things that are ours and unoccupied (`OBJECTIVE.md` §6a)

1. **The human-play calibration instrument.** 24 of 25 replays reproduce step for step through the OFFLINE
   engine; the official scorer assigns human play **89.6774/100** on the same instrument that scores our
   agent **0.2017**. Shipped as a regression check. No public notebook ships one.
2. **The audit of the benchmark's own baseline.** `baseline_actions` is **not** the human play published as
   ground truth: 33 of 183 levels match exactly, the replay human is at-or-faster on 142 of 183, and the
   baseline totals 17,135 actions against the humans' 14,798. A slower baseline means a larger ratio, so
   **scores are inflated relative to observable human play**, by a level-dependent amount (`su15` level 7
   differs by **6.25x**).
3. **The primitive ablation.** `panel` −11, `scale` −5, `collinear` −4 carry the solver; **eight D4 symmetry
   operators, colour mapping, gravity, cropping, counting and the two-stage composition contribute exactly
   zero**. Answers a public open question (`forum 742790` asks *"what primitive would you try first?"*, and
   its two families are the two that pay nothing) and explains 45/1076 against that notebook's 9/1076.

   **The ablation must be reported as a train/eval pair, never as a single number.** Measured
   (`experiments/arc2_baseline.json`):

   | set | tasks | outputs | official metric | coverage |
   | --- | --- | --- | --- | --- |
   | **train** | 1,000 | 1,076 | **4.1822%** | 43/1,000 (4.3%) |
   | **eval** (held out) | 120 | 172 | **0.00%** | 0/120 (0.0%) |
   | **deployed** (leaderboard) | 240 | — | **0.00** | — |

   The held-out column is 0.00%, so the 4.18% is **in-distribution fitting on the 1,000 tasks the
   primitive library was designed against**, and the paper must say so in the same breath. The matching
   candidates are dominated by `colour_map` composites (`largest_object+colour_map` 7, `colour_map` 5,
   `identity+colour_map` 5, …) — the most memorizable family in the set. `eval` `miss_reasons`:
   **164 of 172 outputs echoed the input**, 5 wrong shape, 3 right shape wrong cells.

   **And the artifact-level measurement of what we actually submitted**, from the kernel output
   (`kaggle kernels output ser8147/arc-laya-dual-process-submission`, 259 outputs = 240 tasks x test
   inputs): **209 echoes (80.7%), 46 zero grids (17.8%), 4 with content (1.5%)** — so **98.5% of what we
   submitted carried no information**, while the notebook printed
   `✅ Generated /kaggle/working/submission.json successfully (361.2 KB) with 240 tasks!` in **0.7026 s**.
   The same artifact analysis on the **reference** implementation's commit output gives **238 of 259
   outputs = `[[0]]` (91.9%)** and 21 with content — and **both artifacts passed `problems: 0 / VERDICT
   PASS`**. That contrast is the thesis, measured rather than argued: the reference's filler is a *budget*
   artifact (`ARC_REFERENCE_COMMIT_BUDGET=1800` against the competition's 12 h) while ours is a *capability*
   artifact (it ran to completion on all 240 tasks in under a second and still produced 98.5% filler), and
   **the submission format cannot tell the two apart**. It is why the paper reports coverage alongside score.

Plus the two measured negatives: a learned gate has **zero headroom** (0 of 43 covered tasks), and the
ARC-AGI-3 agent's score is RNG-dominated with an inert heuristic layer (92.8% one-candidate pools).

### 3b. What must be CITED, never claimed (`OBJECTIVE.md` §6b)

The primitive DSL and its coverage verdict (`arc-baseline-rebuild`), the `hash(str)` non-determinism and its
`zlib.crc32` remedy (`forum 742027`), the 8-symmetries-plus-colour-map solver (`forum 742790`), and the
745-of-2,172 concentration in the 30–40 band. **Claiming any of these would be a checkable error.**

## 4. What must be corrected, in one pass

| Artifact | Currently says | Must say |
|---|---|---|
| live writeup | `+16.7% relative improvement` (0.24 → 0.28) | one draw from a 6.3x-wide distribution |
| live writeup | `0.28 (28% Solved)` | the metric is 0–100 where 100 = human, so **0.28%** |
| live writeup | `synthesizing rules for 17.9% of benchmark tasks` | **2.1%** coverage deployed, 6.2% local |
| live writeup | code submission reference → the ARC-AGI-2 kernel | → the **ARC-AGI-3** kernel, where we hold the **median** of 3,284 teams |
| `notebooks/arc2_reference_kernel/submission.ipynb` (kernel header, **public**) | *"three points on one axis: a minimal public solver (0.84%), our primitive ablation (4.18%), and this reference (the field's ~30 band)"* | the first two are **train** pass@2 on 1,076 public training outputs; the third is a **leaderboard**. Putting an in-distribution number and a held-out number on one axis is the exact error this project forbids — label all three, and state that 4.18% is train against 0.00% held out |
| any artifact describing the deployed ARC-AGI-2 solver | a working symbolic program synthesiser | the deployed v2 carried a **dead `rot270`**: `[list(x) for x in zip(*g)[::-1]]` raises `TypeError: 'zip' object is not subscriptable`, so every rot270 candidate was silently dropped (deployed notebook md5 `a9acb4eeb2`, 666 lines, against `src/`'s 746) |
| any artifact quoting the solver's capability as one number | a single percentage | the honest triple is **train 4.1822% (45/1076) / eval 0.00% (0/172) / deployed 0.00%**, and of the 259 outputs actually submitted **80.7% echoed the input, 17.8% were zero grids, 1.5% carried content** |
| model card (model level) | `Brier 0.0818`, no ECE, "System 1 decision gating" | v2's `0.1020` **and `ECE 0.2392`**, plus the measured status |
| model card (instance level) | v1's text with `versionNumber: 2` | just needs a **re-push**; the repo file is already correct |
| `paper/draft.md` §4.3 | the dual-process trade-off ablation | **delete**; and the System 1 taxonomy (geometry, flood_fill, counting, extrapolation) is contradicted by the ablation |

Also add the two variance caveats wherever a score is quoted: the leaderboard uses **50% of the test data**,
and a nondeterministic notebook's posted score is a rerun maximum.

## 5. Non-goals

- Does not "defend" the architecture in prose. It declares the deviation and reports the measurement.
- Does not claim any of §3b.
- Does not edit any public artifact before §2's coordinated pass.
- Does not exceed the **1,500-word** writeup limit (live is 849 words, the local draft 1,134 — headroom is thin).

## 6. Allowed edit surfaces

- `odd/OBJECTIVE.md`
- `odd/tasks/paper-pivot.md`
- `submissions/KAGGLE_WRITEUP.md`
- `paper/draft.md`
- `models/kaggle_model_hub/model-metadata.json`
- `models/laya_arc_finetuned_v2/laya_finetuned_typed_decisions/model-instance-metadata.json`
- `notebooks/kaggle_laya_arc_train.ipynb`
- `notebooks/arc2_submission_kernel/kernel-metadata.json`
- `notebooks/arc2_reference_kernel/submission.ipynb`
- `README.md`

## 7. Tasks

| # | Task | Status | Evidence |
|---|---|---|---|
| P1 | Write the pivoted paper and writeup (≤1,500 words) from §3a, citing §3b | pending | draft in `submissions/KAGGLE_WRITEUP.md` |
| P2 | Correct the model-level card: v2's numbers + ECE + the measured status | pending | prepared, withheld per §2 |
| P3 | Re-push the instance metadata (fixes the live v1 text with no edit) | pending | card shows `/2` and v2's numbers |
| P4 | Correct the three false numbers, repoint the code reference, delete `paper/draft.md` §4.3 | pending | `kwriteup.sh diff` clean |
| P5 | Artifact honesty that falls outside `system1-wiring`'s surfaces. **Split, because the original wording was dangerous — see §11** | in progress | §11 |
| P6 | **The coordinated publish**: update the live writeup, the card, the draft and the README in one pass | pending | all artifacts agree |
| P7 | Independent verification | pending | verifier report |
| P8 | Correct the **reference kernel header** (§4): label the provenance of its three points, and carry the artifact-content finding. This surface was outside every feature when this audit was made | pending | `grep -rn "4.18" notebooks/` returns one hit |
| P9 | Put the deployed v2's **dead `rot270`** into the deployment description, and decide whether `notebooks/arc2_submission_kernel/` gets the same builder treatment as the ARC-AGI-3 notebook | pending | the notebook is 666 lines against `src/`'s 746 |

## 8. Acceptance criteria

1. **No two public artifacts contradict each other** after §6's pass — checked by reading the writeup, the
   card and `paper/draft.md` against each other.
2. Every number in the writeup is reproducible from a committed file in the repository.
3. Every finding in §3b is attributed, not claimed.
4. The writeup is ≤1,500 words and its code reference points at the track where we hold the median.
5. The card publishes ECE, and states the model's measured status honestly.
6. `paper/draft.md` §4.3 is gone and the System 1 is presented as a hypothesis that was tested and rejected.

## 9. Known risks carried

| Risk | Impact | Owner |
|---|---|---|
| A public writeup correction is itself public and time-stamped | the "entered first" tie-break question is moot (§3.7.b: no hackathon tiebreakers) but the edit is visible | stated |
| Publishing "not wired into any submission" may read as an admission | could be scored down by a judge skimming for a working system | the honest framing is also the admissibility condition (`OBJECTIVE.md` §2) |
| The ARC-AGI-2 reference may still be blocked by R8 | Accuracy on that track stays 0.00 | `arc2-reference` |
| 1,500-word limit with three contributions plus citations | the writeup may have to drop detail | this feature |
| Rewriting the whole narrative late | little time to iterate before 2026-11-08 | this feature |

## 10. Evidence log

- 2026-09-24 — created. Records the owner's "publish together" decision (§2) and the fact that the
  model-card correction was **prepared and deliberately withheld**.
- 2026-09-24 — the model card was found to be CLI-editable (`kaggle models update` and
  `kaggle models instances update`), which makes P2/P3 possible at all.
- 2026-09-24 — **claim audit for the unlabelled train number.** `grep` for `4.18`/`1076`/`0.84` across the
  repository found the claim **already correctly labelled everywhere except one public artifact**:
  `README.md`, `EXECUTIVE-SUMMARY.md`, `PROJECT_DOCUMENTATION.md` and `paper/draft.md` contain no mention of
  the ablation at all; `experiments/arc2_baseline.json` is keyed `train`/`eval`; and `odd/OBJECTIVE.md`'s
  table column is already headed *"public training outputs (pass@2)"*. The single exception is the
  **reference kernel's header cell**, which places `0.84%` and `4.18%` (both train) on one axis with
  `~30` (a leaderboard) — and `notebooks/arc2_reference_kernel/submission.ipynb` was in **no feature's edit
  surfaces**, so it would have survived the coordinated pass. Recorded as P8 and added to §6.
- 2026-09-24 — **artifact-level evidence added to §3a.** `kaggle kernels output` reaches the real submission
  artifacts. Ours: 80.7% echo / 17.8% zero grids / 1.5% content out of 259 outputs. The reference's commit
  output: 238 of 259 = `[[0]]`. Both passed `problems: 0 / VERDICT PASS`. Two new correction rows added to
  §4 and P9 recorded for the dead `rot270` in the deployed v2.

## 11. P5, split — and the wording that nearly broke a kernel

P5's original text read *"pin `revision=` in the training kernel, drop the spurious `arc-laya-finetune-data`
source"*. **The second clause is true only of the ARC-AGI-2 submission kernel.** Verified 2026-09-24:

```text
kaggle datasets files ser8147/arc-laya-finetune-data
  train.jsonl        1,405,656
  train_v2.jsonl     3,470,067      <- the training notebook globs for exactly these names
```

So the **training kernel's** copy of that source is **essential** and must stay; a literal reading of P5
would have removed the fine-tune's data access. The original wiring audit was correct — it said the *ARC-2
kernel* declares the source and never reads it — but the summary here dropped the qualifier.

### P5a — DONE: the base-model revision is pinned

`notebooks/kaggle_laya_arc_train.ipynb` called `snapshot_download("convaiinnovations/laya")` with **no
revision**, so a fine-tune is not reproducible: it takes whatever `main` points at on the day it runs.

**Upstream moved the same day this was found** — `main` was at `55cf4c4ebb4e` as of 2026-09-24, and the
commit history is all *"README: ... laya 0.3.x notes"*. Whether the weights moved with it was **measured,
not assumed**: `model.safetensors` has LFS oid `891102d372688fc2…` and size 842,609,210 bytes at `55cf4c4`,
at the previous commit `d51a650`, and at the older `c225928` — **byte-identical across all three**. The
drift was documentation-only, so the shipped model's base is unaffected. The pin (`MODEL_REVISION =
"55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851"`) makes that a guarantee rather than an observation from
commit titles. Applied and committed; takes effect on the next kernel push.

### P5b — DONE: the spurious source was on the submission kernel

`notebooks/arc2_submission_kernel/kernel-metadata.json` declared `ser8147/arc-laya-finetune-data` while its
notebook — a model-free symbolic solver whose only input is the competition's own test challenges — never
reads it. Emptied in commit `3b9179e`.

### P5c — the legacy lineage: document, do not delete

Three early kernels plus a dataset sit disconnected from the model:

| artifact | last run | note |
| --- | --- | --- |
| `ser8147/laya-gate-finetune` | 2026-09-21 | superseded |
| `ser8147/fork-of-laya-gate-finetune` | 2026-09-21 | a fork of the above |
| `ser8147/arc-laya-fine-tune` | 2026-09-22 | superseded |
| `ser8147/arc-laya-finetune` | 2026-09-23 | **the current training kernel** — note the hyphen |

Nothing consumes any of them. `arc-laya-finetune` and `arc-laya-fine-tune` differ by one hyphen, which is
exactly the kind of pair that invites editing the wrong one. **They stay published and are documented as
retired**, with the spelling hazard named, because deleting a published kernel is a worse failure mode than
describing it.

### P2 — PREPARED, withheld per §2: the model-level card

`models/kaggle_model_hub/model-metadata.json` currently states `Brier Score Calibration: 0.0818` — **v1's
number while the model serves v2** — omits ECE entirely, and describes the intended use as *"System 1
decision gating"*, the claim that would contradict the rewritten writeup. The replacement text, ready to
apply in one pass:

```markdown
# ARC Laya Decision Engine

A specialized 421M non-autoregressive decision model fine-tuned on ARC-AGI task traces and
sequential game environments.

### Model Characteristics
* **Parameters:** 421M
* **Architecture:** Non-autoregressive encoder with multi-head calibrated decision heads
* **Inference Latency:** < 50ms per state on CPU / GPU
* **Token Overhead:** 0 tokens (logit/classification output)
* **Accuracy (v2, 1,720 ARC cases):** 89.92%
* **Brier Score Calibration:** 0.1020
* **Expected Calibration Error:** 0.2392

### Intended Use
A declared **offline** component. It is **not wired into any submission**: `model_sources` is empty
on all six competition kernels, and neither submission kernel imports it. It ships as a measured
artifact and as a hypothesis this project tested and rejected -- a gate built on it has zero headroom
(0 of 43 covered tasks; `odd/tasks/system1-wiring.md` §11).

### Known limitation
The probabilities are **not** well calibrated: Brier 0.1020 with ECE 0.2392. Do not use this model
as a confidence source.
```

### P3 — already satisfied, verify only

The live instance list shows **version 2** with `"Version 2: Fine-tuned on 1720 ARC cases, 89.92%
accuracy, Brier 0.102…"`, which matches the repo's corrected instance file at
`models/laya_arc_finetuned_v2/laya_finetuned_typed_decisions/model-instance-metadata.json`. No re-push is
needed; `§4`'s "v1's text with `versionNumber: 2`" no longer describes the live artifact. Confirm at apply
time rather than trusting this note.

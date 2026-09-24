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
- `README.md`

## 7. Tasks

| # | Task | Status | Evidence |
|---|---|---|---|
| P1 | Write the pivoted paper and writeup (≤1,500 words) from §3a, citing §3b | pending | draft in `submissions/KAGGLE_WRITEUP.md` |
| P2 | Correct the model-level card: v2's numbers + ECE + the measured status | pending | prepared, withheld per §2 |
| P3 | Re-push the instance metadata (fixes the live v1 text with no edit) | pending | card shows `/2` and v2's numbers |
| P4 | Correct the three false numbers, repoint the code reference, delete `paper/draft.md` §4.3 | pending | `kwriteup.sh diff` clean |
| P5 | The T5 artifact-honesty items that fall outside `system1-wiring`'s surfaces: pin `revision=` in the training kernel, drop the spurious `arc-laya-finetune-data` source, retire/document the legacy lineage | pending | wiring audit + kernel metadata |
| P6 | **The coordinated publish**: update the live writeup, the card, the draft and the README in one pass | pending | all artifacts agree |
| P7 | Independent verification | pending | verifier report |

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
| Rewriting the whole narrative late | little time to iterate before 2026-11-09 | this feature |

## 10. Evidence log

- 2026-09-24 — created. Records the owner's "publish together" decision (§2) and the fact that the
  model-card correction was **prepared and deliberately withheld**.
- 2026-09-24 — the model card was found to be CLI-editable (`kaggle models update` and
  `kaggle models instances update`), which makes P2/P3 possible at all.

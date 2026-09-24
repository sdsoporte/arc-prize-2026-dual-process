<!-- P1 DRAFT — NOT LIVE. Withheld from publication until the coordinated publish pass (odd/tasks/paper-pivot.md §2, P6). Do not paste this into the competition writeup before then. -->

# The Metric Cannot Measure the Agents: A Human-Play Ruler, a Baseline Audit, and a Primitive Ablation

**Author:** Sergio Alberto Dominguez ([@ser8147](https://www.kaggle.com/ser8147)) · Independent Researcher
**Repository:** [github.com/sdsoporte/arc-prize-2026-dual-process](https://github.com/sdsoporte/arc-prize-2026-dual-process) · **License:** MIT / CC0

---

## 1. What this is

Three measured instruments and one diagnosis. A human-play ruler for ARC-AGI-3. An audit of the benchmark's own baseline. A leave-one-out ablation of a primitive DSL. The diagnosis they support: this benchmark's metric cannot measure the agents competing on it — and the submission validator cannot see the difference either. This is a measurement report, not an architecture pitch. Every number below is traceable to a file in the repository.

## 2. A human-play calibration instrument

ARC-AGI-3 scores an agent against human play, but no public tool prices the human. We built one. The published human replay archive ([`jihangli1121/arc-agi-3-replays-v1`](https://www.kaggle.com/datasets/jihangli1121/arc-agi-3-replays-v1)) is fed action by action into our **OFFLINE** engine, and the results are scored by the official scorer.

- **24 of 25** published replays reproduce step for step, for the whole recorded run.
- Scored under the official scorer, human play is **89.6774 / 100**.
- Our own agent, on the same instrument and the same game set, has a reproducible baseline of **0.2017** (3 of 183 levels).

Two caveats are part of the instrument. The single non-reproducing game, `cn04`, is a known local-build mismatch, not an agent failure. And the instrument's job is **calibrating human play and serving as a regression check** — not predicting the leaderboard. It agreed once with a randomly seeded run and later missed a specific prediction; we do not use it to forecast submitted scores.

What the instrument buys is comparability without spending a submission. Anyone can measure an agent against known-good human play, on the identical scorer, offline, without spending a competition submission per day. Sources: `experiments/arc3_calibration_reference.json`, `experiments/arc3_local_eval.py`, `experiments/arc3_baseline.json`, `docs/ARC3_REPLAYS_FINDINGS.md`.

## 3. Audit of the benchmark's own baseline

The ARC-AGI-3 level score is `min(100, baseline_actions / actions_taken × 100)`. The implicit premise is that `baseline_actions` is the published human play. It is not the same runs.

- Only **33 of 183** levels match exactly; the replay human is at-or-faster on **142 of 183**.
- Baseline totals **17,135** actions against the humans' **14,798**.
- A slower baseline means a larger ratio, so scores are inflated relative to observable human play, by a level-dependent amount: **`su15` level 7 differs by 6.25x**.

The audit is checkable by anyone holding the public replays and the environment metadata. It matters because every published ARC-AGI-3 score, ours included, is priced against a reference that is not the human it is described as. Source: `docs/ARC3_REPLAYS_FINDINGS.md`.

## 4. A measured primitive ablation

The public notebook `yusuketogashi/arc-baseline-rebuild` reports that exact symbolic rules covered nothing of ARC-AGI-2; forum **742790** asks *"what primitive would you try first?"*. We answered it by measurement: disable one operator family at a time and count the public **training** outputs that stop being solved. Measured on **1,000 training tasks / 1,076 outputs**:

| family | outputs lost when disabled |
|---|---|
| `panel` | **−11** |
| `scale` | **−5** |
| `collinear` | **−4** |
| `holes`, `kronecker`, `overlay` | **−2** each |
| `object`, `tiling` | **−1** each |

**Eight D4 symmetry operators, colour mapping, gravity, cropping, counting and the two-stage composition contribute exactly zero.** The two families the public post proposes are the two that pay nothing.

This is a **train** result and must be read as one. The honest triple is **train 4.1822% (45/1076) / eval 0.00% (0/172) / deployed 0.00**. The 4.18% is in-distribution fitting on the tasks the library was designed against, and the held-out column is zero. Sources: `experiments/arc2_ablation.py`, `experiments/arc2_ablation_results.json`, `experiments/arc2_baseline.json`.

## 5. The thesis: the metric cannot measure the agents on it

The official scorer is a ratio with a fixed denominator. In `arc_agi/scorecard.py`, `EnvironmentScoreCalculator.add_level` appends a hard **0.0** for every uncompleted level, and `to_score` divides by `len(level_scores)` — the **total** level count. A game's score is (a few small numbers) / (its total level count), then averaged over games.

Two measured runs of the same code with the same seed, differing only in `--max-steps`, show what that does:

| run | mean score | levels cleared |
|---|---|---|
| `--max-steps 80` | **0.3175** | 1 / 183 |
| `--max-steps 500` | **0.4042** | 2 / 183 |

**The entire difference is one level cleared in one game**, `r11l`, whose level score is **2.17** — and `(0.4042 − 0.3175) × 25 = 2.17` closes exactly. Every other game scores exactly 0.00.

Our deployed ARC-AGI-3 run measured **0.4042** locally and scored **0.02** on the leaderboard, against a best posted **0.28**. The leaderboard is computed on **approximately 50% of the test data**, so a public half containing none of the one or two cleared levels scores ~0. **Every one of these numbers is a draw from a one-to-two-event statistic.**

The same agent can be reported at 0.3175 or 0.4042 with no change in its reasoning, only in how many steps it was allowed. Scores at this scale are near-binary event statistics rendered as floats.

The validator cannot see any of it. Measured on the actual submitted artifacts via `kaggle kernels output`:

| artifact | non-informative outputs | with content |
|---|---|---|
| reference implementation, commit submission | **238 of 259 = `[[0]]` (91.9%)** | 21 |
| ours | **209 echoes (80.7%) + 46 zero grids (17.8%) = 98.5%** | 4 (1.5%) |

**Both passed `problems: 0 / VERDICT PASS`.** The format validates keys and shapes, not content. That is the mechanism behind the large block of teams at 0.00 — in ARC-AGI-2, **722 of 2,172** teams sit there.

The two failures compound. A submission can clear nothing, pass validation, and be reported at 0.00 alongside a large block of other teams. A submission can clear one level, measure 0.4042 locally, and be reported at 0.02 because the public half of the test data did not contain it. Neither the number nor the verdict distinguishes reasoning from filler, and a metric whose denominator is the total level count combined with a validator that cannot see whether an answer carries information cannot separate the agents competing in this benchmark. The honest report is coverage alongside score, never score alone.

## 6. The negatives of our own architecture

The instrument made our own failures visible, so we report them as measurements.

- **System 1 (Laya, 421M, offline):** measured **ECE 0.2392**, and zero gate headroom — **0 of 43** covered tasks, because candidates are functionally redundant wherever coverage exists. It is not wired into any submission: `model_sources` is empty on all six competition kernels. It stands as a hypothesis we tested and rejected.
- **The deployed ARC-AGI-2 solver** covered **15 of 240** hidden test tasks, and **all fifteen were wrong**. Coverage means a candidate reproduced the training pairs, not that the test answer was right.
- **The ARC-AGI-3 agent's heuristic layer is inert:** **92.8%** of its choice calls receive a one-candidate pool.

Any score we quote carries two caveats: the leaderboard uses ~50% of the test data, and our agent's score spreads across a wide range because it is RNG-dominated.

## 7. What is cited, and what is ours

**Ours:** the human-play calibration instrument, the baseline audit, the ablation, and the negatives above.

**Cited, never claimed:** the primitive DSL and the finding that exact symbolic rules covered nothing — `yusuketogashi/arc-baseline-rebuild`. The `hash(str)` non-determinism and its `zlib.crc32` remedy — forum **742027**. The 8-symmetries-plus-colour-map solver at **9/1076 = 0.84%**, whose open question our ablation answers and whose two families pay nothing — forum **742790**. Our ablation's 4.18% is a train figure measured on the same 1,076 outputs, not a comparison on one axis.

**Reference implementation:** we deploy upstream work — [`mikelou1/arc-agi2-lb33-89-minimal-perfpatch`](https://www.kaggle.com/code/mikelou1/arc-agi2-lb33-89-minimal-perfpatch) and the fork [`manderson240/arc-agi-2-fork-lb33-89-20260903`](https://www.kaggle.com/code/manderson240/arc-agi-2-fork-lb33-89-20260903) — as a **declared reference**, not as our contribution. Our own deployment of it deviates by exactly one parameter (`use_gradient_checkpointing=True`), plus a commit-mode budget default of 1500 → 1800.

## 8. Code and reproducibility

Everything runs offline, with no network access and no third-party APIs. The instruments are `experiments/arc3_calibration.py`, `experiments/arc3_local_eval.py` and `experiments/arc2_ablation.py`; their recorded outputs are `experiments/arc3_calibration_reference.json`, `experiments/arc3_baseline.json`, `experiments/arc2_baseline.json` and `experiments/arc2_ablation_results.json`; the audit is `docs/ARC3_REPLAYS_FINDINGS.md`. MIT / CC0.

# Objective — ARC Prize 2026

> **Status:** governing document. Supersedes nothing; it was simply missing until now.
> **Owner:** Sergio D (@ser8147)
> **Created:** 2026-09-24
> **Deadlines:** code freeze **2026-11-02 23:59 UTC** · paper due **2026-11-09 23:59 UTC**

---

## 1. The objective, in one line

> **Win the paper ranking, not the leaderboard: make the writeup true, complete and theoretically
> strong, with the highest leaderboard score that comes cheap — and let the project's own quantified
> diagnosis be the contribution.**

## 2. Why the goal is the Paper Track, and what ARC-AGI-2/3 are for

The Paper Track is the target. ARC-AGI-2 and ARC-AGI-3 participation is an **eligibility requirement**,
not the goal. From the competition Evaluation page:

> *"To be eligible to win the ARC 2026 Paper Award Prize, you must join this competition and submit a
> Writeup that documents your solution for either ARC-AGI-2 or ARC-AGI-3."*

And the Competition-Specific Rules, §2.1.b:

> *"Team must match the team making a submission to either ARC-AGI-2 or ARC-AGI-3."*

Both conditions are **already satisfied**: two live submissions (ARC-AGI-3 `0.28`, ARC-AGI-2 `0.00`) under
the same individual (user id `36828078`, sole member of all three teams). No further submission is required
for eligibility.

**The consequence that matters:** the writeup must *document the solution*. Its honesty is therefore not a
virtue, it is the admissibility condition. A paper describing an architecture its submission does not
implement fails the very premise that makes it eligible.

## 3. The ranking arithmetic, stated before it is too late

Six criteria, each 0–5, averaged: **Accuracy · Universality · Progress · Theory · Completeness · Novelty**.

The $375K pool is reserved for papers scoring **above 4.5/5**, which requires 27 of 30 points:

| if Accuracy = | the other five must sum to | i.e. average | verdict |
|---|---|---|---|
| 1 | 26 | 5.2 | **impossible** |
| 2 | 25 | 5.0 | perfect on all five |
| 3 | 24 | 4.8 | very hard |

**With a near-zero Accuracy the pool is arithmetically out of reach.** What remains is the **top three
places** ($50K / $20K / $5K), which are a **ranking among ~202 papers**, not a threshold. That is the
contestable prize, and it is decided by paper quality.

## 4. Sub-objectives, ranked by where the points are

| # | Criterion | Our position | What to do |
|---|---|---|---|
| 1 | **Completeness** | **broken** | Make the writeup describe what the submissions actually do, or state explicitly what they do not. Remove every number that cannot be reproduced from the repository. |
| 2 | **Theory** | **strongest asset** | Four measured results, none of them argued: see §6. |
| 3 | **Progress** | strong | The calibration method lets anyone measure an agent against human play **without spending a submission per day**. |
| 4 | **Universality** | strong | Human-replay calibration is domain-agnostic: it validates any agent harness against known-good human play. |
| 5 | **Novelty** | strong | No public ARC work ships an offline human-replay calibration benchmark. |
| 6 | **Accuracy** | **unwinnable** | 0.28 of 100 against a leader at 19.40. Improve only if cheap; never at the cost of criteria 1–5. |

## 5. Non-goals

- **Do not chase the leaderboard.** The gap is ~80x on ARC-AGI-3 and total on ARC-AGI-2. Days spent there
  are days not spent on the ranking.
- **Do not add heuristic tuning to the agent.** Measured: its weighted-heuristic layer is inert (see §6).
- **Do not re-argue the architecture.** Make it true or declare it separate; do not defend it in prose.
- **Do not overstate.** Every published number must be reproducible from the repository.

## 6. The evidence base — four measured results

These are the paper's real contribution. Each is measured, not asserted.

1. **Human-play calibration benchmark.** The `jihangli1121/arc-agi-3-replays-v1` replays drive the OFFLINE
   engine: **24 of 25 reproduce step for step**, and the official scorer assigns human play **89.6774/100**
   on the same instrument that scores our agent **0.2017**. Shipped as a regression check
   (`experiments/arc3_calibration.py`).
2. **The primitive-DSL coverage wall (ARC-AGI-2).** Coverage — a candidate that reproduces every training
   pair — is **0/120 on the public evaluation set** for both solver generations, **5/240 (2.1%)** deployed
   and **15/240 (6.2%)** local on the test set. Neither solves anything on eval. The measured ceiling of a
   whole-grid primitive pool is ~7% even on training data.
3. **Score is dominated by the RNG.** 13 sweeps of byte-identical agent code spread over **0.173 to 1.090
   — a 6.3x range**. Consequence: the published "+16.7% improvement" (0.24 → 0.28) is one draw from that
   distribution.
4. **The agent's heuristic layer is inert.** `CLICK_WEIGHT` set to `1e6` **and** to `0.0` produce identical
   scores; likewise `REPEAT_PENALTY` at `0.0` and `1e6`. Cause, instrumented: **92.8% of `random.choices`
   invocations receive a pool of exactly one candidate**, where weights cannot matter. Real decisions come
   from the BFS frontier planner.

## 7. Honest current state — what is false today

| Artifact | Claim it makes | Measured reality |
|---|---|---|
| `submissions/KAGGLE_WRITEUP.md` | `+16.7% relative improvement` (0.24 → 0.28) | one draw from a 6.3x-wide distribution |
| `submissions/KAGGLE_WRITEUP.md` | `0.28 (28% Solved)` | the metric is 0–100 where 100 = human, so **0.28%** |
| `submissions/KAGGLE_WRITEUP.md` | `synthesizing rules for 17.9% of benchmark tasks` | **2.1%** coverage deployed, 6.2% local |
| `submissions/KAGGLE_WRITEUP.md` | cites the ARC-AGI-2 kernel as the code submission reference | that kernel scores `0.00`; the ARC-AGI-3 one scores `0.28` |
| `paper/draft.md`, `models/kaggle_model_hub` | Laya v1 numbers (88.24%, Brier 0.0818) | the model hub serves **v2** (89.92%, Brier 0.1020, **ECE 0.2392**) |
| Model card | "System 1 decision screening" | `model_sources: []` on all six kernels — **nothing consumes it** |
| Both submission kernels | describe a dual-process System 1 | neither imports the model; both are model-free |

## 8. Definition of done

1. **No number in the paper without a reproducible source in the repository.** Today three fail this.
2. **The paper describes what the submissions do**, or states explicitly what they do not.
3. **The four results in §6 are written up and independently checkable.**
4. **Everything is public in the repository**, which is also the eligibility requirement of Rule 2.5.b
   ("a link to a code repository with complete and detailed instructions so that the results can be
   reproduced").
5. **The live writeup is corrected and current**, because that is what the judge reads.

The writeup is editable until the deadline. The **code freezes first** (Nov 2), so any submission change
must land before it; the paper can continue to Nov 9.

## 9. Decision rights

| Decision | Owner | Notes |
|---|---|---|
| Whether System 1 is part of the submission or an offline-measured component | **Sergio** | determines the paper's central claim |
| Which track carries the Accuracy reference | **Sergio** | currently misfiled on the 0.00 kernel |
| Spending a competition submission | **Sergio** | 1/day per track, resets 00:00 UTC |
| Merging branches, pushing to GitHub | **Sergio** | |
| Task decomposition and implementation | el Gentleman | within an approved objective |

## 10. How this document is used

This is the governing objective. Work is tracked under `odd/tasks/<feature>.md`, one feature per coherent
unit, and each feature references the sub-objective it serves. A feature that serves no sub-objective in §4
is out of scope by definition.

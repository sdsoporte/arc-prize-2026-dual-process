# Objective — ARC Prize 2026

> **Status:** governing document. Supersedes nothing; it was simply missing until now.
> **Owner:** Sergio D (@ser8147)
> **Created:** 2026-09-24
> **Deadlines:** code freeze **2026-11-02 23:59 UTC** · paper due **2026-11-08 23:59 UTC** ([arcprize.org/competitions/2026](https://arcprize.org/competitions/2026) → Key Dates; the rules page governs over Kaggle's Nov 9 platform field)

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

> **Unconfirmed: the Paper Track prize structure.** The prize figures this section reasons over
> (top-3 $50K/$20K/$5K and the $375K threshold pool) — and the total the repo elsewhere quotes as
> "$450K" — are **unconfirmed pending the paper page**. The ARC Prize overview page fetched on 2026-09-24
> shows only *"Paper Prize — Awards for papers that advance our understanding"* with **no breakdown**.
> Treat every Paper Track prize amount here as unverified until `https://arcprize.org/competitions/2026/paper`
> is read. The figures are kept, not asserted; see the matching entry under `docs/KAGGLE_OPS.md` §10
> ("What is NOT verified").

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

### 3.1 There is a SECOND rubric-judged prize, and we are not playing it

ARC-AGI-2 carries a **$275,000 Grand Prize awarded to the single highest-scoring Solution Writeup**, judged on
the **same six criteria**, equally weighted. So the same body of work can be entered twice, with the Progress
narrative adapted:

| | Paper Track | ARC-AGI-2 Grand Prize |
|---|---|---|
| places | 3 ($50K/$20K/$5K) + the threshold pool | **one, winner-take-all, $275K** |
| field | ~203 writeups, with a visible "View Writeups" tab that invites entry | **unknown — currently zero**. The mechanism is a forum post designated from the team page *after* the competition closes, described only in a four-year-old product announcement |
| Accuracy | we may cite ARC-AGI-3 (0.28, the **median** of 3,284 teams) | fixed to ARC-AGI-2 (`0.00`, bottom third) |
| Progress | "a top score on ARC Prize" | "**85% on ARC-AGI-2**" |

Verified: the **Solution filter returns "No discussions found" in both ARC-AGI-2 and ARC-AGI-3** — nobody has
designated a writeup yet, because designation cannot happen before close. **That is a low-visibility
instrument, and here it is an advantage.** It is also monitorable: if the filter is still near-empty close to
the deadline, the effective field is a handful.

### 3.2 A third variance source nobody quotes

Both leaderboards state: *"This leaderboard is calculated with approximately **50% of the test data**. The
final results will be based on the other 50%, so the final standings may be different."* Every published
score is therefore measured on half the test set. Added to the agent's RNG (6.3x) and to submission sampling,
this means **no score in this project may be quoted without those caveats.**

## 4. Sub-objectives, ranked by where the points are

| # | Criterion | Our position | What to do |
|---|---|---|---|
| 1 | **Completeness** | **broken** | Make the writeup describe what the submissions actually do, or state explicitly what they do not. Remove every number that cannot be reproduced from the repository. |
| 2 | **Theory** | **the primary asset** | Four measured results, three of them unoccupied by anyone else: see §6a. |
| 3 | **Progress** | strong | The calibration instrument lets anyone measure an agent against human play **without spending a submission per day**; the ablation tells the field which primitives pay and which pay nothing. |
| 4 | **Novelty** | **narrowed — cite, never claim** | Only three things are unoccupied (§6a). The primitive DSL, the coverage wall and the `hash(str)` non-determinism are **already public** (§6b); claiming them would be a checkable error. |
| 5 | **Universality** | strong | Human-replay calibration is domain-agnostic: it validates any agent harness against known-good human play. |
| 6 | **Accuracy** | weak, but **improvable on ARC-AGI-2** | 0.28 is the **median** of 3,284 teams on ARC-AGI-3. On ARC-AGI-2 we sit at the floor with 722 teams (33.2%) — and a **declared reference implementation** can reach ~30. |

## 5. Non-goals

- **Do not chase the leaderboard.** The gap is ~80x on ARC-AGI-3 and total on ARC-AGI-2. Days spent there
  are days not spent on the ranking.
- **Do not add heuristic tuning to the agent.** Measured: its weighted-heuristic layer is inert (see §6).
- **Do not re-argue the architecture.** Make it true or declare it separate; do not defend it in prose.
- **Do not overstate.** Every published number must be reproducible from the repository.

## 6. The evidence base

### 6a. Ours, and unoccupied — the actual contribution

1. **A human-play calibration instrument (ARC-AGI-3).** The `jihangli1121/arc-agi-3-replays-v1` replays drive
   the OFFLINE engine: **24 of 25 reproduce step for step**, and the official scorer assigns human play
   **89.6774/100** on the same instrument that scores our agent **0.2017**. Shipped as a regression check
   (`experiments/arc3_calibration.py`). **No public notebook ships one**, and the replays dataset has no
   linked notebook.
2. **An audit of the benchmark's own baseline.** The score is `min(100, baseline_actions / actions × 100)`,
   but `baseline_actions` is **not the human play published as ground truth**: only **33 of 183** levels match
   exactly, the replay human is at-or-faster on **142 of 183**, and the baseline totals **17,135** actions
   against the humans' **14,798**. A slower baseline means a larger ratio, so **scores are inflated relative
   to observable human play**, by a level-dependent amount (`su15` level 7 differs by **6.25x**). Checkable by
   anyone with the public replays and the environment metadata.
3. **A measured primitive ablation (ARC-AGI-2).** Leave-one-out over the operator families:
   **`panel` −11, `scale` −5, `collinear` −4** carry the solver, while **eight D4 symmetry operators, colour
   mapping, gravity, cropping, counting and the two-stage composition contribute exactly ZERO.** This answers
   a public open question — forum `742790` asks *"what primitive would you try first?"*, and its two families
   are the two that pay nothing — and it explains our **45/1076** against that notebook's **9/1076**.
4. **Two negative results, measured.** A learned System-1 gate has **zero headroom** (0 of 43 covered tasks;
   the first matching candidate is always the correct one, because the candidates are functionally
   redundant). And the ARC-AGI-3 agent's score is RNG-dominated with an inert heuristic layer (92.8% of
   selection calls receive a one-candidate pool).

### 6b. Already published elsewhere — cite, never claim

| Finding we held | Who published it first |
|---|---|
| A primitive DSL covers ~nothing of ARC-AGI-2 | `yusuketogashi/arc-baseline-rebuild` — *"exact symbolic rules did not cover any"* |
| Run-to-run variance from `hash(str)` seeding, and the `zlib.crc32` remedy | ARC-AGI-2 forum **742027** (19-sep) — the same diagnosis and the same fix |
| A minimal solver = 8 symmetries + a global colour map | ARC-AGI-2 forum **742790** (23-sep) |
| The ARC-AGI-2 field converges on one public pipeline | measured: **745 of 2,172 teams (34.3%) sit in the 30–40 band, only 4 exceed 40** |

Our DSL's overlap with that published solver is **~4–5 families out of ~30** (`d4_exact`, `d4_color_map`,
`integer_upscale`, `integer_tile`); the other families do not appear there. Real, but partial.

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
| `notebooks/arc2_reference_kernel/submission.ipynb` (public kernel header) | `0.84%`, `4.18%` and `~30` presented as **one axis** | the first two are **train** (pass@2 on 1,076 public training outputs); the third is a **leaderboard**. Our ablation is **4.18% train against 0.00% held out**. This surface was in **no feature's edit surfaces** until 2026-09-24, so it would have survived the coordinated pass — see `paper-pivot` P8 |
| the deployed ARC-AGI-2 solver (v2) | a working symbolic program synthesiser | it carried a **dead `rot270`** (`zip(*g)[::-1]` raises `TypeError: 'zip' object is not subscriptable`), so every rot270 candidate was silently dropped |
| what we actually submitted | a submission of 240 tasks | 259 outputs: **80.7% echoed the input, 17.8% were zero grids, 1.5% carried content** — 98.5% non-informative, and the notebook printed `✅ … 240 tasks!` in 0.7026 s |
| `paper/draft.md` §4.3 and the System 1 taxonomy | System 1 classifies "geometry, flood_fill, counting, extrapolation" | the ablation shows geometry, flood-fill and counting contribute **zero** while `panel`/`scale`/`collinear` carry the solver — **a working gate would classify the wrong axes** |
| anywhere a leaderboard score is quoted | a bare number | it is measured on **50% of the test data**, and our agent's score spreads **6.3x** across seeds |

## 8. Definition of done

1. **No number in the paper without a reproducible source in the repository.** Today three fail this.
2. **The paper describes what the submissions do**, or states explicitly what they do not.
3. **The four results in §6 are written up and independently checkable.**
4. **Everything is public in the repository**, which is also the eligibility requirement of Rule 2.5.b
   ("a link to a code repository with complete and detailed instructions so that the results can be
   reproduced").
5. **The live writeup is corrected and current**, because that is what the judge reads.

The writeup is editable until the deadline. The **code freezes first** (Nov 2), so any submission change
must land before it; the paper can continue to Nov 8.

## 9. Decision rights

| Decision | Owner | Notes |
|---|---|---|
| Whether System 1 is part of the submission or an offline-measured component | **Sergio** | determines the paper's central claim |
| Which track carries the Accuracy reference | **Sergio** | currently misfiled on the 0.00 kernel |
| Spending a competition submission | **Sergio** | 1/day per track, resets 00:00 UTC |
| Merging branches, pushing to GitHub | **Sergio** | |
| Task decomposition and implementation | el Gentleman | within an approved objective |

## 10. The pivot

**What the paper becomes.** Not "our architecture works", but:

> *An offline human-play ruler for ARC-AGI-3, an audit of the benchmark's own baseline, and a measured map of
> which primitive families actually pay — plus the negative results of our own architecture, which the
> instrument made visible.*

**Three points on one measured axis (ARC-AGI-2):**

| approach | public training outputs (pass@2) |
|---|---|
| the minimal public solver (symmetries + colour map) | 9/1076 (0.84%) |
| **our ablation** — the families that pay | **45/1076 (4.18%)** |
| the field's full neural pipeline (declared reference implementation) | ~30 on the leaderboard |

**What must be declared, not claimed:** the DSL, the coverage wall and the `hash(str)` finding are public
(§6b). The Laya gate moves from "our architecture" to "a hypothesis we test and reject", with §6a's zero
headroom as the evidence.

**What must be corrected in the live writeup:** the three false numbers (§7), the code reference repointed to
the ARC-AGI-3 kernel where we hold the **median** of 3,284 teams, and `paper/draft.md` §4.3 deleted.

**Publishing is coordinated, not piecemeal.** The owner decided on 2026-09-24 that **no public artifact is
edited until the writeup rewrite is drafted**, because a corrected model card would otherwise contradict the
live writeup mid-flight — a second, self-inflicted Completeness failure. The writeup, the model card,
`paper/draft.md` and the README change in one pass. Owned by `odd/tasks/paper-pivot.md`.

## 11. How this document is used

This is the governing objective. Work is tracked under `odd/tasks/<feature>.md`, one feature per coherent
unit, and each feature references the sub-objective it serves. A feature that serves no sub-objective in §4
is out of scope by definition.

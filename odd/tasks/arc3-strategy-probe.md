# ODD Feature — `arc3-strategy-probe`

> **Status:** in progress
> **Branch:** `feat/arc3-strategy-probe`
> **Base commit:** `fd6508f`
> **Created:** 2026-09-23
> **Engram mirror topic key:** `arc-paper-track/odd/arc3-strategy-probe/tasks`

---

## 1. Goal

Answer one question with the instrument that now exists: **do the two replay-evidenced strategy hypotheses
produce a signal larger than seed noise?**

The owner asked for a short experiment and then a decision, so this feature is explicitly bounded: measure,
report the deltas, decide nothing itself.

## 2. Why the design is paired, and why that matters

`arc3-eval-harness` established that the agent's score is dominated by its RNG. With the seed fixed, 13
sweeps of identical code spread over **0.173 to 1.090** (6.3x). Therefore:

- Comparing two variants on **one** seed is worthless - the difference could be seed luck.
- Comparing **means of independent runs** needs enough samples to see through a 6.3x spread.
- Comparing variants **at the same seed** cancels the seed effect: the paired delta is the informative
  quantity, and a consistent sign across seeds is real evidence where an absolute number is not.

So the probe runs a factorial: several seeds x {baseline, variant A, variant B, ...}, and reports the
per-seed paired delta rather than only the means.

## 3. The hypotheses, and where they come from

Both come from measured human play in `docs/ARC3_REPLAYS_FINDINGS.md`, not from intuition:

| # | Hypothesis | Human evidence | What our agent does today |
|---|---|---|---|
| H1a | Clicks should be **spatially concentrated** | ACTION6 is 29.7% of human actions, the most of any; the top 8 destinations are ~18% of all clicks | `self.rng.randint(0, 63)` over the whole grid |
| H1b | Clicks should be **weighted up** relative to movement | humans click 29.7% vs 16.5/14.1/18.0/18.9% for A1-A4 | click weight 0.5 vs movement weight 4.0 |
| H2 | **Repetition helps**, our agent penalises it | `repPrev` 36-65% in 15 of 25 games, `maxRun` up to 18 | `if is_oscillating and a.value == last_action_val: w *= 0.1` |

H1 is one hypothesis with two distinct mechanisms; the probe separates them so a null result on one does not
hide a positive result on the other.

## 4. Non-goals

- Does not choose a strategy. It measures five arms and stops.
- **Does not change the shipped agent's behaviour.** It adds three *behaviour-preserving* seams to
  `MyAgent` so variants can be expressed as subclasses instead of by copying a 90-line method into the
  probe (which would silently drift from the real agent): the click weight and the repeat penalty become
  named class attributes, and click-target selection becomes an overridable method. The baseline arm
  running the real `MyAgent` must still reproduce `0.2017`, which is the proof the seams changed nothing.
- Does not deploy, and does not touch Kaggle.
- Does not chase the leaderboard. The honest ceiling here is small: the agent is at `0.2017` and human play
  on the same instrument is `89.68`, so even a 10x improvement is 2.0 against a leader at 19.40.

## 5. Constraints

| Constraint | Value | Source |
|---|---|---|
| Interpreter | `data/arc-agi-3-agents/.venv/bin/python` (3.12) | `arc3-eval-harness` §7 |
| Engine | `OperationMode.OFFLINE` only; never `data/arc-agi-3/environment_files/` | `arc3-eval-harness` §3 |
| Scorer | reuse `experiments/arc3_local_eval.py`; do not reimplement | verified |
| Seed | `ARC3_AGENT_SEED` (added in `9d39c09`) | this feature depends on it |
| Baseline reference | mean `0.2017`, 3/183 levels, reproducible | `experiments/arc3_baseline.json` |
| Human ceiling | `89.6774` | `experiments/arc3_calibration_reference.json` |
| Cost | 25 games is ~15 s, so a factorial of 5 arms x 6 seeds is ~8 min | measured |

## 6. Allowed edit surfaces

- `experiments/arc3_strategy_probe.py`
- `experiments/arc3_probe_results.json`
- `src/arc3_spatial_memory_agent.py`
- `odd/tasks/arc3-strategy-probe.md`

## 7. Tasks

| # | Task | Status | Evidence |
|---|---|---|---|
| P1 | Scaffold ODD tracking: branch, feature doc, Engram mirror, todo | in progress | this file |
| P2a | Add three behaviour-preserving seams to `MyAgent` (click weight, repeat penalty, click-target method) | done | baseline seam check rerun: `mean 0.2017`, `3/183` exactly |
| P2b | Implement the probe: 5 arms subclassing `MyAgent`, paired factorial over seeds, results JSON | done | `experiments/arc3_strategy_probe.py`, `experiments/arc3_probe_results.json` |
| P3 | Run it and analyse: per-seed deltas, sign consistency, whether any effect exceeds seed noise | done | seeds 1-6: null; no arm distinguishable (see §9) |
| P4 | Independent verification | pending | verifier report |

## 8. Acceptance criteria

1. Reuses `arc3_local_eval`'s sweep and the official scorer; reimplements neither.
2. The only `src/` change is the three behaviour-preserving seams, and the baseline arm with the default seed
   still reproduces `0.2017` / 3-of-183 exactly. (An earlier wording said "`src/` is untouched", which
   contradicted this feature's own Step P2a; the worker caught the inconsistency, and the behaviour check is
   the criterion that actually matters.)
3. Every arm runs the **same seeds**, so paired deltas are computable.
4. Reports, per variant: the per-seed delta against baseline, the count of seeds where the sign is positive,
   the mean/median delta, and the range - not just a single mean.
5. States plainly whether any effect is distinguishable from seed noise, including the possibility that none
   is.
6. Does not write to `data/arc-agi-3/environment_files/`; verified with `git status` afterwards.
7. The `src/` seams are behaviour-preserving: the baseline arm, running the unmodified `MyAgent`, reproduces
   the recorded `0.2017` / 3-of-183 exactly. If it does not, the seams changed behaviour and the probe is
   measuring something other than what it claims.

## 9. Evidence log

- 2026-09-23 - created. Depends on the seed fix (`9d39c09`) and the calibration benchmark (`545dd95`).
- 2026-09-23 - P2a. Added the three behaviour-preserving seams to `src/arc3_spatial_memory_agent.py`:
  `CLICK_WEIGHT = 0.5`, `REPEAT_PENALTY = 0.1`, and `_choose_click_coords()`. Nothing else in `choose_action`
  was restructured. Proof the seams changed nothing: `data/arc-agi-3-agents/.venv/bin/python
  experiments/arc3_local_eval.py` (default seed, `ARC3_AGENT_SEED` unset) reports `mean score: 0.2017`,
  `levels completed: 3/183` exactly.
- 2026-09-23 - P2b/P3. Ran the 5-arm x 6-seed paired factorial (30 sweeps, ~6 min) with
  `data/arc-agi-3-agents/.venv/bin/python experiments/arc3_strategy_probe.py`; results in
  `experiments/arc3_probe_results.json`. **Null result.** Baseline arm across seeds 1-6: 0.3390 / 0.4063 /
  0.2269 / 0.5980 / 0.3982 / 0.1367, i.e. range 0.4614 and stdev 0.1459 - larger than every arm's mean
  paired delta. No arm is sign-consistent: `click_local` -0.0237 (0/2/4), `click_reuse` +0.1630 (3/2/1),
  `click_weight` 0.0000 (0/0/6), `no_rep_penalty` 0.0000 (0/0/6). `click_reuse` is the only arm with a
  positive mean and is the only follow-up candidate worth more seeds; it is not evidence of an effect.
- 2026-09-23 - interpretation of the exact-zero weight arms: the heuristic pool where `CLICK_WEIGHT` and
  `REPEAT_PENALTY` are read (Step 5 of `choose_action`) is a fallback behind the BFS frontier planner
  (Step 4); the click arms move the score only because they also perturb the RNG stream via their extra
  draws. The weight seams consume no extra RNG, so a zero delta there is expected wherever the selection
  boundary never moved - it does not prove the weights are irrelevant in general.
- 2026-09-23 - safety: `git status --short -- data/arc-agi-3/environment_files/` prints nothing after every
  run; the engine ran `OperationMode.OFFLINE` against the `/tmp/arc3-env` copy only.

## 10. Known risks carried

| Risk | Impact | Owner |
|---|---|---|
| Seed noise is large, so a small real effect is undetectable at this sample size | The probe may be inconclusive, which is itself a finding | this feature |
| Variants subclass and override `MyAgent` methods, so they can drift from the shipped agent | Probe results may not transfer exactly | documented, accepted for a probe |
| Even a large relative gain is a small absolute score | The score will not become competitive | paper positioning |

## 11. Result: a clean null, and a mechanism that explains two of the three arms

The factorial is a **clean null**. No arm's paired delta exceeds seed noise.

| arm | paired delta: +ve / -ve / zero | mean | median | max |
|---|---|---|---|---|
| `click_local` | 0 / 2 / 4 | -0.0237 | 0.0000 | 0.0000 |
| `click_weight` | 0 / 0 / **6** | 0.0000 | 0.0000 | 0.0000 |
| `no_rep_penalty` | 0 / 0 / **6** | 0.0000 | 0.0000 | 0.0000 |
| `click_reuse` | 3 / 2 / 1 | +0.1630 | +0.0434 | +0.6075 |

**Seed-noise scale**, measured on the baseline arm across the six seeds: `0.1367` to `0.5980`, range
**0.4614**, stdev **0.1459**. So any effect below roughly 0.15 is undetectable at this sample size, and
`click_reuse`'s mean barely exceeds the noise it is trying to beat.

`click_reuse` is not evidence even though its mean is positive: it consumes a **variable number of RNG
draws**, so its delta conflates the strategy with a diverged random trajectory. Only `click_local`,
`click_weight` and `no_rep_penalty` are same-stream paired comparisons.

### The two zero arms are not "no effect" - they are provably inert

The writer's first explanation was that the weights sit behind the Step-4 BFS planner and never fire. **That
is wrong**, and the correction is the most interesting thing this feature produced. Measured over one full
25-game run (12,525 decisions):

| decision source | share |
|---|---|
| `System2_BFS_Frontier` | 66.1% |
| `System1_HeuristicFrontier` (Step 5, where the weights live) | **25.6%** |
| `System1_DeadlockUndo` | 7.1% |
| `System1` (RESET) | 1.1% |
| `System1_DeadlockProbe` | 0.1% |

So Step 5 runs a quarter of the time. The weights are still inert, for a different reason, found by
instrumenting `random.Random.choices` itself:

```
random.choices calls: 3204
  pool of 1 candidate: 2974   92.8%
  pool of 2:            111    3.5%
  pool of 3:             99    3.1%
  pool of 4+:            20    0.6%
```

**In 92.8% of invocations the candidate pool has collapsed to exactly one element**, and with one element the
weights cannot change the outcome - `random.choices` returns it whatever they are. Decisive confirmation:
setting `CLICK_WEIGHT` to `1e6` **and** to `0.0`, and `REPEAT_PENALTY` to `0.0` **and** to `1e6`, all produce
the **identical** score on three seeds. A value of one million and a value of zero cannot both be right
unless the number is never read.

This also explains the asymmetry that made the null informative: `click_local` changes the click
*coordinates*, i.e. the content of an action that is already being taken, and it did move the result
(mildly negative). The weight arms change a *selection* that, nine times in ten, has nothing to select
between.

### What this means

The agent's weighted-heuristic layer is **decorative most of the time**. Its decisions really come from the
BFS frontier planner and the deadlock branches; when Step 5 does run it is usually a single forced move.
So "improving the agent" by tuning weights was never going to show a signal, and the negative result is a
statement about the agent's **decision structure**, not about the hypotheses. Real progress would need a
different architecture - a learned policy, or planning that actually branches - not better constants.

That is a Theory-section finding, and it is worth more than the score would have been.

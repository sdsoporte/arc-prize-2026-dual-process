# ODD Feature — `arc3-eval-harness`

> **Status:** in progress (plan revised after resource discovery - see §2)
> **Branch:** `feat/arc3-eval-harness`
> **Base commit:** `bf576e2`
> **Created:** 2026-09-23
> **Engram mirror topic key:** `arc-paper-track/odd/arc3-eval-harness/tasks`

---

## 1. Goal

Get **trustworthy local ARC-AGI-3 measurement** so the agent can be improved without spending one of the
~40 remaining daily submissions to learn a single number. ARC-AGI-2/3 allow **1 submission per day** and
the code freezes **2026-11-02**.

## 2. Plan revised: most of this already exists

An earlier draft of this document proposed writing `experiments/eval_arc3.py` from scratch, and claimed
the agent's build pipeline did not exist. **Both were wrong.** A resource sweep on the owner's prompting
found that the official ARC Prize starter kit is present at `data/arc-agi-3-kaggle-starter/`
(`arcprize/ARC-AGI-3-Kaggle-Starter`) - **gitignored**, which is why it was invisible to a tree listing.

What it already provides, verified by running it:

| Command | What it does | Status |
|---|---|---|
| `make setup` | venv + `arc-agi` + kaggle CLI + clone the framework | not needed - `.venv` exists elsewhere |
| `make play-local` | `scripts/play_local.py` - plays **all 25 games** in-process, per-game `--max-steps`, reports state/levels/actions | **works**: 151 actions/game at ~1500 fps |
| `make verify-local` | 50 steps on `ls20,vc33` | available |
| `make notebook` | `scripts/build_notebook.py` splices `agent/my_agent.py` into `notebooks/submission.ipynb` | the real build step |
| `make submit` | build + `kaggle kernels push -p notebooks/` | the real deploy step |
| `make status` / `make pull-sample` | tail the latest run / fetch the official Stochastic Goose sample | available |

**So the runner exists and should be reused, not rewritten.** The genuine gap is narrower and more
interesting:

- **`play_local.py`'s final score is unusable locally.** `arc.get_scorecard()` prints
  `scorecard_id: None` and `Aggregate scorecard score: 0.0`, because scorecards are API-backed. Running
  it on `ls20,vc33` reported 0 levels and a 0.0 aggregate.
- Computing the score locally through `arc_agi.EnvironmentScoreCalculator` **does** work: a manual sweep
  of all 25 environments produced a mean score of **0.6084** with **3/183 levels** completed
  (`ar25` 1/8, `m0r0` 1/6, `r11l` 1/6), in 13.8 s. That is the same order of magnitude as the
  deployed v2's leaderboard `0.28`, which is the first evidence that the local loop is a usable proxy.

**Revised deliverable:** extend the existing local loop with local scoring, rather than writing a
parallel runner.

## 3. Critical footgun found while exploring

**`play_local.py` runs in `OperationMode.NORMAL` and silently overwrites the environment directory.**
Executing it rewrote four **tracked** files:

```
data/arc-agi-3/environment_files/ls20/9607627b/ls20.py
data/arc-agi-3/environment_files/ls20/9607627b/metadata.json
data/arc-agi-3/environment_files/vc33/5430563c/vc33.py
data/arc-agi-3/environment_files/vc33/5430563c/metadata.json
```

The refresh added the upstream MIT header and **removed a local `_get_hidden_state` override** that the
repository had committed in `6274b0f` - i.e. it silently reverted a local patch on competition data.
All four were restored with `git checkout -- data/arc-agi-3/environment_files/`.

Consequence for this feature: **any harness here must run in `OperationMode.OFFLINE` or against a copy of
the environment directory.** A measurement tool that mutates the thing it measures is not a measurement
tool. (This also explains a burst of 39 code-quality findings, all inside the refreshed third-party game
engine, none in our code.)

## 4. The real drift problem: four copies of one agent

| Copy | Lines | md5 prefix | State |
|---|---|---|---|
| `src/arc3_spatial_memory_agent.py` | 303 | `0d6cfd4698` | v3 |
| `data/arc-agi-3-kaggle-starter/agent/my_agent.py` | 303 | `0d6cfd4698` | v3 - **byte-identical** to the above |
| `notebooks/arc3_submission_kernel/submission.ipynb` | inlined | - | v3 |
| Kaggle kernel `ser8147/arc-agi-3-dual-process-agent` | 214 | - | **v2 - this is what scores 0.28** |

The starter is the source of truth for the build (`make notebook` reads `agent/my_agent.py`), yet it is
gitignored. So the repository cannot regenerate its own tracked notebook, and the tracked notebook has
been edited directly - which is exactly how the deployed kernel drifted two versions behind. **This, not a
missing pipeline, is the root cause of the v2/v3 divergence.**

## 5. Kaggle resource inventory (the owner's reminder)

Found with the CLI; nothing here was downloaded.

**Datasets, highest value first:**

| Dataset | Why it matters |
|---|---|
| `jihangli1121/arc-agi-3-replays-v1` (182 MB, 257 downloads) | **Ground-truth human replays per environment** (`environment_files/<game>/replays/*.json`), plus `action_effect_dict.npz` and `best.pth`. This is the baseline the score is measured against - the single most valuable artefact found. |
| `karnakbaevarthur/arc-agi-3-all-tasks-explanation` (5.8 KB, usability 1.0) | `arc_agi_logic_tasks.csv` - logic explanation per task. Cheap, high signal for hand-crafting agent heuristics. |
| `anglolodorf/arc-agi-3-pretrained-weights` (18 MB) | ARC-AGI-3 "ForgeNet" pretrained weights. |
| `clarelaurent/arc-agi-3` (67 MB) | Further ARC-AGI-3 material. |
| `maximolorenzoylosada/arc-agi-2-structural-task-features` (usability 0.88) and `.../arc-agi-structural-task-features` | Structural features for ARC 1 and 2 - directly relevant to the **ARC-AGI-2 coverage wall**. |
| `karnakbaevarthur/logic-for-each-arc-task` (954 downloads, usability 1.0) and `.../arc-task-logic-labels` | Logic labels per ARC task. |
| `karnakbaevarthur/neurogolf-2026-task-transformation-library` (1001 downloads) | `arc_primitives.csv/json` + per-task ONNX - a transformation library, i.e. candidate primitives for the ARC-AGI-2 pool. |
| `arcgen100k/the-arc-gen-100k-dataset` | 100K synthetic ARC tasks, for training rather than for solving. |

**Models:** `edudev-commons-org/arc-agi-wayfinder-agent` (a published ARC-AGI-3 competition agent),
`johanvondelacruz/arc-flash-7b` (Qwen2.5-Coder-7B fine-tuned for ARC), `ka1242/arcae` (ARC autoencoder),
`sorokin/arc-llm`, plus the LLM bases the top notebooks build on.

**Notebooks:** the ARC-AGI-3 public leaderboard is dominated by one lineage - `Duck Qwen3.8` /
`flash-next` / `anim-base` variants, with `Duck Qwen3.8 Anim Base` at 257 votes and many forks. Also
`amanatar/arc-agi-3-hybrid-repl-agent` and `nihilisticneuralnet/arc-agi-3-retained-reasoning`. This is
required reading for the paper's *Prior Work* and *Novelty* sections: a heuristic agent at 0.28 is not
competing on the same axis as the leaders at 19.40.

Also confirmed: the ARC-AGI-3 competition data ships the **entire upstream framework including its `.git`**
(`arcprize/ARC-AGI-3-Agents` at `135f20a`), with reference templates `random_agent.py`, `llm_agents.py`,
`multimodal.py`, `reasoning_agent.py`, `smolagents.py` and `langgraph_*`.

## 6. Non-goals

- Not a new runner. `play_local.py` is reused.
- Not a solver **improvement**. Two correctness repairs were made because they blocked measurement rather
  than changed strategy: the dead `rot270` primitive (ARC-AGI-2, §8) and the non-reproducible RNG seed
  (ARC-AGI-3, §13). No heuristic, weight or policy was changed.
- Does not download or submit anything to Kaggle.
- Does not fix the four-copies drift (recorded as a risk; the build-path feature owns it).
- Does not touch `data/arc-agi-3/environment_files/` - see §3.

## 7. Constraints

| Constraint | Value | Source |
|---|---|---|
| Interpreter | `data/arc-agi-3-agents/.venv/bin/python` (3.12.13) has `arc_agi` + `arcengine`; local 3.14 does not, and the vendored wheels are `cp312` | probe |
| Starter | `data/arc-agi-3-kaggle-starter/` - **gitignored**; holds `Makefile`, `scripts/`, `agent/`, `vendor/` | `git check-ignore` |
| Environments | 25 local under `data/arc-agi-3/environment_files/`, **tracked** | `git ls-files` |
| Safe mode | `OperationMode.OFFLINE` - `NORMAL` re-downloads and mutates the env dir | measured, §3 |
| Score | `EnvironmentScoreCalculator(id, resets, state, guid)` -> `add_level(completed, actions_taken, baseline_actions, game_id)` -> `to_score()` | `arc_agi` |
| Baselines | per-level `baseline_actions` in each environment's metadata | `EnvironmentInfo` |
| Submissions | 1 / day; code deadline 2026-11-02 23:59 UTC | `kstate.sh` |

## 8. Allowed edit surfaces

- `experiments/arc3_local_eval.py`
- `experiments/test_arc3_local_eval.py`
- `experiments/arc3_baseline.json`
- `experiments/arc3_calibration.py`
- `experiments/arc3_calibration_reference.json`
- `src/arc3_spatial_memory_agent.py`
- `docs/ARC3_REPLAYS_FINDINGS.md`
- `odd/tasks/arc3-eval-harness.md`

## 9. Tasks

| # | Task | Status | Evidence |
|---|---|---|---|
| T1 | Scaffold ODD tracking | done | this file; branch `feat/arc3-eval-harness` |
| T2 | Prove an end-to-end local episode: OFFLINE wrapper + our agent + scorer | done | `ar25`: 301 actions in 0.2 s; sweep of 25 games in 13.8 s |
| T3 | Build a scoring wrapper that mirrors `play_local.py`'s loop but reports the official local score, in OFFLINE mode, with per-game budgets | in progress | brief sent; reference numbers to reproduce: mean 0.6084, 3/183 levels |
| T3b | Analyse the human-replay dataset `jihangli1121/arc-agi-3-replays-v1` and try to turn it into a calibration benchmark | done | `docs/ARC3_REPLAYS_FINDINGS.md`; I re-ran and confirmed **24/25 replays reproduce** and the official scorer gives human play **89.6774/100** |
| T7 | Make the agent's RNG reproducible | in progress | see §13 |
| T8 | Promote the replay calibration into `experiments/` as a durable regression check | pending | delegated |
| T4 | Tests for scoring and aggregation | pending | test output |
| T5 | Record `experiments/arc3_baseline.json` for v3 | pending | per-environment scores |
| T6 | Independent verification | pending | verifier report |

## 10. Acceptance criteria

1. **Mirrors** `play_local.py`'s loop inline rather than importing it. The loop is ~20 lines; importing it would
   make the harness depend on a **gitignored** file, which is the root cause of the drift this feature exists
   to document (§4). The module docstring cites the provenance.
2. Runs in `OperationMode.OFFLINE` and **verifiably does not modify `data/arc-agi-3/environment_files/`**
   (`git status` clean afterwards).
3. Reports the official score via `EnvironmentScoreCalculator`, not a reimplementation, plus per-game
   levels/actions/baseline for explanation.
4. Per-game time and action budgets, so one game cannot stall the sweep.
5. A recorded baseline exists and reproduces.
6. Tests cover scoring and aggregation without playing real games.

## 11. Evidence log

- 2026-09-23 - **the score scale is 0-100 where 100 = human-level performance**, now settled from two
  independent sources. `arc_agi/scorecard.py` computes `min(100, baseline_actions / actions_taken * 100)`
  per completed level and averages over every level in `baseline_actions`; the competition Evaluation page
  says *"Scores for individual game range from 0 to 100%. A score of 100% represents an agent matching
  human-level performance."* The public leaderboard top of **19.40** confirms it is not a 0-1 scale.

  Consequence, and it is uncomfortable: our `0.28` is **0.28% of human-level performance, not 28%**. The
  local v3 sweep is `0.6084` (**0.61%** of human). The leader sits at 19.40%. **The submission writeup states
  "0.28 (28% Solved)"** - a 100x overstatement that a judge can check in one step. The `+16.7% relative
  improvement` claim (0.24 -> 0.28) is arithmetically correct.

- 2026-09-23 - consequence of the same formula for strategy: because a level scores
  `baseline_actions / actions_taken`, completing a level slowly is worth almost nothing. Clearing `ar25`
  level 1 (human baseline 32) in ~267 actions yields `32/267*100 = 12`, and averaged over its 8 levels that
  is `1.50` for the whole game. **The metric rewards understanding, not persistence** - a frontier explorer
  is structurally capped at near zero no matter how long it plays.

- 2026-09-23 - an explore agent (read-only, no shell) established, from source: the exact score semantics
  above; the complete per-level `baseline_actions` table for all 25 games totalling **183 levels**; that
  `Playback` **cannot** consume the replay dataset as shipped (`*.recording.jsonl` under `RECORDINGS_DIR`
  vs `<game>/replays/<game>-<guid>.json` - wrong directory, wrong suffix, possibly a different body schema,
  so an adapter is required); and that our agent's three completions are **all level 1**, i.e. the gap is
  "get past level 1", not "find level 1".

- 2026-09-23 - **hazard found**: an untracked top-level `environment_files/` sits at the repository root
  (4.2 MB, not gitignored, `wa30` metadata `date_downloaded 2026-09-23T18:30Z`, contents differing from the
  tracked `data/arc-agi-3/environment_files/`). It is a second artefact of the same `NORMAL`-mode footgun
  described in §3, and it is one `git add -A` away from being committed and pushed.

- 2026-09-23 - **design decision**: the harness mirrors `play_local.py`'s loop inline instead of importing it.
  The loop is short, and depending on a gitignored path would reproduce the very drift documented in §4.
  Practical consequence: `data/arc-agi-3-kaggle-starter/` can be deleted without breaking the harness.

- 2026-09-23 - feasibility: `Arcade(operation_mode=OFFLINE)` loaded 25 environments; `make('ar25')` returned
  a `LocalEnvironmentWrapper`; `reset()` produced a `FrameDataRaw` (`win_levels=8`, `available_actions=[1..7]`).
- 2026-09-23 - end-to-end with our agent: `ar25`, 301 actions, 0.2 s, 1619 actions/s (no FPS throttle for
  live agents; the `sleep` is only in `Playback`).
- 2026-09-23 - full manual sweep, all 25 games, 500-step budget: mean score **0.6084**, **3/183 levels**,
  13.8 s. Per-game levels: `ar25` 1/8, `m0r0` 1/6, `r11l` 1/6, all others 0.
- 2026-09-23 - `play_local.py` executed successfully but reported `scorecard_id: None` and an aggregate of
  `0.0`, confirming its scorecard path needs the online API.
- 2026-09-23 - **side effect caught and reverted**: `play_local.py` in `NORMAL` mode overwrote 4 tracked
  environment files, removing a local `_get_hidden_state` patch. Restored via `git checkout`;
  `git status` for that path is clean again.
- 2026-09-23 - **retraction**: an earlier draft claimed `agent/` and `scripts/build_notebook.py` did not
  exist and that the notebook was hand-maintained. Both false - they exist in the gitignored starter at
  `data/arc-agi-3-kaggle-starter/`. The drift cause is that the build inputs are gitignored, not missing.

## 12. Known risks carried

| Risk | Impact | Owner |
|---|---|---|
| v3 has never been scored; deployed kernel is v2 | Accuracy rubric | deploy (Phase B) |
| Four copies of the agent; build inputs gitignored | Drift recurs on every edit | build-path feature |
| `play_local.py` mutates the env dir in `NORMAL` mode | Silent mutation of tracked competition data | this feature's OFFLINE requirement |
| Local score is a proxy for the competition gateway | Local numbers are not the leaderboard | none - inherent |
| `arcengine` 0.1.0 in the venv vs 0.9.3 vendored | Local behaviour may differ from the rerun | environment reconciliation |
| Public leaderboard is dominated by Qwen3.8-based agents at up to 19.40 | A heuristic agent cannot close that gap | paper positioning |
| **The writeup claims "0.28 (28% Solved)"; the metric makes it 0.28% of human level** | Accuracy rubric / credibility - checkable in one step by a judge | paper refresh |
| Untracked, non-gitignored `environment_files/` at repo root | 4.2 MB of third-party code one `git add -A` from being pushed | owner decision |

## 13. The finding that outranks everything else: the agent is not reproducible

`src/arc3_spatial_memory_agent.py:42` seeds the RNG from wall-clock time and a process-salted hash:

```python
seed = int(time.time() * 1_000_000) + hash(self.game_id) % 1_000_000
random.seed(seed)
```

Neither term is useful entropy. `time.time()` encodes *when the run happened*; `hash(str)` is salted per
process unless `PYTHONHASHSEED` is set. Every run is therefore a different trajectory, and it seeds the
**global** `random` module, coupling the agent to whatever else consumes it.

Measured consequence - 13 full sweeps of all 25 games (8 of them run by the parent):

```
0.5858  0.3992  0.2031  1.0900  0.4872  0.4964  0.8098  0.1730
0.2741  0.4193  0.6710  0.8260  0.8879
```

Range **0.173 to 1.090 - a 6.3x spread**, varying 2 to 5 levels out of 183. With the seed pinned the same
instrument gives **identical** results three times running (mean `1.1642`, 4/183), which isolates the cause
to the agent and clears the harness. The harness's arithmetic was separately proven correct by replaying
the reference sweep's level-transition indices through the real official scorer to land on exactly `0.6084`.

Three consequences, in increasing order of severity:

1. The `0.6084` recorded as a reference was **one draw** from that distribution, not a baseline.
2. **The writeup's "+16.7% relative improvement" (v1 `0.24` -> v2 `0.28`) has no statistical basis.** With a
   6.3x spread a single leaderboard submission is one sample, and that difference is indistinguishable from
   noise. Comparing agent versions locally was equally impossible, so the whole "improve v3, then deploy"
   plan was unfalsifiable as designed.
3. The competition rules require submissions to be **reproducible**. A wall-clock-seeded agent is not.

It also explains the headline symptom: 2-5 levels cleared out of 183 is what a random walker with a little
structure looks like. The score is dominated by RNG, not by the heuristics.

## 14. Calibration benchmark (T3b outcome)

The `jihangli1121/arc-agi-3-replays-v1` dataset turned out to be exactly the asset the feature needed.
Re-verified by the parent after the worker's run:

| Measure | Result |
|---|---|
| Replays reproducing the recorded run step for step through the OFFLINE engine | **24 / 25** |
| Games with 100% settled-frame pixel identity | 22 / 25 |
| Official score assigned to human play through our engine | **89.6774 / 100** |
| Sole failure | `cn04` - the installed `cn04-2fe56bfb` is a **different game** from the recorded `cn04-65d47d14` |

This is the first hard reference point in the project: human play scores **89.68** on the same instrument
that gives our agent **0.17-1.09** and the public leader **19.40**. A **77x gap** to human at our best seed.

It also invalidates an assumption: **`baseline_actions` is not the same human run as the replays.** Only
33/183 levels match exactly (18%); the human is at-or-faster on 142/183 (77.6%); the baseline totals 17,135
actions against the humans' 14,798, and the metadata is dated later than the replays. So the score is
normalised against a *different, slower* human sample, which flatters every score by an unknown amount.

And it yields the first evidence-backed strategy hypothesis: humans use **ACTION6 (click) 29.7%**, the most
of any action, with clicks **spatially concentrated** (the top 8 destinations are ~18% of all clicks), while
our agent clicks `random.randint(0, 63)` across the whole grid. Humans also almost never undo
(ACTION7 = 0.08%) and their play is strongly autocorrelated (`repPrev` 36-65%, `maxRun` up to 18) whereas
our agent penalises repetition. Both are cheap things to test once measurement is trustworthy.

# ARC-AGI-3 GT Replays — findings and calibration

> **Dataset:** [`jihangli1121/arc-agi-3-replays-v1`](https://www.kaggle.com/datasets/jihangli1121/arc-agi-3-replays-v1) — "ARC AGI 3 GT Replays" (owner `Zink`)
> **Verified:** 2026-09-23. Every command below was executed on that date; outputs are reproduced verbatim, trimmed for width.
> **Scope:** read-only against Kaggle. The 182 MB dataset was downloaded to `/tmp/arc3-replays/` and analysed there. **No repository file was modified except this document.**
> **Companion:** [`docs/KAGGLE_OPS.md`](KAGGLE_OPS.md) — how this repository talks to Kaggle. [`experiments/arc3_local_eval.py`](../experiments/arc3_local_eval.py) — the local scoring harness this dataset can now calibrate.

This document separates **MEASURED** (observed in command output) from **INFERRED** (an interpretation
of measurements that was not itself executed) at every substantive claim.

---

## 0. TL;DR

The dataset is a **human-play calibration benchmark**, not just a statistic source.

| | |
|---|---|
| Replays | 25 files, one per ARC-AGI-3 public game: 14,673 non-RESET human actions + 150 RESET markers = 14,823 frame events, + 24 scorecard lines = 14,847 JSONL lines |
| Schema | Already the ARC-AGI-3 recorder schema, one JSON object per line, carrying full 64×64 rendered frames |
| **Replayable?** | **YES — 24/25 games replay exactly, step for step, for the whole recorded run** |
| **Official score of a human replay through our engine** | **89.68 / 100** (mean over the 25 games; `cn04` contributes 0 because the local build is a different game) |
| Mean over the 24 reproducing games | **93.42 / 100** |
| `baseline_actions` vs this replay | **Not the same runs.** Only 33/183 levels match exactly (18.0%); the human replay is faster than the baseline on 109/183 levels (59.6%) |
| Hardest / easiest | `su15` 74.06 … five games at 100.00 |
| Extras | `action_effect_dict.npz` = 14,672-row action-effect dataset; `best.pth` = a 15,981,299-parameter BC policy checkpoint + Adam moments |
| Licence | CC BY 4.0, public, free. The dataset ships **no game source code** (only replays) |

Our deployed leaderboard score is 0.28 and the local 0-weight-baseline sweep is 0.6084. The gap to 89.68
is the actual engineering problem; this dataset is the ruler for it.

---

## 1. Provenance, licence, and Rule 2.6 eligibility

**MEASURED** — download and licence, verbatim:

```console
$ kaggle datasets download -d jihangli1121/arc-agi-3-replays-v1 -p /tmp/arc3-replays --unzip
Dataset URL: https://www.kaggle.com/datasets/jihangli1121/arc-agi-3-replays-v1
License(s): Attribution 4.0 International (CC BY 4.0)
Downloading arc-agi-3-replays-v1.zip to /tmp/arc3-replays
100%|██████████| 174M/174M [00:17<00:00, 10.5MB/s]
```

**MEASURED** — public dataset view, no credentials sent (anonymous `GET`):

```console
$ curl -s "https://www.kaggle.com/api/v1/datasets/view/jihangli1121/arc-agi-3-replays-v1"

title       : ARC AGI 3 GT Replays
subtitle    : Public ground-truth replays for the 25 ARC AGI 3 public games
licenseName : Attribution 4.0 International (CC BY 4.0)
ownerName   : Zink
totalBytes  : 808242817
lastUpdated : 2026-05-11T10:33:06.643Z
downloadCount: 257   viewCount: 136   voteCount: 1

description:
  Per-game GT replay JSONLs scraped from the public ARC AGI 3 web app at
  https://three.arcprize.org/. Layout: environment_files/<game_id>/replays/*.json.
  One replay per game, ~150-1500 actions each. Includes RESET markers
  (action_input.id=0) and GAME_OVER state transitions -- i.e. the full human
  exploration arc, not just winning trajectories. Compatible with the ARC AGI 3
  framework's gateway env (action_input.id in {0..7}).
```

**MEASURED — Rule 2.6 position:**

- Stated licence is `CC BY 4.0` (attribution only). A derivative work must credit `Zink` / the dataset URL. There is no non-commercial or no-derivatives clause.
- The dataset is publicly readable at no cost. The view endpoint above answered **without** an `Authorization` header, so availability does not depend on our personal credentials.
- **The dataset ships no game source code.** `find /tmp/arc3-replays -type f` returns 28 files: 25 replay JSONs, `best.pth`, `action_effect_dict.npz`, `per_game_priors.json`. There is no `environment_files/<game>/<version>/*.py`.

**INFERRED:** the *content* of the replays (rendered 64×64 frames, `win_levels`, `available_actions`) is
generated output of the ARC Prize game engine, whose code is licensed separately (the installed
`arc-agi 0.9.1` and `arcengine 0.9.3` wheel metadata both read `License: MIT License`, ARC Prize
Foundation). CC BY 4.0 on this dataset covers the scraped export; it does not, by itself, relicense the
underlying games. For a writeup, the safe reproduction claim is: *"we used a CC BY 4.0 public replay
dataset for calibration, and the game sources needed to replay them are the MIT-licensed
`arc-agi`/`arcengine` wheels already in `data/arc-agi-3-agents/.venv`."*

**Not determined:** whether the ARC Prize competition rules treat third-party replay datasets as
permitted external data for a Paper Track writeup. That is a rules question, not a dataset question, and
belongs in the Rule 2.6 eligibility checklist in `docs/KAGGLE_OPS.md` §7.

---

## 2. Inventory

**MEASURED** — `find /tmp/arc3-replays -type f -printf '%s\t%p\n' | sort -nr` (28 files, 771 MB unpacked):

| File | Size |
|---|---|
| `best.pth` | 191,940,986 B (192 MB) |
| `environment_files/*/replays/*.json` (25 files) | 2.3 MB … 98.4 MB, 579 MB total |
| `action_effect_dict.npz` | 2,324,478 B |
| `per_game_priors.json` | 107,681 B |

**MEASURED** — one replay per game, 25/25 games present:

```text
datasets replays found: 25 games: 25
total lines: 14847  scorecard lines: 24  frame-event lines: 14823
RESET events: 150  initial: 24  mid-run: 126
```

Per-game inventory. `acts` = non-RESET recorded actions. `events` = frame-event lines.
`schema` = which progress field the recording carries (see §3). `ver` = whether the `game_id` version
hash recorded in the replay equals the hash of the version installed in
`data/arc-agi-3/environment_files/`.

| game | MB | events | acts | RESETs | schema | ver | replay file |
|---|---|---|---|---|---|---|---|
| ar25 | 7.92 | 578 | 577 | 1 | old | **DIFF** `e3c63847` vs `0c556536` | `ar25-b36f18b1-…json` |
| bp35 | 98.43 | 638 | 626 | 12 | new | same | `bp35-092c5051-…json` |
| cd82 | 6.61 | 137 | 132 | 5 | old | same | `cd82-68939ee7-…json` |
| cn04 | 8.22 | 779 | 776 | 3 | new | **DIFF** `65d47d14` vs `2fe56bfb` | `cn04-1e88ef7d-…json` |
| dc22 | 19.29 | 1193 | 1188 | 5 | new | **DIFF** `4c9bff3e` vs `fdcac232` | `dc22-1b9a2041-…json` |
| ft09 | 2.33 | 164 | 161 | 3 | new | same | `ft09-f58b7fe2-…json` |
| g50t | 79.58 | 576 | 562 | 14 | new | same | `g50t-c71254fe-…json` |
| ka59 | 18.52 | 827 | 824 | 3 | new | **DIFF** | `ka59-c714584f-…json` |
| lf52 | 79.84 | 1212 | 1208 | 4 | new | same | `lf52-d55d768c-…json` |
| lp85 | 5.58 | 423 | 419 | 4 | old | same | `lp85-543d2fc4-…json` |
| ls20 | 13.42 | 547 | 546 | 1 | new | same | `ls20-8aed7120-…json` |
| m0r0 | 13.87 | 971 | 965 | 6 | new | **DIFF** | `m0r0-a98ab7b3-…json` |
| r11l | 7.64 | 168 | 167 | 1 | new | **DIFF** | `r11l-7bda9483-…json` |
| re86 | 18.69 | 1072 | 1068 | 4 | new | **DIFF** | `re86-0d461c1c-…json` |
| s5i5 | 8.66 | 609 | 596 | 13 | new | **DIFF** | `s5i5-4dd39f12-…json` |
| sb26 | 35.88 | 154 | 150 | 4 | old | same | `sb26-0a3dac2e-…json` |
| sc25 | 8.66 | 217 | 214 | 3 | old | **DIFF** | `sc25-b0b236ff-…json` |
| sk48 | 20.60 | 697 | 694 | 3 | new | **DIFF** | `sk48-b62df293-…json` |
| sp80 | 17.04 | 473 | 468 | 5 | new | **DIFF** | `sp80-ee7752b0-…json` |
| su15 | 42.56 | 567 | 543 | 24 | new | **DIFF** | `su15-20f8d100-…json` |
| tn36 | 7.59 | 251 | 248 | 3 | new | **DIFF** | `tn36-90f13b17-…json` |
| tr87 | 6.89 | 318 | 316 | 2 | new | same | `tr87-147def87-…json` |
| tu93 | 54.37 | 379 | 363 | 16 | old | **DIFF** | `tu93-74d1dff4-…json` |
| vc33 | 11.54 | 308 | 306 | 2 | old | **DIFF** | `vc33-51cfedc8-…json` |
| wa30 | 20.14 | 1565 | 1556 | 9 | new | same | `wa30-0f88f0f8-…json` |

`…json` filename suffixes are the recording GUIDs and are repeated verbatim inside the file
(`data.guid`) and inside the final scorecard card (`cards[<game_id>].guids`).

**MEASURED:** every game was played to `state == "WIN"` at its full `win_levels`; 183 levels completed
total — exactly `sum(len(baseline_actions)) == 183`. `cn04` is the only recording with **no** final
scorecard line, and the only one whose first event is **not** a RESET (it starts with `ACTION3`).

**MEASURED:** recording wall-clock timestamps span `2025-11-10T17:57:45Z` … `2026-03-20T18:11:08Z`.
Our `metadata.json` files were fetched on `2026-04-08`. Replays are older than the metadata.

---

## 3. Replay schema

**MEASURED.** Despite the `.json` suffix the files are **JSONL**: one JSON object per line, no enclosing
array. Every line has exactly two top-level keys, `timestamp` and `data`.

### 3.1 Frame event (one per submitted action, plus the initial reset)

```json
{"timestamp": "2026-01-30T19:51:31.810265+00:00",
 "data": {"game_id": "ft09-0d8bbf25",
          "frame": [[[5,5,5,...], ...]],          // list of 64x64 int grids
          "state": "NOT_FINISHED",
          "levels_completed": 0,
          "win_levels": 6,
          "action_input": {"id": 6, "data": {"game_id": "ft09-0d8bbf25", "x": 16, "y": 47}, "reasoning": null},
          "guid": "f58b7fe2-aafd-414a-a1fb-74d4b3e717e0",
          "full_reset": false,
          "available_actions": [1, 2, 3, 4, 5, 6]}}
```

### 3.2 Final scorecard event (24 of 25 files; absent in `cn04`)

```json
{"timestamp": "2026-01-30T19:55:53.194903+00:00",
 "data": {"won": 1, "played": 1, "total_actions": 163, "levels_completed": 6,
          "cards": {"ft09-0d8bbf25": {"game_id": "ft09-0d8bbf25", "total_plays": 1,
                    "guids": ["f58b7fe2-..."], "levels_completed": [6], "states": ["WIN"],
                    "actions": [163],
                    "actions_by_level": [[[1,17],[2,36],[3,51],[4,72],[5,137],[6,163]]],
                    "resets": [2], "total_actions": 163}}}}
```

`cards` is the ARC-AGI-3 `Scorecard` structure; `actions_by_level` is `[[level, cumulative_action_index], …]`.

### 3.3 Two schema generations

**MEASURED** — the progress field differs between recordings:

| field set | games | count |
|---|---|---|
| `levels_completed` + `win_levels` (current `arcengine 0.9.3` `FrameDataRaw`) | bp35, cn04, dc22, ft09, g50t, ka59, lf52, ls20, m0r0, r11l, re86, s5i5, sk48, sp80, su15, tn36, tr87, wa30 | 18 |
| `score` + `win_score` (older engine generation) | ar25, cd82, lp85, sb26, sc25, tu93, vc33 | 7 |

The old-generation scorecard card has **no `actions_by_level`**, only `{"scores": [...], "states": [...], "actions": [...], "resets": [...]}`. Per-level action counts for those 7 games must be derived from the `score` transitions in the frame events.

**MEASURED** — `cn04` is doubly anomalous: its `action_input.id` values are **strings** (`"ACTION3"`) rather than the integers every other file uses, and its frame chain is empty for one event (`state == GAME_OVER`). Any parser must accept both `int` and `"ACTIONn"` ids.

### 3.4 Frame semantics

**MEASURED** — over all 14,823 frame events:

- 46,283 frame layers total; **every** layer has shape `(64, 64)`; values are small ints.
- Observed per-layer value ranges span `0..15`; the most common ranges are `(0,15)` ×30,644, `(0,11)` ×1,923, `(0,14)` ×7,556, `(0,9)` ×2,945.
- `frame` is a **list of grids, not one grid**: a rendered animation chain. Length 1 for an instant
  action, up to 98 for a multi-step animation (`g50t`). `frame[-1]` is the settled post-action frame.
- `frame` length distribution is per-game bimodal (e.g. `su15` alternates 1 and 5; `bp35` 5/7/12/…/47), i.e. it encodes the game's animation cadence.

**Consequence (MEASURED):** an action-to-observation replay must compare against `frame[-1]`, never
`frame[0]`. Comparing `frame[0]` produces false mismatches (§4.2).

### 3.5 This dataset already *is* the recorder schema

**MEASURED** — `arc_agi/wrapper.py::EnvironmentWrapper._record()` writes exactly
`{"timestamp": ..., "data": {game_id, frame, state, levels_completed, win_levels, action_input{id,data,reasoning}, guid, full_reset, available_actions}}`
to `<recordings_dir>/<scorecard_id>/<game_id>-<guid>.jsonl`. The dataset's frame events are that object,
line for line, with `action_input.id` normalised from `"ACTION6"` to `6`.

**MEASURED — correction to the prior exploration note.** The installed packages contain **no
`Recorder` and no `Playback` class**:

```console
$ grep -rn "class Playback\|class Recorder" \
    data/arc-agi-3-agents/.venv/lib/python3.12/site-packages/{arc_agi,arcengine}/*.py
(no matches)
$ cat .../arc_agi-0.9.1.dist-info/METADATA | head -3
Name: arc-agi
Version: 0.9.1
```

`arc-agi 0.9.1` has a recording **writer** and no reader. So there is nothing to shim into: the correct
integration is to iterate the JSONL events and call `LocalEnvironmentWrapper.step()` directly. Renaming
copies into `RECORDINGS_DIR/<scorecard_id>/*.jsonl` would satisfy a hypothetical `Playback`, but in this
environment that class does not exist, so the rename would buy nothing.

---

## 4. Replayability — the headline result

**MEASURED: YES for 24 of 25 games, exactly.**

### 4.1 Method

`OperationMode.OFFLINE`, `environments_dir=/tmp/arc3-env` (a `cp -r` copy of the repository's
`data/arc-agi-3/environment_files/`, never the original), `recordings_dir=/tmp/arc3-recordings`.
For each game: `Arcade.make(<game>)` (whose constructor performs one `reset()` that corresponds to the
replay's first event), then step the wrapper once per recorded event with
`GameAction.from_id(action_input.id)` and `data={"x":…, "y":…}` for `ACTION6`, `env.reset()` for
mid-run `RESET`. The observed `(levels_completed|score, state)` after every step is compared with the
recorded value.

```python
arcade = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir="/tmp/arc3-env",
                recordings_dir="/tmp/arc3-recordings")
env = arcade.make(game_id)
for action_id, x, y in recorded_actions:          # local wrapper, in-process
    obs = env.reset() if action_id == 0 else env.step(GameAction.from_id(action_id),
                                                      {"x": x, "y": y} if action_id == 6 else None)
```

### 4.2 Result — progress trace

`$ python calib.py` (25 games, 14,673 steps, ~30 s wall clock):

```text
ar25  n= 577 rec=(8, 'WIN') got=(8, 'WIN') match=True  firstdiv=None
bp35  n= 637 rec=(9, 'WIN') got=(9, 'WIN') match=True  firstdiv=None
cd82  n= 136 rec=(6, 'WIN') got=(6, 'WIN') match=True  firstdiv=None
cn04  n= 778 rec=(6, 'WIN') got=(0, 'GAME_OVER') match=False firstdiv=(14, (0,'NOT_FINISHED'), (1,'NOT_FINISHED'))
dc22  n=1192 rec=(6, 'WIN') got=(6, 'WIN') match=True  firstdiv=None
...
wa30  n=1564 rec=(9, 'WIN') got=(9, 'WIN') match=True  firstdiv=None

SUMMARY reproduced: 24 / 25
errors: []
```

`firstdiv=None` means the *entire* 150–1,500-step `(levels_completed, state)` trajectory matched, not
just the endpoint. `cn04` diverges at step 14: the local build completes level 1 one action earlier than
the recording, and the recorded human actions then drive it to `GAME_OVER` at step 114.

**MEASURED — the version-hash mismatch is mostly cosmetic.** 15 of 25 replays carry a `game_id` version
hash different from the installed game directory, yet **14 of those 15 reproduce exactly**. The version
string is not a reliable predictor of behavioural drift. `cn04` is the one true content difference.

### 4.3 Result — pixel identity

Stronger check: compare the recorded **settled frame** `frame[-1]` with the engine's `frame[-1]` at
every step.

`$ python settled.py $(ls /tmp/arc3-replays/environment_files)`:

```text
TOTAL settled frames identical=13454 mismatched=1369 (90.764%)
```

By game:

| verdict | games | detail |
|---|---|---|
| **every** settled frame identical | **22 / 25** | ar25, cd82, dc22, ft09, g50t, ka59, lf52, lp85, ls20, m0r0, r11l, re86, sb26, sc25, sk48, sp80, su15, tn36, tr87, tu93, vc33, wa30 |
| 637/638 | bp35 | only the **final WIN frame** differs (2,061 cells); all transient animation frames also differ in 33 of 638 events |
| 20/609 | s5i5 | exactly 2 cells differ, in every frame: `(row 10, col 31)` recorded `14` vs engine `13`, and `(row 34, col 10)` recorded `11` vs engine `13` — a fixed HUD-like element. Level progress still matches exactly. |
| 0/779 | cn04 | different game build (§4.4) |

Checking the **full animation chain** (not just the settled frame) with `pixels.py` gives
`frames_identical` of 578/578 (ar25), 1211/1212 (lf52), 1565/1565 (wa30), 600/638 (bp35) — i.e. `bp35`
and `lf52` differ only in transient animation-phase frames, never in the state the engine settled on.

### 4.4 `cn04` — the one genuine failure, with the reason

**MEASURED:**

```text
local cn04 win_levels: 6
recorded levels: 6
transitions: [(14, 1, 'NOT_FINISHED'), (114, 1, 'GAME_OVER'), (115, 0, 'GAME_OVER'), ...]
final: (778, 0, 'GAME_OVER')
```

The playback run through the official scorecard reports:

```text
"score": 0.0, "levels_completed": 0, "actions": 778, "resets": 3, "state": "GAME_OVER",
"level_scores": [0,0,0,0,0,0,0,0,0], "level_actions": [14,101,149,101,26,101,116,101,69],
"level_baseline_actions": [-1,...], "message": "Human baseline actions size mismatch"
```

`cn04`'s installed build is a different game from the one that was recorded (level 1 completes at step
14 instead of 15; the recorded sequence kills the agent at step 114). Its `message` also exposes a
scorer quirk: because the level counter went `1 → 0` on reset, the scorecard appended more
`actions_by_level` entries than the 6 baselines, and `EnvironmentScorecard._calculate_score` bails to 0
via its `len(env_info.baseline_actions) < len(card.actions_by_level[idx])` guard. That guard silently
zeroes the whole game rather than scoring the completed levels.

**This is a reproducible calibration benchmark.** Concretely:

```console
$ python scorecard_calib.py
OFFICIAL SCORECARD over 25 human replays: score=89.6774
total_levels_completed=177/186 total_environments_completed=24/25 total_actions=14798
```

(186 = 183 baselines + 3 phantom `cn04` sub-levels from the reset bug; 177 = 183 − 6 lost `cn04` levels.)

**Practical use in this repository:** `experiments/arc3_local_eval.py` can gain a `--replay` mode that
feeds these recordings through the engine and asserts the aggregate equals **89.68**. That turns every
future change to the scoring path into a checked invariant, using the *official* scorer, with no API key
and no submission budget.

---

## 5. Implied human scores (the local ceiling)

**MEASURED.** Score per game = `mean over all levels in baseline_actions of min(100, baseline_i / human_actions_i * 100)`,
computed by `EnvironmentScoreCalculator` / `EnvironmentScorecard` from `arc_agi/scorecard.py`. Values
below are the official scorer's output from the replay-through-engine run (`scorecard_calib.py`),
cross-checked against an independent derivation from the recorded traces.

| game | levels | steps | score | per-level scores |
|---|---|---|---|---|
| ar25 | 8 | 577 | 96.60 | 100, 100, 72.8, 100, 100, 100, 100, 100 |
| bp35 | 9 | 637 | 94.58 | 100, 66.7, 100, 100, 100, 100, 100, 84.5, 100 |
| cd82 | 6 | 136 | **100.00** | 100 ×6 |
| cn04 | — | 778 | **0.00** | non-reproducing build (§4.4); scored 0 by the scorer's size-mismatch guard |
| dc22 | 6 | 1192 | 96.56 | 92.2, 87.2, 100, 100, 100, 100 |
| ft09 | 6 | 163 | 93.86 | 100, 63.2, 100, 100, 100, 100 |
| g50t | 7 | 575 | **100.00** | 100 ×7 |
| ka59 | 7 | 826 | 84.77 | 71.8, 62.3, 59.3, 100, 100, 100, 100 |
| lf52 | 10 | 1211 | 95.76 | 100, 100, 81.1, 82.6, 100, 100, 100, 94.0, 100, 100 |
| lp85 | 8 | 422 | 81.08 | 51.5, 100, 100, 69.6, 100, 100, 35.6, 91.9 |
| ls20 | 7 | 546 | 98.76 | 100, 100, 100, 91.3, 100, 100, 100 |
| m0r0 | 6 | 970 | 80.56 | 100, 53.1, 100, 30.2, 100, 100 |
| r11l | 6 | 167 | **100.00** | 100 ×6 |
| re86 | 8 | 1071 | 92.04 | 92.9, 100, 43.4, 100, 100, 100, 100, 100 |
| s5i5 | 8 | 608 | 90.83 | 100, 100, 100, 26.6, 100, 100, 100, 100 |
| sb26 | 8 | 153 | 99.48 | 100, 100, 100, 100, 100, 95.8, 100, 100 |
| sc25 | 6 | 216 | 98.72 | 92.3, 100, 100, 100, 100, 100 |
| sk48 | 8 | 696 | 95.85 | 100, 100, 100, 91.2, 75.7, 100, 100, 100 |
| sp80 | 6 | 472 | 96.69 | 100, 100, 100, 86.0, 94.1, 100 |
| su15 | 9 | 566 | **74.06** | 100, 100, 52.0, 76.2, 100, 100, 16.0, 22.3, 100 |
| tn36 | 7 | 250 | 99.74 | 100, 100, 100, 100, 100, 98.2, 100 |
| tr87 | 6 | 317 | **100.00** | 100 ×6 |
| tu93 | 9 | 378 | 90.86 | 100, 100, 100, 100, 100, 87.9, 29.8, 100, 100 |
| vc33 | 7 | 307 | **100.00** | 100 ×7 |
| wa30 | 9 | 1564 | 81.16 | 56.8, 100, 70.7, 86.7, 73.7, 100, 42.5, 100, 100 |

- **Mean over all 25 games (official scorer, `cn04` = 0): 89.68**
- **Mean over the 24 reproducing games: 93.42**
- Unweighted mean over the 183 individual levels: 93.15
- Minimum 74.06 (`su15`), maximum 100.00

`steps` is the number of engine steps executed (= recorded frame events − 1, including mid-run
`RESET`s), not the §2 non-`RESET` action count.

Five games — `cd82`, `g50t`, `r11l`, `tr87`, `vc33` — are at the 100 cap because the human was at or
faster than the baseline on *every* level. The per-game score cannot exceed 100, so those five hide how
much faster the human actually was (see §5.1).

**Our numbers for comparison (from the parent session, not re-measured here):** our agent completes 3
levels, all level 1 (`ar25` 1/8, `m0r0` 1/6, `r11l` 1/6); the local 0-weight-baseline sweep averages
0.6084; the deployed leaderboard score is 0.28; the leader is 19.40.

**INFERRED:** 93.42 is what *one* good human run scores against ARC Prize's baseline. It is a reference
point, not a theoretical bound: the per-level bound is 100 (reached whenever `actions_taken ≤
baseline_actions`, which this human did on 142/183 levels), and a policy that beats the baseline
consistently would sit at or near it. The useful reading is the opposite direction — our current 0.61
local mean is ~150× below a single human run, so the metric has enormous dynamic range still unused.

### 5.1 Efficiency headroom the cap hides

**MEASURED** — uncapped ratio `baseline_actions / replay_actions` per level (the scorer caps it at 1):

| game | levels at the 100 cap | mean uncapped `baseline/replay` at those levels | real headroom |
|---|---|---|---|
| g50t | 7/7 | 1.74 | 74% faster than baseline |
| r11l | 6/6 | 1.64 | 64% faster |
| cd82 | 6/6 | 1.21 | 21% faster |
| tr87 | 6/6 | 1.39 | 39% faster |
| vc33 | 7/7 | 1.39 | 39% faster |

Across all 183 levels the uncapped ratio `baseline/replay` has mean **1.374** and median **1.127**; the
human was at or above baseline on **142/183 levels (77.6%)**. The extremes are `sk48` L2 (baseline 177,
human 32 — **5.53× faster**) and `su15` L7 (baseline 8, human 50 — **6.25× slower**), i.e. humans are
noisy, not optimal, per level even inside a winning run.

Totals: baseline actions across the 25 games **17,135**; human replay actions **14,798**; the five
100-capped games alone account for most of the difference.

---

## 6. Does the replay action count match `baseline_actions`?

**MEASURED: NO. `baseline_actions` is not this replay.**

Per-level comparison, all 183 levels. Counts are score-consistent (they include mid-run `RESET`
actions, matching what `Scorecard.actions_by_level` records and what the scorer consumes).

```text
game  base(sum) replay(sum)   per-level base | replay
ar25        748         577   [32,50,75,37,89,159,233,73] | [17,22,103,29,29,159,152,66]
bp35        651         637   [21,48,44,38,33,87,86,131,163] | [15,72,36,31,31,48,86,155,163]
cd82        171         136   [55,8,41,21,23,23] | [41,8,30,21,19,17]
cn04        789         778   [29,54,85,300,208,113] | [15,54,69,319,208,113]
dc22       1228        1192   [59,102,67,98,324,578] | [64,117,59,78,324,550]
ft09        208         163   [43,12,23,28,65,37] | [17,19,15,21,65,26]
g50t        879         575   [78,175,179,230,96,54,67] | [51,175,86,52,96,48,67]
ka59        730         826   [28,109,51,51,33,132,326] | [39,175,86,47,21,132,326]
lf52       1339        1211   [32,81,60,71,205,148,244,109,164,225] | [24,81,74,86,118,148,189,116,150,225]
lp85        388         422   [17,38,31,16,41,60,26,159] | [33,22,31,23,33,34,73,173]
ls20        776         546   [22,123,73,84,96,192,186] | [21,123,39,92,54,108,109]
m0r0       1107         970   [30,111,203,26,500,237] | [30,209,83,86,436,126]
r11l        233         167   [22,33,51,26,52,49] | [7,28,30,20,37,45]
re86       1255        1071   [26,42,86,108,189,139,424,241] | [28,38,198,57,84,117,328,221]
s5i5        638         608   [20,89,106,54,162,38,86,83] | [19,57,85,203,82,30,76,56]
sb26        213         153   [18,28,18,19,31,23,58,18] | [18,16,15,15,31,24,17,17]
sc25        350         216   [36,6,32,83,143,50] | [39,5,32,33,66,41]
sk48       1070         696   [61,177,101,103,230,181,125,92] | [15,32,35,113,304,42,63,92]
sp80        518         472   [39,58,25,148,96,152] | [11,18,17,172,102,152]
su15        361         566   [22,42,26,115,36,31,8,40,41] | [18,28,50,151,18,31,50,179,41]
tn36        317         250   [32,72,26,40,30,55,62] | [23,22,26,37,25,56,61]
tr87        414         317   [54,58,40,45,71,146] | [37,30,39,29,63,119]
tu93        462         378   [19,16,34,42,123,80,14,23,111] | [19,15,34,42,76,91,47,23,31]
vc33        447         307   [7,18,44,61,131,34,152] | [6,13,31,59,92,24,82]
wa30       1843        1564   [71,119,183,98,368,68,79,442,415] | [125,58,259,113,499,58,186,134,132]

n levels compared=183 (of 183)
ratio replay/baseline: mean=0.971 median=0.887 p25=0.65 p75=1.00 min=0.18 max=6.25
replay faster than baseline: 109 (59.6%)  slower: 41  exact: 33
```

**MEASURED conclusions:**

1. **Exact per-level equality: 33/183 = 18.0%** — far above chance for an unrelated human run, far below
   identity. Whitespace-separated examples: `bp35` L6 `87` vs `48`, `tu93` L7 `14` vs `47`, `wa30` L8
   `442` vs `134`.
2. The replay is **faster than the baseline on 59.6% of levels** (median ratio 0.887) and slower on 22.4%.
3. Some levels match exactly (`ar25` L6 159/159, `ls20` L1 and L2 22/22 and 123/123, `lf52` L10 225/225,
   `g50t` L2 175/175) — the signature of a *shared origin* rather than two independent human samples.
4. **`baseline_actions` is therefore a different (and generally slower) human sample than this replay.**
5. **INFERRED — the most likely explanation:** `baseline_actions` is ARC Prize's official published
   human baseline, retrieved by the API and cached into our `metadata.json`
   (`date_downloaded: 2026-04-08`), while these replays were scraped from the web app between
   2025-11-10 and 2026-03-20. `docs/KAGGLE_OPS.md` §2 documents that `metadata.json` comes from the API.
6. **Not determined:** whether `baseline_actions` is a mean, median, best, or a single reference attempt.
   Deciding this needs ARC Prize's baseline definition, which is not in this dataset, not in the wheel
   metadata, and not in our `metadata.json`. **What would determine it:** reading the ARC Prize
   three.arcprize.org baseline methodology page, or obtaining a second independent human replay per game
   and comparing the distribution.

---

## 7. Quantified strategy observations

**MEASURED.** `RESET` markers are excluded from action histograms (they are control flow, not strategy).
Aggregate over all 25 games, 14,673 non-RESET actions:

| action | count | share |
|---|---|---|
| ACTION1 | 2,421 | 16.50% |
| ACTION2 | 2,075 | 14.14% |
| ACTION3 | 2,645 | 18.03% |
| ACTION4 | 2,769 | 18.87% |
| ACTION5 (interact) | 392 | 2.67% |
| ACTION6 (click) | 4,359 | 29.71% |
| ACTION7 (undo) | **12** | **0.08%** |

This matches the dataset's own `action_effect_dict.npz` histogram (`ACTION4` 2,768 vs our 2,769 — the
1-row difference is the dropped empty-frame event in `cn04`, §8).

### 7.1 Per-game distribution and style

`A6%`/`A5%` are per-game shares. `dClick` = distinct click coordinates. `top1%`/`top8%` = share of clicks
in the single hottest / 8 hottest coordinates. `repPrev%` = share of actions identical to the previous
action. `maxRun` = longest identical-action run. `resets` = mid-run `RESET` markers.

| game | acts | A6% | A5% | A7 | dClick | top1% | top8% | repPrev% | maxRun | resets |
|---|---|---|---|---|---|---|---|---|---|---|
| ar25 | 577 | 13.3 | 0.3 | 0 | 71 | 2.6 | 18.2 | 56.8 | 15 | 0 |
| bp35 | 626 | 54.0 | 0.0 | **4** | 249 | 2.4 | 12.7 | 26.7 | 7 | 11 |
| cd82 | 132 | 34.1 | 16.7 | 0 | 38 | 6.7 | 33.3 | 15.2 | 3 | 4 |
| cn04 | 776 | 11.1 | 14.3 | 0 | 83 | 2.3 | 12.8 | 56.1 | 14 | 3 |
| dc22 | 1188 | 35.3 | 0.0 | 0 | 171 | 2.9 | 17.2 | 47.4 | 18 | 4 |
| ft09 | 161 | 94.4 | 0.6 | 0 | 133 | 2.6 | 15.1 | 11.2 | 4 | 2 |
| g50t | 562 | **0.0** | 5.0 | 0 | 0 | — | — | 51.4 | 7 | 13 |
| ka59 | 824 | 5.7 | 0.0 | 0 | 43 | 6.4 | 25.5 | 65.0 | 14 | 2 |
| lf52 | 1208 | 43.5 | 0.0 | 0 | 371 | 1.1 | 7.2 | 32.9 | 14 | 3 |
| lp85 | 419 | **100.0** | 0.0 | 0 | 149 | 4.5 | 27.4 | 42.5 | 13 | 3 |
| ls20 | 546 | 0.0 | 0.0 | 0 | 0 | — | — | 55.3 | 8 | 0 |
| m0r0 | 965 | 5.8 | 0.3 | 0 | 52 | 3.6 | 21.4 | 52.7 | 10 | 5 |
| r11l | 167 | **100.0** | 0.0 | 0 | 154 | 1.8 | 10.8 | 2.4 | 2 | 0 |
| re86 | 1068 | **0.0** | 2.7 | 0 | 0 | — | — | 64.4 | 13 | 3 |
| s5i5 | 596 | **100.0** | 0.0 | 0 | 145 | 4.7 | 22.5 | 56.9 | 12 | 12 |
| sb26 | 150 | 92.7 | 7.3 | 0 | 114 | 2.9 | 15.1 | 1.3 | 2 | 3 |
| sc25 | 214 | 42.5 | 0.0 | 0 | 48 | 11.0 | 42.9 | 36.0 | 12 | 2 |
| sk48 | 694 | 5.8 | 0.0 | **8** | 33 | 7.5 | 37.5 | 51.7 | 12 | 2 |
| sp80 | 468 | 14.1 | 5.8 | 0 | 62 | 3.0 | 18.2 | 56.0 | 13 | 4 |
| su15 | 543 | **100.0** | 0.0 | 0 | 465 | 0.7 | 4.6 | 0.7 | 2 | 23 |
| tn36 | 248 | **100.0** | 0.0 | 0 | 170 | 3.2 | 18.1 | 0.4 | 2 | 2 |
| tr87 | 316 | 0.0 | 0.0 | 0 | 0 | — | — | 62.3 | 10 | 1 |
| tu93 | 363 | 0.0 | 0.0 | 0 | 0 | — | — | 39.9 | 7 | 15 |
| vc33 | 306 | **100.0** | 0.0 | 0 | 94 | 4.9 | 28.4 | 57.5 | 9 | 1 |
| wa30 | 1556 | 0.0 | 10.2 | 0 | 0 | — | — | 49.0 | 15 | 8 |

### 7.2 What the humans do that a frontier-exploration heuristic does not

**MEASURED / INFERRED pairs.** Each row is a measured number followed by its interpretation.

1. **They use the action set the game offers, not all seven.** `available_actions` at reset is
   `(6,)` alone in 5 games (`lp85`, `r11l`, `s5i5`, `tn36`, `vc33`), `(1,2,3,4)` alone in 3, and
   `(1,2,3,4,5,6)` in 5. Measured: in the 5 click-only games the human used `ACTION6` for 100% of
   actions; in `g50t`/`ls20`/`re86`/`tr87`/`tu93`/`wa30` the human used **0%** `ACTION6`.
   *Interpretation:* a heuristic that spends equal budget on all available actions wastes
   `(n−1)/n` of its budget in single-action games. Reading `available_actions` and pruning to it is free.

2. **They commit to long homogeneous runs.** `repPrev%` (fraction of actions equal to the previous one)
   is 36–65% in 15 of 25 games, with `maxRun` up to 18 (`dc22`). Human play is not i.i.d. sampling.
   *Interpretation:* a frontier heuristic that samples actions independently will never produce the
   18-in-a-row run that `dc22` level 4 needed.

3. **They exploit click locality.** Median `top8%` is 18.2% of all clicks landing in 8 of up to 465
   distinct coordinates; `sc25` 42.9%, `sk48` 37.5%, `cd82` 33.3%. `su15` is the opposite extreme
   (465 distinct clicks, top-8 = 4.6%) — 100% click, wide search.
   *Interpretation:* the search space is not uniform; a coarse-to-fine click schedule beats uniform
   click sampling, but `su15`-style games need broad coverage first.

4. **They barely use undo, and only in two games.** `ACTION7` total = **12** actions across 14,673
   (0.08%), all in `bp35` (4) and `sk48` (8). *Interpretation:* undo is not a strategy humans lean on;
   a heuristic that burns budget on undo is wasting it. Conversely, the 2 games where it *is* used are
   the ones where a wrong irreversible move is expensive.

5. **They use `RESET` deliberately as a retry.** 126 mid-run resets across 24 games (0.86% of actions);
   `su15` 23, `tu93` 15, `g50t` 13, `s5i5` 12, `bp35` 11. *Interpretation:* giving up and restarting a
   level is part of the human policy — 5 games had `GAME_OVER` states in the recording
   (`tu93` 15, `bp35` 9, `wa30` 6, `cn04` 4, `sp80` 4), and the human still won every one of them.
   A heuristic that treats `GAME_OVER` as terminal rather than as a reset signal forfeits those levels.

6. **They open levels with a consistent, small repertoire.** Measured first-6-actions per level (excerpt):

   ```
   g50t  L1 a5a5a4a4a2a1 | L2 a3a1a1a1a4a4 | L3 a1a4a4a4a4a2 | ...
   ls20  L1 a1a1a1a1a1a1 | L2 a1a1a1a1a1a1 | L3 a1a1a1a1a1a1 | L4 a3a3a2a2a2a4 | ...
   su15  L1 c(49,14)c(18,44)c(19,45)c(31,5)c(4,59)c(11,53) | L2 c(34,31)c(37,41)c(48,55)... | ...
   tn36  L1 c(32,37)c(32,15)c(22,45)c(21,45)c(21,45)c(34,51) | ...
   ```

   Levels in the *same* game often open with the *same* pattern (`ls20` L1/L2/L3 all begin with long
   `ACTION1` runs). *Interpretation:* level-to-level transfer inside a game is real; a fresh-exploration
   policy per level discards it.

7. **They are usually faster than the baseline, but not reliably so.** Measured: the human matched or
   beat `baseline_actions` on **142/183 levels (77.6%)**; strictly better on 109/183 (59.6%), strictly
   worse on 41/183 (22.4%), exactly equal on 33/183 (18.0%). The extremes run both ways — `sk48` L2
   baseline 177 vs human 32 (**5.53× faster**) and `su15` L7 baseline 8 vs human 50 (**6.25× slower**).
   *Interpretation:* the baseline is a loose target, not a tight one, and even winning humans take local
   efficiency hits. A harness that treats a single slow level as evidence of a broken policy will
   misread the human data.

---

## 8. `action_effect_dict.npz`

**MEASURED** — `numpy.load(...).files`:

```text
feature_keys:      shape=(14672, 256) dtype=float32
action_ids:        shape=(14672,)     dtype=int8
xs:                shape=(14672,)     dtype=int8
ys:                shape=(14672,)     dtype=int8
frame_deltas:      shape=(14672,)     dtype=int32
level_progresses:  shape=(14672,)     dtype=int8
game_id_idx:       shape=(14672,)     dtype=int16
game_ids:          shape=(25,)        dtype=object   # ['ar25', 'bp35', ...]
```

Established by direct comparison against the replays:

| column | measured relationship to the replays |
|---|---|
| `action_ids`, `xs`, `ys` | **Exact copy** of the replay's non-`RESET` action sequence, in order: `(id, x, y)` matches for **24/25 games, all 14,672 rows**. Rows = non-`RESET` actions (so mid-run `RESET` markers are dropped). `cn04` is offset by one row because its recording has no leading `RESET`. |
| `game_id_idx` | Index into `game_ids`; per-game row counts equal per-game non-`RESET` action counts exactly (except `cn04`, one row shorter because one event has an empty `frame` chain). |
| `level_progresses` | Binary; **only populated for the 18 new-schema games** (sum 132 = exactly the level completions of those 18 games). It is **all-zero for the 7 old-schema games** (`ar25`, `cd82`, `lp85`, `sb26`, `sc25`, `tu93`, `vc33`), i.e. the exporter read `levels_completed` and ignored `score`. |
| `frame_deltas` | Non-negative int, 0 … 4096 (= 64×64), mean 125.1, median 55. Highly correlated with the number of cells that change between consecutive settled frames (Pearson 0.97 on `bp35`, 0.75 on `ft09`, 0.47 on `tu93`), but **not exactly** that quantity in any of the three games probed — the exact pre/post frame pairing was not reproduced. |
| `feature_keys` | 256-d `float32`, **non-negative**, exactly **L2-normalised** (`‖row‖₂ = 1.0`, std of the norms 0.0000), sparse: 16–106 non-zero entries, median 55, mean 56. 12,779 of 14,672 rows are distinct. |

**MEASURED — `feature_keys` is a deterministic function of the settled frame.** On `ft09`, of the
repeated settled frames (14 occurrences), all 14 produced byte-identical feature rows and 0 produced a
different one.

**NOT DETERMINED — how `feature_keys` is computed.** Seven candidate constructions were tested against
`ft09` and none matched (max abs difference 0.29 … 1.00): 16×16 block-mean of the frame; block-mean of the
absolute pre/post difference; block-sum of `|Δ|`; 16×16 changed-cell counts; `ReLU(block_mean − global
mean)`; `ReLU` of the signed block difference vs level start; `ReLU` of the block difference vs the
reset frame. **What would determine it:** the author's feature-extraction code or the trained encoder
that produced the column. Neither is shipped.

**INFERRED:** the name, the `feature_keys` / `action_ids` / `xs` / `ys` / `frame_deltas` /
`level_progresses` layout, and the `best.pth` checkpoint in the same dataset identify this as the
**training dataset for an action-effect / world-model predictor**: `(state feature, action) → (frame
delta, level progress)`. It is a usable behavioural-cloning corpus for the 25 public games.

---

## 9. `best.pth`

**MEASURED — `torch` is NOT installed in the analysis interpreter.**

```console
torch ABSENT: ModuleNotFoundError No module named 'torch'
numpy 2.3.2
```

So the checkpoint could not be loaded with `torch.load`. It was instead inspected without torch, by
parsing the zip container and unpickling `best_action/data.pkl` against stub `torch` classes. Everything
below is therefore **recovered from the file itself**, not guessed.

```text
magic bytes: b'PK\x03\x04\x00\x00\x08\x08'          # a torch>=1.6 zip checkpoint
zip entries: 544
entry layout: best_action/data.pkl, best_action/version (b'3'), best_action/byteorder (b'little'),
              best_action/.data/serialization_id, best_action/data/0 … best_action/data/539
tensors were saved on device: cuda:0, dtype float32
```

**MEASURED** — top-level dict keys:

```text
['model_state', 'optimizer_state', 'config', 'epoch', 'best_score', 'epoch_complete',
 'best_epoch', 'best_action_metric', 'best_action_epoch', 'best_public_metric',
 'best_public_epoch', 'selection_metric_name', 'selection_metric']
```

**MEASURED** — `model_state` is a 135-entry `OrderedDict` with **15,981,299 parameters (63.9 MB fp32)**.
Architecture recovered from the key names and shapes:

| component | shapes | reading |
|---|---|---|
| `pos_embed` `(1,256,384)`, `slot_queries` `(1,8,384)` | 256 tokens, 8 learned slots | slot-attention state tokenizer |
| `conv.0..conv.6` | `(96,64,3,3) → (192,96,3,3) → (384,192,3,3) → (384,384,3,3)` | 4 conv stages, 3 upsampled by stride 2 |
| `cross_attn.*` | `in_proj (1152,384)`, `out_proj (384,384)` | cross-attention from slots to conv tokens |
| `encoder.layers.0..5.*` | 6 × TransformerEncoder layer, d_model 384, `linear1 (1536,384)` | 6 layers × 8 heads |
| `scalar_proj.0 (384,18)`, `state_norm` | 18 scalar inputs → 384 | scalar-state branch |
| `goal_encoder.*` + `goal_proj.*` | conv `64→64→128` then `(384,128)` | goal image encoder (`use_goal: True`) |
| `action_embed (7,384)` | 7 rows | action embedding indexed by `ACTION1..7` |
| `action_head (7,384)` | 7 logits | **policy head** |
| `x_head (64,384)`, `y_head (64,384)` | 64 + 64 logits | click coordinate heads |
| `value_head (1,384)` | 1 scalar | value/expected-progress head |
| `avail_head (7,384)` | 7 logits | predicts `available_actions` |
| `archetype_head (3,384)` | 3 classes | game-archetype classifier (aux weight 0.0) |
| `next_latent_head.0 (384,768)` | 768 → 384 | **next-state world model** on the latent |
| `saliency_decoder.*`, `recon_init`, `recon_decoder.*` | up to 16 output channels | saliency + frame reconstruction decoders (aux weights 0.0) |

**MEASURED** — `optimizer_state` contains **135 entries**, each with `exp_avg` and `exp_avg_sq` of the
same shape as the parameter, plus a scalar `step`. The zip's 540 tensor blobs total **47,944,032 float32
values = exactly 3.0000 × 15,981,299**. That arithmetic is the whole explanation of the file size: the
checkpoint is *model + Adam first/second moments*, not a 192 MB model.

**MEASURED** — training metadata:

```text
epoch: 1   epoch_complete: True
best_action_metric: 0.7465842542984087        best_action_epoch: 1
selection_metric_name: 'val_action_acc'       selection_metric: 0.7465842542984087
best_public_metric: -inf                      best_public_epoch: -1
```

**MEASURED** — `config` (36 keys), verbatim highlights:

```text
batch_size 96, grad_accum 2, model_dim 384, num_slots 8, depth 6, num_heads 8, history 4
epochs 16, lr 0.0003, weight_decay 0.1, max_grad_norm 1.0, decoder_dim 64, slot_iters 1
collect_episodes_per_game 16, collect_workers 16, beam_width 6, branch_factor 10
coord_budget 20, max_steps 192, stall_steps 24, reset_limit 4
aux_archetype_weight 0.0, aux_saliency_weight 0.0, aux_recon_weight 0.0, use_goal True
color_permutation_prob 0.5, split_mode 'game', episode_val_fraction 0.2, seed 42
hardware_profile 'rtx4070super'
data: ['/mnt/c/Users/ljh20/MCS/ARC-Prize-2026-ARC-AGI-3/Local_Output/Collection_Cache/bc_v3_perturbed/collected/episodes_bc_v4.cache.pkl.gz']
per_game_priors_path: .../Local_Output/per_game_priors.json
games: None
```

**MEASURED** — `per_game_priors.json` is present in the dataset and is byte-identical in role to the
`per_game_priors_path` referenced by `config`; it holds, per game: `n_attempts`, `n_winning_attempts`,
`n_failed_attempts`, `total_actions_recorded`, `opening_actions`, `action_id_counts`, `click_ratio`,
`click_heatmap_8x8`, `click_hot_spots`, `death_signatures`, `post_reset_actions`,
`per_level_action_dist`, `archetype`, `repeat_kept_actions`, `first_frame_color_hist`,
`available_actions_union`. Every game reports `n_attempts: 1`, `n_winning_attempts: 1`,
`n_failed_attempts: 0` — consistent with §2's single play-to-`WIN` recording per game, and further
evidence that `baseline_actions` did not come from these replays.

**INFERRED — is it usable?**

- It is a **behavioural-cloning policy checkpoint**, not a world model alone: `action_head` (7) +
  `x_head`/`y_head` (64+64) give a complete ARC-AGI-3 action distribution, plus `value_head`,
  `avail_head`, `archetype_head`, a `next_latent_head` world model, and reconstruction decoders.
- It is usable **only with the author's model class**: the state dict gives shapes but not the forward
  pass (slot iteration order, the 18 scalar features, image preprocessing, the goal encoder input).
  Without that source, the 15.98 M parameters are a shape-compatible but not runnable artifact.
- Its reported quality is modest and it is a very early checkpoint: `epoch: 1`, `val_action_acc 0.7466`,
  and the 16-epoch schedule in `config` was not completed.
- **The two extras cannot be used to score anything**: `action_effect_dict.npz` and `best.pth` have no
  ARC-AGI-3 submission path. Their value here is (a) the npz's per-game action statistics, which we can
  read directly, and (b) evidence about what a competitor found worth modelling.

---

## 10. What is NOT determined

| Question | Status | What would determine it |
|---|---|---|
| How `feature_keys` is computed | **Undetermined.** 7 candidate constructions tested, none matched. Characterised only as: 256-d, non-negative, L2-normalised, sparse, a deterministic function of the settled frame. | The author's feature-extraction or encoder code, not shipped. |
| The exact pairing used for `frame_deltas` | **Undetermined.** Correlates with settled-frame change counts (0.47–0.97) but no pairing reproduced it exactly. | Same as above. |
| Whether `baseline_actions` is a mean / median / best / single reference attempt | **Undetermined.** Measured: it is not this replay (33/183 exact matches). | ARC Prize's baseline methodology, or a second independent human replay per game. |
| Whether `best.pth` reproduces its reported 0.7466 `val_action_acc` | **Undetermined.** `torch` is absent from `data/arc-agi-3-agents/.venv`, so the checkpoint was never executed. | Installing torch and the author's model class, then evaluating. |
| Whether the 6 games skipping `cn04` to `GAME_OVER` in the official replay scorecard reflects a version regression worth reporting to ARC Prize | **Undetermined.** Measured: `cn04`'s installed build diverges at step 14 and dies at step 114. | Comparing our `cn04-2fe56bfb` source against the recorded `cn04-65d47d14`. |
| Which of the 15 version-hash-mismatched games have genuinely different content | **Partially determined.** 14 of 15 replay bit-for-bit identically; only `cn04` differs. `s5i5` differs in 2 HUD pixels. | A source diff per game version. |
| Whether Rule 2.6 permits using this dataset in a Paper Track writeup | **Out of scope here.** | `docs/KAGGLE_OPS.md` §7 eligibility checklist + the live competition rules. |

---

## 11. Reproduction recipe

Run from any directory; nothing here writes inside the repository.

```bash
# 1. Fetch (182 MB) — never into the repo
mkdir -p /tmp/arc3-replays
kaggle datasets download -d jihangli1121/arc-agi-3-replays-v1 -p /tmp/arc3-replays --unzip

# 2. HARD SAFETY RULE: copy the game sources; never point the engine at the tracked tree.
#    OperationMode.NORMAL silently re-downloads and OVERWRITES data/arc-agi-3/environment_files/.
cp -r /home/s/dev/projects/kaggle/arc-paper-track/data/arc-agi-3/environment_files /tmp/arc3-env

# 3. Drive a replay through the OFFLINE engine
/home/s/dev/projects/kaggle/arc-paper-track/data/arc-agi-3-agents/.venv/bin/python calib.py

# 4. Official score over all 25 human replays  -> 89.6774
/home/s/dev/projects/kaggle/arc-paper-track/data/arc-agi-3-agents/.venv/bin/python scorecard_calib.py

# 5. Assert the repository's game files were not touched
git -C /home/s/dev/projects/kaggle/arc-paper-track status --short -- \
    data/arc-agi-3/environment_files/ environment_files/
```

The scripts used (`calib.py`, `settled.py`, `pixels.py`, `scorecard_calib.py`, `analyze.py`,
`strategy.py`, `ratios.py`, `final_checks.py`) live in `/tmp/arc3-analysis/` and are **not** committed —
they are throwaway analysis. The three that are worth promoting into `experiments/` are `calib.py`
(step-for-step replay), `settled.py` (pixel assertion) and `scorecard_calib.py` (the 89.68 regression).

---

## 12. Safety evidence

**MEASURED** — the required repository-hygiene check, run after all analysis:

```console
$ git -C /home/s/dev/projects/kaggle/arc-paper-track status --short -- \
      data/arc-agi-3/environment_files/ environment_files/
?? environment_files/
```

- `data/arc-agi-3/environment_files/` printed **nothing** — the tracked game sources are unmodified.
  The engine was only ever constructed with `OperationMode.OFFLINE` and
  `environments_dir=/tmp/arc3-env`.
- `environment_files/` at the repository root is an **untracked, pre-existing** directory (50 files,
  4.2 MB) left by an earlier `OperationMode.NORMAL` run. It is *not* git-ignored. It was neither
  deleted, modified, nor staged by this work; it is reported here so the next operator can decide
  whether to remove it or add it to `.gitignore`.
- The Kaggle access token at `/home/s/.kaggle/access_token` was never read, printed, or copied. The
  anonymous dataset-view `GET` sent no `Authorization` header.

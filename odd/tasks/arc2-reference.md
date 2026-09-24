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
| Public pipeline | `mikelou1/arc-agi2-lb33-89-minimal-perfpatch` (113 votes) — **the canonical one returns 403 on pull** |
| Pullable forks | `manderson240/arc-agi-2-fork-lb33-89-20260903` (we hold it: 1174 lines, `kgmon`/`augment`/`dfs`/`qwen3_4b`/`LoRA`, scored **31.81**, rank 211), `qiuqiuh/arc-highscore-lb3389-replica` (50 votes), `rokaiyasomapti/reproduce-nvarc-2025-results`, `luxluxshan/arc2-nvarc-v1` |
| Base model, offline | `qwen-lm/qwen-3` on Kaggle Models — the **~2.67 GB Qwen3-4B** instance exists, so it can be a `model_sources` entry |
| Runtime budget | ARC-AGI-2 allows **12 h** CPU/GPU; L4x4 machines (96 GB) available |
| Internet | **disabled** at rerun → no `pip install`; the base model must arrive via `model_sources` |
| Submissions | **1/day**, resets 00:00 UTC; `kernels push` does **not** consume one |
| GPU quota | 30 h/week, 28.63 h remaining, refreshes 2026-09-26 |
| Deadline | **2026-11-02** |

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
| R3 | Adapt the reference pipeline to run **offline**: `model_sources` for Qwen3-4B, no `pip install`, `enable_internet: False` | pending | commit-mode run completes |
| R4 | Push and verify in commit mode (free — does not consume a submission) | pending | kernel status + log |
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
| The fork's dependencies (`unsloth`, `transformers` versions) may not exist in the offline image | The rerun fails and the submission is wasted | this feature — verify in commit mode first |
| Per-task LoRA TTT costs real GPU hours | Quota pressure (28.63 h/week) | this feature |
| 12 h runtime cap | The pipeline may time out | measure in commit mode |
| The score lands anywhere in the field's 28–34 band | No distinction, and it is nondeterministic | stated in the paper |
| A fork declared honestly can still be read as padding | Reviewer perception | the paper keeps it to one comparison row |

## 10. Evidence log

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

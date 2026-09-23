# ODD Feature — `kaggle-ops`

> **Status:** in progress
> **Branch:** `feat/kaggle-ops-runbook`
> **Base commit:** `fa49131` (feat(arc3): implement version 3 agent with system 2 bfs frontier planning and deadlock recovery)
> **Created:** 2026-09-23
> **Engram mirror topic key:** `arc-paper-track/odd/kaggle-ops/tasks`

---

## 1. Goal

Make every Kaggle interaction in this repository **explicit, verified and reproducible**, so that no
future session has to reverse-engineer the credentials model, the CLI surface, the writeups API, the
submission limits or the artifact lifecycle again.

Secondary goal: document how to **deliberately consume Kaggle's provided services** (notebooks, GPU/TPU
quota, datasets, models, writeups) instead of discovering them by accident.

## 2. Why now

Reconnaissance on 2026-09-23 established that:

1. The Paper Track submission is a **Writeup**, not a `competitions submissions` entry. The Kaggle CLI
   2.2.4 has **no** writeup commands, and the public `www.kaggle.com/api/v1/writeups/*` paths return 404.
   The working route is `kagglesdk` / `/api/v1/competitions/{name}/hackathon-write-ups`. This single fact
   cost a full investigation cycle.
2. Kaggle artifact state had drifted from the repository: the ARC-AGI-3 and ARC-AGI-2 kernels on Kaggle
   are **older** than the local code, and the most obvious command for checking model versions
   (`kaggle models instances list`) **silently hides every version except the latest** — which made a
   published v1 checkpoint look deleted. There was no documented, trustworthy place to check any of this.
3. Hard constraints were unknown until measured: code freezes **2026-11-02**, the paper is due
   **2026-11-09**, ARC-AGI-2/3 allow **1 submission/day**, and GPU quota is **30 h/week**.

Without a written runbook each of these is rediscovered expensively, and the drift repeats.

## 3. Non-goals

- Not a rewrite of the ARC solvers or the agents (separate feature).
- Not a change to any Kaggle artifact. **This feature is read-only against Kaggle**; it only writes local
  documentation and helper scripts.
- Not a replacement for `PROJECT_DOCUMENTATION.md` (project narrative) — this is operational.
- Does not resolve the paper/evidence narrative gap (paper cites a missing v1 model, no submission kernel
  runs the Laya System 1). That is recorded here as a known risk, to be fixed by its own feature.

## 4. Constraints

| Constraint | Value | Source |
|---|---|---|
| Auth | `/home/s/.kaggle/access_token`, `auth_method: ACCESS_TOKEN`, user `ser8147` | `kaggle config view` |
| Kaggle CLI | 2.2.4 — no writeup commands | `kaggle --help` |
| ARC-AGI-2 / ARC-AGI-3 code deadline | 2026-11-02 23:59 UTC | competition Timeline page |
| Paper Track deadline | 2026-11-09 23:59 UTC | competition Timeline page |
| ARC-AGI-2 / ARC-AGI-3 submissions | 1 / day | `ApiGetCompetition.maxDailySubmissions` |
| Paper Track submissions | 1 total (hackathon rule) | Competition-Specific Rule 2.2.a |
| GPU quota | 30 h/week, 28.63 h remaining, refreshes 2026-09-26 | `kaggle quota` |
| TPU quota | 20 h/week, 0 used | `kaggle quota` |
| Local interpreter | Python 3.14.4, numpy only; `torch`/`transformers`/`arcengine`/`pandas` MISSING | local probe |

## 5. Allowed edit surfaces

- `docs/KAGGLE_OPS.md`, `docs/ENVIRONMENT.md`
- `scripts/kaggle/**`
- `odd/tasks/kaggle-ops.md`
- `README.md` (link only)

No edits to `src/**`, `notebooks/**`, `models/**`, `data/**`, `paper/**`, `submissions/**`.

## 6. Tasks

| # | Task | Status | Evidence |
|---|---|---|---|
| T1 | Scaffold ODD tracking: branch, feature doc, Engram mirror, visible todo | done | branch `feat/kaggle-ops-runbook`; this file |
| T2 | Build `scripts/kaggle/` verified helpers (state dump + writeup access) | done | all subcommands exit 0; see evidence log |
| T3 | Write `docs/KAGGLE_OPS.md` master runbook | pending | review + every command re-executed |
| T4 | Write `docs/ENVIRONMENT.md` (local bootstrap + Kaggle GPU strategy) | pending | local probe output |
| T5 | Independent verification pass and close | pending | `gentle-ai-verify` report |

## 7. Acceptance criteria

1. Every command documented in `docs/KAGGLE_OPS.md` has been executed and its observed output shape
   recorded. No command is documented from memory or inference.
2. A single command produces a complete live inventory of Kaggle artifact state, and it runs.
3. The writeup read path is scripted so the Paper Track submission can be inspected without
   re-deriving the API.
4. Deadlines, submission limits and quota are stated with their source and are checkable.
5. Limitations are explicit: what was NOT verifiable (e.g. logged-in writeup visibility) is marked as such.

## 8. Evidence log

- 2026-09-23 — reconnaissance complete (auth, CLI surface, writeups API, artifact inventory, limits,
  deadlines, quota). Recorded in Engram `#4510` and `#4517`.
- 2026-09-23 — branch `feat/kaggle-ops-runbook` created from `fa49131`.
- 2026-09-23 — `scripts/kaggle/{lib.sh,kstate.sh,kwriteup.sh}` written and executed against the live
  account. Defects found **by running them** and fixed: a malformed f-string in the writeup listing, a
  non-enum `framework` argument, and a nested `trap ... RETURN` that replaced the caller's cleanup trap
  and aborted under `set -u` while leaking scratch files. Verified after the fix: `list`, `get`, `dump`,
  `diff`, `check` all exit 0; `diff` returns 0 on differences; an unknown subcommand returns 1; no
  `/tmp/kwriteup.*` leftovers.
- 2026-09-23 — **Correction to an earlier finding.** An earlier reconnaissance note claimed Kaggle Models
  held only v2 and that the paper's v1 checkpoint no longer existed. That was **wrong**. Versions 1 and 2
  are both published and both return HTTP 200. The misleading signal was `kaggle models instances list`,
  which lists only the latest version. The correct command is
  `kaggle models variations versions list <owner>/<model>/<framework>/<instance>`. This trap is now
  encoded in `kstate.sh` so it cannot recur.

## 9. Known risks carried (not fixed by this feature)

| Risk | Impact | Owner feature |
|---|---|---|
| ~~Paper cites Laya **v1**, which no longer exists in Kaggle Models (only v2)~~ | **RETRACTED** — v1 and v2 are both published and public. The real hazard is that `models instances list` shows only the latest version | platform trap, now encoded in `kstate.sh` |
| No submission kernel runs the Laya System 1 (both are model-free heuristic/DSL) | Completeness / Theory rubric | solver integration |
| Live writeup is stale vs `submissions/KAGGLE_WRITEUP.md`; lacks the required GitHub repo link (Rule 2.8.b) | Eligibility + Accuracy rubric | writeup refresh |
| Pre-existing uncommitted work in `src/arc3_spatial_memory_agent.py`, the ARC-3 kernel notebook, `.gitignore` and untracked `environment_files/` | Not part of this feature | ARC-3 v3 deploy |

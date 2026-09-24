# Feature: arc2026-rules-correction

**Branch:** `feat/arc2026-rules-correction` (base `main` @ `b099e67`)
**Status:** in progress
**Objective link:** `odd/OBJECTIVE.md` §4 criterion **Completeness** and §7 (a wrong number is a false
claim). A deadline is the one number whose error is unrecoverable.

---

## 1. Why

A survey of open Kaggle competitions turned up the authoritative ARC Prize 2026 rules page, and it
contradicts this repository on the project's most important date.

**Source of truth, quoted from `https://arcprize.org/competitions/2026` → Key Dates:**

```text
March 25, 2026       Competition starts
June 30, 2026        ARC-AGI-3 Milestone #1
September 30, 2026   ARC-AGI-3 Milestone #2
November 2, 2026     Submissions due
November 8, 2026     Papers due
December 4, 2026     Results announced
```

**The paper is due 2026-11-08, not 2026-11-09.** Kaggle's own `deadline` field for
`arc-prize-2026-paper-track` reads `2026-11-09T23:59:00` — and `docs/KAGGLE_OPS.md` line 816 asserts that
*"rules page Timeline says 'November 9, 2026'"*, which the rules page does **not** say. The platform field
and the governing rules disagree, and **the rules govern**.

**The error is in the unsafe direction.** `EXECUTIVE-SUMMARY.md` line 38 already recorded the correct date
with the correction noted; nine other sites across six files say Nov 9. Planning to Nov 9 while the true
deadline is Nov 8 means the paper is a day late, and the Paper Track allows **one submission, editable in
place** (Rule 2.2.a) — there is no late path.

## 2. The correction, exhaustively

Every site that states the paper deadline, verified by `grep` on 2026-09-24:

| file | line | currently | must say |
| --- | --- | --- | --- |
| `docs/KAGGLE_OPS.md` | 237 | `2026-11-09T23:59:00.000Z` (in a `kaggle competitions list` dump) | keep the dump verbatim — it is recorded CLI output — but annotate that the rules page supersedes it |
| `docs/KAGGLE_OPS.md` | 816 | `2026-11-09T23:59:00.000Z`, citing the rules Timeline as "November 9, 2026" | **`2026-11-08T23:59:00.000Z`**, citing `https://arcprize.org/competitions/2026` Key Dates; record that Kaggle's field says 11-09 and that the rules page supersedes it |
| `docs/KAGGLE_OPS.md` | 838 | "the paper is due a week later (`2026-11-09`)" | `2026-11-08` |
| `README.md` | 5, 97 | November 9, 2026 / `Nov 09, 2026` | November 8 / `Nov 08` |
| `odd/OBJECTIVE.md` | 6 | `paper due **2026-11-09 23:59 UTC**` | `2026-11-08 23:59 UTC` |
| `odd/OBJECTIVE.md` | 161 | "the paper can continue to Nov 9" | Nov 8 |
| `odd/tasks/kaggle-ops.md` | 33, 53 | `2026-11-09` | `2026-11-08`, with the source named |
| `odd/tasks/paper-pivot.md` | 156 | "before 2026-11-09" | `2026-11-08` |
| `EXECUTIVE-SUMMARY.md` | 3, 19, 38, 297 | Nov 9 / Nov 8 mixed | **Nov 8 everywhere**, and line 38's parenthetical resolved rather than left as an open correction |

## 3. What else the rules page settled, and one prize the repo never recorded

**ARC-AGI-3 prizes total $850K**, and the breakdown is not what the repo implies:

```text
Grand Prize (100%)          $700K   first eligible agent to score 100%; rolls over if unwon
Top Score Award             $ 75K   guaranteed: 40/15/10/5/5 K
Milestone Prizes            $ 75K   guaranteed, and this is the part the repo never mentions
   Milestone #1 (Jun 30)            1st $25K, 2nd $10K, 3rd $2.5K
   Milestone #2 (Sep 30)            1st $25K, 2nd $10K, 3rd $2.5K
```

*"Participants who open source their solutions by the milestone deadlines are eligible for milestone prize
money."*

**Assessed honestly: not actionable.** Our ARC-AGI-3 entry is rank **1729 of 3287 at 0.28**, while the top
of that leaderboard is **19.40 / 7.10 / 7.01**. A top-3 placing needs roughly 7 — twenty-five times our
score. Eligibility is already satisfied (the repository is public), so **no action is required by
2026-09-30**, and this section exists to close the question rather than open a false one.

**Also settled by the rules page, and worth recording because they are eligibility conditions:**

- *"All leading participants are expected to open source their solutions to be eligible for a prize."*
- *"Internet access is not available during Kaggle evaluation (no API-based systems like GPT/Claude/etc.)"* —
  both our submissions comply.
- *"All prizes require reproducible, open-source submissions"*, and solutions must go through the
  designated Kaggle competition per track.

**Not verified and therefore not claimed:** the Paper Track's own prize structure. `odd/OBJECTIVE.md` §3
records *"$450K: top-3 $50K/$20K/$5K + a $375K threshold pool"*; the overview page fetched here shows only
"Paper Prize — Awards for papers that advance our understanding of how to achieve strong performance on
ARC-AGI" with no breakdown. Treat the §3 figure as **unconfirmed** until the paper page is read.

## 4. Tasks

| id | task | status | evidence |
| --- | --- | --- | --- |
| D1 | Correct all nine sites to 2026-11-08 with the source named, and resolve `EXECUTIVE-SUMMARY.md` line 38 | **done** | all nine sites edited across six files; `EXECUTIVE-SUMMARY.md` line 38 now reads "Nov 08, 2026 ─── Papers due (rules page; Kaggle's platform field says Nov 9 and is superseded)"; `docs/KAGGLE_OPS.md` line 237 kept verbatim with an adjacent superseding note; see §7 |
| D2 | Record the ARC-AGI-3 prize breakdown and the milestone assessment in `docs/KAGGLE_OPS.md` | **done** | new section 11 (`$850K` breakdown, the milestone-eligibility quote, the eligibility conditions, and the rank 1729/3287 @ 0.28 vs 19.40/7.10/7.01 assessment recorded as non-actionable) |
| D3 | Mark the Paper Track's prize breakdown as unconfirmed until its page is read | **done** | `odd/OBJECTIVE.md` §3 figure flagged **unconfirmed** at its site, and repeated under `docs/KAGGLE_OPS.md` §10 "What is NOT verified"; the figure was kept, not deleted or asserted |
| D4 | Independent verification: `grep` returns no Nov 9 deadline claim | **done** | `grep -rn` over `*.md`/`*.sh`/`*.py` for the Nov 9 variants: every non-`.venv` hit is the annotated verbatim CLI dump, this document, or an explicit "Kaggle's field disagrees and is superseded" statement |

Verified patterns for D4: `Nov 9`, `Nov 09`, `November 9`, `11-09`, `2026-11-09`.

## 5. Invariants

1. **No site states the paper deadline as Nov 9** except where recording Kaggle's disagreeing field
   verbatim, and that exception must be labelled as superseded.
2. **Every corrected site names its source**: `https://arcprize.org/competitions/2026` Key Dates.
3. The recorded `kaggle competitions list` output at `docs/KAGGLE_OPS.md` line 237 is **kept verbatim** — it
   is evidence of what the CLI returned, not a claim — with an adjacent note that the rules page supersedes
   it. Do not silently rewrite recorded output.

## 6. Non-goals

- No change to any public Kaggle artifact. This is a documentation correction.
- Does not decide whether to pursue any competition; that is the survey's output and the owner's call.
- Does not re-litigate the ARC-AGI-2/3 code freeze of 2026-11-02, which the rules page confirms.

## 7. Commit log

| commit | work unit |
| --- | --- |
| _(uncommitted)_ | `docs(arc2026): correct the Paper Track deadline to 2026-11-08 and record the ARC-AGI-3 prize structure` |

**Status:** D1–D3 implemented and D4 verified; the change is **uncommitted** in the worktree on the
parent's authority (this feature does not commit). Suggested commit subject above; body should note the
nine corrected sites, the `docs/KAGGLE_OPS.md` §11 prize addition, and the `odd/OBJECTIVE.md` §3
unconfirmed marker.

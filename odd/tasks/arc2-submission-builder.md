# Feature: arc2-submission-builder

**Branch:** `feat/arc2-submission-builder` (base `main` @ `2ac687b`)
**Status:** ASB1–ASB4 done (see §4); ASB5 verification and ASB6 re-deploy decision open
**Objective link:** `odd/OBJECTIVE.md` §4 criterion **Completeness**, §7 (the artifact is one of the
surfaces whose claim is currently false), and §13 (no number in the paper without a reproducible source).

---

## 1. Why

The ARC-AGI-2 submission kernel has no generator, and it drifted. Measured 2026-09-24:

```text
notebooks/arc2_submission_kernel/kaggle_arc2_submission.ipynb   cell 2 (the solver): 666 lines, md5 a9acb4eeb2
src/arc2_dual_process_solver.py                                                    746 lines, md5 a3734ca6ee
diff: 111 lines, 11 hunks
```

`odd/tasks/arc-eval-harness.md` line 178 already recorded the consequence as a carried risk: *"Both
ARC-AGI-2 notebooks still inline the PRE-repair solver, so the repair and the trace hook do not reach the
submitted kernel… The fix does not affect the leaderboard until re-deployed."* This feature is the
structural repair of that risk, in the same way `scripts/build_arc3_notebook.py` was for the ARC-AGI-3
notebook.

**And the drift hid two dead paths in the artifact we actually submitted.**

## 2. The two dead paths, and the proof they imply

### 2a. A dead primitive — `rot270`

```python
# DEPLOYED (666-line cell)                          # src/
("rot270", lambda g: [list(x) for x in zip(*g)[::-1]]),   ("rot270", lambda g: [list(x) for x in zip(*g)][::-1]),
```
Executed to confirm: `TypeError: 'zip' object is not subscriptable`. Every `rot270` candidate raised and was
dropped. `src/` carries the comment that names this exact failure mode — *"A primitive that raises on EVERY
task is a dead primitive, and a dead primitive is indistinguishable from a useless one in every metric —
exactly how `rot270` stayed broken and invisible"* — which is why `src/` records first-failure per candidate.

### 2b. NOT a dead path — the `matching_solvers` change is a consistent refactor

The element type of `matching_solvers` changed between revisions, and the call sites changed with it:

```python
# DEPLOYED                                          # src/
matching_solvers: list[Callable] = []               matching_solvers: list[tuple[str, Callable]] = []
...                                                 ...
matching_solvers.append(fn)                         matching_solvers.append((name, fn))
...                                                 ...
attempt_1 = matching_solvers[0](inp)                attempt_1 = matching_solvers[0][1](inp)
```

**Both revisions are internally consistent. There is no dead dispatch, and no coverage branch ever raised.**
An earlier reading of this feature doc claimed otherwise and derived from it a "proof" that the deployed
solver's hidden-set coverage was exactly zero. **That claim was false and is retracted** — see §8 for how it
was found and §2c for the measured value.

### 2c. The measured coverage, and the finding that survives

Forcing `grids_equal` to return `False` and diffing the outputs identifies exactly which tasks reached a
matching candidate: **the deployed solver covered 15 of the 240 hidden test tasks**, and the repaired solver
covers the same 15 — the `rot270` repair does not change coverage, because no covered task's matching
candidate is `rot270`.

**That is the stronger finding, and it is the one the paper should carry: the deployed submission contained
15 tasks with a genuine solver answer, and it scored `0.00`. All fifteen were wrong.** `coverage` means "a
candidate reproduced the *training* pairs", not "the test answer is correct" — which is precisely the
`4.1822%` train / `0.00%` eval pair of `paper-pivot` §3a, now confirmed on the deployed artifact.

The artifact buckets are the visible subset of those 15, not the whole story:

| bucket | count | share | which code path |
| --- | --- | --- | --- |
| echoes the input | 209 | 80.7% | `fallback_1`'s same-dimension branch, or a matching solver whose output coincides with the input |
| zero grid | 46 | 17.8% | `fallback_1`'s `is_const_dim` branch |
| other content | 4 | 1.5% | the 4 covered tasks whose answer differs from both an echo and a zero grid |

The submission cell printed `with 240 tasks` because `len(submission)` is a **key count**.

## 3. The second defect: two divergent copies

```text
notebooks/kaggle_arc2_submission.ipynb                          3 cells [3, 16, 666]   -> no submission writer
notebooks/arc2_submission_kernel/kaggle_arc2_submission.ipynb   4 cells [4, 16, 666, 10]
solver cell: identical in both, 666 lines, md5 a9acb4eeb2
```

`notebooks/arc2_submission_kernel/kernel-metadata.json` resolves `code_file` **relative to its own
directory**, so the pushed notebook is the 4-cell one. No `kernel-metadata.json` references the top-level
copy and only `docs/KAGGLE_LLM_INTEGRATION_GUIDE.md` mentions it by name, as an example. **It is an orphan
that silently duplicates a 666-line solver** — a trap for whoever edits the wrong one.

## 4. Tasks

| id | task | status | evidence |
| --- | --- | --- | --- |
| ASB1 | Write `scripts/build_arc2_notebook.py`: generate all cells from `src/arc2_dual_process_solver.py` + templates, with `--check` | done | `scripts/build_arc2_notebook.py` (stdlib only). `python3 scripts/build_arc2_notebook.py --check` exits 0 on the generated notebook; two consecutive builds leave the notebook at md5 `3202aeefe12beb4d7cfc6f96bce509db` |
| ASB2 | Tests: byte-identity of the solver cell to `src/`, idempotency, `--check` exits non-zero on a hand-edit, the header carries no false provenance | done | `scripts/test_build_arc2_notebook.py` — `python3 -m pytest scripts/test_build_arc2_notebook.py -q` → `10 passed`. Covers the six required cases (byte-identity, idempotency, `--check` 0, `--check` non-zero on a hand-edit, working `rot270`, no key-count success line) plus a header case, a live dispatch-execution case, and a committed-notebook `--check` case |
| ASB3 | Regenerate the notebook; truthful header; and make the submission cell report **coverage**, not a key count | done | Regenerated by the script (never hand-edited): 4 cells, cell 2 = 746 lines, md5 `a3734ca6eeb0738ddc0b1bb1b457e6af` == `src/arc2_dual_process_solver.py`. Local run of the generated cell 3 logic over `data/arc-agi-2/arc-agi_test_challenges.json` (240 tasks, 259 test inputs, write redirected to `/dev/null`, no scoring): `Coverage: 15/240`, `Fallback: 225/240`, `attempt_1 buckets: echo_input=203 zero_grid=40 other=16` |
| ASB4 | Delete the orphan `notebooks/kaggle_arc2_submission.ipynb`; drop the spurious `ser8147/arc-laya-finetune-data` dataset source from the kernel metadata; fix the doc that names it | done | `git rm notebooks/kaggle_arc2_submission.ipynb` (staged deletion, absent from the worktree); `notebooks/arc2_submission_kernel/kernel-metadata.json` now emits `"dataset_sources": []`; `docs/KAGGLE_LLM_INTEGRATION_GUIDE.md` names the surviving path `notebooks/arc2_submission_kernel/kaggle_arc2_submission.ipynb` |
| ASB5 | Independent verification | pending | — |
| ASB6 | Re-deploy / re-submit decision | pending | **owner's**; see §6 |

## 5. Invariants

1. `python3 scripts/build_arc2_notebook.py --check` exits 0 on the committed notebook, and **non-zero on a
   hand-edit**.
2. The solver cell body is **byte-identical** to `src/arc2_dual_process_solver.py`.
3. The header states provenance and **does not claim the live kernel is current**: the version deployed as
   the ARC-AGI-2 submission on 2026-09-23 was the pre-repair revision carrying both dead paths of §2.
4. **The submission cell reports coverage.** It must print how many outputs differ from their input, so the
   artifact cannot again report "240 tasks" while carrying none. This is a deliberate behaviour change and
   is declared as such.
5. The repo notebook is the current source of truth; the deployed version is described, not silently
   redefined.

## 6. The re-deploy question (owner's)

Re-deploying the repaired notebook would change the submitted artifact. Two facts bound it: the
D4 family contributes **exactly zero** to the ablation, so the `rot270` repair cannot move the score; and the
`matching_solvers` repair only matters where coverage exists.

**That second premise was measured wrong.** ASB3's coverage report shows the repaired kernel finds a candidate
reproducing every training pair on **15 of 240** test tasks — not zero. The deployed revision covers the **same
15** (measured in §2c), so the `rot270` repair does not change coverage at all, and the inference *"a
re-submission is expected to score 0.00 again"* is **now untested** and belongs to ASB5/ASB6.
The output buckets also moved: deployed `echo=209 zero=46 other=4` against repaired
`echo=203 zero=40 other=16`. See §8.

A re-submission is still cheap to bound, but it must not be predicted. **Cost: one submission day on the
ARC-AGI-2 track.** Not decided here.

## 7. Non-goals

- No change to the solver's algorithm or its primitive set. `src/` is the source of truth and is not edited.
- No publication and no `kaggle kernels push`. Deploy is ASB6.
- Does not touch `notebooks/arc2_reference_kernel/` — that surface belongs to `paper-pivot` P8.

## 8. Commit log

| commit | subject | scope |
| --- | --- | --- |
| (uncommitted) | `feat(arc2): generate the submission notebook from src/ and report coverage` | `scripts/build_arc2_notebook.py`, `scripts/test_build_arc2_notebook.py`, `notebooks/arc2_submission_kernel/kaggle_arc2_submission.ipynb`, `notebooks/arc2_submission_kernel/kernel-metadata.json`, `notebooks/kaggle_arc2_submission.ipynb` (deletion), `docs/KAGGLE_LLM_INTEGRATION_GUIDE.md`, `odd/tasks/arc2-submission-builder.md` |

Not committed by the implementer: the parent owns the terminal git action (`ASB6`/owner decisions are
separate).

### RETRACTION — the §2b "proof" was false, and it matters more than the original claim

This doc's first revision asserted that the deployed revision called `matching_solvers[0](inp)` on a **tuple**,
and derived from that a "proof" that hidden-set coverage was exactly zero. **Both are false.** The deployed
revision appends the bare `fn` (`matching_solvers.append(fn)`, typed `list[Callable]`) and therefore calls it
correctly; the tuple form exists only in `src/`, which appends `(name, fn)` and unwraps with `[0][1]`. **The two
revisions are internally consistent refactors of each other, and no coverage branch ever raised.** The error came
from reading a truncated side-by-side diff and attributing the `+` lines to the wrong side.

The false proof was replaced by a measurement. Running the deployed solver over all 240 test tasks never raised
— which is consistent with *either* reading and therefore proved nothing. The coverage question was settled by
**forcing `grids_equal` to return `False` and counting the tasks whose output changes: 15 of 240.** The repaired
solver also covers 15.

**The surviving finding is sharper than the invented one: the deployed submission carried 15 genuine solver
answers and scored `0.00` — all fifteen were wrong.** That is the `4.1822%` train / `0.00%` held-out gap of
`paper-pivot` §3a, confirmed on the deployed artifact rather than inferred. And the artifact buckets are the
visible subset of those 15, not a story about a submission made entirely of fallbacks.

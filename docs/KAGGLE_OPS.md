# Kaggle Operations Runbook

> **Account:** `ser8147`
> **Verified:** 2026-09-23 (every command below was executed on that date; outputs are reproduced verbatim, trimmed for width)
> **Scope:** read-only against Kaggle. This document, `scripts/kaggle/kstate.sh`, and `scripts/kaggle/kwriteup.sh` never upload, submit, edit, or delete anything on Kaggle.
> **Companion:** [`docs/ENVIRONMENT.md`](ENVIRONMENT.md) — local interpreter, packages, and Kaggle GPU/TPU strategy.

This repository competes in three ARC Prize 2026 Kaggle competitions:

| Competition | Ref |
|---|---|
| Paper Track | `arc-prize-2026-paper-track` |
| ARC-AGI-2 | `arc-prize-2026-arc-agi-2` |
| ARC-AGI-3 | `arc-prize-2026-arc-agi-3` |

The single most expensive misunderstanding in this repository has been this: **the Paper Track
submission is a Writeup, not a `competitions` submission.** Nothing in the Kaggle CLI 2.2.4 can read
or write a Writeup. Section 7 is the highest-value part of this document.

---

## ⚠️ Safety: mutating commands are forbidden from this repository

Against Kaggle you may **only** run read-only commands. The following families upload, submit, create,
edit, or delete, and must not be run casually — a stray submit on ARC-AGI-2/3 consumes the day's single
submission:

```
kaggle competitions submit          # consumes a submission slot
kaggle kernels push                 # creates/updates a kernel and may run it
kaggle datasets create / version    # creates/uploads a dataset
kaggle datasets delete
kaggle models create / variations versions create
kaggle models variations versions delete
kaggle competitions pages create / update / delete
```

On 2026-09-23, ARC-AGI-2 and ARC-AGI-3 both had **0 submissions remaining today** (`maxDailySubmissions = 1`).
A single accidental `submit` would burn the day. When this document needs to explain what a mutating
command does, it quotes the command's `--help` text and labels it
**"documented from `--help`, not executed"**.

---

## 0. TL;DR — quick reference

Start here. `kstate.sh` answers "what does Kaggle actually hold right now?" in one command.

| I want to… | Command |
|---|---|
| Dump the complete live Kaggle state | `scripts/kaggle/kstate.sh` |
| See who I am / how I authenticate | `kaggle config view` |
| See GPU/TPU quota left this week | `kaggle quota` |
| List my notebooks | `kaggle kernels list --user ser8147 --page-size 30` |
| Check whether a deployed notebook is current | `kaggle kernels status ser8147/<kernel>` |
| See the files a notebook produced | `kaggle kernels files ser8147/<kernel>` |
| Stream a notebook's log | `kaggle kernels logs ser8147/<kernel>` |
| List my datasets | `kaggle datasets list --user ser8147` |
| List the files in a dataset | `kaggle datasets files ser8147/<dataset>` |
| List my models | `kaggle models list --owner ser8147` |
| **See all published versions of a model** | `kaggle models variations versions list ser8147/arc-laya/Transformers/typed-decisions` |
| Check the list of ARC-AGI-2/3 submissions | `kaggle competitions submissions -c arc-prize-2026-arc-agi-2` |
| Check submissions remaining today | `kaggle competitions submission-limits -c arc-prize-2026-arc-agi-2` |
| List Paper Track writeups (owner-scoped) | `scripts/kaggle/kwriteup.sh list` |
| Read the live Paper Track writeup | `scripts/kaggle/kwriteup.sh dump` |
| Diff the live writeup vs the repo draft | `scripts/kaggle/kwriteup.sh diff` |
| Audit writeup eligibility | `scripts/kaggle/kwriteup.sh check` |
| Read a competition rules page | `kaggle competitions pages list --page-name rules --content <competition>` |

Observed shapes:

```
$ kaggle config view
Configuration values from /home/s/.kaggle
- username: ser8147
- auth_method: ACCESS_TOKEN
- path: None
- proxy: None
- competition: None
```

```
$ kaggle quota
resource  used   remaining  total   refreshAt
--------  -----  ---------  ------  -------------------
GPU       1.37h  28.63h     30.00h  2026-09-26T00:00:00
TPU       0.00h  20.00h     20.00h  2026-09-26T00:00:00
```

```
$ kaggle kernels status ser8147/arc-laya-finetune
ser8147/arc-laya-finetune has status "KernelWorkerStatus.COMPLETE"
```

```
$ kaggle datasets status ser8147/arc-laya-finetune-data
ready
```

```
$ kaggle competitions submission-limits -c arc-prize-2026-arc-agi-2
Submissions today: 1
Lifetime submissions: 2
Remaining today: 0
```

---

## 1. Authentication and access model

### Token file

The CLI authenticates from a single file. There is **no `kaggle.json`** in this environment.

```
$ ls -la /home/s/.kaggle/
-rw------- 1 s s 38 Sep 21 11:17 access_token
```

The file holds a single line: an access token. `kaggle config view` reports the scheme as
`auth_method: ACCESS_TOKEN`. `scripts/kaggle/lib.sh` reads it via `kaggle_token()` and uses it as an
`Authorization: Bearer <token>` header against the HTTP API.

> **Never print, echo, or commit the token.** Avoid `kaggle auth print-access-token` — it exists
> (`kaggle auth print-access-token [-h] [--expiration EXPIRATION_DURATION]`) and prints a live
> credential to stdout. `scripts/kaggle/lib.sh` deliberately keeps token values out of every code path
> that produces output.

Preconditions enforced by the helper scripts (`require_kaggle`): `kaggle`, `curl`, and `python3` must be
on `PATH`, and the token file must exist.

### The CLI has no Writeup command at all

```
$ kaggle --help    # top-level commands
{competitions,c,datasets,d,kernels,k,models,m,files,forums,f,benchmarks,b,config,auth,quota}
```

There is no `writeups`, `hackathon`, or `paper` command. The word "writeup" does not appear anywhere in
the top-level command list (`kaggle --help`). The Paper Track submission therefore cannot be read (or
edited) through `kaggle`.

### API hosts

`scripts/kaggle/lib.sh` documents the base as:

```
KAGGLE_API_BASE=https://www.kaggle.com/api/v1
```

Authenticated and anonymous GETs against this base work for the routes this repository uses
(`/competitions/list`, `/competitions/<name>/pages`, `/competitions/<name>/hackathon-write-ups`,
`/kernels/list`, `/datasets/list`, `/models/<owner>/<model>/get`).

> **Disagreement with the reconnaissance note — read this.** An earlier note claimed
> `https://api.kaggle.com/v1/...` returns 404 for these routes while
> `https://www.kaggle.com/api/v1` works. **That was not reproduced on 2026-09-23.** Every route tested
> returned the *same* status code on both hosts, often byte-identical bodies:

| Route | `www.kaggle.com/api/v1` | `api.kaggle.com/v1` |
|---|---|---|
| `/competitions/list?search=arc-prize-2026-paper-track` | 200 | 200 |
| `/competitions/arc-prize-2026-paper-track/hackathon-write-ups?pageSize=5` | 200 | 200 |
| `/competitions/arc-prize-2026-arc-agi-2/leaderboard/view` | 200 | 200 |
| `/models/ser8147/arc-laya/get` | 200 | 200 |
| `/kernels/list?user=ser8147` | 200 | 200 |
| `/datasets/list?user=ser8147` | 200 | 200 |
| `/competitions/arc-prize-2026-arc-agi-2/submissions` | 404 | 404 |
| `/competitions/arc-prize-2026-arc-agi-2/episodes` | 404 | 404 |

`api.kaggle.com/v1` is in fact live: the CLI's own error messages name it
(`... https://api.kaggle.com/v1/kernels.KernelsApiService/GetKernelSessionStatus`). For consistency this
repository keeps `www.kaggle.com/api/v1` as its REST base, but do not repeat the "one host 404s" claim —
it is unsupported.

> **The prefix is what differs, not the host.** Re-verified 2026-09-23 by executing all four:
>
> | URL | Result |
> |---|---|
> | `https://www.kaggle.com/api/v1/competitions/list?search=...` | 200, 2775 bytes |
> | `https://api.kaggle.com/v1/competitions/list?search=...` | 200, 2775 bytes (identical body) |
> | `https://api.kaggle.com/api/v1/competitions/list?search=...` | 404, empty |
> | `https://www.kaggle.com/v1/competitions/list?search=...` | 404 |
>
> Both hosts serve the same REST data, each under exactly one prefix. Separately, `api.kaggle.com/v1`
> also serves the CLI's connect-style endpoints (`POST /v1/<Service>/<Method>`, e.g.
> `models.ModelApiService/ListModels` -> 200, while the same POST on `www.kaggle.com/v1` returns 400).
> There is no functional reason to prefer either host; `lib.sh` pins one so the repository is consistent.

Some intuitive-looking routes **do** 404 on both hosts and should not be reached for:
`/competitions/<name>` (single-competition lookup), `/competitions/<name>/submissions`,
`/competitions/<name>/episodes`, `/writeups/86160`, `/quota`. Competition metadata comes from the
`kagglesdk` client, not a REST path (see section 8).

---

## 2. Artifact inventory and drift detection

**Why it matters.** Kaggle artifact state drifts from the repository silently, and nothing in the repo
can reveal it. On 2026-09-23 the ARC-AGI-3 and ARC-AGI-2 kernels on Kaggle were older than the local
code, and the most obvious model-version command hid a published checkpoint. Two different people could
read the same repository and report two different states of the world.

**The command.** `scripts/kaggle/kstate.sh` dumps everything in one read-only pass, in nine labelled
sections:

1. Identity and auth (`kaggle config view`)
2. Accelerator quota (`kaggle quota`)
3. Deadlines and submission limits (via `kagglesdk`: `deadline`, `maxDailySubmissions`, `maxTeamSize`)
4. Kernels (`kaggle kernels list --user ser8147`)
5. Datasets (`kaggle datasets list --user ser8147`)
6. Models and published versions (`kaggle models list --owner`, then the *correct* version listing)
7. ARC-AGI-2 and ARC-AGI-3 submissions (`kaggle competitions submissions`) and remaining-today
8. (combined with 7) submission limits
9. Paper Track writeups (owner-scoped)

Observed snapshot, 2026-09-23 (trimmed):

```
== 2. Accelerator quota (weekly) ==
   resource  used   remaining  total   refreshAt
   GPU       1.37h  28.63h     30.00h  2026-09-26T00:00:00
   TPU       0.00h  20.00h     20.00h  2026-09-26T00:00:00

== 3. Deadlines and submission limits ==
   competition                       deadline (UTC)         subs/day  max team
   arc-prize-2026-paper-track        2026-11-09T23:59:00.000Z        5         8
   arc-prize-2026-arc-agi-2          2026-11-02T23:59:00.000Z        1         5
   arc-prize-2026-arc-agi-3          2026-11-02T23:59:00.000Z        1         8

== 4. Kernels (notebooks) ==
   ref                                       title                             author    lastRunTime                 totalVotes
   ser8147/arc-laya-finetune                 arc-laya-finetune                 Sergio D  2026-09-23 16:41:42.937000           0
   ser8147/arc-laya-dual-process-submission  arc-laya-dual-process-submission  Sergio D  2026-09-22 23:08:31.650000           0
   ser8147/arc-agi-3-dual-process-agent      arc-agi-3-dual-process-agent      Sergio D  2026-09-22 22:54:10.063000           0
   ser8147/arc-laya-fine-tune                ARC Laya Fine-Tune                Sergio D  2026-09-22 19:55:41.380000           0
   ser8147/fork-of-laya-gate-finetune        Fork of laya-gate-finetune        Sergio D  2026-09-21 15:20:57.387000           0
   ser8147/laya-gate-finetune                laya-gate-finetune                Sergio D  2026-09-21 15:58:52.377000           0

== 6. Models and published versions ==
         version  variation        title                       private
               2  typed-decisions  ARC Laya Decision Engine      False
               1  typed-decisions  ARC Laya Decision Engine      False

== 9. Paper Track writeups (owner-scoped; anonymous sees none) ==
   totalCount=2  returned=2
   - [TEMPLATE] writeupId=57408 writeUpId=72510 state=published
   - [PARTICIPANT] writeupId=86160 writeUpId=114706 state=published
```

**How to read it for drift.** Compare each section against the repo:

| Section | Repo counterpart | Drift signal |
|---|---|---|
| Kernels `lastRunTime` | local notebook/`src` mtime, git HEAD | Kernel older than the cited code ⇒ deployed artifact ≠ repo |
| Model versions | `models/**` checkpoints | A repo checkpoint with no published version is not reproducible by anyone |
| Datasets | `data/`, `submissions/` | Missing version ⇒ training data not reproducible |
| Writeups | `submissions/KAGGLE_WRITEUP.md` | Live writeup ≠ repo draft (see section 7) |

Deployed code is current only once it has been pushed and its `lastRunTime` is after the last relevant
commit. `kstate.sh` is read-only; it does not push anything.

---

## 3. Notebooks / kernels

Command group: `kaggle kernels` (alias `kaggle k`). Read-only subcommands used here: `list`, `status`,
`files`, `logs`, `pull` (downloads into the working tree), `output` (downloads output files), `get`
(alias of `pull`). Mutating: `push`, `update`, `delete`.

### Listing, status, files, logs

```
$ kaggle kernels list --user ser8147 --page-size 30
ref                                       title                             author    lastRunTime                 totalVotes
----------------------------------------  --------------------------------  --------  --------------------------  ----------
ser8147/arc-laya-finetune                 arc-laya-finetune                 Sergio D  2026-09-23 16:41:42.937000           0
...
```

```
$ kaggle kernels status ser8147/arc-laya-finetune
ser8147/arc-laya-finetune has status "KernelWorkerStatus.COMPLETE"
```

Observed statuses on 2026-09-23: the three current kernels (`arc-laya-finetune`,
`arc-laya-dual-process-submission`, `arc-agi-3-dual-process-agent`) are `KernelWorkerStatus.COMPLETE`.
The legacy `arc-laya-fine-tune` and `laya-gate-finetune` are `KernelWorkerStatus.ERROR`. Querying status
for `fork-of-laya-gate-finetune` returns `404 Client Error: Not Found` even though it appears in
`kernels list` — a fork may be listed while not resolvable by the status endpoint.

```
$ kaggle kernels files ser8147/arc-laya-finetune
name                                        size  creationDate
------------------------------------------  ----  ----------------------------------------
benchmark_report.json                        936  4:41 pm, Wednesday 23 September 2026 UTC
config.json                                  922  4:41 pm, Wednesday 23 September 2026 UTC
model.safetensors                            934  4:41 pm, Wednesday 23 September 2026 UTC
...
```

```
$ kaggle kernels logs ser8147/arc-laya-finetune | head -3
[{"stream_name":"stderr","time":6.409935486,"data":"0.00s - Debugger warning: ...\n"}
,{"stream_name":"stderr","time":6.409970637,"data":"0.00s - make the debugger miss breakpoints...\n"}
,{"stream_name":"stdout","time":8.053736765,"data":"Wed Sep 23 16:41:57 2026       \r\n"}
```

`kernels logs` emits newline-delimited JSON objects, each with `stream_name`, `time`, and `data`. The
`-f/--follow` flag streams live logs from a running session — **not executed here** (it would block on a
running kernel; no kernel was running).

`kernels output` downloads a kernel's output files onto disk and `kernels pull` writes the notebook and
metadata into the working tree. Both were **not executed in this repository** to keep the tree clean;
`--help` shapes are captured below.

### Metadata fields

A kernel is described by `kernel-metadata.json`. This repository has three, and they show the fields the
CLI actually uses:

```json
{
  "id": "ser8147/arc-agi-3-dual-process-agent",
  "title": "arc-agi-3-dual-process-agent",
  "code_file": "submission.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": false,
  "enable_gpu": true,
  "enable_tpu": false,
  "enable_internet": false,
  "keywords": [],
  "dataset_sources": [],
  "kernel_sources": [],
  "competition_sources": ["arc-prize-2026-arc-agi-3"],
  "model_sources": []
}
```

| Field | Meaning |
|---|---|
| `id` | `<owner>/<kernel-slug>`. This is the identity — changing it pushes to a different kernel. |
| `title` | Human title. Must be at least five characters (enforced by the push code path). |
| `code_file` | The notebook/script file inside the metadata folder that is uploaded. |
| `kernel_type` | Observed value `notebook`; a notebook/script selector. |
| `is_private` | Visibility. Code-competition submissions must be reachable by the grader. |
| `enable_gpu` / `enable_tpu` | Legacy boolean accelerator switches. **Deprecated** in favour of `machine_shape` (see section 4 of `docs/ENVIRONMENT.md`). |
| `enable_internet` | Must be `false` for a competition run whose rules require offline execution. |
| `dataset_sources` / `model_sources` / `kernel_sources` | Kaggle resources mounted at `/kaggle/input/...`. |
| `competition_sources` | Mounts competition data; required for a code competition that reads its own test set. |

`notebooks/kernel-metadata.json` (repo root of `notebooks/`) points at `ser8147/arc-laya-finetune` with
`code_file: kaggle_laya_arc_train.ipynb`, `is_private: true`, `enable_gpu: true`,
`enable_internet: true`, and both ARC competitions as sources.

### Deploying a kernel (mutating — documented from `--help`, not executed)

```
$ kaggle kernels push --help
usage: kaggle kernels push [-h] [-p FOLDER] [-t TIMEOUT] [--accelerator ACC]
```

- `-p/--path` — folder containing `kernel-metadata.json` (defaults to the current directory).
- `-t/--timeout` — cap the run time in seconds (the global maximum still applies).
- `--accelerator` — accelerator type; **overrides** the `enable_gpu`/`enable_tpu` booleans in the
  metadata. It is sent as `machine_shape`; see `docs/ENVIRONMENT.md` for the supported values.

A push **runs** the kernel and consumes GPU/TPU quota. It is a mutating operation and is out of scope for
this repository's read-only policy.

### How a code-competition submission actually happens

Both ARC tracks are **kernel-only** competitions: the competition metadata reports
`isKernelsSubmissionsOnly: True`. You do not upload a CSV; you submit a *kernel version* whose notebook
writes the required output file. The output filename is set by the competition (observed as
`submission.json` for ARC-AGI-2 and `submission.parquet` for ARC-AGI-3 in the submission table).

The submit command (mutating — documented from `--help`, not executed):

```
$ kaggle competitions submit --help
usage: kaggle competitions submit [-h] [-f FILE_NAME] [-k KERNEL] -m MESSAGE
                                  [-v VERSION] [-q] [--sandbox] [competition]
```

- `-k/--kernel` — the `<owner>/<kernel>` to submit to a code competition.
- `-v/--version` — the kernel version to submit, e.g. `"3"`.
- `-f/--file` — for a code competition, the **name of the output file** the kernel produced (not a path).
- `-m/--message` — required; the submission description shown in the submissions table.

Each call consumes a submission slot against `maxDailySubmissions`. On ARC-AGI-2/3 that cap is **1/day**.

---

## 4. Datasets

Command group: `kaggle datasets` (alias `kaggle d`). Read-only: `list`, `files`, `status`, `metadata`.
Mutating: `create`, `version`, `delete`.

```
$ kaggle datasets list --user ser8147
ref                              title                             size  lastUpdated                 downloadCount  voteCount  usabilityRating
-------------------------------  --------------------------  ----------  --------------------------  -------------  ---------  ---------------
ser8147/arc-laya-finetune-data   ARC Laya Fine-Tune Dataset      156079  2026-09-23 16:38:34.800000              3          0  0.3125
ser8147/laya-gate-finetune-data  laya-gate-finetune-data         256241  2026-09-21 15:58:17.427000              0          0  0.25
```

```
$ kaggle datasets files ser8147/arc-laya-finetune-data
name                        size  creationDate
-----------------------  -------  --------------------------
dataset_v2_summary.json      300  2026-09-23 16:38:35.997000
holdout.jsonl             349600  2026-09-23 16:38:36.015000
holdout_v2.jsonl          598173  2026-09-23 16:38:36.014000
train.jsonl              1405656  2026-09-23 16:38:36.207000
train_v2.jsonl           3470067  2026-09-23 16:38:36.017000
```

```
$ kaggle datasets status ser8147/arc-laya-finetune-data
ready
```

`datasets status` prints the bare status with no trailing newline; `--format json` also includes the
current version number.

### Creating / versioning (mutating — documented from `--help`, not executed)

```
$ kaggle datasets create --help
usage: kaggle datasets create [-h] [-p FOLDER] [-u] [-q] [-t] [-r {skip,zip,tar}] [--ignore-patterns ...]
```

`create` uploads a new dataset from a folder containing a `datasets-metadata.json`; `-u/--public`
creates it publicly (default is private), `-t/--keep-tabular` disables tabular→CSV conversion, and
`-r/--dir-mode` controls directory handling.

```
$ kaggle datasets version --help
usage: kaggle datasets version [-h] -m VERSION_NOTES [-p FOLDER] [-q] [-t] [-r {skip,zip,tar}] [-d] [...]
```

`version` requires `-m`, and `-d/--delete-old-versions` destroys history. Both upload to Kaggle and are
outside this repository's read-only policy.

---

## 5. Models

Command group: `kaggle models` (alias `kaggle m`). Read-only: `list`, `instances list`,
`variations list`, `variations versions list`, `variations versions files`. Mutating: `create`,
`variations versions create`, `variations versions delete`.

### 🪤 The version-listing trap (documented prominently because it already caused a false alarm)

`kaggle models instances list` shows **only the latest version of each instance**. On 2026-09-23 this made
a published v1 checkpoint look deleted:

```
$ kaggle models instances list ser8147/arc-laya
      version  notes                                                                  created                     size
-------------  ---------------------------------------------------------------------  --------------------------  ----------------------
            2  Version 2: Fine-tuned on 1720 ARC cases, 89.92% accuracy, Brier 0.102  {...}                       842613036
```

Only version 2 appears. The correct command lists every version:

```
$ kaggle models variations versions list ser8147/arc-laya/Transformers/typed-decisions
      version  variation        title                       private
-------------  ---------------  ------------------------  ---------
            2  typed-decisions  ARC Laya Decision Engine      False
            1  typed-decisions  ARC Laya Decision Engine      False
```

Both v1 and v2 are published and public. The web pages for both respond `HTTP 200`.

> **Rule:** never conclude that a model version does not exist from `instances list`. Use
> `variations versions list`.

Note the argument shape: `instances list` and `variations versions list` both take
`<owner>/<model>/<framework>/<instance>`. The framework segment is `Transformers` (the
`ModelFramework` enum), and the instance slug here is `typed-decisions`.

`kaggle models list --owner ser8147`:

```
    id  ref               title                     subtitle                                              author
------  ----------------  ------------------------  ----------------------------------------------------  --------
762380  ser8147/arc-laya  ARC Laya Decision Engine  Fast non-autoregressive System 1 decision screening  Sergio D
```

### Model version files

```
$ kaggle models variations versions files ser8147/arc-laya/Transformers/typed-decisions/2
name                        size  creationDate
---------------------  ---------  --------------------------
README.md                   1418  2026-09-23 18:03:58.367000
benchmark_report.json       1581  2026-09-23 18:03:58.367000
model.safetensors      842609220  2026-09-23 18:03:58.367000
rl_agent_config.json         817  2026-09-23 18:03:58.367000
```

The version-suffix argument is `<owner>/<model>/<framework>/<variation>/<version-number>`.

### Creating / versioning (mutating — documented from `--help`, not executed)

```
$ kaggle models create --help
usage: kaggle models create [-h] [-p FOLDER]

$ kaggle models variations versions create --help
usage: kaggle models instances versions create [-h] [-p FOLDER] [-n VERSION_NOTES] [-q]
                                               [-r {skip,zip,tar}] [--ignore-patterns ...] model_instance
```

`models create` reads a `model-metadata.json`; `... versions create` uploads a new version of an
instance and takes `-n/--version-notes`. Both are mutating and outside this repository's policy.

---

## 6. Competition submissions and score checking

Command group: `kaggle competitions` (alias `kaggle c`).

```
$ kaggle competitions submissions -c arc-prize-2026-arc-agi-2
     ref  fileName         date                        description                                                           status                     publicScore  privateScore
--------  ---------------  --------------------------  --------------------------------------------------------------------  -------------------------  -----------  ------------
56477009  submission.json  2026-09-23 00:09:07.723000  Dual-Process Solver v2 (System 2 Bounded Symbolic Program Synthesis)  SubmissionStatus.COMPLETE  0.00
56474960  submission.json  2026-09-22 22:11:11.300000                                                                        SubmissionStatus.COMPLETE  0.00
```

```
$ kaggle competitions submissions -c arc-prize-2026-arc-agi-3
     ref  fileName            date                        description                                                               status                     publicScore  privateScore
--------  ------------------  --------------------------  ------------------------------------------------------------------------  -------------------------  -----------  ------------
56477006  submission.parquet  2026-09-23 00:09:05.667000  Dual-Process Agent v2 (In-Episode Spatial Memory & Frontier Exploration)  SubmissionStatus.COMPLETE  0.28
56474962  submission.parquet  2026-09-22 22:11:16.410000                                                                            SubmissionStatus.COMPLETE  0.24
```

The `ref` column is the submission **id**, which is what `competitions episodes` takes as input.

### Submission limits

```
$ kaggle competitions submission-limits -c arc-prize-2026-arc-agi-2
Submissions today: 1
Lifetime submissions: 2
Remaining today: 0
```

`--json` emits `{"numToday": 1, "numTotal": 2}`.

### ARC-AGI-3 episodes, replay, and logs

ARC-AGI-3 is an interactive competition, so it exposes per-episode tooling:

```
$ kaggle competitions episodes --help
usage: kaggle competitions episodes [-h] ... submission_id     # find ids via `competitions submissions`

$ kaggle competitions replay --help
usage: kaggle competitions replay [-h] [-p PATH] [-q] episode_id

$ kaggle competitions logs --help
usage: kaggle competitions logs [-h] [-p PATH] [-q] episode_id agent_index   # agent_index is 0-based
```

Observed on 2026-09-23:

```
$ kaggle competitions episodes 56477006
No episodes found
```

```
$ kaggle competitions episodes 56474962
No episodes found
```

Both ARC-AGI-3 submissions exist and are `COMPLETE` with public scores, yet **no episodes are currently
returned** for either. Treat "No episodes found" as "none exposed right now", not as "the submission did
not run". `replay` and `logs` were **not executed** because they download files; their `--help` shape is
above, and both require an `episode_id` that is not currently obtainable.

---

## 7. The Paper Track writeup (highest-value section)

### Why it is not a `competitions` submission

The Paper Track is a **hackathon**. Its submission is a *Writeup* — a document plus a media gallery and
attached public notebooks. It never appears in `kaggle competitions submissions`:

```
$ kaggle competitions submissions -c arc-prize-2026-paper-track
No submissions found
```

**That output is not evidence of no submission.** It is the normal, expected result. The same competition
reports platform limits as if a normal submission existed:

```
$ kaggle competitions submission-limits -c arc-prize-2026-paper-track
Submissions today: 0
Lifetime submissions: 0
Remaining today: 5
```

The real submission is the writeup. Check it with `scripts/kaggle/kwriteup.sh list`, never with
`competitions submissions`.

### The working API route

```
kaggle --help      # no writeup command exists
```

`https://www.kaggle.com/api/v1/writeups/<id>` returns **404**. The route that works is:

```
GET /api/v1/competitions/<competition>/hackathon-write-ups?pageSize=50   # list (owner-scoped)
GET /api/v1/competitions/<competition>/hackathon-write-ups/<writeupId>   # single writeup
Authorization: Bearer <token>
```

`scripts/kaggle/lib.sh` wraps both as `writeups_list()` and `writeup_get()`; `scripts/kaggle/kwriteup.sh`
builds the operator surface on top (`list`, `get`, `dump`, `diff`, `check`).

Observed identity, 2026-09-23:

| Field | Value |
|---|---|
| Participant hackathonWriteUp id | `86160` |
| Participant `writeUp.id` | `114706` |
| Title | `Dual-Process Heuristic Gating for ARC-AGI` |
| `contentState` | `published` |
| `publishTime` | `2026-09-22T21:48:18.897Z` |
| `updateTime` | `2026-09-22T22:00:48.7493602Z` |
| Track id | `[346]` |
| URL | `https://www.kaggle.com/competitions/arc-prize-2026-paper-track/writeups/dual-process-heuristic-gating-for-arc-agi` |
| Host template hackathonWriteUp id | `57408` |

`writeups_list` returns **`totalCount: 2`**: the participant writeup (`86160`) and the host's sample
template (`57408`, "Less is More: Recursive Reasoning with Tiny Networks"). There is no third writeup,
and there must never be: see the one-submission rule below.

```
$ scripts/kaggle/kwriteup.sh list
totalCount=2  returned=2
  [TEMPLATE] writeupId=57408 messageId=72510 state=published
             title       = 'Less is More: Recursive Reasoning with Tiny Networks'
             url         = https://www.kaggle.com/competitions/arc-prize-2026-paper-track/writeups/arc-prize-2026-sample-writeup
             publishTime = 2026-03-24T20:36:10.153Z
  [PARTICIPANT] writeupId=86160 messageId=114706 state=published
             title       = 'Dual-Process Heuristic Gating for ARC-AGI'
             url         = https://www.kaggle.com/competitions/arc-prize-2026-paper-track/writeups/dual-process-heuristic-gating-for-arc-agi
             publishTime = 2026-09-22T21:48:18.897Z
```

### Owner-scoped vs anonymous visibility

Participant writeups are **not anonymously readable** during the competition. Observed:

| Request | Result |
|---|---|
| Anonymous GET of the participant writeup URL | HTTP **404** |
| Anonymous GET of the host template URL | HTTP **200** |
| Anonymous API listing (`hackathon-write-ups`) | `totalCount` absent, **0** rows |
| Authenticated API listing | `totalCount: 2`, 2 rows |
| Anonymous API GET of writeup `86160` | HTTP **401** |
| Authenticated API GET of writeup `86160` | HTTP **200** |

> **A 404 on the writeup URL does not mean the writeup is unpublished.** The authoritative field is
> `contentState` (`published` here), plus a non-null `publishTime`, both readable only with the token.

### Editing rules

- **One submission, total.** Competition-Specific Rule `2.2.a` (rendered as
  `###2. COMPETITION-SPECIFIC RULES` → `####2. SUBMISSION LIMITS` → `a`) states: *"For Hackathons, each
  Team may submit one (1) Submission only."* The same section adds that for hackathons each team is
  allowed one submission and that pre-merge submissions are unsubmitted. The platform would allow 5/day,
  but the competition rule is binding.
- Therefore the writeup must be **edited in place** — never delete and re-create it, and never create a
  second one. `scripts/kaggle/kwriteup.sh` cannot and does not edit; editing is done in the Kaggle web UI
  (the "Submit" button appears in the top-right after the writeup is saved, per the Submission
  Requirements page).
- A writeup must have a **Track selected** to be submittable. That is why the API exposes
  `hackathonTrackIds` (observed `[346]`) — an empty value means it cannot be submitted.

### Eligibility requirements (from the live rules and Submission Requirements pages)

`kaggle competitions pages list --page-name rules --content arc-prize-2026-paper-track` and
`--page-name "Submission Requirements"` produce the following, verified 2026-09-23:

| Requirement | Source (as rendered on the page) |
|---|---|
| Word limit: **1,500 words**; over-limit may be penalised | Submission Requirements, §1 |
| **Cover image required** | Submission Requirements, §1.a (Media Gallery) |
| **Public notebook** attached in Project Links; must not require login or paywall | Submission Requirements, §1.b |
| Winner license **CC-BY-4.0** | Competition-Specific `2.5.a` (WINNER LICENSE → Open Source) |
| Winner must deliver code + docs capable of reproducing the submission | Competition-Specific `2.8.a` (WINNER'S OBLIGATIONS) |
| Winning description must link a **code repository with reproduction instructions** | Competition-Specific `2.5.b`; also referenced by `2.8` |
| Six-criterion rubric, each 0–5, averaged | Evaluation page |
| No tiebreakers in hackathons | General `3.7.b` (DETERMINING WINNERS) |
| Winner notified by email; must respond within one week | General `3.8.b` (NOTIFICATION) |
| Entry/team-merge deadline | competition metadata `newEntrantDeadline` (section 8) |

> **Rule-numbering note.** The rendered rules page does **not** print the strings "2.2.a", "2.5.a",
> "3.7.b" or "3.8.b". Those labels are the repository's shorthand for `§ <section>.<rule>.<paragraph>`.
> The substance was verified directly against the page text quoted above; the literal labels are a
> convention, not a quotation.
>
> **Nuance on the repository link.** The Submission Requirements page calls the Project Link *"(Optional)"*,
> while the winner-obligation text says the description *"should also include a link to a code repository
> with complete and detailed instructions so that the results obtained can be reproduced."* Treat the link
> as optional to *submit* but required to *win*.

### Compliance checklist

Run `scripts/kaggle/kwriteup.sh check`. Observed output, 2026-09-23:

```
$ scripts/kaggle/kwriteup.sh check
== Eligibility and integrity audit -- writeup 86160 ==
  [PASS] word count              849 / 1500   (Submission Requirements page)
  [PASS] contentState            published
  [PASS] published               2026-09-22T21:48:18.897Z
  [PASS] track selected          [346]   (required in order to submit)
  [PASS] cover image             /writeups/114706/images/cover
  [PASS] kaggle license field    Attribution 4.0 International (CC BY 4.0)
  [FAIL] body license text       states CC-BY: False   (Competition-Specific 2.5.a requires CC-BY-4.0)
  [PASS] public notebooks        2 of 2 attached are public
  [FAIL] repository project link MISSING - Competition-Specific 2.5.b (winner obligation: repo link + reproduction steps)
  [FAIL] repository in body      False

  [INFO] Competition-Specific 2.2.a: a hackathon team may submit ONE submission only.
         Edit this writeup in place. Never create a second one.

  [INFO] anonymous GET https://www.kaggle.com/competitions/arc-prize-2026-paper-track/writeups/dual-process-heuristic-gating-for-arc-agi
         -> HTTP 404   (404 means the public cannot read it)

  Attached project links:
    - kernels     private=False upvotes=0  'ARC-AGI-2 Dual-Process Submission Notebook'
      https://www.kaggle.com/code/ser8147/arc-laya-dual-process-submission
    - kernels     private=False upvotes=0  'ARC-AGI-3 Dual-Process Interactive Agent'
      https://www.kaggle.com/code/ser8147/arc-agi-3-dual-process-agent
    - datasets    private=False upvotes=0  'ARC Laya Fine-Tune Dataset'
      https://kaggle.com/datasets/ser8147/arc-laya-finetune-data
    - unspecified private=None upvotes=0  ''
      /writeup-links/195991/images/cropped
```

Three FAILs are open on the live writeup:

1. Body text does not state CC-BY (the Kaggle license *field* is correct; the prose still says
   "Permissive Public Domain (MIT / CC0)").
2. No GitHub repository link attached as a project link.
3. No `github.com` reference in the body.

**The live writeup is also stale relative to the repository.** `kwriteup.sh diff` compares the live
markdown against `submissions/KAGGLE_WRITEUP.md`, and they differ substantially: the local draft already
contains the GitHub repository link, the ARC-AGI-3 leaderboard score, and additional structure that the
live writeup lacks. What the judges currently read is **not** what the repo holds.

```
$ scripts/kaggle/kwriteup.sh diff | head -8
== live writeup 86160  vs  submissions/KAGGLE_WRITEUP.md ==
--- .../local.md
+++ .../live.md
@@ -1,27 +1,21 @@
 ...
+**Author:** Sergio Alberto Dominguez ([@ser8147](https://www.kaggle.com/ser8147))
+**Affiliation:** Independent Researcher
+**Kaggle Code Submission Reference:** [ser8147/arc-laya-dual-process-submission](...)
+**License:** Permissive Public Domain (MIT / CC0)
```

### Reading the rules without a browser

```
$ kaggle competitions pages list --page-name rules --content arc-prize-2026-paper-track
```

The same content is available as structured JSON at `GET /api/v1/competitions/<competition>/pages`
(each page object carries `name`, `postTitle`, and `content`). This is how the rule text in this section
was verified.

---

## 8. Deadlines, submission limits and quotas

| Item | Value | Source (command + field) |
|---|---|---|
| ARC-AGI-2 final submission deadline | `2026-11-02T23:59:00.000Z` | `kagglesdk` `ApiGetCompetition.deadline` |
| ARC-AGI-3 final submission deadline | `2026-11-02T23:59:00.000Z` | `kagglesdk` `ApiGetCompetition.deadline` |
| Paper Track final submission deadline | `2026-11-09T23:59:00.000Z` | `kagglesdk` `ApiGetCompetition.deadline`; rules page Timeline says "November 9, 2026" |
| ARC-AGI-2 entry / team-merger deadline | `2026-10-26T23:59:00.000Z` | `kagglesdk` `ApiGetCompetition.newEntrantDeadline` |
| ARC-AGI-3 entry / team-merger deadline | `2026-10-26T11:59:00.000Z` | `kagglesdk` `ApiGetCompetition.newEntrantDeadline` |
| Paper Track entry deadline | none exposed | `newEntrantDeadline` absent for the Paper Track |
| ARC-AGI-2 submissions | 1/day (`kaggle competitions submission-limits` → Remaining today: 0) | `maxDailySubmissions` + live limits |
| ARC-AGI-3 submissions | 1/day (Remaining today: 0) | `maxDailySubmissions` + live limits |
| Paper Track platform limit | 5/day shown (`Remaining today: 5`) | `maxDailySubmissions` |
| **Paper Track binding limit** | **1 submission total** | Competition-Specific Rule `2.2.a` |
| ARC-AGI-2 max team size | 5 | `maxTeamSize` |
| ARC-AGI-3 max team size | 8 | `maxTeamSize` |
| Paper Track max team size | 8 | `maxTeamSize` |
| GPU quota | 30 h/week; 1.37 h used; 28.63 h remaining; refreshes `2026-09-26T00:00:00` | `kaggle quota` |
| TPU quota | 20 h/week; 0 used; refreshes `2026-09-26T00:00:00` | `kaggle quota` |
| ARC-AGI-2 entries (teams) | 2165 | `kagglesdk` `ApiGetCompetition.teamCount` |
| ARC-AGI-3 entries (teams) | 3274 | `teamCount` |
| Paper Track entries (teams) | 202 | `teamCount`; `README.md` says "~199 (as of Sep 22, 2026)" — expected, it grew |

`newEntrantDeadline` is how the entry/team-merger deadline is exposed; the Paper Track has no such field.
Note the ARC-AGI-3 entry deadline is `11:59:00Z`, not `23:59:00Z`. Publishing the day without the time
would be wrong by 12 hours.

Since all three deadlines are in **November 2026**, code must be frozen on ARC-AGI-2/3 first
(`2026-11-02`), and the paper is due a week later (`2026-11-09`).

---

## 9. Known traps and gotchas

Each entry is **symptom → cause → correct command**.

### 9.1 "No submissions found" on the Paper Track

- **Symptom:** `kaggle competitions submissions -c arc-prize-2026-paper-track` prints `No submissions found`.
- **Cause:** the Paper Track submission is a Writeup, which never appears in the `competitions submissions` listing.
- **Correct command:** `scripts/kaggle/kwriteup.sh list`

### 9.2 A published model version looks deleted

- **Symptom:** `kaggle models instances list ser8147/arc-laya` shows only one version.
- **Cause:** that command lists only the **latest** version of each instance and silently hides older published versions.
- **Correct command:** `kaggle models variations versions list ser8147/arc-laya/Transformers/typed-decisions`

### 9.3 Writeups have no CLI, and the obvious API path 404s

- **Symptom:** no writeup command in `kaggle --help`; `GET /api/v1/writeups/86160` returns 404.
- **Cause:** the CLI never implemented writeups, and `/writeups/*` is not the route.
- **Correct route:** `GET /api/v1/competitions/<competition>/hackathon-write-ups` (wrapped by
  `scripts/kaggle/kwriteup.sh`).

### 9.4 `framework must be of type ModelFramework`

- **Symptom:** `TypeError: framework must be of type ModelFramework` when calling a `kagglesdk` model
  version request with `framework="transformers"`.
- **Cause:** the SDK setter validates against the `ModelFramework` enum, not a string. The enum member is
  `ModelFramework.MODEL_FRAMEWORK_TRANSFORMERS` (there is no `.TRANSFORMERS` attribute).
- **Correct usage:**
  ```python
  from kagglesdk.models.types.model_enums import ModelFramework
  req.framework = ModelFramework.MODEL_FRAMEWORK_TRANSFORMERS
  ```

### 9.5 A 404 on the writeup URL looks like "unpublished"

- **Symptom:** anonymous GET of the participant writeup URL returns HTTP 404; the listing is empty.
- **Cause:** participant writeups are not anonymously readable during the competition.
- **Correct check:** read `contentState` and `publishTime` via the authenticated API
  (`scripts/kaggle/kwriteup.sh get`).

### 9.6 Creating a second writeup would be a rules violation

- **Symptom:** the platform shows "Remaining today: 5" and invites a new submission.
- **Cause:** the platform's `maxDailySubmissions = 5` is not the binding rule. Competition-Specific Rule
  `2.2.a` allows a hackathon team **one submission total**.
- **Correct action:** edit writeup `86160` in place; never create a new one. `kwriteup.sh check` prints
  this reminder every run.

### 9.7 The live writeup would fail the winner-obligation check

- **Symptom:** `kwriteup.sh check` reports the repository link MISSING and the body CC-BY text `False`.
- **Cause:** the live writeup is an older revision than `submissions/KAGGLE_WRITEUP.md`; the repo draft has
  the link, the live one does not.
- **Correct action:** `scripts/kaggle/kwriteup.sh diff` to see the gap, then update the writeup in the
  Kaggle UI (one submission — edit in place).

### 9.8 Tie-break behaviour for edit-early vs edit-late

- **Symptom:** fear that editing a writeup after the first submission forfeits a tie-break.
- **Cause:** the two rule sources appear to disagree, and nobody has reconciled them:
  - **General `3.7.b`** (rendered `###3. GENERAL COMPETITION RULES` → `####7. DETERMINING WINNERS` → `b`):
    *"For Hackathon Competitions, each of the top Submissions will get a unique ranking and there will be
    no tiebreakers."*
  - **The Paper Track Evaluation page** says: *"In the event of a tie, the Paper that was entered first to
    the Competition will be the winner."*
- **Working assumption (the repository's, from the reconnaissance note):** for a hackathon there are no
  tiebreakers, so an early writeup submission confers no tie-break advantage and editing it carries no
  tie-break risk.
- **Caveat:** the Evaluation-page sentence is real and quotes were verified on 2026-09-23. The conflict
  with General `3.7.b` is **unresolved** and is listed in section 10. Do not assert either side as settled
  without a host clarification.
- **Why it does not gate the decision either way:** Competition-Specific `2.2.a` allows a hackathon team
  only **one** submission, so editing the existing writeup is the only mechanism that exists for improving
  it. The tie-break question therefore cannot be allowed to block the update — the real choice is "edit" or
  "never improve". Two further independent reasons it is safe: the General-rules carve-out above, and the
  fact that "entered first" refers to the Submission entry time (`createTime` `2026-09-22T21:23:05Z`),
  which a content edit does not change.
- **Correct action:** treat the tie-break question as open; it does not change the binding constraints
  (one submission total, edit in place).

### 9.9 `api.kaggle.com/v1` does not 404 where `www.kaggle.com/api/v1` works

- **Symptom:** an earlier note said the reverse.
- **Cause:** the claim was never reproduced (see section 1).
- **Correct action:** use `KAGGLE_API_BASE=https://www.kaggle.com/api/v1` for consistency, but do not
  repeat the false claim about the other host.

### 9.10 There is no per-command `--user` for authentication

- **Symptom:** `kaggle config view` reports `username: ser8147` and you assume the CLI can be pointed at
  another account per command.
- **Cause:** there is no per-command account flag; the token file is the only identity. Unset `path`/`proxy`
  mean defaults apply.
- **Correct action:** confirm identity with `kaggle config view`; do not attempt to set a different user
  with `kaggle config set` (it only supports `competition`, `path`, `proxy`).

---

## 10. What is NOT verified

Be explicit about the edges of this document. The following were **not** observed and must not be treated
as established:

- **Mutating commands.** `kernels push`, `datasets create/version`, `models create`/`versions create`,
  `competitions submit`, `competitions pages create/update/delete`, and `kernels output`/`pull` were not
  executed. Their descriptions come from `--help` text and the CLI's Python source, and are labelled as
  such. No upload, submit, create, edit, or delete was performed against Kaggle.
- **`kaggle auth print-access-token`.** Runs were deliberately avoided; it prints a live credential.
- **`kaggle kernels logs -f`.** Not executed (no running kernel; it would block).
- **`kaggle competitions replay` / `competitions logs`.** Not executed (they download files, and their
  required `episode_id` is not currently obtainable — `competitions episodes` returns "No episodes found").
- **A second Kaggle account.** Every host/route test used the `ser8147` token and the anonymous,
  unauthenticated path. Behaviour for other accounts is untested.
- **The writeup edit/submit API.** No POST/PUT route was exercised. Editing writeups is done in the
  Kaggle web UI; whether a supported write API exists is unresolved here.
- **Exact accelerator enum values accepted by the live API.** The SDK docstring lists
  `NvidiaTeslaT4`, `NvidiaTeslaP100`, `Tpu1VmV38`; none was exercised because that requires a real push.
  See `docs/ENVIRONMENT.md`.
- **Whether `kaggle models instances list` will change.** The trap is current behaviour of CLI 2.2.4, not
  a documented guarantee.
- **Why ARC-AGI-3 episodes return "No episodes found".** Not determined; possibly a timing/phase
  condition. No conclusion is drawn.
- **The exact ARC-AGI-2/3 code-freeze semantics** (e.g. whether a kernel version submitted before the
  deadline but run after it scores). Only the deadline values themselves were verified.
- **The tie-break question.** General `3.7.b` says hackathon competitions have no tiebreakers, while the
  Paper Track Evaluation page says a tie goes to the entry submitted first. The two were both read
  verbatim on 2026-09-23 and they conflict. Not resolved here.
- **Quota accounting for kernels.** `kaggle quota` reports remaining hours; how a specific run is
  measured against them was not tested.

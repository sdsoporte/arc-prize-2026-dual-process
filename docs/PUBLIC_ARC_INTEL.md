# ARC Prize 2026 — public-notebook field intel and positioning

> **Tracks:** `arc-prize-2026-arc-agi-3`, `arc-prize-2026-arc-agi-2` (also `arc-prize-2026-paper-track`, out of scope)
> **Verified:** 2026-09-24, 03:34–04:10 UTC. Every `kaggle kernels list` / `leaderboard` / `kernels pull` below was
> executed on that date. 30 notebooks were pulled to `/tmp/arc-intel/nb/` and parsed from `cells[].source`.
> **Scope:** **read-only** against Kaggle. Nothing was pushed, submitted, created, updated or deleted. No
> notebook, kernel, dataset or model was touched. **No repository file was modified except this document.**
> **House style:** MEASURED (observed in command output or notebook text) / INFERRED (my reading of a
> measurement) / NOT DETERMINED (stated with what would settle it), as in [`ARC3_REPLAYS_FINDINGS.md`](ARC3_REPLAYS_FINDINGS.md).
> **Audience:** the *Prior Work* and *Novelty* sections of `paper/draft.md`. §7 and §8 are the payload.

---

## 0. TL;DR

| Question | Answer (one line) |
|---|---|
| ARC-AGI-3 dominant approach | **LLM tool-use agent on a served 27B chat model** — the Tufa Labs "Duck/TAAF" harness and its forks |
| ARC-AGI-2 dominant approach | **One lineage**: the 2025 winner `qwen3_4b_grids15_sft139` (NVARC) + per-task LoRA test-time training + batched DFS + augmentation re-scoring |
| Is the field LLM-based? | ARC-AGI-2: **9/10** sampled. ARC-AGI-3: **13/20** sampled (rest is a *neural*-heuristic family, not model-free) |
| Qwen3.8 / "duck" / "flash-next" cluster | **CONFIRMED and dominant** in ARC-AGI-3 — 11/100 notebooks in the top-100 vote window are literally titled "duck"; 9/30 pulled notebooks reference a `Qwen3.8-Flash-Next` checkpoint |
| Public score claims | ARC-AGI-2 clusters at **33.89** ("LB33.89", ≥10 notebooks); ARC-AGI-3 milestone claims 0.86 (Gemma-4-31B) and 1.21 (Duck) |
| What nobody ships | (a) a genuinely model-free agent, (b) a hand-built DSL as the **primary** ARC-AGI-2 path, (c) an offline human-replay calibration harness, (d) a **learned System-1 gating** layer |
| Where we fit | Our ARC-AGI-2 DSL is **already public** (and reported near-zero-coverage); our ARC-AGI-3 agent is **public in kind and beaten** (0.36–0.46 public vs our 0.28); our **replay calibration harness is the only unoccupied niche**; the dual-process gate is now **measured to have zero headroom** (0 of 43 tasks — repo §7.3). See §7. |

---

## 1. Method, provenance, and its limits

**MEASURED — commands used** (all read-only):

```console
$ kaggle kernels list --competition arc-prize-2026-arc-agi-3 --page-size 30 --sort-by scoreDescending
$ kaggle kernels list --competition arc-prize-2026-arc-agi-2 --page-size 30 --sort-by voteCount
$ kaggle kernels list --dataset <owner/slug> --page-size 15          # dataset→notebook reverse index
$ kaggle kernels pull <owner/slug> -p /tmp/arc-intel/nb
$ kaggle competitions leaderboard -c arc-prize-2026-arc-agi-3 --download
```

**NOT DETERMINED — the three hard measurement limits.** State these in the paper if any number is quoted.

1. **Per-notebook public LB score is not exposed.** `kaggle kernels list --format json` returns `totalVotes`
   but no score field; the CLI has no `score` column; `/api/v1/kernels/view/...` returns HTTP 404 and the
   public kernel HTML is a shell. The `--sort-by scoreDescending` ordering is the **only** score signal the
   public API gives, and it is ordinal, not numeric. *What would settle it:* a logged-in browser scrape of
   `bestPublicScore` per kernel, or an owner's own writeup.
2. **The API returns a 100-notebook window per sort.** `pageSize=100` caps out; there is no page token in the
   response headers, so counts below are *window-relative*, not census counts.
3. **Sample is stratified, not random.** I pulled the highest-vote and highest-score-sorted notebooks, plus the
   lineages those hit. Percentages describe *the visible public surface*, not all 3,283 / 2,172 teams.

---

## 2. The sample — 30 pulled notebooks

`V` = `totalVotes` (MEASURED). `Claim` = score stated in the notebook itself (MEASURED quote; scale varies by
date — the 2026-06 "milestone" scale is **not** the current 0–100 scale). Technique/LLM/search/learn/lineage
are **INFERRED** from notebook code and markdown.

### ARC-AGI-3 (n=20)

| # | ref | V | technique | LLM | search | learns | framework | lineage |
|---|---|---:|---|:-:|:-:|:-:|---|---|
| 1 | `inversion/arc3-sample-submission-random-agent` | 622 | random-walk sample | – | – | – | `arcengine` | official sample |
| 2 | `jeroencottaar/tufa-labs-duck-harness-june-30-milestone-winner` | 332 | LLM tool-use agent; *solver code lives in a dataset* | ✅ Qwen3.6-27B-FP8 | ✗ | ✗ | vLLM | **DUCK root** ("milestone-winning 1.21") |
| 3 | `foysalemonshanto/lb-9-arc3-duck-v12-with-qwen-3-8-27b` | 288 | Duck + 27B FP8 | ✅ Qwen3.8-27B-FP8 | ✗ | ✗ | vLLM | DUCK |
| 4 | `wuliao0/duck-qwen3-8-anim-base` | 257 | Duck serving swap → NVFP4 + MTP | ✅ Qwen3.8-Flash-Next | ✗ | ✗ | vLLM | DUCK (byte-identical cluster) |
| 5 | `ashvinsingh/ash-s-arc-agi-3-agent` | 238 | BFS + pretrained CNN reward | ✗ | ✅ BFS | ✅ | torch | **FORGE** (`pretrained_weights.pt`) |
| 6 | `keithtyser/duck-qwen3-8-flash-next-nvfp4-mtp` | 179 | NVFP4/MTP serving patch | ✅ Qwen3.8-Flash-Next | ✗ | ✗ | vLLM | DUCK (byte-identical cluster) |
| 7 | `projectforty2/forge-arc-agi-3-agent` | 165 | FORGE v18, BFS solver + ForgeNet | ✗ | ✅ BFS | ✅ | torch | **FORGE** ("v10 scored 0.39, v15 0.36") |
| 8 | `nihilisticneuralnet/0-46-arc-agi-3-persistent-memory-bfs-arc-solver` | 152 | BFS/A\*/MCTS/beam + ForgeNet + PER replay | ✗ | ✅ all | ✅ TTT | torch | FORGE (title claims **0.46**) |
| 9 | `jakobbrggen/taaf-anim-arc-agi-3-solver` | 120 | TAAF deployment harness for Tufa solver | ✅ Qwen3.6-27B-FP8 | ✗ | ✗ | vLLM | DUCK/TAAF |
| 10 | `romantamrazov/arc-real-agi-solution` | 111 | Duck + 27B, "MTP3+async … produced 2.66 LB" | ✅ Qwen3.8-27B-FP8 | ✗ | ✗ | vLLM | DUCK |
| 11 | `mbmmurad/arc-agi-3-lb-0-86-3rd-place-candidate-milestone` | 102 | Gemma reflection agent ("**LB 0.86**", 3rd) | ✅ **Gemma-4-31B** | ✗ | ✗ | vLLM | GEMMA |
| 12 | `vyankteshdwivedi/arc-agi-3-hybrid-solver-bfs-cnn-heuristics` | 100 | BFS + ChangeNet + goal/introspection heuristics | ✗ | ✅ BFS | ✅ | torch | FORGE |
| 13 | `ko0kip/arc-agi-3-gemma-4-31b-reflection-agent` | 81 | Gemma agent, structured-output action pick | ✅ Gemma-4-31B | ✗ | ✗ | vLLM | GEMMA |
| 14 | `tantan0327/arc3-flashnext-asis` | 32 | "Based on Tufa Labs' Duck harness", serving only | ✅ Qwen3.8-Flash-Next | ✗ | ✗ | vLLM | DUCK (byte-identical cluster) |
| 15 | `kaiwalyaatulraut/arc-agi-3-solution` | 20 | Duck-derived (Jaccard 0.61 to DUCK) | ✅ Qwen3.8-Flash-Next | ✗ | ✗ | vLLM | DUCK |
| 16 | `amanatar/arc-agi-3-hybrid-repl-agent` | 16 | Duck + REPL; internal gate `mean_score >= 6.3` | ✅ Qwen3.8-Flash-Next | ✗ | ✗ | vLLM | DUCK (Jaccard 0.75) |
| 17 | `anhadmahajan06/arc-agi-3-fluid-intelligence-agent` | 15 | "Production Hybrid Agent (2.31%+)" | ✅ Qwen3.8-27B-FP8 | ✗ | ✗ | vLLM | DUCK |
| 18 | `anglolodorf/forge-arc-agi-3-simplified-baseline` | 4 | "ForgeAgent — v17 best-scoring solver" + CNN | ✗ | ✗ | ✅ | torch | **FORGE root** |
| 19 | `themightiestman/arc3-forge-v1-open-source-baseline` | 0 | byte-identical to #18 | ✗ | ✗ | ✅ | torch | FORGE root |
| 20 | `yocybercode/thui-fast-b78-mtp0-full25-r1` | 0 | Duck Flash-Next, MTP 3→0 ablation | ✅ Qwen3.8-Flash-Next | ✗ | ✗ | vLLM | DUCK (Jaccard 0.71) |

### ARC-AGI-2 (n=10)

| # | ref | V | technique | LLM | search | learns | framework | lineage |
|---|---|---:|---|:-:|:-:|:-:|---|---|
| 21 | `koushikrudra/failed-in-aimo` | 552 | NVARC base **+ "Leg C"**: 7B coder samples DSL programs, AST-whitelisted, verified on all train pairs | ✅ Qwen3-4B + Qwen2.5-Coder-7B | ✅ batched DFS + DSL verify | ✅ per-task LoRA TTT | unsloth/transformers | **NVARC public root** ("LB33.89") |
| 22 | `nihilisticneuralnet/baseline-nvarc-arc-25-winning-solution-for-t4x2` | 209 | 2025 winner, verbatim | ✅ Qwen3-4B | ✅ DFS | ✅ TTT | unsloth | **NVARC root** |
| 23 | `junaid512/arc-agi-enhanced-solution` | 136 | Qwen3-4B SFT + beam + augmented-view NLL | ✅ Qwen3-4B | ✅ beam/DFS | ✅ TTT | unsloth | NVARC-adjacent (Apr) |
| 24 | `mikelou1/arc-agi2-lb33-89-minimal-perfpatch` | 113 | GPU-logits/KV micro-patch on `failed-in-aimo` (md names it as master) | ✅ Qwen3-4B | ✅ DFS | ✅ TTT | unsloth | NVARC fork |
| 25 | `koushikrudra/arc-agi2-original-kg` | 102 | NVARC + KGMon ordering | ✅ Qwen3-4B | ✅ DFS | ✅ TTT | unsloth | NVARC |
| 26 | `yusuketogashi/arc-baseline-rebuild` (Program079) | 83 | NVARC primary + **exact-symbolic fallback** (D4 + color map + up/tile/downscale) + singleton retry | ✅ Qwen3-4B | ✅ DFS | ✅ TTT | unsloth/triton | NVARC + hand-written symbolic |
| 27 | `luxluxshan/arc2-nvarc-v1` | 76 | NVARC run **twice** with independent seeds, pooled | ✅ Qwen3-4B | ✅ DFS | ✅ TTT | unsloth | NVARC (credits mikelou1, koushikrudra) |
| 28 | `baidalinadilzhan/prev-year-s-compressarc-method-p100-gpu-lb-1-67` | 75 | CompressARC: per-task test-time neural compressor | ✗ | ✗ | ✅ TTT | torch | CompressARC ("LB 1.67") |
| 29 | `fayche/arc-prize-2026-solver` | 58 | LB33.89 baseline, Chinese md | ✅ Qwen3-4B | ✅ DFS | ✅ TTT | unsloth | NVARC |
| 30 | `karnakbaevarthur/logic-profiler-for-each-task` | 48 | DeepSeek labels logic primitives; **submission is off-the-shelf NVARC** | ✅ DeepSeek-V3/R1 + Qwen3-4B | ✅ DFS | ✅ TTT | unsloth | NVARC + LLM diagnostic |

---

## 3. Quantified distribution (MEASURED → INFERRED)

**Over the 30-notebook stratified sample:**

| Class | ARC-AGI-3 | ARC-AGI-2 | All |
|---|---:|---:|---:|
| LLM-driven (agent or grid transducer) | 13 | 9 | **22 / 30 (73%)** |
| Neural-heuristic hybrid, **no LLM** (BFS + tiny CNN) | 6 | 0 | 6 / 30 (20%) |
| Learned, no LLM, no search (neural compressor) | 0 | 1 | 1 / 30 |
| Random baseline | 1 | 0 | 1 / 30 |
| **Model-free heuristic agent as the submission path** | **0** | **0** | **0 / 30** |
| **Hand-built DSL as the primary solver** | 0 | **0** | **0 / 30** |
| **Learned System-1 gate over a System-2 solver** | **0** | **0** | **0 / 30** |

**Corroboration over the 100-notebook API window per track** (title keywords, INFERRED from titles):

| keyword | ARC-AGI-3 (n=100) | ARC-AGI-2 (n=100) |
|---|---:|---:|
| `duck` | 11 | 0 |
| `qwen` (any) | 11 | 3 |
| `nvarc` | 0 | 8 |
| `33-89` / `33.89` | 0 | 4 |
| `perfpatch` | 0 | 5 |
| `baseline`/`starter`/`random`/`sample` | 9 | 8 |
| `gemma`\|`reflection` | 1 | 0 |
| `bfs` / `forge` | 0 / 1 | 0 / 0 |

**Framework incidence in the pulled sample:** `torch` 20/30, `vLLM` 14/30, `transformers` 11/30,
`unsloth` 9/30 (all 9 are ARC-AGI-2), `triton` 10/30. **Two ecosystems, no overlap**: ARC-AGI-3 = vLLM-served
chat model; ARC-AGI-2 = Unsloth/LoRA test-time training on a 4B grid-vocabulary model.

**Verified from notebook text — the LLM's role differs by track:**

- ARC-AGI-3: LLM is an **agent** — tool-use loop, prompts, game policy
  (`tufa-labs-duck-harness`, md: *"The Duck prompts, tool-use loop, game policy, and scorer remain unchanged"*);
  action choice by structured output (`arc-agi-3-gemma-4-31b-reflection-agent`,
  `gemma-4-31b-it` + `models/google` + `response_format`).
- ARC-AGI-2: LLM is a **grid transducer** — `arc_loader.convert_grid_to_string` emits digits + `\n`, a
  13–16-token vocabulary, then batched DFS over the LM's NLL (`fayche/arc-prize-2026-solver` md: *"13-token
  vocabulary shrinking (digits '0'..'9', '\n', start/end tokens)"*).

---

## 4. Lineage is measurable, and it is concentrated

**MEASURED — byte-identical cell bodies** (SHA-256 of concatenated `cells[].source`, 10 hex chars):

```console
a609c35888  duck-qwen3-8-flash-next-nvfp4-mtp.ipynb
a609c35888  duck-qwen3-8-anim-base.ipynb
a609c35888  arc3-flashnext-asis.ipynb
1.00 Jaccard  arc3-forge-v1-open-source-baseline  <->  forge-arc-agi-3-simplified-baseline
```

**MEASURED — shared library fingerprint, ARC-AGI-2.** The NVARC pipeline is shipped as `%%writefile`
libraries. Eight notebooks embed a **byte-identical** `arc_loader.py`:

```console
arc_loader.py = d01cd561(12640 chars) in:
  failed-in-aimo · arc-agi2-original-kg · arc-agi2-lb33-89-minimal-perfpatch · arc-baseline-rebuild
  arc2-nvarc-v1 · baseline-nvarc-arc-25-winning-solution-for-t4x2 · logic-profiler-for-each-task
  (+ fayche/arc-prize-2026-solver uses a 13173-char sibling)
arc_decoder.py = 965cfd91(4156) in 6 of the same 9
```

**MEASURED — the documented lineage chain**, from `arc2-nvarc-v1`'s own markdown:
> *"Credits: NVARC (Ivan Sorokin & Jean-François Puget), the ARChitects (Daniel Franzen & Jan Disselhoff),
> mikelou1's throughput patch, koushikrudra's deterministic-seed fixes."*

and `mikelou1/arc-agi2-lb33-89-minimal-perfpatch`'s own markdown (zh):
> *"本 notebook 以 `baseline_LB33.89_failed-in-aimo.ipynb` 为母版"* — "this notebook uses
> `baseline_LB33.89_failed-in-aimo.ipynb` as its master template."

Chain: **NVARC 2025 winner → `failed-in-aimo` (koushikrudra, public "LB33.89") → perfpatch (mikelou1) →
NVARC+ seed-ensemble (luxluxshan) → Program079 (yusuketogashi)**. Model asset in every link:
`/kaggle/input/models/sorokin/qwen3_4b_grids15_sft139/transformers/bfloat16/1`.

**INFERRED lineage labels:** ARC-AGI-3 = DUCK/TAAF **13/20**, FORGE/BFS **6/20**, GEMMA **2/20**, official
sample **1/20**. ARC-AGI-2 = NVARC **9/10**, CompressARC **1/10**.

**Qwen3.8 / duck / flash-next — CONFIRMED, not refuted.** The checkpoint string
`RadixArk/Qwen3.8-Flash-Next-NVFP4` (or `Qwen/Qwen3.8-27B-FP8`) appears in 9/30 pulled notebooks, and
`thui-fast-b78` writes an explicit provenance assert:
`assert provenance["model_hf_repo"] == "RadixArk/Qwen3.8-Flash-Next-NVFP4"`. The original Duck paper served
`Qwen3.6-27B-FP8` (`driessmit1/vrfai-qwen3-6-27b-fp8-hf-snapshot`); the public field has since standardised on
its successor. **The cluster is dominant by vote, by copy-count, and by public score claim — but it is a
*serving/quantisation* fork ecosystem, not an algorithmic one.**

---

## 5. Verified top-10 leaderboards

**MEASURED** — `kaggle competitions leaderboard --download`,
`arc-prize-2026-arc-agi-3-publicleaderboard-2026-09-24T03:34:59.csv` and `…-arc-agi-2-…T03:35:03.csv`.

**ARC-AGI-3** — 3,283 entries, max 19.40, median 0.30, 8 teams ≥ 10, 1,219 ≥ 1, 160 at 0.

| # | team | score | # | team | score |
|---:|---|---:|---:|---|---:|
| 1 | Lord Han Solo | 19.40 | 6 | Matija Ludvig & Zhongwei Wang | 11.64 |
| 2 | **Tufa Labs** | 18.81 | 7 | the last dance 🕺 | 11.09 |
| 3 | **NVARC3** | 16.07 | 8 | Tong Hui Kang | 10.78 |
| 4 | Yi-Chia Chen | 15.98 | 9 | Ebi | 8.68 |
| 5 | Daniel Franzen | 13.12 | 10 | Fususu | 8.65 |

**ARC-AGI-2** — 2,172 entries, max 83.06, median 28.06, 1,340 ≥ 10, 722 at 0.

| # | team | score | # | team | score |
|---:|---|---:|---:|---|---:|
| 1 | **Tufa Labs** | 83.06 | 6 | Junhua Yang | 37.22 |
| 2 | rabbithole | 76.94 | 7 | Nguyen | 36.94 |
| 3 | **nvbanana** | 74.17 | 8 | AI Winter | 35.42 |
| 4 | Yi-Chia Chen | 55.14 | 9 | Hiểu Vy | 34.86 |
| 5 | Kha Vo | 37.50 | 10 | **Team BlackBox** | 33.89 |

**MEASURED — the leaderboard and the public notebooks are the same people (INFERRED mapping):**
`NVARC3` = `cpmpml, darraghdog, eladsarafian, galkaplun, sorokin, yeyinzhu`; `nvbanana` =
`cpmpml, darraghdog`; `Team BlackBox` includes **`koushikrudra`** — author of the public "LB33.89"
notebook. `Tufa Labs` includes `jeroencottaar` and `driessmit1` — authors of the public Duck harness and its
model snapshot. So **#2 ARC-AGI-3, #1 ARC-AGI-2, #3 ARC-AGI-3 and #3 ARC-AGI-2 all publish working code.**

**MEASURED — the absolute top does not.** `kaggle kernels list --user lordhansolo` → `Not found`;
`--user cookizesong` (rabbithole, ARC-AGI-2 #2) → `Not found`. The 19.40 ARC-AGI-3 leader and the 76.94
ARC-AGI-2 runner-up have **zero** public notebooks.

**Score claims inside pulled notebooks** (MEASURED quote; scales differ by date):

| notebook | claim as written |
|---|---|
| `tufa-labs-duck-harness-june-30-milestone-winner` | "the notebook that scored our milestone-winning **1.21**" |
| `mbmmurad/…lb-0-86-3rd-place-candidate…` | "It reached **0.86** on the public leaderboard and put me in 3rd place at the time" |
| `forge-arc-agi-3-agent` | "**v10 scored 0.39** on Kaggle. **v15 scored 0.36** (regression from …)" |
| `0-46-…-persistent-memory-bfs-arc-solver` | title: "**[0.46]**" |
| `arc-real-agi-solution` | "the exact MTP3+async serving that produced **2.66 LB**" |
| `arc-agi-3-fluid-intelligence-agent` | "Production Hybrid Agent (**2.31%+**)" |
| `arc2-nvarc-v1` | "Re-running the *identical* NVARC notebook gives public-LB scores of **26.9 – 32.2** … (29.7 / 31.8 / 26.9 / 32.2 / 31.4)" |
| `failed-in-aimo` / `arc-prize-2026-solver` | "LB **33.89**" |
| `prev-year-s-compressarc…` | title: "LB[**1.67**]" |

> **This is the most important single sentence in the whole document** — from `arc2-nvarc-v1`:
> *"Re-running the identical NVARC notebook gives public-LB scores of 26.9 – 32.2 … because test-time
> training and batched DFS are stochastic."* The field's dominant method has the **same RNG-dominance
> property we diagnosed in our own agent (6.3× seed spread)**, and the field's response is engineering
> (`arc2-nvarc-v1` pools two independent-seed passes; `failed-in-aimo` adds "koushikrudra's deterministic-seed
> fixes"), not a research finding. Our 6.3× observation is **not novel**; our *use* of it might be.

---

## 6. What the field does *not* do

Ranked by how defensible the gap is after pulling 30 notebooks.

**(a) A genuinely model-free agent — 0/30, and this is the sharpest negative result.** Everything I could
classify is either an LLM agent or a **neural**-heuristic hybrid. The six "heuristic-looking" ARC-AGI-3
notebooks are hybrid: they all load
`forge-pretrained-weights/pretrained_weights.pt` into a `ForgeNet`/`ChangeNet` CNN and keep an
`optim.Adam` + train loop (`0-46-…`, `forge-arc-agi-3-agent`, `arc-agi-3-hybrid-solver-bfs-cnn-heuristics`,
`ash-s-arc-agi-3-agent`, `forge-arc-agi-3-simplified-baseline`, `arc3-forge-v1-open-source-baseline`).
Their BFS is a **planner over a learned heuristic**, not a model-free policy. `arc-agi-3-hybrid-repl-agent`'s
own import list (`vllm_server_watchdog`, `inference.tools.eval`) shows the same. **Our model-free agent is a
category the public field has essentially vacated** — *but* §8 explains why that is a weaker claim than it
sounds.

**(b) A hand-built DSL as the primary ARC-AGI-2 solver — 0/10.** The two symbolic things that exist are:
(i) `failed-in-aimo` "Leg C" — a **35-primitive numpy DSL** (`rot90/crop/get_objects/flood_fill/gravity/…`,
53 functions, AST-whitelist enforced, `prog_timeout=4.0`s per candidate, verified on *all* train pairs) whose
**programs are sampled by `Qwen2.5-Coder-7B-Instruct`**, not searched; and (ii) `arc-baseline-rebuild`
Program079's `discover_exact_rules()` — `d4_exact`, `d4_color_map`, `integer_upscale`, `integer_tile`,
`uniform_block_reduce`, bbox crop, gated at `MIN_SYMBOLIC_CONFIDENCE = 94`, used **only as an unused-slot
fallback**. Its own markdown states the result:
> *"Program078 showed that exact symbolic rules did not cover any of the observed 10 empty and 17 singleton
> outputs."*

This is a ~1:1 overlap with our `src/arc2_symbolic_dsl.py` (rot90/rot180/rot270/flip_h/flip_v/transpose/
crop/tile/gravity + `make_color_substitutor`) and their coverage finding matches our 0/120 EVAL result.
**Our ARC-AGI-2 DSL is not novel, and the field already published that it does not work.**

**(c) An offline human-replay calibration harness — 0 public notebooks.** `kaggle kernels list --dataset
jihangli1121/arc-agi-3-replays-v1` → `Not found`, while two control datasets
(`anglolodorf/arc-agi-3-pretrained-weights`, `poonszesen/arc-interactive-community`) return linked notebooks,
so the reverse index does work. **No public ARC-AGI-3 notebook is attached to the GT-replays dataset.**
*Caveat (INFERRED risk): a notebook could `kaggle datasets download` it at runtime without attaching it; I
found no notebook that does, but "no attachment" is what I measured.* The nearest public analogue is
`justforgags/duck-eval-results` ("ARC-AGI-3 Duck-Harness Offline Eval Logs", 410 downloads, **0 linked
notebooks**) — offline logs exist as data, not as a calibration methodology. **This is our best-defended gap.**

**(d) A learned System-1 gate over a System-2 solver — 0/30.** A regex for
`gating|router|routing|classifier|system 1|system 2|dual-process|arbitrat|selector` over all 30 notebooks
matched only `import selectors` in `arc-baseline-rebuild`. The field's hybrids are **pipelines, not gates**:
"neural primary → symbolic fallback" (`arc-baseline-rebuild`), "LLM induction pre-pass → neural TTT"
(`failed-in-aimo`), "two stochastic passes pooled" (`arc2-nvarc-v1`), "DeepSeek labels the task offline"
(`logic-profiler`). Nobody ships a **learned decision model that routes** between a fast path and a slow path.
*Caveat: the Duck/TAAF solver body lives in a dataset, not in its notebook, so a routing layer inside *that*
code would be invisible to this scan; I only measured the notebooks.*

**(e) Also absent:** any public notebook scoring the paper's central mechanism. No pulled notebook contains a
fine-tuned non-autoregressive decision model, an impasse/deadlock classifier, or a Brier-calibrated gate.

---

## 7. Where our approach fits — plainly

**Let the two facts the field gives us set the frame.** The public field has **already published our
ARC-AGI-2 DSL**, with our primitive set and our coverage verdict; and it has **already built our ARC-AGI-3
agent's *kind*, and scores it 0.28–0.64 points above us**.

| Our claim (`paper/draft.md`) | Field reality (MEASURED/INFERRED) | Verdict |
|---|---|---|
| §2.3 "structured search over DSLs is the primary engine of accuracy" | True for the DSL literature — but the *leaderboard* engine is **test-time training + stochastic decoding + re-scoring** (`qwen3_4b_grids15_sft139`, `score_kgmon`), i.e. NVARC. `failed-in-aimo` puts a DSL **on top of** NVARC as a pre-pass, never as the primary. | Reframe: the field treats symbolic methods as a **monotonic add-on** to a learned base, and reports they add little. Our DSL-only pipeline is a strictly weaker configuration than what public notebooks already run. |
| Two-stage `<spatial op> + colour_map` composition; D4 + `crop_nonzero` + `tile_2x2/3x3` + `gravity` | `arc-baseline-rebuild` returns `d4:{op}` **and** `d4_color_map:{op}` (learned color map, injectivity-scored), plus `integer_upscale`, `integer_tile`, `uniform_block_reduce`. | **Not novel.** Cite it as concurrent/prior work and say so explicitly; a reviewer comparing `src/arc2_symbolic_dsl.py` line-by-line to Program079 will find the overlap in ~30 seconds. |
| Our own diagnostic: **0/120 coverage on public EVAL** (repo `odd/tasks/system1-wiring.md`: TEST 5/240 deployed = 2.1%, 15/240 local = 6.2%; EVAL 0/120) | `arc-baseline-rebuild` reports the same outcome for a *superset* of our primitives ("did not cover any of the observed 10 empty and 17 singleton outputs"). | Honest, and now **corroborated by an independent public notebook**. Report it as the *reason* for the dual-process design, not as a surprise. |
| The gate is the contribution: System 1 "prunes program synthesis search trees by over 80%" | Repo `system1-wiring.md` §11 measured the gate on the only set with both coverage *and* ground truth (TRAIN): **43/1000 covered; 18 had >1 matching candidate; 1 disagreed; the first matching candidate is always correct; maximum ranker headroom = 0 tasks of 43.** T3/T4/T7 cancelled. | **This is the strongest available framing.** The candidates are functionally redundant (`identity+color_map` ≡ `color_map`), so neither pruning nor ranking can move the score. Publishing a *measured null* with this mechanism is more defensible than claiming a gate that cannot act — and it is a direct rebuttal to any "just wire the model" review. |
| §4.3 "System 2 Only (Unguided DFS) … timeouts on 18 tasks; Dual-Process … 28 minutes" | Nobody reproduces this; it is a local 50-task experiment presented as an ablation with no seed or variance reporting. The repository's own `system1-wiring.md` §11 already retracted the premise: pruning can only *lower* coverage, and re-casting the gate as a **ranker** returned **zero headroom**. | **Remove it.** It is contradicted by the project's own later measurement (§11: max ranker headroom = 0/43) and is the single most likely target for a reviewer's reproducibility objection. |
| §4.2 "88.24% composite … 91.76% transformation classification … Brier 0.0818" | The measured population is the **model's own typed decision schema**, i.e. offline, self-generated labels — not task solve rate. `paper/draft.md` §5.1 concedes this, but the abstract does not. | A reviewer will call this headline/measured-target mismatch. Separate "gate accuracy" from "score" everywhere, including the abstract. |
| The 421M Laya model "gates the submission" | `odd/tasks/system1-wiring.md`: *"`model_sources: []` on all six kernels — the model is published and consumed by nothing."* It runs in **no** submission kernel, and §11 concluded *"the System 1 has no measured role in either track."* | **This is the admissibility problem, not a rubric detail.** The repository has now resolved it the defensible way: measure the null, declare System 1 separate, and stop claiming it gates the score. Keep that resolution; do not re-inflate the abstract. |
| ARC-AGI-3 agent (CRC32 in-episode memory, wall-edge map, BFS frontier, deadlock undo-probing), 0.28 | The **FORGE/BFS family** (`ash-s`, `0-46`, `projectforty2`, `vyankteshdwivedi`) is a public, same-kind lineage scoring **0.36–0.46**, with an optional learned reward instead of our hand-tuned one. | Our agent is **unremarkable in kind and below the public best in the same family.** Reporting 0.28 as a contribution invites an unfavourable comparison to public notebooks *and* to the 18.81 of the published Duck harness. |
| Weighted-heuristic layer "inert" (92.8% one-candidate pools); RNG-dominated (6.3×) | `arc2-nvarc-v1` publishes the same stochasticity for the *winning* method (26.9–32.2 across reruns) and **engineers around it** with seed pooling. | Our diagnosis is correct but **not novel**. The publishable angle is the *method*: quantified seed-variance + an offline replay harness. |
| "Dual-process" as the framing | **No public notebook ships a learned gate** (§6d). | **This is the genuinely unoccupied slot** — but it is currently a gap with a hypothesis and no measured score. |

**What is genuinely ours, ranked:**

1. **The offline human-replay calibration harness** (`docs/ARC3_REPLAYS_FINDINGS.md`, `experiments/arc3_local_eval.py`).
   Human replay scores **89.68/100** through our engine; the public field has the dataset (258 downloads) and
   no linked notebook (§6c). Nobody else turns human replays into a *ruler* with per-game attribution.
2. **A model-free ARC-AGI-3 policy** (§6a). The gap is real, but note the field did not abandon model-free
   heuristics because it cannot build them — it abandoned them because the learned variants score higher.
   Frame this as an *interpretability/ablation* contribution (how much score do you lose with zero learning?),
   not as a competitive one.
3. **A measured null on the dual-process gate** (§6d + repo §11) — *this is now the most citable result in the
   repository*, but it must be presented as a **negative result**, not as an architecture. The measured chain is:
   43/1000 train tasks covered → 18 with more than one matching candidate → 1 disagreement → **max ranker
   headroom 0/43**, because the DSL candidates are functionally redundant wherever coverage exists
   (`identity+color_map` ≡ `color_map`). Nobody in the 30-notebook sample reports anything like this about
   their own gating layer. Claim it as *"we designed a System-1 gate, then showed it cannot act"* — that is a
   contribution; *"our System 1 gates the solver"* is not, because it provably does not.

**What a reviewer will compare us against, in order of likelihood:**
(i) NVARC / 2025 winner (`qwen3_4b_grids15_sft139` TTT + batched DFS) — the ARC-AGI-2 ceiling and our
DSL's better; (ii) the ARChitects' DSL work (Franzen, already cited in our §2.3); (iii) Tufa Labs' Duck
harness — the ARC-AGI-3 paradigm, publicly documented, 18.81; (iv) TRM (Jolicoeur-Martineau, cited in §2.2);
(v) CompressARC — the one non-LLM learned public approach we found in-track.

**Bottom line for §8 of the paper.** As written, the paper's named contribution is a **gating architecture
that no submission runs and that the project has since measured to have zero headroom**, an ARC-AGI-2 DSL
that is **a subset of a public notebook's exact-rule fallback which is already reported not to work**, and an
ARC-AGI-3 agent that is **0.08–0.18 below the public same-family best**. The defensible contributions are the
**offline human-replay calibration methodology** (the only unoccupied niche in §6) and the **measured null on
the gate** (repo §11). Recommend: retitle around the negative result and the instrument, move the Laya gate
from "our architecture" to "a hypothesis we test and reject", and delete §4.3's ablation table. Do not
flatter it — as it stands the work is a careful negative result with one genuinely novel measurement
instrument and one genuinely novel calibration ruler.

---

> **Provenance note (MEASURED).** `odd/tasks/system1-wiring.md` was revised by another process at 00:47:13,
> ~3 s after this document was first written. The §11 numbers above are quoted from that later revision, not
> from the state I read during research. The file is outside this task's allowed edit surface and was **only
> read** here; it is reported, not reverted.

---

## 8. Not determined

| Open question | Why it is open | What would settle it |
|---|---|---|
| Whether the zero-headroom null generalises past `test[0]` | Repo §11 notes the gate "was measured on `test[0]`; tasks with several test inputs give the official metric more outputs" | Re-run the ranker headroom measurement over every test input of all 43 covered train tasks |
| Per-notebook public LB score | No score field in the CLI/JSON API; kernel HTML is a shell; `/api/v1/kernels/view` 404 | Authenticated browser scrape of `bestPublicScore`, or owner writeups |
| Total count of public notebooks per track | API window caps at 100 per sort, no page token | Kaggle UI pagination, or an authenticated internal search endpoint |
| Whether the highest-claimed fork results are reproducible | Fork authors report single runs; only `arc2-nvarc-v1` reports a rerun distribution | Independent reruns under the same 12 h budget |
| The Duck solver's internal prompting / memory / any internal routing | Solver ships in `jeroencottaar/taaf-kaggle-source-share`, not in the notebook; the notebook's md says so explicitly | Download and inspect the TAAF source dataset (read-only; not done here) |
| Whether a hidden notebook downloads the replay dataset at runtime | Reverse index shows no *attachment*; runtime downloads are invisible to it | Full-text search over notebook sources for `arc-agi-3-replays-v1` |
| Whether "0.86" / "2.66" / "6.3" claims share one scoring scale | Milestone-era notebooks predate the current 0–100 rescale; not all state the scale | Cross-check each claimer's submission date against the leaderboard CSV `LastSubmissionDate` |
| Whether our agent's 0.28 and FORGE's 0.36–0.46 were measured on the same game set/scale | Both are self-reported; FORGE's claims are from June–July milestone runs | Same-budget head-to-head on the same env set |

---

## 9. Reproducibility appendix

- Leaderboard CSVs: `/tmp/arc-intel/arc-prize-2026-arc-agi-{2,3}-publicleaderboard-2026-09-24T03:3{4,5}*.csv`
- 30 pulled notebooks: `/tmp/arc-intel/nb/` (sources as of 2026-09-24; fork counts will drift)
- Scanner: `/tmp/arc-intel/scan.py` (parses `cells[].source`; reports LLM/framework/search/learn/fork markers)
- Kernel metadata: `/tmp/arc-intel/list_arc-prize-2026-arc-agi-{2,3}.json` (top-100 by voteCount)
- Kaggle contact was **read-only** throughout: `kernels list`, `kernels pull`, `datasets list`,
  `competitions leaderboard`, `competitions list`, plus unauthenticated-data/authenticated-metadata `GET`s.
  No `push`, `submit`, `create`, `update`, `delete`, `datasets download` or `models` call was made.

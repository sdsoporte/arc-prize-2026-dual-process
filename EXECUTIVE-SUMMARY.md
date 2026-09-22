# ARC Prize 2026 — Paper Track: Executive Summary

> **Generated:** 2026-09-22 | **Deadline:** November 9, 2026 (48 days)
> **Source:** [arcprize.org/competitions/2026/paper](https://arcprize.org/competitions/2026/paper)

---

## The Competition at a Glance

| | ARC-AGI-2 | ARC-AGI-3 |
|---|---|---|
| **Paradigm** | Static grid puzzles | Interactive agentic environments |
| **Prize** | $700,000 | $850,000 |
| **Paper Track** | $450,000 shared across both |
| **What you do** | Predict output grids from input examples | Navigate and solve dynamic games |
| **Core challenge** | Pattern recognition + program synthesis | Scientific discovery + hypothesis testing |
| **Data** | 1,000 train / 120 eval / 240 test tasks | 25 interactive environments |
| **Code deadline** | Nov 2, 2026 | Nov 2, 2026 |
| **Paper deadline** | Nov 9, 2026 | Nov 9, 2026 |

---

## 📄 Paper Track — Complete Rules (from arcprize.org)

### What Is It?

> *"The Paper Prize rewards conceptual progress that best advances our understanding of how to achieve strong performance on ARC-AGI."*

It's **NOT** a standalone paper competition. Your paper MUST be linked to a working Kaggle code submission in either the ARC-AGI-2 or ARC-AGI-3 track. **The code does NOT need to achieve a high score** — the paper is evaluated independently.

### Key Dates (Official)

```
Mar 25, 2026  ─── Competition started
Jun 30, 2026  ─── ARC-AGI-3 Milestone #1
Sep 30, 2026  ─── ARC-AGI-3 Milestone #2
Nov 02, 2026  ─── Code submissions due (ARC-AGI-2 & ARC-AGI-3)
Nov 08, 2026  ─── Papers due ← CORRECTED from Kaggle (Nov 9 on Kaggle, Nov 8 on arcprize.org)
Dec 04, 2026  ─── Results announced
```

### Prize Structure — $450K Total

| Prize | Amount | Condition |
|-------|--------|-----------|
| 🥇 1st Place Paper | $50,000 | Highest-scoring paper across both tracks |
| 🥈 2nd Place Paper | $20,000 | Second highest |
| 🥉 3rd Place Paper | $5,000 | Third highest |
| 🏆 Outstanding Papers Pool | $375,000 | Shared equally among ALL papers scoring **> 4.5/5** |

> **Critical insight:** Top 3 prizes = $75K (guaranteed). Outstanding pool = $375K (discretionary).
> The REAL money is in the pool. If only 5 papers score > 4.5, that's $75K EACH.

### Evaluation Rubric (0–5 per category, equally weighted)

| Category | What judges evaluate | How to score high |
|----------|---------------------|-------------------|
| **Accuracy** | Performance on the Kaggle leaderboard | Higher leaderboard score = higher rubric score |
| **Universality** | Does it generalize beyond ARC to other problems? | Show it works on related benchmarks/domains |
| **Progress** | Does it increase the chance of reaching 85% on ARC-AGI? | Present ideas the community can build on |
| **Theory** | Does the paper explain *WHY* it works, not just *HOW*? | Provide formal reasoning, not just implementation |
| **Completeness** | How thoroughly does it cover the leaderboard submission? | Document every detail of your approach |
| **Novelty** | Is the approach original vs. existing public research? | Unique angle, not just engineering on known methods |

**Final Score** = average across the 6 categories.

> ⚠️ Rubric evaluations will NOT be shared with participants.
> ⚠️ In case of a tie, the paper submitted FIRST wins.

### Required Paper Structure (Official)

The organizers specify exactly what to include:

1. **Abstract** — What the contribution is (e.g., "we present a method to solve ARC-AGI, with the following characteristics...")
2. **Intro** — What ARC-AGI is, why it's important, and the inspiration behind the approach
3. **Prior Work** — Previous approaches related to yours — highlight similarities and differences
4. **Approach** — How it works, including an algorithm-level description
5. **Results** — Scores on Kaggle leaderboard and public eval. **Do NOT report train set performance**
6. **Conclusion** — Summary of the contribution and what was achieved

> *"Shorter and clearer is always better. No filler, no unnecessary equations. Papers are about communicating ideas clearly so others can learn from and reuse them."*
> — arcprize.org

### Eligibility Rules

| Rule | Detail |
|------|--------|
| **Linked code** | Paper must link to a valid Kaggle submission in ARC-AGI-2 or ARC-AGI-3 |
| **Open source** | All code must be open-sourced under CC0 or MIT-0 (permissive/public domain) |
| **3rd party code** | Must be available under at least Apache-2.0 or GPLv3 |
| **Open source BEFORE scores** | You must open source BEFORE receiving official private evaluation scores |
| **No internet during eval** | Kaggle evaluation runs WITHOUT internet (no GPT/Claude/API calls) |
| **Reproducible** | All solutions must be reproducible |
| **Discretionary** | Prizes awarded at ARC Prize Inc.'s sole discretion |

### Previous Winners Reference

- [2025 Results and Analysis](https://arcprize.org/blog/arc-prize-2025-results-analysis)
- [2024 Results and Analysis](https://arcprize.org/competitions/2024)

---

## ARC-AGI-2: Static Grid Puzzles

### Task Format
Each task provides 2-10 training examples (input → output grid pairs) demonstrating a hidden transformation rule, plus 1-4 test inputs to predict.

```
Train Example:        Test:
Input:  [0,0,0]      Input:  [0,0]    → Predict output
        [0,1,0]              [0,2]
        [0,0,0]

Output: [0,1,0]
        [1,1,1]
        [0,1,0]
```

### Key Statistics

| Metric | Value |
|--------|-------|
| Training tasks | 1,000 |
| Evaluation tasks | 120 |
| Test tasks | 240 |
| Grid size range | 1×1 to 30×30 |
| Avg grid size | ~12×12 |
| Avg training examples per task | 3.21 |
| Avg test inputs per task | 1.11 |
| Colors used | 10 (integers 0–9) |
| Avg unique colors per task | 6.44 |

### Color Distribution

| Color | ID | Usage | Role |
|-------|--:|------:|------|
| Black | 0 | 44.0% | Background (dominant) |
| Teal/Cyan | 8 | 12.2% | Most used non-background |
| Blue | 1 | 9.2% | Common foreground |
| Yellow | 4 | 6.8% | |
| Orange | 7 | 6.5% | |
| Green | 3 | 6.4% | |
| Red | 2 | 6.3% | |
| Grey | 5 | 3.8% | |
| Magenta | 6 | 2.9% | |
| Maroon | 9 | 1.7% | Least common |

### Task Categories
- **Pattern Completion & Extrapolation** — continuing sequences, filling gaps
- **Symmetry & Geometry** — reflection, rotation, translation
- **Topology & Connectivity** — flood fill, connected components, path tracing
- **Counting & Sorting** — object count → color/size mapping
- **Object Interactions** — collision, occlusion, intersection effects
- **Scale & Resizing** — zoom, tiling small patterns

### Submission Format
For each test task, 2 attempts allowed:
```json
{
  "task_id": [
    {"attempt_1": [[grid]], "attempt_2": [[grid]]}
  ]
}
```

---

## ARC-AGI-3: Interactive Agentic Environments

### Paradigm Shift
ARC-AGI-3 is NOT a grid-prediction task. It's a **real-time game-playing benchmark** where AI agents must:

1. **Start blind** — no rules are explained
2. **Experiment** — take actions to discover how the environment works
3. **Form hypotheses** — build a mental model of the game mechanics
4. **Solve levels** — apply understanding to complete multi-level puzzles

### Action Space (6 discrete + 1 coordinate)

| Action | Meaning |
|--------|---------|
| ACTION1-4 | Movement (Up/Down/Left/Right) |
| ACTION5 | Interact (Enter/Spacebar) |
| ACTION6 | Click at (x, y) coordinates |
| ACTION7 | Undo |

### Environment Types (25 total)
Interactive games involving:
- **Navigation & Physics** — move avatars, avoid walls, push objects
- **Object Interaction** — rotators change shape/color, doors, keys
- **Resource Management** — energy limits, step budgets, game-over conditions
- **Multi-level Progression** — typically 6 levels per environment, sequential solving

### State Representation
- Grids of color values (0–15)
- Score (levels completed)
- Game state enum (PLAYING, WIN, GAME_OVER)

### Available Agent Approaches

| Agent | Strategy | Key Tech |
|-------|----------|----------|
| **Random** | Random action selection | Baseline |
| **Reasoning** | Scientific method with LLM | o4-mini, PNG grid rendering with coordinate overlays |
| **Text LLM** | Grid → text, two-step observe→act | OpenAI API |
| **Multimodal** | Grid → pixel art + visual diffs | Vision LLM, two-stage (NL command → action) |
| **LangGraph** | Stateful graph with chat history | LangGraph functional API |
| **SmolAgents Code** | LLM writes & executes Python (BFS/DFS) | HuggingFace smolagents |
| **SmolAgents Vision** | Vision-based tool calling | HuggingFace smolagents |

---

## Fundamental Differences: AGI-2 vs AGI-3

```
ARC-AGI-2                           ARC-AGI-3
─────────                           ─────────
Static puzzles                      Dynamic games
All info given upfront              Must discover rules through play
Pattern recognition                 Scientific experimentation
Program synthesis                   Sequential decision-making
Single-step prediction              Multi-step interactive solving
Grid → Grid                         Agent → Environment → Feedback loop
"What's the pattern?"               "How does this world work?"
```

---

## Strategic Assessment for the Paper Track

### Path A: Focus on ARC-AGI-2
- **Pros:** Well-understood problem space, extensive prior art to build on, easier to show formal theory
- **Cons:** Harder to score on Novelty — approaches are more explored
- **Good for:** If you have ideas about program synthesis, DSLs, or compositional reasoning

### Path B: Focus on ARC-AGI-3
- **Pros:** Brand new paradigm → massive Novelty score potential, less competition, unique contribution
- **Cons:** Harder to achieve high Accuracy, less prior art to reference
- **Good for:** If you have ideas about active learning, world models, or scientific reasoning agents

### Path C: Both (Unified Theory)
- **Pros:** Maximum Theory and Progress scores — showing a unified approach to both static and interactive reasoning
- **Cons:** Most ambitious, risk of spreading too thin
- **Good for:** If you can frame both as instances of the same underlying reasoning problem

### Paper Scoring Reminder

| Category | Weight | Where to win |
|----------|--------|-------------|
| Accuracy | Equal | Higher leaderboard = higher score |
| Universality | Equal | Approach generalizes beyond ARC |
| Progress | Equal | Helps community reach 85% on ARC-AGI |
| Theory | Equal | Explain WHY, not just WHAT |
| Completeness | Equal | Cover every detail of your submission |
| Novelty | Equal | Original vs. existing research |

> **Key insight:** The $375K bonus pool (for papers scoring ≥ 4.5/5) rewards QUALITY over RANKING.
> A novel, well-theorized approach with modest accuracy can earn more than a high-scoring but undocumented submission.

---

## Project Data Inventory

```
arc-paper-track/
├── README.md                              ← Project overview
├── data/
│   ├── arc-agi-2/                         ← 7.1 MB — Competition dataset (1,360 tasks)
│   │   ├── arc-agi_training_challenges.json
│   │   ├── arc-agi_training_solutions.json
│   │   ├── arc-agi_evaluation_challenges.json
│   │   ├── arc-agi_evaluation_solutions.json
│   │   ├── arc-agi_test_challenges.json
│   │   └── sample_submission.json
│   ├── arc-agi-3/                         ← 90 MB — Interactive environments
│   │   ├── ARC-AGI-3-Agents/              ← Agent starter kit
│   │   ├── arc_agi_3_wheels/              ← Runtime wheels
│   │   └── environment_files/             ← 25 game environments (.py + metadata)
│   ├── arc-agi-3-agents/                  ← 1.2 MB — Agent templates (GitHub)
│   └── arc-agi-original/                  ← 6.1 MB — ARC-AGI-1 (800 tasks)
├── paper/                                 ← Paper drafts
├── notebooks/                             ← Experiments
├── src/                                   ← Your approach code
├── references/                            ← Related papers
└── submissions/                           ← Final artifacts
```

---

## Immediate Next Steps

1. **Decide your track**: ARC-AGI-2, ARC-AGI-3, or both
2. **Explore tasks hands-on**: Run a few ARC-AGI-2 tasks visually, play some ARC-AGI-3 environments
3. **Literature review**: Read top papers from ARC Prize 2024/2025
4. **Formulate your approach**: What's your unique angle?
5. **Start the code submission**: You need working code on the leaderboard before Nov 2
6. **Write the paper**: Due Nov 9 — start the structure early

# ARC Prize 2026 — Paper Track

> **Competition:** [ARC Prize 2026 - Paper Track](https://www.kaggle.com/competitions/arc-prize-2026-paper-track)
> **Prize Pool:** $450,000 USD
> **Deadline:** November 9, 2026
> **Teams:** ~199 (as of Sep 22, 2026)
> **Status:** ✅ Enrolled

---

## What Is This Competition?

The Paper Track is **NOT a coding competition**. It's a **research paper competition** where you document your conceptual approach to solving ARC-AGI-2 or ARC-AGI-3.

You need to:
1. Have a **working code submission** in either the ARC-AGI-2 or ARC-AGI-3 Kaggle track
2. Write a **comprehensive paper** explaining your methodology, theory, and implementation
3. Open-source all your code under a permissive license (CC0, MIT-0)

---

## What Is ARC-AGI?

**ARC (Abstraction and Reasoning Corpus)** is a benchmark created by **François Chollet** (creator of Keras) to measure **fluid intelligence** — the ability to reason through novel problems never seen before.

### Core Idea
- Tasks are **easy for humans** (~100% accuracy) but **hard for AI**
- Each task shows 2-5 input/output grid examples, then asks the AI to solve a new one
- You CANNOT brute-force it — each task requires understanding an abstract rule
- Intelligence is defined as **skill-acquisition efficiency**, not memorization

### Benchmark Versions

| Version | Status | What It Tests |
|---------|--------|---------------|
| **ARC-AGI-1** | Solved (~top end) | Original 2019 grid puzzles |
| **ARC-AGI-2** | Active competition ($700K) | Compositional reasoning, novel symbol interpretation — resists memorization |
| **ARC-AGI-3** | Active competition ($850K) | **Interactive/agentic** — agents navigate environments, take actions, observe feedback |

### Example Task (Conceptual)
```
Input:  ■ □ □     Output: ■ ■ □
        □ ■ □             □ ■ ■
        □ □ ■             □ □ ■

Rule: "Extend each filled cell one step to the right"
```
The AI must discover this rule from 2-3 examples and apply it to a new grid.

---

## Prize Structure

| Prize | Amount |
|-------|--------|
| 🥇 1st Place Paper | $50,000 |
| 🥈 2nd Place Paper | $20,000 |
| 🥉 3rd Place Paper | $5,000 |
| 🏆 Outstanding Papers Pool (score ≥ 4.5/5) | $375,000 shared |
| **Total** | **$450,000** |

> **Key insight:** The $375K outstanding papers pool means you DON'T need to win 1st place to earn significant money. If your paper scores 4.5+ on the rubric, you share that pool.

---

## Evaluation Rubric (0–5 per category)

| Category | What They Evaluate |
|----------|-------------------|
| **Accuracy** | Your performance on the ARC-AGI-2 or ARC-AGI-3 leaderboard |
| **Universality** | Does your approach generalize beyond this specific competition? |
| **Progress** | Could your work help the community reach 85% on ARC-AGI? |
| **Theory** | How well do you explain *WHY* your approach works? |
| **Completeness** | How thoroughly does your paper cover your submission? |
| **Novelty** | Is your approach original relative to existing public research? |

---

## Requirements Checklist

- [ ] Working code submission on ARC-AGI-2 OR ARC-AGI-3 Kaggle track
- [ ] Comprehensive paper documenting methodology
- [ ] Abstract summarizing the contribution
- [ ] Link from paper to Kaggle code submission
- [ ] Public Kaggle notebook
- [ ] Media gallery (where applicable)
- [ ] All code open-sourced under permissive license (CC0, MIT-0)

---

## Timeline

```
Mar 25, 2026  ─── Competition started
Sep 22, 2026  ─── TODAY (48 days remaining)
Nov 02, 2026  ─── ARC-AGI-2 & ARC-AGI-3 code deadline
Nov 09, 2026  ─── Paper Track submission deadline
```

> ⚠️ Your CODE must be submitted to ARC-AGI-2/3 BEFORE Nov 2. The paper is due 1 week later.

---

## Strategy Notes

### Why This Track Is Interesting
- **Only 199 teams** competing for $450K — extremely low competition/dollar ratio
- The $375K pool rewards quality, not just ranking — a well-written paper can earn money even without a top score
- **Theory and Novelty** are 2 of 6 categories — you can score well by having a creative, well-explained approach even without state-of-the-art accuracy

### What Makes a Strong Paper
1. **Clear theory** — Don't just describe what you did, explain WHY it works
2. **Honest evaluation** — Acknowledge limitations and failure modes
3. **Novelty** — A unique approach matters more than a high score with known techniques
4. **Reproducibility** — Open-source code, clear instructions, public notebook

---

## Key Resources

- [ARC Prize Official Site](https://arcprize.org/)
- [ARC-AGI Benchmark Overview](https://arcprize.org/arc-agi)
- [ARC Prize 2026 Competition Details](https://arcprize.org/competitions/2026)
- [ARC-AGI-2 Kaggle Track](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2)
- [ARC-AGI-3 Kaggle Track](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3)
- [ARC Prize Discord](https://discord.gg/9b77dPAmcA)
- [ARC Prize GitHub](https://github.com/arcprize)
- [ARC Tasks Explorer](https://arcprize.org/tasks)
- [François Chollet — On the Measure of Intelligence (Paper)](https://arxiv.org/abs/1911.01547)
- [Kaggle LLM Integration Guide](docs/KAGGLE_LLM_INTEGRATION_GUIDE.md) — Comprehensive guide for offline LLM inference on Kaggle
- [Kaggle Operations Runbook](docs/KAGGLE_OPS.md) and [Local & Kaggle Environment](docs/ENVIRONMENT.md) — Verified Kaggle CLI/API commands, the Paper Track writeup workflow, and local environment bootstrap

---

## Project Structure

```
arc-paper-track/
├── README.md              ← This file
├── docs/                  ← Technical guides & LLM integration reports
├── data/                  ← Competition data (NOTE.md)
├── paper/                 ← Paper drafts and figures
├── notebooks/             ← Kaggle notebooks and experiments
├── src/                   ← Source code for the approach
├── references/            ← Related papers and resources
└── submissions/           ← Submission artifacts
```

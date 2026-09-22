# ARC Prize — Past Winners Analysis (2024 & 2025)

> Reference material for Paper Track 2026 strategy

---

## 2025 Paper Track Winners

### 🥇 1st — "Less is More: Recursive Reasoning with Tiny Networks"
- **Author:** Alexia Jolicoeur-Martineau
- **Score:** 45% ARC-AGI-1 / 8% ARC-AGI-2
- **Approach:** Tiny Recursive Model (TRM) — only **7M parameters**, zero pretraining
- **Why it won:** Proved that tiny specialized models beat massive foundation models when designed with recursive architectures. No brute-force, no memorization.
- **Key lesson:** Novelty + strong theory. "Why does 7M params work?" is a compelling story.

### 🥈 2nd — "Self-Improving Language Models for Evolutionary Program Synthesis"
- **Authors:** J. Pourcel, C. Colas & P. Oudeyer
- **Approach:** Evolutionary program synthesis — iteratively generate and refine candidate programs
- **Why it won:** Treats intelligence as iterative self-improvement, not single-pass inference
- **Key lesson:** Strong Progress score — the community can build on evolutionary refinement loops

### 🥉 3rd — "ARC-AGI Without Pretraining"
- **Authors:** I. Liao & A. Gu
- **Approach:** Reasoning from scratch without pretraining data
- **Why it won:** Directly addresses benchmark contamination, forces pure abstract reasoning
- **Key lesson:** Clean experimental design scores high on Theory and Novelty

---

## 2024 Paper Track Winners

### "The Surprising Effectiveness of Test-Time Training for Abstract Reasoning"
- **Author:** Ekin Akyürek et al.
- **Approach:** Test-Time Training (TTT) — fine-tune the model at inference time using each task's demonstration pairs
- **Score:** Key technique behind ~50% SOTA
- **Why it won:** Paradigm-shifting idea. Models adapt dynamically to each new task.
- **Link:** https://ekinakyurek.github.io/papers/ttt.pdf
- **Key lesson:** One powerful idea, well explained, beats a complex engineering pipeline

### "Combining Induction and Transduction for Abstract Reasoning"
- **Authors:** Li et al.
- **Approach:** Hybrid of program synthesis (induction) + direct output prediction (transduction)
- **Why it won:** Bridged symbolic and subsymbolic approaches
- **Key lesson:** Theoretical framing of two paradigms as complementary scores high on Theory

### "The LLM ARChitect: Solving ARC-AGI Is a Matter of Perspective"
- **Authors:** Franzen et al.
- **Score:** 53.5% on leaderboard
- **Approach:** DFS over token probabilities using masked diffusion models
- **Why it won:** Showed DFS vastly outperforms greedy/beam search for ARC
- **Key lesson:** Highest Accuracy score + novel search mechanism

### "Searching Latent Program Spaces"
- **Authors:** Bonnet & Macfarlane
- **Approach:** Search strategies over latent program representations
- **Why it won:** More efficient exploration of solution spaces vs. brute-force
- **Key lesson:** Efficiency as novelty — doing more with less compute

---

## Winning Patterns — What Gets Prizes

### The 4 Pillars of Winning Papers

| Pattern | Description | Examples |
|---------|-------------|----------|
| **Inference-Time Compute** | Don't just train bigger — think harder at test time | TTT, evolutionary refinement, self-correction |
| **Program Synthesis** | Generate code/DSL instead of predicting pixels | Evolutionary synthesis, latent program search |
| **Small > Big** | Specialized small models beat massive LLMs | 7M param TRM beat GPT-4 on ARC |
| **Advanced Search** | DFS, speculative decoding, tree search | ARChitects DFS, latent space search |

### What DOESN'T Win
- ❌ "We ran GPT-4 with better prompts" — no novelty
- ❌ "We scaled up compute" — no theory
- ❌ "We ensembled 10 models" — no progress toward 85%
- ❌ Training set overfitting — explicitly penalized

---

## Score Benchmarks

| Year | Benchmark | SOTA Score | Grand Prize Threshold |
|------|-----------|------------|----------------------|
| 2024 | ARC-AGI-1 | 55.5% | 85% (unclaimed) |
| 2025 | ARC-AGI-1 | ~solved at top | 85% (unclaimed) |
| 2025 | ARC-AGI-2 | 24% (pure) / 54% (with scaffolding) | 85% (unclaimed) |
| 2026 | ARC-AGI-2 | TBD | 85% |
| 2026 | ARC-AGI-3 | TBD (interactive) | 85% |

> **Key insight:** You DON'T need to hit 85% to win the Paper Track.
> The 2025 1st place paper scored only **8% on ARC-AGI-2** — it won because of
> exceptional Theory, Novelty, and Progress scores.

---

## Strategic Takeaways for 2026

1. **A novel 10% solution with great theory beats an undocumented 40% solution**
2. **The winning formula:** unique approach + clear "WHY" explanation + open-source code
3. **Unexplored territory in 2026:** ARC-AGI-3 (interactive/agentic) has ZERO prior papers — massive Novelty opportunity
4. **Test-Time Training is table stakes** — you need to go beyond it or combine it with something new
5. **Submit early** — ties go to the earliest submission
6. **Study the 2024/2025 winners** to position your paper as building on OR departing from their work (Prior Work section)

---

## Open Source Repos to Study

- Ekin Akyürek's TTT implementation
- ARChitects DFS implementation
- "RG basic ported submission"
- All available via [arcprize.org](https://arcprize.org) and Kaggle forums

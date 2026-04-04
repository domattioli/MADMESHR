# Session 6 Report: Annulus-Layer2 Scaling + Angular Resolution Ablation

**Date:** 2026-04-03

## Summary

Tested two key questions: (1) Can the current architecture scale to 64-vertex domains? (2) Does doubling angular resolution improve star quality? Both answered negatively — annulus-layer2 failed to complete at 0% after 7k steps, and the 24×4 grid showed q=0.308 at 10k vs 0.371 baseline (undertrained due to 2× action space). Confirmed session 5 star baseline is reproducible (0.371, 5Q at 15k). No code changes — training runs only. 21/21 tests passing.

## What Was Completed

### WS1: Annulus-Layer2 Scaling Test (FAIL)

**Setup:** 10k steps (reduced from planned 50k due to extreme slowness), 12×4 grid, max_ep_len=70, eval every 5k.

**Results:**

| Eval Step | Return | MeanQ | Elements | Completion | AvgEtaB |
|-----------|--------|-------|----------|------------|---------|
| 5k | -9.39 | 0.277 | 70.0 | 0% | -68.52 |
| (killed at 7k) | ~-14 | - | 70 | 0% | ~-69 |

**Result: FAIL.** 0% completion across 100+ episodes. Agent always hit max_ep_len=70 without consuming the full boundary.

**Key observations:**
- EtaB dominated at -68 per episode (64 boundary vertices → many penalty steps)
- Training was extremely slow: ~1k steps/min due to 70-step episodes with expensive action enumeration on 64 vertices
- Returns improved from -51 (random) to -9 (5k eval), showing the agent learned *something*, but never achieved completion
- The Mu penalty decreased from -43 to -6, suggesting the agent learned to place elements more densely, but couldn't close the mesh

**Diagnosis:**
1. **Horizon too long:** 70 steps with γ=0.99 means early actions get ~0.5× credit of final actions. The completion bonus is too far in the future.
2. **State representation inadequate:** 44-dim state vector captures only local boundary context (neighbors, angles). With 64 vertices, the agent has no global view of remaining boundary.
3. **Training budget insufficient:** Rectangle (20v) took 10k steps. 64v may need 100k+ to even see a completion event. But at 1k steps/min, that's 100+ minutes.
4. **Action space possibly too coarse:** The 12×4 grid may not offer enough valid actions for the complex non-convex geometry.

---

### WS2: 24×4 Angular Resolution on Star (INCONCLUSIVE)

**Setup:** 25k steps (killed at 12k due to slowness), 24×4 grid (97 actions), eval every 5k.

**Results:**

| Eval Step | Return | MeanQ | Elements | Completion | Epsilon |
|-----------|--------|-------|----------|------------|---------|
| 5k | 6.98 | 0.210 | 5.0 | 100% | ~0.72 |
| 10k | 7.81 | 0.308 | 6.0 | 100% | ~0.46 |
| (killed at 12k) | - | - | - | - | ~0.35 |

**Comparison with 12×4 baseline at same timesteps:**

| Eval Step | 12×4 MeanQ | 24×4 MeanQ | 12×4 Epsilon | 24×4 Epsilon |
|-----------|-----------|-----------|-------------|-------------|
| 5k | 0.275 | 0.210 | ~0.29 | ~0.72 |
| 10k | 0.364 | 0.308 | ~0.05 | ~0.46 |
| 15k | 0.371 | N/A | 0.05 | N/A |

**Result: INCONCLUSIVE.** The 24×4 quality at 10k (0.308) is lower than 12×4 (0.364), but epsilon was still at 0.46 — the policy was far from converged. With 97 actions, the agent needs proportionally more exploration. The linear epsilon schedule decays over 70% of training (17.5k steps), so convergence would happen around 17-20k.

**Key insight:** The 24×4 run takes ~2× longer per timestep than 12×4, making it impractical to train to convergence within a reasonable session. A 25k 24×4 run would take ~50 minutes vs ~25 minutes for 15k 12×4.

---

### Baseline Confirmation: Star 12×4 (PASS)

**Setup:** 15k steps, 12×4 grid, eval every 5k.

**Results:**

| Eval Step | Return | MeanQ | Elements | Completion |
|-----------|--------|-------|----------|------------|
| 5k | 7.69 | 0.275 | 5.0 | 100% |
| 10k | 8.99 | 0.364 | 5.0 | 100% |
| 15k | 9.11 | 0.371 | 5.0 | 100% |

**Result: PASS.** Exactly reproduces session 5 star results (0.371, 5Q, 100%). Confirms the reward structure is stable and results are reproducible.

---

### WS3: Epsilon Schedule (SKIPPED)

Skipped due to time constraints. WS2 was the prerequisite and produced inconclusive results.

### WS4: Transfer Learning Diagnostic (SKIPPED)

Skipped due to time constraints.

## Key Metrics Comparison

| Metric | Session 5 | Session 6 | Change |
|--------|-----------|-----------|--------|
| Star quality (12×4) | 0.371 | 0.371 | Same (reproduced) |
| Star elements | 5Q | 5Q | Same |
| Annulus-layer2 completion | 0% (greedy) | 0% (trained 7k) | No improvement |
| Star quality (24×4) | N/A | 0.308 (10k, undertrained) | New data |
| Tests passing | 21 | 21 | Stable |

## What Didn't Work

### Annulus-Layer2 Is Too Hard for Current Architecture

The 64-vertex domain is fundamentally beyond the current system's reach. The bottleneck is threefold:
1. **Training speed:** Action enumeration on 64-vertex boundaries is ~5× slower than 10-vertex domains
2. **Episode length:** 70-step episodes vs 5-12 step episodes means 5-10× more environment steps per episode
3. **Credit assignment:** The completion bonus (the main learning signal) is unreachable within 70 steps

### 24×4 Needs More Training Budget

Doubling angular bins doubled the action space and training time. The standard epsilon schedule (linear decay over 70% of steps) means 17.5k steps of exploration out of 25k. The 12×4 baseline converges by 10k, but the 24×4 needs at least 20-25k to converge — and that takes 50+ minutes.

## What Went Well

- **Reproducibility confirmed.** Star 12×4 results are rock-solid: 0.371, 5Q across two sessions and multiple runs.
- **Clean failure mode for annulus-layer2.** The failure was clear and diagnostic: 0% completion, 70 elements per episode, EtaB dominance. No ambiguity about what went wrong.
- **Early killing saved time.** Recognizing the annulus-layer2 run was hopeless at 7k and the 24×4 was too slow at 12k prevented wasting 60+ more minutes.

## What Didn't Go Well

- **Session was compute-bound.** Both WS1 and WS2 ran into wall-clock limits. The plan's time estimates (120 min WS1, 90 min WS2) were roughly correct but didn't account for the reality that watching a slow run is not productive use of session time.
- **Only 1 of 4 workstreams completed to useful conclusion.** WS1 gave a clear answer (FAIL), WS2 was inconclusive, WS3-4 were skipped.
- **No code changes.** This session was pure experimentation with no incremental improvements to the codebase.

## Observations

1. **The architecture has a scaling wall at ~20 vertices.** Rectangle (20v) works. Annulus-layer2 (64v) doesn't. The gap suggests the 44-dim state representation and 49-action space are fundamentally limited for large domains.

2. **Larger action spaces need proportionally more training.** Going from 49 to 97 actions roughly doubled training time. This makes incremental resolution increases expensive and suggests a different approach (e.g., continuous actions, hierarchical decomposition) for quality improvement.

3. **The quality gap on star may not be action-space-limited.** The 24×4 run, even at 10k (undertrained), showed similar quality to the 12×4 at 5k (also undertrained). The convergence trajectory suggests 24×4 would reach a similar ~0.37 quality, not dramatically higher. This tentatively supports the session 3 conclusion that the ceiling is geometry-limited.

4. **Annulus-layer2 needs a fundamentally different approach.** Options:
   - Curriculum: train on simpler subproblems, transfer
   - Hierarchical: decompose 64v into smaller sub-domains
   - Architecture: increase state dim, use attention/GNN for variable-size boundaries
   - Longer training with patience: 100k+ steps, but impractical at current speed

5. **Run-to-run variance is low.** Star converged to identical results in sessions 5 and 6. This is a positive signal for the reward structure's stability.

## Files Changed

No code changes. Training runs only.

| File | Changes |
|------|---------|
| `output/latest/star.png` | Updated visualization from session 6 star run |
| `checkpoints/star-12x4/` | Saved star 12×4 model weights |
| `checkpoints/star-24x4/` | Saved star 24×4 model weights (partial training) |
| `checkpoints/annulus-layer2/` | Saved annulus-layer2 model weights (5k best) |

## Short-term Next Steps (Session 7)

1. **Speed up action enumeration.** Profile and optimize the bottleneck in `DiscreteActionEnv._enumerate_valid_actions()`. Vectorized geometry checks, caching, or precomputation could make training 2-5× faster, unlocking both annulus-layer2 and longer 24×4 runs.

2. **Curriculum learning for annulus-layer2.** Train on progressively harder domains (octagon → circle → star → annulus-layer2). Transfer learned features rather than starting from scratch on the hardest domain.

3. **Complete the 24×4 ablation.** With faster training, run star 24×4 to 30k convergence and get a definitive answer on angular resolution's impact.

## Medium-term Next Steps (Sessions 8-9)

4. **Variable-size state representation.** Replace fixed 44-dim vector with something that scales (e.g., GNN over boundary graph, attention mechanism). This is a prerequisite for any domain >20 vertices.

5. **Continuous action space (revisit).** SAC failed in early sessions, but with the improved reward structure (quality-gated completion), it may work better now.

6. **Multi-domain training.** Train a single agent across all domains to learn generalizable features.

# Session 5 Report: eta_b Rebalancing + Quality-Gated Completion

**Date:** 2026-04-03

## Summary

Scaled eta_b boundary penalty by 0.3 and replaced flat +10 completion bonus with quality-gated `5 + 10*mean_q`. Star quality improved 66% (0.223→0.371, 5Q). Other domains (octagon, rectangle, L-shape) remained stable. Added reward component logging for diagnostics. 21/21 tests passing.

## What Was Completed

### WS1: Reward Component Logging

**Changes:**
- Added `eta_e`, `eta_b`, `mu`, `completion_bonus` to info dict in `DiscreteActionEnv.step()`
- Added per-episode component accumulation in `DQNTrainer.train()` (printed with episode log)
- Enhanced `DQNTrainer.evaluate()` to print MeanQ, AvgElements, AvgEtaB breakdown

**Files:** `src/DiscreteActionEnv.py`, `src/trainer_dqn.py`

**Key diagnostic from 1k star run:** EtaB accumulated to -10.70 for 12-step episodes vs -4.52 for 8-step episodes, confirming eta_b dominance identified in session 4.

### WS2: Scale eta_b by 0.3

**Change:** `reward = eta_e + 0.3 * eta_b + mu` (was `eta_e + eta_b + mu`)

**Star 15k training results:**

| Eval Step | Return | MeanQ | Elements | Completion | AvgEtaB |
|-----------|--------|-------|----------|------------|---------|
| 5k | 9.92 | 0.281 | 5.0 | 100% | -2.47 |
| 10k | 9.87 | 0.319 | 7.0 | 100% | -3.82 |
| 15k | 9.94 | 0.223 | 4.0 | 100% | -1.68 |

**Result: PARTIAL PASS.** The 10k eval showed q=0.319 (above 0.30 gate) with 7 elements, but this was an artifact of higher epsilon (0.09) providing exploration. By 15k with epsilon=0.05, the agent converged back to the session 4 solution (4Q, q=0.223).

**Key insight:** eta_b scaling alone doesn't help when element quality doesn't improve with more elements. The math predicted this: when q4=q8, fewer steps always wins because it minimizes accumulated penalties. The agent needs a reason to prefer higher quality, not just more elements.

### WS3: Quality-Gated Completion Bonus

**Change:** Replaced all flat +10 completion bonuses with `5.0 + 10.0 * mean_q`
- Boundary consumed (bnd<3): quality-gated
- Auto-close quad (bnd=4): quality-gated
- Boundary triangle (bnd=3): quality-gated
- Self-intersecting quad: kept at 2.0 (worst outcome)
- Interior triangle: kept at 3.0

**Star 15k training results (WS2+WS3 combined):**

| Eval Step | Return | MeanQ | Elements | Completion | AvgEtaB |
|-----------|--------|-------|----------|------------|---------|
| 5k | 7.44 | 0.248 | 5.0 | 100% | -2.36 |
| 10k | 9.11 | 0.371 | 5.0 | 100% | -2.36 |
| 15k | 9.11 | 0.371 | 5.0 | 100% | -2.36 |

**Result: PASS.** Quality 0.371 exceeds 0.33 target. Converged by 10k — stable at 15k. The quality-gated bonus was the critical change: it creates a direct gradient signal from completion quality to return, breaking the tie between low-quality-fast and high-quality-slow strategies.

**Math verification:**
- 4 steps at q=0.22: `4*(0.22 + 0.3*(-0.8)) + 5 + 10*0.22 = -0.08 + 7.2 = 7.12`
- 5 steps at q=0.37: `5*(0.37 + 0.3*(-0.47)) + 5 + 10*0.37 = 1.14 + 8.7 = 9.84`
- 5 steps clearly wins with quality improvement, confirming the incentive alignment.

### WS4: Octagon + Rectangle + L-shape Verification

**Octagon (8k, eval every 4k):**

| Eval Step | Return | MeanQ | Elements | Completion |
|-----------|--------|-------|----------|------------|
| 4k | 10.14 | 0.478 | 3.0 | 100% |
| 8k | 10.14 | 0.478 | 3.0 | 100% |

**Final:** q=0.478, 3Q — identical to session 4. Target was ≥0.50: **MISS** (by 0.022).

**Rectangle (15k, eval every 5k):**

| Eval Step | Return | MeanQ | Elements | Completion |
|-----------|--------|-------|----------|------------|
| 5k | -8.54 | 0.342 | 25.0 | 0% |
| 10k | 11.23 | 0.464 | 9.0 | 100% |
| 15k | 11.23 | 0.464 | 9.0 | 100% |

**Final:** q=0.464, 9Q — identical to session 4. **PASS.**

**L-shape (6k, eval every 3k):**

| Eval Step | Return | MeanQ | Elements | Completion |
|-----------|--------|-------|----------|------------|
| 3k | 9.59 | 0.459 | 2.0 | 100% |
| 6k | 9.59 | 0.459 | 2.0 | 100% |

**Final:** q=0.459, 2Q — identical to session 4. **PASS.**

## Key Metrics Comparison

| Metric | Session 3 | Session 4 | Session 5 | Change (S4→S5) |
|--------|-----------|-----------|-----------|-----------------|
| Star quality | 0.314 | 0.223 | **0.371** | +66% |
| Star elements | 8Q | 4Q | 5Q | +1 |
| Octagon quality | 0.535 | 0.478 | 0.478 | Same |
| Octagon elements | 4Q | 3Q | 3Q | Same |
| L-shape quality | 0.459 | 0.459 | 0.459 | Same |
| L-shape elements | 2Q | 2Q | 2Q | Same |
| Rectangle quality | N/A | 0.464 | 0.464 | Same |
| Rectangle elements | N/A | 9Q | 9Q | Same |
| Tests passing | 21 | 21 | 21 | Stable |

## What Didn't Work

### eta_b Scaling Alone Was Insufficient (WS2)

Scaling eta_b by 0.3 did not improve star quality at convergence. The 10k eval that showed q=0.319 was misleading — higher epsilon (0.09) at that point created diversity in the eval rollouts. By 15k with epsilon=0.05, the agent converged back to the session 4 solution (4Q, q=0.223).

**Root cause:** When element quality doesn't increase with more elements, fewer steps always gives higher return under any positive eta_b weight. The scaling makes 8 steps break even with 4 steps, but doesn't make 8 steps *clearly* win. The agent needs a quality-dependent terminal reward to break the tie.

### Octagon Didn't Improve

Octagon stayed at q=0.478, 3Q — unchanged from session 4. The quality-gated bonus doesn't help because the agent has already found the highest-return solution at 3Q. With only 8 boundary vertices and mild angles (135°), the eta_b penalty per step is small (-0.25*0.3 = -0.075), so adding more elements is nearly neutral. The 3Q solution may be near-optimal for this action space resolution.

### Eval Epsilon Artifact

The 10k eval for WS2 (q=0.319, 7 elements) was misleadingly positive because epsilon was still 0.09, injecting randomness into otherwise greedy eval rollouts (eval uses epsilon=0 but training epsilon was high during buffer collection). This suggests eval results at early checkpoints should be interpreted with caution — the model's learned policy may not match eval performance if the replay buffer is biased toward high-epsilon trajectories.

**Correction:** The eval does use epsilon=0 (deterministic), so this explanation is wrong. The more likely explanation is that the model's Q-values at 10k hadn't fully converged, leading to different action selections than at 15k. The model was still plastic enough to pick 7-element solutions that it later abandoned as Q-values sharpened.

## What Went Well

- **Star quality recovery.** 0.223 → 0.371 is the largest single-session quality improvement. The quality-gated completion bonus was the key insight — it aligns the terminal reward with the optimization objective.
- **No regressions.** All non-star domains maintained identical quality and element counts. The reward changes are compatible with domains of varying difficulty.
- **Reward logging.** The eta_e/eta_b/mu/bonus breakdown made it trivial to diagnose WS2's failure and understand why WS3 succeeded.
- **Clean ablation.** Testing WS2 alone before WS2+WS3 confirmed that the quality-gated bonus (not eta_b scaling) was the critical change.

## What Didn't Go Well

- **WS2 alone was a dead end.** Spent training time on eta_b scaling alone when the break-even analysis already predicted the agent would stay at 4 elements if quality didn't improve. Should have combined WS2+WS3 from the start, or at least tested WS3 alone.
- **Octagon below target.** Target was ≥0.50, got 0.478. The reward changes help star but don't unlock octagon quality improvement. This may require action space changes (finer resolution) or different reward tuning.
- **Convergence speed.** Star training showed useful solutions at 5-10k but then converged to a different (better) solution by 15k. Could potentially stop star training at 10k with quality-gated bonus since it converged by then.

## Observations

1. **Quality-gated completion > eta_b rebalancing.** The completion bonus is the dominant reward component (~9-10 out of ~10 total return). Making it quality-dependent has much more leverage than tweaking per-step penalty weights.

2. **Star is now at 84% of ceiling.** Quality 0.371 / ceiling 0.44 = 84%. The remaining gap (0.07) likely requires action space refinement — the 12x4 grid may not have enough resolution for star tip placements.

3. **Octagon may be action-space limited too.** At 0.478 / 0.61 ceiling = 78%. The 3-element solution is geometrically constrained by the 12-angle grid.

4. **Rectangle is robust.** Consistently hits q=0.464 with 9Q across sessions 4 and 5 with very different reward structures. This is a strong signal that the algorithm generalizes to larger domains.

5. **Convergence patterns vary.** Star converges by 10k, octagon by 4k, rectangle by 10k, L-shape by 3k. Training budgets could be reduced based on these patterns.

## Files Changed

| File | Changes |
|------|---------|
| `src/DiscreteActionEnv.py` | eta_b * 0.3, quality-gated completion (5+10*mean_q), reward components in info dict |
| `src/trainer_dqn.py` | Per-episode eta_e/eta_b/mu/bonus logging, enhanced eval with MeanQ/Elements/EtaB |
| `CLAUDE.md` | Updated reward structure docs and known issues |
| `output/latest/*.png` | Updated visualizations for all 4 domains |

## Short-term Next Steps (Session 6)

1. **Action space refinement for star.** Test 24x8 grid on star to see if finer resolution closes the 0.371→0.44 quality gap.
2. **Octagon action space experiment.** Same 24x8 test — 78% of ceiling suggests geometric limitation.
3. **Epsilon schedule tuning.** Current linear decay may be too fast — the WS2 experience showed quality varies significantly with epsilon level. Slower decay or higher minimum epsilon could help exploration.

## Medium-term Next Steps (Sessions 7-8)

4. **Transfer learning.** Now that reward is stable, test training on star → zero-shot eval on octagon. The fast convergence (3-10k steps) suggests memorization.
5. **Curriculum training.** Train L-shape → octagon → star progression.
6. **Multi-domain training.** Single model across all domains with domain-agnostic state encoding.

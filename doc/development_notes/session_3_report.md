# Session 3 Report: Reward Farming Fix + Concave Domain Support

**Date:** 2026-04-03

## Summary

Fixed the critical reward farming problem and enabled concave domain meshing. All three priority workstreams completed. The DQN agent now produces efficient, completion-focused policies across star, octagon, and L-shape domains — element counts dropped from 20-44 (farming) to 2-8 (efficient). Training converges in 2-10k steps, not 50-100k.

## What Was Completed

### 1. Reward Farming Fix (WS1)

**Changes:**
- Quality weight: 2.0 → 0.3 (reduced 6.7x)
- Step penalty: 0.01 → 0.05 (increased 5x)
- Domain-specific `max_ep_len` added to domain registry: star=12, octagon=10, l-shape=10, circle=15, rectangle=25, square=5

**Impact on reward math:**
- 5-step efficient episode: ~1.4 intermediate + 7.1 completion = ~8.5 total
- 12-step farming attempt: ~3.6 intermediate (truncated, no completion) — farming no longer pays
- Completion bonus now 2-5x larger than total intermediates

**Result:** Reward farming completely eliminated. All three domains achieve 100% completion with efficient element counts.

### 2. Convexity Constraint Relaxed (WS2)

**Changes:**
- `_is_valid_quad()`: removed `_is_convex_quad()` gate, now only checks self-intersection
- `_batch_is_valid_quad()`: removed convexity check from vectorized path, returns `no_intersect` only
- Both methods still reject self-intersecting quads

**Impact:** L-shape domain went from ~0 valid actions near the concave vertex to 20 valid actions. Concave quads get lower quality scores naturally through the angle-based quality metric — no hard gate needed.

### 3. Multi-Domain Training (WS3)

All models trained from scratch (no checkpoint loading from session 2).

#### Star (10 vertices, 15k steps, max_ep_len=12)

| Eval Step | Return | Completion | Observation |
|-----------|--------|------------|-------------|
| 3k | 6.03 | 100% | First eval already completing efficiently |
| 6k | 6.82 | 100% | Improving, new best |
| 9k | 7.18 | 100% | Best model, 6-8 elements typical |
| 12k | 6.88 | 100% | Stable, slight return variation |

**Final eval (best model):** return=7.18, quality=0.314, 8Q+0T, complete
**Training completion rate:** 77% (1137/1476 episodes)

#### Octagon (8 vertices, 15k steps, max_ep_len=10)

| Eval Step | Return | Completion | Observation |
|-----------|--------|------------|-------------|
| 3k | 7.08 | 100% | Already strong |
| 6k | 7.64 | 100% | Converging on 3-4 element solutions |
| 9k | 7.64 | 100% | Stable |
| 12k | 8.66 | 100% | New best — highest return across all domains |

**Final eval (best model):** return=8.66, quality=0.535, 4Q+0T, complete
**Training completion rate:** 89% (2449/2747 episodes)

#### L-Shape (6 vertices, concave, 10k steps, max_ep_len=10)

| Eval Step | Return | Completion | Observation |
|-----------|--------|------------|-------------|
| 2k | 7.11 | 100% | Converged immediately — found 2-quad solution |
| 4k | 7.11 | 100% | Stable |
| 6k | 7.11 | 100% | No further improvement (converged at ~2k) |

**Final eval (best model):** return=7.11, quality=0.459, 2Q+0T, complete
**Training completion rate:** 91% (1557/1707 episodes)

**Note:** L-shape converged by 2k steps. In hindsight, should have killed training much earlier and moved on.

### 4. Quality Diagnostic — L-Shape Ceiling

| Domain | Resolution | Max Quality | Mean Max | Greedy Steps | Complete? |
|--------|-----------|-------------|----------|--------------|-----------|
| L-shape (6v) | 12x4 | 0.641 | 0.457 | 4 | Yes (5 elements) |

The agent found a more efficient 2-element solution than the greedy-by-quality baseline (4 steps, 5 elements). The agent optimized for completion over quality — correct behavior under the new reward structure.

### 5. Infrastructure

- **CLAUDE.md** created with project overview, commands, architecture, key patterns, adversarial planning methodology reference
- **Domain registry** now stores `max_ep_len` per domain via `@register_domain(name, desc, max_ep_len=N)` decorator
- Mesh visualizations saved to `output/latest/` and tracked in git (removed .gitignore exclusion)

## What Didn't Work

### Nothing Major Failed This Session

The adversarial planning process from session 2 correctly identified all three problems (reward farming, convexity gate, need for concave domain testing) and the proposed fixes worked on first attempt.

### Minor: GPU OOM on Parallel Training

Three parallel TF training processes exhausted GPU memory (RTX 3060, 12GB). L-shape training crashed at ~50 steps. Fix: ran star + octagon first, then L-shape sequentially. For future: either force CPU mode or run sequentially.

### Minor: Over-training on Converged Models

L-shape converged at ~2k steps but trained to 6k before being manually stopped. Star and octagon were stopped at ~12k when they had converged by ~9k. Future sessions should monitor eval metrics more actively and kill runs when plateau detected.

## Key Metrics Comparison

| Metric | Session 2 | Session 3 | Change |
|--------|-----------|-----------|--------|
| Star completion | 100% (3/4 evals) | 100% (4/4 evals) | Stable |
| Star elements/episode | 20-44 (farming) | 6-8 | **Fixed** |
| Star return | 10-48 (oscillating) | 6.0-7.2 (stable) | **Stabilized** |
| Star mean quality | ~0.37 | 0.314 | Slight decrease (fewer elements = less cherry-picking) |
| Octagon completion | Not tested | 100% | **New** |
| Octagon quality | Not tested | 0.535 | **New** |
| L-shape completion | Would fail (convexity gate) | 100% | **New** |
| L-shape quality | N/A | 0.459 | **New** |
| Training timesteps needed | 43k+ (not converged) | 2k-10k (converged) | **5-20x faster** |
| Tests passing | 21 | 21 | Stable |

## Observations

1. **Reward farming is solved.** The 0.3x quality + 0.05 step penalty + domain-specific max_ep_len eliminates farming completely. Completion bonus dominates intermediate rewards by 2-5x.

2. **Quality vs efficiency trade-off is real.** Star quality dropped from ~0.37 to 0.314 because the agent places fewer elements (8 vs 20-44). With 20+ elements, the agent was cherry-picking easy placements. With 8 elements, each must cover more area, reducing average quality. This is the correct trade-off — we want fewer, complete meshes.

3. **L-shape proved concave domains work.** The relaxed convexity constraint enables meshing of any simple polygon. Quality metric naturally penalizes bad elements without needing a hard gate.

4. **Training converges fast.** 2-10k steps is sufficient for 6-10 vertex domains. The 50-100k budgets from session 2 were massive overkill, especially once farming was eliminated.

5. **Element count correlates with domain complexity.** L-shape: 2Q, Octagon: 4Q, Star: 8Q. Makes geometric sense — star has 10 acute angles requiring more subdivisions.

## Post-Session Analysis: Reward Drift from Pan et al.

Comparing our active reward to Pan et al.'s (FreeMeshRL, Eq. 5-9) reveals we've drifted from the paper's formulation in two structurally important ways that explain the degenerate quads:

**Pan et al. per-step reward:** `r = eta_e + eta_b + mu`
- `eta_e` = element quality (0 to 1) — full weight, not scaled down
- `eta_b` = boundary quality penalty (-1 to 0) — penalizes actions that leave sharp remaining angles on the boundary
- `mu` = density penalty (-1 to 0) — penalizes elements below minimum area threshold; zero above max area; bounded negative
- Completion: flat +10
- Per-step range: roughly -2.2 to +1.0

**Our active per-step reward:** `r = 0.3 * quality + area_consumed - 0.05`
- quality signal (0 to 0.3) — scaled down to prevent farming
- area_consumed (positive, variable per step)
- step penalty

**Problem 1: `area_consumed` inverts Pan's density incentive.** Pan's `mu` is a *bounded negative penalty* that discourages too-small elements. Our `area_consumed` is a *positive reward* that encourages covering area. This inverts the incentive structure: our agent is rewarded for big sloppy elements (high area consumed) while Pan's agent is only penalized for making elements too small. The farming fix reduced quality weight to 0.3x as a blunt countermeasure, but the root cause is that `area_consumed` as a positive reward is structurally wrong.

**Problem 2: No boundary quality term (`eta_b`).** Pan's `eta_b` penalizes actions that create sharp remaining angles on the boundary — it's a forward-looking signal. Without it, our agent has zero incentive to preserve boundary quality for future steps. The 0.04-quality quad visible in the star mesh is a direct consequence: the agent takes a high-area action that leaves a terrible boundary configuration, and nothing penalizes that.

**Implication:** The session 4 adversarial review was partially wrong. It concluded that the quality gap is purely geometric (action space limitation) and that reward tuning cannot help. This is true for the *ceiling* (max achievable quality) but not for the *gap between agent quality and ceiling*. Star agent quality is 0.314 vs ceiling 0.44 — a 29% gap. The agent is not even approaching the ceiling because it has no incentive to avoid boundary-destroying moves. Returning to Pan's reward structure could close much of this gap without changing the action space.

## Files Changed

| File | Changes |
|------|---------|
| `src/DiscreteActionEnv.py` | Quality weight 2.0→0.3, step penalty 0.01→0.05 |
| `src/MeshEnvironment.py` | Removed convexity gate from `_is_valid_quad()` and `_batch_is_valid_quad()` |
| `main.py` | Domain-specific `max_ep_len` in registry, wired to DQNTrainer |
| `.gitignore` | Removed `output/latest/*.png` exclusion |
| `CLAUDE.md` | New: project guide with architecture, commands, planning methodology |
| `output/latest/` | star.png, octagon.png, l-shape.png mesh visualizations |

## Next Session Priority

**Return to Pan et al.'s reward structure.** The reward function has drifted from the paper's formulation in structurally harmful ways: `area_consumed` as a positive reward inverts Pan's bounded density penalty, and the missing `eta_b` boundary quality term removes forward-looking incentives. Fixing these should close the quality gap (star 0.314 vs ceiling 0.44) without re-enabling farming, since Pan's formulation handles this natively through bounded penalties.

See `doc/development_notes/session_4_plan.md`.

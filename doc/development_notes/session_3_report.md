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

**Quality optimization.** Completion is solved across all tested domains. The agent consistently completes but doesn't optimize for element quality (star=0.314 vs ceiling=0.44, octagon=0.535 vs ceiling=0.61). The completion bonus dominates so heavily that quality signal is nearly irrelevant. Session 4 should focus on making quality matter once the mesh is complete.

See `doc/development_notes/session_4_plan.md`.

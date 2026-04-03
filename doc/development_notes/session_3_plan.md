# Session 3 Plan: Fix Reward Farming + Concave Domain Support

**Date:** Planned for next session after 2026-04-03
**Status:** Draft (adversarial-reviewed)

## Context

Session 2 revealed two critical issues:

1. **Reward farming:** The agent places 20-44 elements per episode to accumulate intermediate rewards rather than completing efficiently in ~5 steps. With per-step reward of `2.0*quality + 1.0*area_consumed - 0.01`, a 44-step farming episode earns ~33 in intermediates while the completion bonus is only ~6.4. Training on star (stopped at 43k/100k) showed eval oscillation: 100% completion at 10k (return=48, farming), 0% at 20k (destabilized), 100% at 30k (return=21, moderate), 100% at 40k (return=39, farming returning). The farming behavior is the rational policy under current rewards.

2. **Convexity constraint blocks concave domains:** `_is_valid_quad()` requires all quads to be convex. On L-shaped or any concave domain, valid mesh quads near concavities are non-convex and get rejected. The agent truncates with no valid actions.

Quality ceiling diagnostic showed star caps at 0.44 (geometry-limited), circle at 0.78. Training on circle would prove nothing — the agent already solves it trivially. The real test is concave/asymmetric domains.

## Adversarial Review Summary

Two adversarial agents agreed:
- **Fix reward farming FIRST** — nothing else matters if the agent's rational policy is to farm
- **Circle training is pointless** — too easy, proves nothing about quality optimization
- **Drop SAC migration** — the algorithm is adequate, the reward function is broken
- **Fix convexity constraint** — prerequisite for any concave domain work
- **Reduce intermediate rewards dramatically** so completion bonus always dominates

## Workstreams

### WS1: Fix Reward Farming (Priority 1, ~60 min)
**The critical fix.** Without this, all other work is wasted.

**Root cause:** Per-step intermediate rewards (total ~33 for 44 steps) exceed completion bonus (~6.4). The 0.01 step penalty is negligible (0.44 total across 44 steps vs ~60 in quality rewards).

**Proposed fix (two changes):**

1. **Reduce per-step quality weight 10x:**
   ```python
   # From:
   reward = 2.0 * quality + 1.0 * area_consumed - 0.01
   # To:
   reward = 0.3 * quality + 1.0 * area_consumed - 0.05
   ```
   - Quality signal preserved but bounded (max 0.3 vs 2.0)
   - Area_consumed naturally sums to ~1.0 across full episode (self-limiting)
   - Step penalty 5x larger (0.05 vs 0.01)
   - 5-step episode: ~1.4 intermediate + 7.5 completion = ~8.9 total
   - 44-step farming: ~6.3 intermediate (no completion) -- farming no longer pays

2. **Domain-specific max_ep_len:**
   - star=12, octagon=10, circle=15, l-shape=10, rectangle=25
   - Add to domain registry in main.py
   - Blunt but effective backstop against farming

**Verification:** Train on star 50k steps. Confirm:
- Elements per episode <= 10 (not 20-44)
- Completion rate > 90%
- Mean quality approaches ceiling (~0.42-0.44)

**Files:** `src/DiscreteActionEnv.py`, `main.py`, `src/trainer_dqn.py`

---

### WS2: Relax Convexity Constraint (Priority 2, ~30 min)
**Prerequisite for L-shape and any concave domain.**

**Current behavior:** `_is_valid_quad()` (line 468) calls `_is_convex_quad()` which requires all cross products same sign. Rejects valid concave quads.

**Fix:** Replace convexity check with self-intersection check only. Concave quads have lower quality naturally (the angle quality metric penalizes them), so reward provides the right signal without a hard constraint.

- Modify `_is_valid_quad()` to use `_has_self_intersection()` instead of `_is_convex_quad()`
- Similarly update `_batch_is_valid_quad()` for the vectorized path
- Keep `_is_convex_quad()` method available but don't use it as a gate

**Verification:** Run `quality_diagnostic.py` on L-shape domain — should see valid actions at concave vertex.

**Files:** `src/MeshEnvironment.py`

---

### WS3: Train on L-Shape Domain (Priority 3, ~45 min)
**The real quality optimization test.** L-shape has:
- 6 vertices, concave vertex at (1,1)
- Quality ceiling unknown (run diagnostic first)
- Requires multi-step planning around the concavity
- Not trivially solvable by greedy heuristic (unlike circle)

**Steps:**
1. Run quality diagnostic on L-shape at 12x4 and 24x8
2. Train with fixed reward structure from WS1
3. Measure completion rate and quality
4. Compare greedy-by-quality baseline vs trained agent

**Verification:** >50% completion, quality approaching ceiling.

**Files:** Uses existing `main.py --domain l-shape`

---

### WS4: Add Quality to State Representation (Priority 4, if time)
**Problem:** Agent cannot observe its running mean quality during an episode. It has no information to make quality-aware decisions.

**Fix:** Add `mean_quality_so_far` as a feature in enriched state (float, position 43 replacing the padding zero). This gives the agent information about whether it should prioritize quality or completion.

**Files:** `src/MeshEnvironment.py` (_get_enriched_state)

---

## Execution Order

```
WS1 (fix farming, 60 min)
  |
WS2 (relax convexity, 30 min) -- can partially overlap with WS1 training
  |
WS3 (L-shape training, 45 min)
  |
[If time]
  +-- WS4 (quality in state, 20 min)
```

## Key Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| Reduced quality weight kills quality signal | 0.3*quality still provides 0.15 difference between q=0.3 and q=0.8 per step |
| max_ep_len too tight | Start with 2x optimal, increase if completion drops |
| Relaxing convexity creates self-intersecting elements | _has_self_intersection still gates validity |
| L-shape quality ceiling too low | Run diagnostic first; if ceiling <0.5 at 12x4, try 24x8 |

## What NOT to Do Next Session

- **Don't train on circle** -- too easy, proves nothing
- **Don't migrate to SAC** -- algorithm is fine, reward is broken
- **Don't do curriculum learning** -- solves a problem we don't have (exploration), not the one we have (reward design)
- **Don't increase action resolution** -- diagnostic showed it doesn't help for star; test on L-shape first

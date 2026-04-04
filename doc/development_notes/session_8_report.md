# Session 8 Report: Annulus Dead End + Octagon Quality Breakthrough

**Date:** 2026-04-04

## Summary

Two workstreams completed with a clear architectural diagnosis and a quality breakthrough: (1) Annulus-layer2 feasibility analysis proved the advancing-front formulation with discrete grid is fundamentally unable to mesh the annulus strip — the valid interior region is ~1% of the fan radius, and no grid resolution or distance range produces valid type-1 actions that form non-degenerate quads. The 4 type-1 actions found by 24×8 with the original range are long-distance spanning elements (quality 0.018-0.241). The root cause goes deeper than distance range: the quad topology [ref, ref+1, candidate, ref-1] cannot produce valid elements when all three boundary vertices are on the same side of a narrow strip. (2) Octagon 24×4 training achieved q=0.579 with 5Q — a 21% quality improvement over the 12×4 baseline (0.478, 3Q) and 95% of the theoretical ceiling (0.61). WS2 (annulus training) was skipped per the decision gate. No code changes — this was a diagnostic + training session. 21/21 tests passing.

## What Was Completed

### WS1: Annulus-Layer2 Feasibility Analysis (DEAD END)

**Step 1: Distance-Range Diagnostic**

Checked 10 reference vertices (sorted by interior angle). For each, computed fan radius, distance to nearest valid interior point (via dense 100×100 sampling + PIP check), and the ratio.

| Vertex | Fan Radius | Dist Floor (0.2r) | Nearest Valid | Ratio | Valid Points |
|--------|-----------|-------------------|---------------|-------|-------------|
| 45 | 0.4863 | 0.0973 | 0.0049 | 0.010 | 827 |
| 51 | 0.2929 | 0.0586 | 0.0029 | 0.010 | 7615 |
| 0 | 0.4311 | 0.0862 | 0.0043 | 0.010 | 1295 |
| 36 | 0.4223 | 0.0845 | 0.0042 | 0.010 | 6035 |
| 34 | 0.5403 | 0.1081 | 0.0054 | 0.010 | 8396 |
| 35 | 0.5179 | 0.1036 | 0.0052 | 0.010 | 6568 |
| 1 | 0.4107 | 0.0821 | 0.0041 | 0.010 | 4306 |
| 43 | 0.3833 | 0.0767 | 0.0077 | 0.020 | 972 |
| 30 | 0.4764 | 0.0953 | 0.0048 | 0.010 | 9588 |
| 21 | 0.4182 | 0.0836 | 0.0042 | 0.010 | 6392 |

**Result:** All vertices have valid interior points, but the nearest is at ~1% of the fan radius. The current distance range [0.2, 1.0]*r starts 20× too far. Reachable with [0.2, 1.0]*r: 0/10. Reachable with [0.05, 0.5]*r: 0/10.

**Step 2: Grid Enumeration Across Rollout**

Tested 6 configurations at rollout steps 0, 5, 10, 20:

| Config | Distance Range | Step 0 Type-1 | Step 20 Type-1 |
|--------|---------------|---------------|----------------|
| 12×4 (default) | [0.2, 1.0]*r | 0 | 0 |
| 24×8 | [0.2, 1.0]*r | 4 | 4 |
| 48×8 | [0.2, 1.0]*r | 8 | 8 |
| 12×4 | [0.05, 0.5]*r | 0 | 0 |
| 24×8 | [0.05, 0.5]*r | 0 | 0 |
| 48×8 | [0.05, 0.5]*r | 0 | 0 |

**Key finding:** The type-1 actions found by 24×8 and 48×8 with original range are at 0.66-0.89 × radius — long-distance spanning elements that reach across the annulus. These have quality 0.018-0.241 (degenerate). The modified [0.05, 0.5] range eliminates these but finds nothing closer because even 0.05*r overshoots the strip.

**Step 2b: Ultra-Fine Grids**

| Config | Distance Range | Type-1 Actions |
|--------|---------------|---------------|
| 96×32 | [0.001, 0.02]*r | 0 |
| 96×32 | [0.001, 0.05]*r | 0 |
| 48×32 | [0.001, 0.02]*r | 0 |
| 48×32 | [0.001, 0.05]*r | 0 |

Even with 3072 candidate points and distances starting at 0.001×radius, zero valid type-1 actions. The candidates that pass PIP fail the quad non-self-intersection check.

**Step 3: Oracle Test (with patched distance range)**

Both boundary-reduction and quality-greedy oracles ran for 200 steps with only type-0 actions available. Boundary grew from 64→864 vertices without converging.

**Root Cause (refined from session 7):**

The problem is deeper than distance range — it's the **quad element topology**. Type-1 forms [ref, ref+1, candidate, ref-1]. On the annulus strip:
1. ref, ref+1, and ref-1 are all on the same side of the narrow strip (inner or outer boundary)
2. The candidate vertex must be in the strip interior, perpendicular to the boundary
3. But v2 (ref+1) and v4 (ref-1) are the adjacent boundary vertices — the quad [ref, v2, candidate, v4] degenerates because v2 and v4 subtend a very small angle from the candidate's perspective
4. The resulting quad either self-intersects or has near-zero quality

**Architectural solution identified:** The algorithm needs to form quads from vertices on **opposite sides** of the strip — i.e., connect inner and outer boundary segments. This requires "boundary proximity detection" (finding non-adjacent boundary vertices that are geometrically close), which is a standard feature of production advancing-front algorithms but not implemented in our codebase.

---

### WS2: Annulus Training (SKIPPED)

Skipped per decision gate — WS1 result was DEAD END.

---

### WS3: Octagon 24×4 Quality Push (BREAKTHROUGH)

**Setup:** 15k steps from scratch, 24×4 grid (97 actions), eval every 5k.

**Results:**

| Eval Step | Return | MeanQ | Elements | Completion | Epsilon |
|-----------|--------|-------|----------|------------|---------|
| 5k | 10.14 | 0.478 | 3 | 100% | ~0.55 |
| 10k | 10.14 | 0.478 | 3 | 100% | ~0.10 |
| **15k** | **11.87** | **0.579** | **5** | **100%** | **0.05** |

**Final eval:** q=0.579, 5Q+0T, return=11.87, 100% completion.

**Comparison with 12×4 baseline:**

| Metric | 12×4 (sessions 5-7) | 24×4 (session 8) | Delta |
|--------|---------------------|-----------------|-------|
| Quality | 0.478 | **0.579** | **+21%** |
| Elements | 3 | 5 | +2 |
| Return | 10.14 | **11.87** | **+17%** |
| % of ceiling | 78% | **95%** | +17pp |

**Result: PASS (stretch target exceeded).** Target was q > 0.50, achieved 0.579. The agent discovered that using 5 quads instead of 3 produces much higher quality elements while still completing the mesh. The 24×4 grid's finer angular resolution enables placement of interior vertices that create better-shaped quads. This confirms 24×4 as the standard grid for quality-sensitive training.

**Training dynamics:** Interesting late-stage phase transition — the agent used a 3Q strategy (identical to 12×4) for the first 10k steps, then discovered the 5Q strategy between 10k-15k as epsilon dropped below 0.10. This suggests the 5Q strategy requires precise type-1 placements that are hard to discover through random exploration.

---

## Key Metrics Comparison

| Metric | Session 7 | Session 8 | Change |
|--------|-----------|-----------|--------|
| Star quality (24×4) | 0.405 | 0.405 | Same (not retrained) |
| Octagon quality (12×4) | 0.478 | 0.478 | Same baseline |
| Octagon quality (24×4) | N/A | **0.579** | **New best** |
| Octagon % of ceiling | 78% | **95%** | +17pp |
| Annulus-layer2 feasibility | "action space problem" | **DEAD END confirmed** | Definitive |
| Annulus-layer2 root cause | "grid can't reach strip" | **Quad topology incompatible** | Deeper |
| Tests passing | 21 | 21 | Stable |

## What Didn't Work

### Annulus-Layer2 with Any Grid Configuration
Tested 12 grid/distance-range configurations including ultra-fine grids (96×32, 48×32) with distance ranges from 0.001 to 1.0× radius. Zero viable configurations for meshing the annulus strip. The advancing-front quad topology is fundamentally incompatible with narrow strip geometry because it forms quads from 3 adjacent boundary vertices + 1 interior point, but narrow strips need quads that span from inner to outer boundary.

## What Went Well

- **Exhaustive feasibility analysis** settled the annulus question definitively — no more speculative grid tuning
- **Octagon 24×4 exceeded stretch target** (0.579 vs target 0.50, ceiling 0.61)
- **Phase transition insight** — the 5Q strategy emerged late (10k-15k), suggesting exploration-dependent quality improvements
- **Clean session** — no code changes needed, pure diagnostic + training

## What Didn't Go Well

- WS2 couldn't run due to WS1 dead end — lost the planned training time
- Could have diagnosed the quad topology issue earlier by examining the element formation code more carefully in session 7

## Observations

1. **24×4 is the standard grid for quality.** Both star (+9%, session 7) and octagon (+21%, session 8) improved with 24×4. The finer angular resolution enables better interior vertex placement.

2. **Quality improvements come from strategy changes, not gradual improvement.** The octagon agent used the same 3Q strategy for 10k steps before discovering 5Q. This suggests that epsilon decay schedule matters more than training duration for quality push.

3. **The annulus strip requires architectural changes.** No amount of grid tuning can overcome the quad topology limitation. The two viable approaches are:
   - **Boundary proximity detection (type-2 actions):** Detect non-adjacent boundary vertices that are geometrically close and form quads that span the strip. This is standard in production advancing-front algorithms.
   - **Continuous action space with learned placement:** SAC/PPO with continuous vertex coordinates, bypassing the grid entirely.

4. **Boundary growth is the annulus failure mode.** Type-0 actions grow the boundary (+4 vertices/step) because each quad introduces 2 new vertices from non-boundary edges. This is specific to narrow geometries where type-0 quads don't reduce the remaining area efficiently.

## Files Changed

| File | Changes |
|------|---------|
| `checkpoints/octagon-24x4-s8/` | New: octagon 24×4 converged model (15k steps, q=0.579) |
| `output/latest/octagon.png` | Updated: 24×4 octagon visualization (5Q, q=0.579) |

No source code changes in this session.

## Short-term Next Steps (Session 9)

1. **Boundary proximity detection / type-2 actions:** Implement detection of non-adjacent boundary vertices within a distance threshold. When two parts of the boundary are close, allow forming quads that connect them. This is the key architectural change needed for annulus-layer2.

2. **Epsilon schedule tuning:** The octagon's phase transition at 10k-15k suggests slower epsilon decay could find better strategies. Test with epsilon_min=0.10 for longer before dropping to 0.05.

3. **Star 24×4 extended training:** Star currently at 0.405 (92% of 0.44 ceiling). The octagon's late-phase improvement suggests star might also benefit from extended training beyond 30k steps.

## Medium-term Next Steps (Sessions 10-11)

4. **Multi-domain training with 24×4 grid:** Now that 24×4 is validated on both star and octagon, train a single agent on all domains simultaneously.

5. **Continuous action space (SAC revisit):** With improved reward structure and quality-gated completion, SAC may learn better than early sessions.

6. **Variable-size state representation:** GNN/attention for handling domains with very different vertex counts (8v octagon vs 64v annulus).

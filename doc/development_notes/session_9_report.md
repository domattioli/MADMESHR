# Session 9 Report: Type-2 Prototype + Star Slow Epsilon

**Date:** 2026-04-04

## Summary

Three workstreams completed: (1) Type-2 geometric prototype proved that proximity-based quad formation can mesh portions of annulus-layer2 — the first valid elements ever placed on this domain. The centroid-in-domain check and boundary-crossing rejection were critical additions discovered during debugging; without them, type-2 quads extended outside the domain. Oracle result: 16Q, q=0.487, 34v remaining (incomplete but geometrically valid, 6/7 validation checks pass). (2) Star 24×4 with slow epsilon (decay_frac=1.4) reached q=0.393 at 30k steps — below the 0.405 baseline, still climbing but not efficient. (3) WS3 (circle/octagon) was not attempted due to time spent on type-2 debugging.

## What Was Completed

### WS1: Type-2 Geometric Prototype + Oracle Test (PARTIAL SUCCESS)

**Implemented 3 new methods in MeshEnvironment.py (~150 lines):**

1. **`_find_proximity_pairs(ref_idx, threshold, min_gap)`** — vectorized search for non-adjacent boundary vertices within distance threshold. Finds all 7 coincident pairs on annulus (dist=0.000, gaps 5-23).

2. **`_form_type2_element(ref_idx, far_idx)`** — forms quads from neighbor vertices of coincident/near-coincident pairs. For coincident pairs: quad = [ref-1, ref+1, far+1, far-1] (skips coincident vertices). Includes three validity checks:
   - `_is_valid_quad`: no self-intersection
   - `_type2_crosses_boundary`: quad edges don't cross non-consumed boundary edges
   - Centroid-in-polygon: quad centroid must be inside current boundary

3. **`_update_boundary_type2(element, ref_idx, far_idx, consumed)`** — identifies two contiguous consumed segments, extracts two surviving arcs, reconnects into single loop.

**Oracle test (`annulus_oracle_type2.py`):**
- Greedy priority: type-2 → type-0 → type-1
- All candidates pass through `_try_place_element` which checks: centroid inside domain, no intersection with existing elements or original boundary, no boundary growth
- Result: 16Q, q=0.487, boundary 64→34 (incomplete)
- Only 1 type-2 quad passes all checks (pair 17,25 — q=0.610)
- 15 type-0 quads placed in valid portions of the strip

**7-point mesh validation:**

| Check | Result |
|-------|--------|
| Element-element edge intersections | 0 (PASS) |
| Element-boundary edge intersections | 0 (PASS) |
| Element centroids inside domain | 0 failures (PASS) |
| Boundary vertices inside elements | 2 (FAIL*) |
| Self-intersecting elements | 0 (PASS) |
| Zero/negative area elements | 0 (PASS) |
| Boundary growth steps | 0 (PASS) |

*False positive: v17 and v25 are the coincident pair consumed by the type-2 quad — they sit at the quad's center by construction.

**Key debugging discoveries:**

1. **Initial implementation had no centroid check.** The quad [v0, v2, v57, v55] from pair (1,56) passed `_is_valid_quad` and boundary-crossing checks but spanned a large area mostly outside the domain. The centroid was outside the boundary polygon. Fix: centroid-in-polygon check in `_form_type2_element`.

2. **Boundary-crossing check alone is insufficient.** The boundary edges run *around* the problematic quad without crossing its edges — they weave through the gap between quad edges. The centroid check catches this.

3. **Type-0 on narrow strips causes boundary growth.** Without the growth guard, type-0 quads that span across the strip add +4 vertices/step indefinitely. Fix: save/restore boundary and reject if boundary grew.

---

### WS2: Star 24×4 from Scratch with Slow Epsilon (FAIL)

**Setup:** 30k steps, 24×4 grid (97 actions), epsilon_decay_frac=1.4 (default 0.7, so 2× slower decay).

**Results:**

| Eval Step | Return | MeanQ | Elements | Epsilon |
|-----------|--------|-------|----------|---------|
| 5k | 7.18 | 0.223 | 4 | ~0.88 |
| 10k | 7.25 | 0.230 | 5 | ~0.77 |
| 15k | 7.09 | 0.296 | 8 | ~0.66 |
| 20k | 1.73 | 0.257 | 9 | ~0.55 |
| 25k | 8.19 | 0.353 | 8 | ~0.43 |
| **30k** | **9.02** | **0.393** | **6** | **~0.32** |

**Comparison with session 7 baseline (default epsilon, 30k steps):**

| Metric | Default epsilon (s7) | Slow epsilon (s9) |
|--------|---------------------|-------------------|
| Quality | **0.405** | 0.393 |
| Elements | 5 | 6 |
| Epsilon at 30k | 0.05 | 0.32 |

**Result: FAIL.** q=0.393 < 0.405 baseline. Quality is still climbing at 30k (0.257→0.353→0.393 from 20k-30k), suggesting the slow epsilon approach might beat baseline with 50k+ steps, but it's not more efficient within the same training budget.

**New CLI feature:** `--epsilon-decay-frac` argument added to main.py and wired to DQNTrainer.

---

### WS3: Circle 24×4 + Octagon Reproducibility (NOT ATTEMPTED)

Skipped due to time spent on type-2 debugging (centroid check, boundary growth guard, GIF generation, validation).

---

## Key Metrics Comparison

| Metric | Session 8 | Session 9 | Change |
|--------|-----------|-----------|--------|
| Star quality (24×4) | 0.405 | 0.393 (slow eps) | -3% |
| Octagon quality (24×4) | 0.579 | 0.579 | Same (not retrained) |
| Annulus type-2 quads | N/A | **1 valid** | New |
| Annulus elements placed | 0 | **16** | New |
| Annulus completion | 0% | 47% (34/64 remaining) | New |
| Tests passing | 21 | **24** | +3 |

## What Didn't Work

### Type-2 quads without centroid check
The initial implementation produced quads that extended outside the domain. The quad vertices were all on the boundary and edges didn't cross other edges, but the quad interior covered exterior space. Required centroid-in-polygon check.

### Slow epsilon for star
2× slower epsilon decay didn't improve quality within 30k steps. The agent explores too much in early training and hasn't converged by 30k.

### Type-2 for most coincident pairs
Only 1 of 7 coincident pairs produces a valid type-2 quad (pair 17,25). The others fail centroid-in-domain (4 pairs) or self-intersection (2 pairs). The valid quads tend to be in the wider parts of the strip.

## What Went Well

- **Type-2 geometry is proven.** The prototype places valid, non-intersecting elements on the annulus strip for the first time.
- **Robust validation suite.** 7-point check catches all classes of mesh invalidity.
- **Boundary growth guard.** Prevents the infinite type-0 loop discovered during testing.
- **Clean architecture.** Type-2 methods are additive to MeshEnvironment — no changes to DiscreteActionEnv, DQN, or existing checkpoints.

## What Didn't Go Well

- Spent significant time on debugging mesh validity issues that should have been caught by upfront design (centroid check, boundary growth)
- The GIF/visualization iteration loop was time-consuming
- WS3 couldn't run

## Observations

1. **Centroid-in-polygon is a critical validity check for type-2.** Any quad formed from boundary vertices can extend outside the domain if the boundary is non-convex. This check must be part of the core type-2 implementation, not just the oracle.

2. **Annulus completion requires more type-2 coverage.** Only 1 of 7 coincident pairs works. The remaining boundary needs either: (a) type-2 actions for non-coincident but geometrically close vertices, or (b) a fundamentally different approach (continuous action space, boundary proximity detection at larger thresholds).

3. **The narrow strip problem persists for type-0.** Even with all guards, type-0 quads in the strip interior degrade to q≈0.12-0.23. The strip needs cross-strip quads (type-2) at more locations.

4. **Slow epsilon is not efficient.** The octagon's phase transition at epsilon ~0.15 was the motivation, but star doesn't benefit from extended exploration within 30k steps.

## Files Changed

| File | Changes |
|------|---------|
| `src/MeshEnvironment.py` | +3 new methods: `_find_proximity_pairs`, `_form_type2_element`, `_update_boundary_type2`, `_type2_crosses_boundary` (~150 lines) |
| `tests/test_discrete_env.py` | +3 new tests for type-2 (TestType2Actions class) |
| `annulus_oracle_type2.py` | New: standalone oracle script with intersection/centroid/growth guards |
| `main.py` | +`--epsilon-decay-frac` CLI argument |
| `checkpoints/star-24x4-slow-eps-s9/` | New: star slow-epsilon model (30k steps, q=0.393) |
| `output/latest/annulus_layer2_oracle_type2.png` | New: validated mesh visualization |
| `output/latest/annulus_layer2_oracle_type2.gif` | New: step-by-step animation |
| `output/latest/star.png` | Updated: star 24×4 slow epsilon result |

## Short-term Next Steps (Session 10)

1. **Validate algorithm against Pan et al. paper.** Ensure advancing-front formulation, reward structure, and element formation match the paper. Recreate one of their test domains.

2. **Improve type-2 coverage.** Relax the coincident-only constraint — find geometrically close (not just coincident) boundary segments and form cross-strip quads. May require larger threshold and smarter orientation selection.

3. **DQN integration of type-2.** Once geometry is robust, add type-2 actions to the discrete action space (session 10+).

## Medium-term Next Steps (Sessions 11-12)

4. **Continuous action space (SAC/PPO).** Bypass the grid entirely for domains where discrete resolution is limiting.

5. **Multi-domain training.** Single agent trained on all domains simultaneously.

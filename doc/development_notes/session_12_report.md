# Session 12 Report: Type-2 DQN Architecture + H-shape 24v + Mu Calibration Fix

**Date:** 2026-04-05

## Summary

Five workstreams completed plus project reorganization. (1) H-shape domain revised to 24 vertices with crossbar y=1.5-2.5. (2) Type-2 DQN architecture fully implemented: Discrete(57) with 8 type-2 slots, split bonus, sub-loop completion bonus. (3) DQN stability fixes: hard target updates, smaller buffer, faster epsilon. (4) Mu calibration fix: scale A_min/A_max by expected element count. (5) Type-0 priority vertex selection + boundary distance filter + centroid check on auto-close. After all fixes, H-shape DQN reaches 11Q q=0.677 at 100% completion -- confirmed as theoretical optimum by oracle. Tests grew from 37 to 44. All 8 domains pass 7-point validation. Project reorganized with scripts/, pyproject.toml, __init__.py.

## What Was Completed

### Pre-WS: H-shape Domain Revision (COMPLETE)

**Change:** H-shape revised from 20v (crossbar y=1-3) to 24v (crossbar y=1.5-2.5) per user request. Vertices at y=1.0 and y=3.0 retained on inner walls. 4 new vertices added: (1,1.5), (3,1.5), (3,2.5), (1,2.5).

**Geometry:** 24 vertices, edges 0.5-1.0 unit, area=10.0 (was 12.0 on 20v). Two cutouts each 2x1.5 (was 2x1). Crossbar now 1 unit tall (was 2).

**Validation:** All 8 domains pass 7-point validation. Greedy: 30Q, q=0.401, incomplete (hits step limit).

---

### WS1: H-shape DQN Stability Fix (INITIAL ATTEMPT -- FAILED)

**Problem:** 20v H-shape DQN regressed from 100% (10k) to 0% (15k) in session 11.

**Fix Applied:** Three stability changes:
1. Hard target network updates every 500 steps (was soft Polyak tau=0.005)
2. Smaller replay buffer: 20k capacity (was 100k)
3. Faster epsilon decay: decay_frac=0.5 (was 0.7) -- epsilon reaches 0.05 at 15k

**CLI added:** `--target-update-freq` and `--buffer-size` arguments.

**Training Results (24v, 30k steps, old mu calibration):**

| Eval Step | Return | MeanQ | Elements | Epsilon | Completion |
|-----------|--------|-------|----------|---------|------------|
| 5k | -2.35 | 0.574 | 3.0 | 0.683 | 0% |
| 10k | -2.63 | 0.575 | 3.0 | 0.365 | 0% |
| 15k | -2.64 | 0.576 | 3.0 | 0.050 | 0% |
| 20k | -2.03 | 0.606 | 6.0 | 0.050 | 0% |
| 25k | -2.06 | 0.617 | 6.0 | 0.050 | 0% |
| **30k** | **-2.03** | **0.617** | **6.0** | **0.050** | **0%** |

**Diagnosis:** Mu-avoidance trap. The narrower crossbar (1 unit tall) produces smaller elements that trigger harsh density penalties. With original_area=10.0, A_max=1.0, and element areas ~0.5 in the narrow strips, mu ~= -0.56 per element. The agent correctly learned that placing fewer elements maximizes return.

---

### WS2: Mu Calibration Fix (COMPLETE)

**Problem:** The mu (density) penalty used fixed A_min/A_max thresholds that did not account for domain complexity. Domains requiring many elements (like 24v H-shape) had elements far smaller than A_max, producing harsh per-step penalties that discouraged mesh completion.

**Fix:** Scale A_min/A_max by expected element count:
- `ideal_area = original_area / (n_verts / 2)` -- expected area per element assuming ~n_verts/2 elements
- `A_min = 0.1 * ideal_area`
- `A_max = 0.5 * ideal_area`

This makes the density penalty domain-aware: elements of the expected size receive near-zero mu penalty regardless of domain complexity.

**Training Results (24v, 30k steps, with mu fix):**

| Eval Step | Return | MeanQ | Elements | Epsilon | Completion |
|-----------|--------|-------|----------|---------|------------|
| 5k | 10.92 | 0.482 | 16.0 | 0.683 | 100% |
| 10k | 11.26 | 0.491 | 15.0 | 0.365 | 100% |
| 15k | 11.32 | 0.491 | 15.0 | 0.050 | 100% |
| 20k | 11.32 | 0.491 | 15.0 | 0.050 | 100% |
| **25k** | **11.32** | **0.491** | **15.0** | **0.050** | **100%** |
| **30k** | **11.32** | **0.491** | **15.0** | **0.050** | **100%** |

**Result:** 100% completion, 15Q q=0.491, stable from 15k-30k. The mu fix eliminated the avoidance trap entirely.

---

### WS3: Type-0 Priority + Boundary Distance Filter (COMPLETE)

**Problem:** After mu fix, the agent achieved completion but used 15 elements (vs ~11 expected for optimal coverage). Investigation showed the angle-based vertex selection was not choosing vertices where type-0 actions would be most beneficial.

**Fix 1 -- Type-0 priority vertex selection:** Scan all boundary vertices for the best type-0 opportunity before falling back to angle-based selection. This ensures that whenever a clean type-0 (adjacent vertex connection consuming 2 boundary vertices) is available, it is prioritized.

**Fix 2 -- Boundary distance filter:** Reject interior vertices (type-1) that are closer than 3% of fan radius to non-adjacent boundary edges. This prevents the agent from placing vertices that create near-degenerate configurations.

**Fix 3 -- Centroid check on bnd==4 auto-close:** When the boundary has exactly 4 vertices and auto-closes with a quad, validate that the centroid lies inside the domain. Prevents invalid auto-close on concave final boundaries.

**Training Results (24v, 15k steps, with mu fix + type-0 priority):**

| Eval Step | Return | MeanQ | Elements | Epsilon | Completion |
|-----------|--------|-------|----------|---------|------------|
| 5k | 11.49 | 0.649 | 12.0 | 0.683 | 100% |
| 10k | 11.77 | 0.677 | 11.0 | 0.365 | 100% |
| **15k** | **11.77** | **0.677** | **11.0** | **0.050** | **100%** |

**Result:** 100% completion, 11Q q=0.677, stable from 10k-15k+. Oracle confirms 11Q q=0.677 is the theoretical optimum for this domain. The type-0 priority reduced element count from 15 to 11 by preferring boundary-consuming actions.

---

### WS4: Type-2 DQN Architecture (COMPLETE)

**Problem:** DQN could not use type-2 actions, blocking training on annulus and domains needing interior splits.

**Solution:**

#### Action Space Extension
- `DiscreteActionEnv.max_actions` = 57 (was 49): 1 type-0 + 48 type-1 + 8 type-2 slots
- Type-2 actions enumerated from ALL boundary proximity pairs (threshold=0.02, min_gap=3)
- Pairs sorted by distance, mapped to slots 49-56
- Validated with Check A (original boundary) and Check B (current boundary)
- Domains without proximity pairs (square, octagon, etc.) mask all type-2 slots

#### Type-2 Step Handling
- When action >= 49: look up (ref_idx, far_idx), call `_form_type2_element`, `_update_boundary_type2`
- Check D (overlap) applied before committing
- Boundary self-intersection check (Check C) with rollback
- Saves/restores pending_loops on rollback

#### Type-2 Reward
- Per-step: `eta_e + 0.3*eta_b + 0.3` (split bonus, no mu penalty)
- Sub-loop completion bonus: +2.0 when `pending_loop` activates
- Both reported in info dict as `split_bonus` and `sub_loop_bonus`

#### Refactoring
- Extracted `_fail_step()` helper to reduce code duplication (was 5 identical rollback blocks)
- Saved/restored `pending_loops` on rollback for type-2 consistency

**Verification:**
- Action space: Discrete(57)
- Annulus: 1 type-2 action visible at initial state (coincident pair at distance=0)
- Square: 0 type-2 actions (no proximity pairs)
- 50 random steps on annulus: no crashes
- All 8 domains pass 7-point validation

---

### WS5: Documentation + Project Cleanup (COMPLETE)

#### Project Reorganization
- **scripts/**: Moved `validate_mesh.py`, `quality_diagnostic.py`, `test_domains.py`, `annulus_oracle_type2.py` from root
- **__init__.py**: Added to `src/`, `src/utils/`, `scripts/` for proper package imports
- **pyproject.toml**: Added with build system, dependencies, pytest config
- **Removed:** stale `sandbox-recover_faces_from_edges` JSON file
- **Fixed:** sys.path imports in all moved scripts to use parent directory

#### Documentation
- **CLAUDE.md**: Updated with type-2 architecture (Discrete(57), reward formula, sub-loop bonus), hard target updates, 24v H-shape, validate_mesh command, script paths
- **Session 13 plan**: Written with adversarial review (3 agents: devil's advocate, scope realist, reward analyst)

---

## H-shape DQN Progression Summary

| Stage | Completion | Elements | Quality | Notes |
|-------|-----------|----------|---------|-------|
| Session 11 (20v) | 100% (10k), 0% (15k) | 10Q | 0.533 | Unstable, regressed |
| S12 initial (24v, old mu) | 0% | 3-6 | 0.617 (6 elem) | Mu-avoidance trap |
| S12 after mu fix | 100% | 15Q | 0.491 | Stable 15k-30k |
| S12 after type-0 priority | 100% | **11Q** | **0.677** | **Optimal, stable 10k-15k+** |

---

## Key Metrics Comparison

| Metric | Session 11 | Session 12 | Change |
|--------|-----------|------------|--------|
| H-shape vertices | 20 | **24** | +4 (user request) |
| H-shape crossbar | y=1-3 | **y=1.5-2.5** | Narrower |
| H-shape DQN completion | 100% (10k only) | **100% (stable)** | Fixed |
| H-shape DQN quality | 0.533 (10Q) | **0.677 (11Q)** | +0.144 |
| H-shape DQN stability | Regressed at 15k | **Stable 10k-15k+** | Fixed |
| Action space | Discrete(49) | **Discrete(57)** | +8 type-2 slots |
| Type-2 on annulus | N/A | **1 action** | New capability |
| Tests passing | 37 | **44** | +7 |
| Project structure | Flat scripts | **scripts/, pyproject.toml** | Reorganized |

## What Didn't Work

### Initial H-shape 24v DQN trapped by mu penalty (before mu fix)
The narrower crossbar (1 unit tall vs 2) created a harsher density penalty landscape. Elements in the 0.5-unit-wide inner walls averaged area ~0.5, giving mu ~= -0.56. The DQN learned to avoid placing elements altogether. The stability fixes (hard target updates, smaller buffer, faster epsilon) were valuable architectural improvements but did not address the fundamental reward calibration issue.

### Annulus type-2 coverage limited
Only 1 of 7 proximity pairs passes `_form_type2_element` validation at threshold=0.02. Session 13 will tune the threshold to increase coverage.

## What Went Well

- **Mu calibration fix was transformative.** Scaling A_min/A_max by expected element count turned 0% completion into 100% stable completion. This fix applies to all domains, not just H-shape.
- **Type-0 priority found the optimal mesh.** 11Q q=0.677 matches oracle, reducing from 15 excess elements. Simple heuristic with large impact.
- **Type-2 architecture is clean.** The 8-slot indexed system with proximity pair enumeration works correctly. All validity checks are applied. No crashes in random walk testing.
- **Project structure improved.** Moving scripts, adding pyproject.toml and __init__.py makes the codebase more organized.
- **Stability improvements preserved.** Hard target updates and configurable buffer size are good architectural additions.
- **H-shape fully solved.** From 0% completion (initial) to optimal 11Q solution confirmed by oracle -- all within one session.

## Files Changed

| File | Changes |
|------|---------|
| `main.py` | H-shape revised to 24v; added --buffer-size and --target-update-freq CLI args; wired DQN target_update_freq and buffer capacity |
| `src/DiscreteActionEnv.py` | Type-2 action slots (49-56); type-2 step handling with split bonus; sub-loop completion bonus; _fail_step helper; boundary distance filter (3% fan radius); type-0 priority vertex selection; centroid check on bnd==4 auto-close |
| `src/MeshEnvironment.py` | Mu calibration fix (A_min/A_max scaled by expected element count) |
| `src/DQN.py` | Added target_update_freq param; hard target updates when >0, soft Polyak when 0 |
| `tests/test_discrete_env.py` | Updated action space assertions (49->57); +7 new tests (TestType2DQNIntegration) |
| `CLAUDE.md` | Updated architecture (Discrete(57)), commands (scripts/), known issues |
| `pyproject.toml` | New: build system, deps, pytest config |
| `src/__init__.py` | New: package init |
| `src/utils/__init__.py` | New: package init |
| `scripts/__init__.py` | New: package init |
| `scripts/*.py` | Moved from root; fixed sys.path imports |
| `checkpoints/h-shape-24v-s12/` | 24v H-shape DQN model (best: 11Q, q=0.677) |
| `output/latest/h-shape.png` | DQN eval (24v, 11Q) |
| `output/latest/h-shape_greedy.png` | Greedy baseline (24v) |

## Short-term Next Steps (Session 13)

1. **Type-2 threshold tuning.** Increase threshold to get >= 3 valid type-2 actions on annulus. Verify sub-loop completability before training.

2. **Annulus sub-loop curriculum training.** Pre-place type-2 elements, train DQN on smaller (~20v) sub-loop. Reset original_area correctly. The mu fix and type-0 priority should help training converge faster.

3. **Retrain other domains with mu fix + type-0 priority.** Star (0.44), octagon (0.61), circle (0.78), rectangle (0.464) may improve with the new mu calibration and vertex selection.

## Medium-term Next Steps (Sessions 13-14)

4. **Full annulus end-to-end.** Oracle type-2 + DQN sub-loops, then full DQN with type-2 actions.

5. **Pan et al. benchmark.** Recreate Pan et al. test domains for comparison.

# Session 12 Report: Type-2 DQN Architecture + H-shape 24v

**Date:** 2026-04-05

## Summary

Three workstreams completed plus project reorganization. (1) H-shape domain revised to 24 vertices with crossbar y=1.5-2.5 per user request. DQN training with stability fixes (hard target updates, 20k buffer, faster epsilon) reached 0% completion -- agent converges to 3 elements, trapped by mu penalty in the narrower geometry. (2) Type-2 DQN architecture fully implemented: Discrete(57) action space with 8 type-2 slots, split bonus, sub-loop completion bonus. All validity checks applied. 44 tests passing. (3) Project reorganized with scripts/, pyproject.toml, __init__.py files. Session 13 plan written with adversarial review.

## What Was Completed

### Pre-WS: H-shape Domain Revision (COMPLETE)

**Change:** H-shape revised from 20v (crossbar y=1-3) to 24v (crossbar y=1.5-2.5) per user request. Vertices at y=1.0 and y=3.0 retained on inner walls. 4 new vertices added: (1,1.5), (3,1.5), (3,2.5), (1,2.5).

**Geometry:** 24 vertices, edges 0.5-1.0 unit, area=10.0 (was 12.0 on 20v). Two cutouts each 2x1.5 (was 2x1). Crossbar now 1 unit tall (was 2).

**Validation:** All 8 domains pass 7-point validation. Greedy: 30Q, q=0.401, incomplete (hits step limit).

---

### WS1: H-shape DQN Stability Fix (FAILED)

**Problem:** 20v H-shape DQN regressed from 100% (10k) to 0% (15k) in session 11.

**Fix Applied:** Three stability changes:
1. Hard target network updates every 500 steps (was soft Polyak tau=0.005)
2. Smaller replay buffer: 20k capacity (was 100k)
3. Faster epsilon decay: decay_frac=0.5 (was 0.7) -- epsilon reaches 0.05 at 15k

**CLI added:** `--target-update-freq` and `--buffer-size` arguments.

**Training Results (24v, 30k steps):**

| Eval Step | Return | MeanQ | Elements | Epsilon | Completion |
|-----------|--------|-------|----------|---------|------------|
| 5k | -2.35 | 0.574 | 3.0 | 0.683 | 0% |
| 10k | -2.63 | 0.575 | 3.0 | 0.365 | 0% |
| 15k | -2.64 | 0.576 | 3.0 | 0.050 | 0% |
| 20k | -2.03 | 0.606 | 6.0 | 0.050 | 0% |
| 25k | -2.06 | 0.617 | 6.0 | 0.050 | 0% |
| **30k** | **-2.03** | **0.617** | **6.0** | **0.050** | **0%** |

**Diagnosis:** The agent initially converges to 3 elements (5k-15k), then escapes to 6 elements (20k-30k), but never finds completion. This is a mu-avoidance trap: the narrower crossbar (1 unit tall) produces smaller elements that trigger harsh density penalties. With original_area=10.0, A_max=1.0, and element areas ~0.5 in the narrow strips, mu ~= -0.56 per element. The agent correctly learned that placing fewer elements maximizes return.

**Root cause:** NOT a stability problem. The original 20v instability (session 11) was likely the same mu-avoidance -- the agent found completion at 10k during high-epsilon exploration but then learned that avoiding elements gave better return. The stability fixes (hard target updates, etc.) are still valuable but don't address the reward calibration issue.

**Key insight:** The 24v H-shape with narrower crossbar (y=1.5-2.5) creates a harsher mu landscape than the 20v (y=1-3). Session 13 should address mu calibration for narrow geometries before retraining.

---

### WS2: Type-2 DQN Architecture (COMPLETE)

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

### WS3: Documentation + Project Cleanup (COMPLETE)

#### Project Reorganization
- **scripts/**: Moved `validate_mesh.py`, `quality_diagnostic.py`, `test_domains.py`, `annulus_oracle_type2.py` from root
- **__init__.py**: Added to `src/`, `src/utils/`, `scripts/` for proper package imports
- **pyproject.toml**: Added with build system, dependencies, pytest config
- **Removed:** stale `sandbox-recover_faces_from_edges` JSON file
- **Fixed:** sys.path imports in all moved scripts to use parent directory

#### Documentation
- **CLAUDE.md**: Updated with type-2 architecture (Discrete(57), reward formula, sub-loop bonus), hard target updates, 24v H-shape, validate_mesh command, script paths
- **Session 13 plan**: Written with adversarial review (3 agents: devil's advocate, scope realist, reward analyst). Key decisions: train on smaller sub-loop (~20v), threshold tuning before training, verify sub-loop completability, curriculum reset with correct original_area.

---

## Key Metrics Comparison

| Metric | Session 11 | Session 12 | Change |
|--------|-----------|------------|--------|
| H-shape vertices | 20 | **24** | +4 (user request) |
| H-shape crossbar | y=1-3 | **y=1.5-2.5** | Narrower |
| H-shape DQN completion | 100% (10k) | **0%** | Regression (mu-avoidance) |
| H-shape DQN quality | 0.533 (10k) | **0.617** (6 elements) | Higher but only 6 elements |
| H-shape greedy completion | Incomplete | **Incomplete** | Same (30Q both) |
| Action space | Discrete(49) | **Discrete(57)** | +8 type-2 slots |
| Type-2 on annulus | N/A | **1 action** | New capability |
| Tests passing | 37 | **44** | +7 |
| Project structure | Flat scripts | **scripts/, pyproject.toml** | Reorganized |

## What Didn't Work

### H-shape 24v DQN trapped by mu penalty
The narrower crossbar (1 unit tall vs 2) creates a harsher density penalty landscape. Elements in the 0.5-unit-wide inner walls average area ~0.5, giving mu ~= -0.56. The DQN learns to avoid placing elements altogether. The stability fixes (hard target updates, smaller buffer, faster epsilon) are valuable architectural improvements but don't address the fundamental reward calibration issue.

**Greedy comparison:** Returns -19.59 for 30 elements vs DQN's -2.63 for 3 elements. The DQN is correctly maximizing return by avoiding placement -- the reward structure penalizes the desired behavior.

### Annulus type-2 coverage limited
Only 1 of 7 proximity pairs passes `_form_type2_element` validation at threshold=0.02. Session 13 will tune the threshold to increase coverage.

## What Went Well

- **Type-2 architecture is clean.** The 8-slot indexed system with proximity pair enumeration works correctly. All validity checks are applied. No crashes in random walk testing.
- **Project structure improved.** Moving scripts, adding pyproject.toml and __init__.py makes the codebase more organized for future development.
- **Stability improvements preserved.** Hard target updates and configurable buffer size are good architectural additions even though they didn't solve the H-shape training.
- **Adversarial planning identified key session 13 risks.** The devil's advocate found the WS1/WS2 circular dependency, the reward analyst found the mu calibration risk with original_area, and the scope realist correctly estimated the curriculum implementation time.

## Files Changed

| File | Changes |
|------|---------|
| `main.py` | H-shape revised to 24v; added --buffer-size and --target-update-freq CLI args; wired DQN target_update_freq and buffer capacity |
| `src/DiscreteActionEnv.py` | Type-2 action slots (49-56); _enumerate scans all proximity pairs; type-2 step handling with split bonus; sub-loop completion bonus; _fail_step helper |
| `src/DQN.py` | Added target_update_freq param; hard target updates when >0, soft Polyak when 0 |
| `tests/test_discrete_env.py` | Updated action space assertions (49->57); +7 new tests (TestType2DQNIntegration) |
| `CLAUDE.md` | Updated architecture (Discrete(57)), commands (scripts/), known issues |
| `pyproject.toml` | New: build system, deps, pytest config |
| `src/__init__.py` | New: package init |
| `src/utils/__init__.py` | New: package init |
| `scripts/__init__.py` | New: package init |
| `scripts/*.py` | Moved from root; fixed sys.path imports |
| `checkpoints/h-shape-24v-s12/` | 24v H-shape DQN model (best: 6Q, q=0.617) |
| `output/latest/h-shape.png` | DQN eval (24v, 3Q) |
| `output/latest/h-shape_greedy.png` | Greedy baseline (24v) |

## Short-term Next Steps (Session 13)

1. **Type-2 threshold tuning.** Increase threshold to get >= 3 valid type-2 actions on annulus. Verify sub-loop completability before training.

2. **Annulus sub-loop curriculum training.** Pre-place type-2 elements, train DQN on smaller (~20v) sub-loop. Reset original_area correctly.

3. **H-shape mu calibration.** Investigate adjusting A_min/A_max thresholds for narrow geometries, or exempt elements in narrow regions from mu penalty. Alternatively, lower max_ep_len penalty and let the agent find its own completion strategy.

## Medium-term Next Steps (Sessions 13-14)

4. **Full annulus end-to-end.** Oracle type-2 + DQN sub-loops, then full DQN with type-2 actions.

5. **Reward structure review.** The mu-avoidance trap on H-shape 24v suggests the density penalty needs domain-aware calibration. Consider scaling A_min/A_max by the number of expected elements, not just the original area.

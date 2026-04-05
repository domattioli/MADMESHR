# Session 14 Report: Rectangle Regression Root Cause + Annulus Type-2 Feasibility

**Date:** 2026-04-05

## Summary

Two major findings: (1) The rectangle "regression" from q=0.464 to q=0.426 reported in session 13 was caused by **wrong domain definitions in the eval scripts** (3.0×1.5 rectangle instead of 4.0×1.0, star inner radius 0.5 instead of 0.4) — NOT by the boundary distance filter. The actual rectangle_s5 checkpoint produces q=0.464 regardless of filter threshold. No regression exists. (2) Annulus oracle stuck at 21v diagnosed: type-2 threshold too tight (0.02, only 1 valid action at init) + greedy strategy creates geometric dead-ends. Threshold increased to 0.10 (5 valid type-2 actions). DQN training on annulus reached 4k steps with 0% completion and 2-3 elements avg — confirms the domain needs a true sub-loop curriculum with multi-vertex selection, not direct training.

## What Was Completed

### WS1: Rectangle Boundary Distance Filter Investigation (COMPLETE — 15 min)

**Goal:** Controlled experiment to isolate cause of rectangle regression.

**Experiment:** Evaluated rectangle_s5 checkpoint at three filter thresholds:

| Threshold | Quality | Elements | Complete |
|-----------|---------|----------|----------|
| 0.0 (OFF) | **0.464** | 9Q+0T | Yes |
| 0.01 (LOW) | **0.464** | 9Q+0T | Yes |
| 0.03 (ON) | **0.464** | 9Q+0T | Yes |

**All three produce identical results.** The boundary distance filter is NOT the cause.

**Root cause:** Session 13 eval scripts (`scripts/eval_ablation.py`, `scripts/eval_checkpoints.py`) had **wrong domain definitions**:
- Rectangle: 3.0×1.5 with 0.5 spacing (eval) vs 4.0×1.0 with linspace(0,4,9) (main.py)
- Star: inner radius 0.5 (eval) vs 0.4 (main.py)

The checkpoints were evaluated on the wrong geometry, producing misleading quality numbers.

**Fix:** Replaced duplicate domain constructors in eval scripts with imports from `main.py`:
```python
from main import DOMAINS
def make_rectangle(): return DOMAINS['rectangle']()
```

**Bonus:** Added `bnd_dist_threshold` as a configurable parameter on MeshEnvironment (default 0.03) and per-domain in domain registry. Not needed for rectangle, but useful for future narrow geometries.

---

### WS2 Phase 1: Annulus Oracle Debugging (COMPLETE — 30 min)

**Goal:** Diagnose why oracle gets stuck at 21 boundary vertices.

**Diagnosis:**

1. **Type-2 threshold too tight (0.02):** Only 4 valid type-2 elements at initial state; at threshold=0.10, 18 valid type-2 elements available.
2. **DQN env only exposes 1 type-2 at 0.02 vs 5 at 0.10:** Bottleneck in `DiscreteActionEnv._enumerate()` hardcoded threshold.
3. **Greedy strategy creates dead-ends:** After ~20 type-0 placements, remaining boundary becomes irregular:
   - Most type-0 centroids fall outside boundary (annulus is curved)
   - Valid type-0 actions intersect existing elements or original boundary
   - Type-1 finds 0 valid actions at stuck vertices
4. **With threshold=0.10:** Oracle places 4 type-2 elements + 14 type-0/1, creates 4 pending sub-loops (sizes 7, 3, 18, 3), but still gets stuck at 8 vertices on active loop.

**Fix:** Made `type2_threshold` configurable in `DiscreteActionEnv.__init__()` (default 0.02). Set annulus to 0.10 via domain registry.

**Conclusion:** The oracle dead-end is a **greedy strategy limitation**, not a fundamental geometry issue. The annulus CAN be split into sub-loops via type-2. The problem is completing each sub-loop with the current single-ref-vertex selection approach.

---

### WS2 Phase 2: DQN Training Feasibility Test (COMPLETE — killed at 4k steps)

**Goal:** Test if DQN can learn on annulus with improved type-2 threshold.

**Training config:** 12×4+type2 (57 actions), ε-decay 50%, buffer 20k, target update 500, batch 64.

**Results:**

| Metric | t=2000 | t=4000 |
|--------|--------|--------|
| Return | -5.72 | -5.46 |
| Completion | 0% | 0% |
| Avg Elements | 2.0 | 3.0 |
| Mean Quality | 0.474 | 0.385 |
| Loss | 0.038 | 0.011 |

**Training killed at t=4000** — plateau confirmed (memory: kill runs early when metrics plateau). The agent places 2-3 elements but never completes. This is the expected "reward farming" pattern: the agent learns to collect intermediate rewards from a few element placements but can't chain enough valid actions to reach completion.

**Step timing:** ~215ms/step (env) + ~185ms/step (training overhead) ≈ 400ms/step total. 10k steps would take ~67 min.

**Root cause of failure:** The environment selects a SINGLE reference vertex per step (minimum angle). If that vertex has no productive actions (boundary growth guard reverts), the step is wasted. The annulus needs either:
1. **True sub-loop curriculum:** Pre-place type-2 elements externally, initialize env at each sub-loop
2. **Multi-vertex selection:** Allow agent to choose which vertex to operate on
3. **Relaxed boundary growth guard:** Allow boundary-maintaining (not just shrinking) type-0 actions

---

### WS2 Phase 3: Full Type-2 Integration (SKIPPED)

Skipped per decision gate: Phase 2 showed 0% completion.

However, type-2 integration is **verified working**:
- 5 valid type-2 actions at initial state (threshold=0.10)
- Agent can select type-2 actions, no crashes
- Type-2 elements produce correct boundary splits

---

### WS3: Validation + Stretch Goals (COMPLETE)

**7-point validation:** All 8 domains pass (7/7 critical checks each).

**Baseline verification:**

| Domain | Quality | Elements | Complete | Status |
|--------|---------|----------|----------|--------|
| Rectangle (DQN s5) | 0.464 | 9Q+0T | Yes | ✓ Restored (was "0.426") |
| Octagon 24×4 (DQN s8) | 0.579 | 5Q+0T | Yes | ✓ Stable |
| Octagon (greedy 24×4) | 0.370 | 19Q+0T | Yes | - |
| Octagon (greedy 24×8) | 0.421 | 16Q+0T | Yes | +0.051 vs 24×4 |
| H-shape (greedy) | 0.500 | 24Q+0T | Yes | - |
| L-shape (greedy) | 0.323 | 2Q+2T | Yes | - |
| Rectangle (greedy) | 0.570 | 25Q+0T | Yes | - |

**Stretch: Octagon 24×8 greedy eval:**
- Greedy at 24×8: q=0.421 vs 24×4: q=0.370 (+0.051)
- Marginal improvement — the quality gap (0.579 DQN vs 0.421 greedy) is mostly strategy, not resolution
- Not worth training DQN at 24×8 for this small greedy delta

**Code improvements:**
- Fixed `run_greedy()` to handle type-2 actions (was silently ignoring them)
- Fixed eval scripts to import domain definitions from main.py (prevents drift)

---

## Key Metrics

| Metric | Session 13 | Session 14 | Change |
|--------|-----------|------------|--------|
| Rectangle quality | 0.426 (apparent) | **0.464** | Regression was eval script bug |
| Tests passing | 44 | **44** | Unchanged |
| Validation | 8/8 | **8/8** | Unchanged |
| Annulus type-2 valid | 1 (at 0.02) | **5** (at 0.10) | +4 via threshold increase |
| Annulus DQN completion | N/A | **0%** | Needs different approach |

## What Didn't Work

### Direct DQN training on annulus
Even with 5 type-2 actions available, the DQN only places 2-3 elements per episode. The single-ref-vertex selection bottleneck means most steps are wasted — the chosen vertex has no productive actions, and the boundary growth guard reverts the placement. The annulus needs a curriculum approach or multi-vertex architecture.

### Greedy on annulus
The greedy evaluator in `run_greedy()` was silently broken for type-2 actions (fixed). Even after fixing, greedy only places 2 elements because the same single-ref-vertex + boundary growth guard limitations apply.

## What Went Well

- **Controlled experiment was decisive.** 3 threshold values, identical results — boundary filter eliminated as cause in 5 minutes.
- **Root cause found.** The eval script domain definition bug would have caused confusion in future sessions. Fixed permanently by importing from main.py.
- **Type-2 threshold improvement.** Going from 0.02 to 0.10 increased valid type-2 actions from 1 to 5. This is a meaningful improvement for annulus feasibility.
- **Quick kill on training.** Killed at 4k steps (plateau at 0% completion) instead of running full 10k. Saved ~25 min.

## Files Changed

| File | Changes |
|------|---------|
| `madmeshr/mesh_environment.py` | Added `bnd_dist_threshold` parameter to `__init__()` |
| `madmeshr/discrete_action_env.py` | Added `type2_threshold` parameter to `__init__()` |
| `madmeshr/utils/visualization.py` | Added `bnd_dist_threshold` and `type2_threshold` to `run_dqn_eval_and_save()` |
| `main.py` | Added `bnd_dist_threshold` and `type2_threshold` to domain registry; fixed `run_greedy()` to handle type-2 actions; increased greedy max steps to 100 |
| `scripts/annulus_oracle_type2.py` | Made `type2_threshold` configurable (default 0.10) |
| `scripts/eval_ablation.py` | Fixed: import domain definitions from main.py instead of duplicating |
| `scripts/eval_checkpoints.py` | Fixed: import domain definitions from main.py instead of duplicating |

## Short-term Next Steps (Session 15)

1. **True sub-loop curriculum for annulus:** Create a `SubLoopEnv` that initializes with a pre-placed sub-loop boundary (not the full 64-vertex annulus). Use oracle's type-2 placements to create 18-20v sub-loops, train DQN on those.
2. **Multi-vertex selection:** Currently the env picks the minimum-angle vertex. For annulus, the agent needs to choose WHICH vertex to operate on. This is an architecture change: add vertex index to the action space, or rotate the boundary so different vertices become the reference.
3. **Boundary growth tolerance:** The boundary growth guard blocks type-0 actions that maintain boundary count. For annulus, allow boundary-maintaining (not growing) actions. This alone could dramatically improve element placement rate.

## Medium-term Next Steps

4. **Pan et al. benchmark domains.** Standardized test cases for quality comparison.
5. **Gymnasium migration.** Replace deprecated `gym` with `gymnasium`.
6. **DQN architecture improvements:** Larger networks, curriculum learning across domains.

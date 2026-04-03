# Session 2 Plan: Quality Optimization & Infrastructure

**Date:** 2026-04-03
**Status:** In Progress

## Context

The DQN achieves 100% completion on the 10-vertex star domain but with mean quality 0.37 (target 0.7). The agent rushes to completion because `area_consumed` (x2.0) dominates `quality` (x1.0) in per-step reward. Before tuning rewards, we must determine whether the action space resolution (12x4=48 type-1 actions) even permits quality >0.5. If the best achievable quality is ~0.45, no reward function will reach 0.7.

## Adversarial Review Summary

Two adversarial agents stress-tested the initial plan. Key findings both converged on:

1. **Quality ceiling risk:** The 12x4 action grid may cap achievable quality at ~0.45. Reward tuning is pointless if the agent physically cannot select high-quality actions. A diagnostic must run first.
2. **Quality gate cliff is dangerous:** A hard threshold (mean_quality < 0.5 => downgrade) creates a 6+ reward discontinuity that destabilizes TD targets. Replace with smooth quadratic `mean_quality**2`.
3. **Area signal must stay meaningful:** Dropping area_consumed to 0.5x risks recreating the "reward farming" failure from session 1 (agent places many tiny elements for intermediate rewards without completing).
4. **Triangle detection solves a non-problem:** Agent produces 0 triangles on star. Deprioritize.
5. **Checkpoints first:** Independent, zero risk, enables all iteration.

## Workstreams (revised after adversarial review)

### WS1: Checkpoint Save/Load (30 min) -- Do First
**Why first:** Independent, zero risk, enables all subsequent iteration.

**Changes:**
- `src/DQN.py`: Add `save_weights(path)` and `load_weights(path)` using TF's `model.save_weights()`/`model.load_weights()`
- `src/trainer_dqn.py`: Auto-save best model (by eval return) and final model at each eval checkpoint
- `main.py`: Add `--save-dir` and `--load-path` CLI args

**Files:** `src/DQN.py`, `src/trainer_dqn.py`, `main.py`
**Verification:** Train 5k steps on star, save, load, verify identical eval performance.

---

### WS2: Quality Ceiling Diagnostic (20 min) -- Critical Decision Point
**Why:** Both adversarial agents identified the same risk: the 12x4 action grid may cap quality at ~0.45. If true, reward tuning is futile. This diagnostic determines whether we need WS3a (resolution increase) or can skip straight to WS3b (reward tuning).

**What to build:** A script that runs a greedy rollout on the star domain. At each step, for every valid action, compute the element quality. Log:
- `max_quality` across all valid actions per step
- `mean_quality` across all valid actions per step
- Quality of the greedy-chosen action
- Same metrics at 24x8=192 resolution for comparison

**Decision gate:**
- If max achievable quality >= 0.6 at most steps: proceed to WS3b (reward tuning), keep 12x4
- If max achievable quality < 0.55: do WS3a (increase resolution to 24x8) first

**Files:** New script `quality_diagnostic.py` (temporary)
**Verification:** Script runs, prints per-step quality table, makes the decision clear.

---

### WS3a (conditional): Increase Action Resolution
**Only if diagnostic shows quality ceiling < 0.55 with 12x4.**

**Changes:**
- `src/DiscreteActionEnv.py`: Make `n_angle`, `n_dist` configurable via constructor
- `src/DQN.py`: `num_actions` parameter already configurable
- `main.py`: Add `--n-angle` and `--n-dist` CLI args

**Risk:** Enumerate time increases ~4x (192 vs 48 candidates). With vectorized numpy, adds ~10-20ms per step. Acceptable.

**Files:** `src/DiscreteActionEnv.py`, `main.py`
**Verification:** Rerun quality diagnostic at new resolution, confirm ceiling >= 0.6.

---

### WS3b: Reward Rebalancing (45 min)
**Changes to per-step reward** (`DiscreteActionEnv.py`):
- From: `quality + 2.0 * area_consumed - 0.01`
- To: `2.0 * quality + 1.0 * area_consumed - 0.01`
- Rationale: Increase quality weight but keep area_consumed meaningful

**Changes to completion bonus** (`DiscreteActionEnv.py`):
- From: `5.0 + 10.0 * mean_quality` (linear)
- To: `5.0 + 10.0 * mean_quality**2` (smooth quadratic)
- Effect at various quality levels:
  - quality 0.37: 6.37 (was 8.7 -- bigger penalty for low quality)
  - quality 0.50: 7.5
  - quality 0.70: 9.9
  - quality 0.90: 13.1

**Files:** `src/DiscreteActionEnv.py`
**Verification:** Train 100k steps on star domain. Target: mean quality > 0.5, completion > 90%.

---

### WS4: Boundary vs Interior Triangle Detection (30 min) -- If Time Permits
**Why deprioritized:** Agent produces 0 triangles on star. Infrastructure for future sessions.

**Implementation:**
- Store `self.original_boundary` at init
- Check if triangle edges lie on original boundary segments (collinear + contained test)
- Boundary triangle (>= 1 original edge): reward `2.0 + 4.0 * mean_quality`
- Interior triangle (0 original edges): reward `0.5 + 1.0 * mean_quality`

**Files:** `src/MeshEnvironment.py`, `src/DiscreteActionEnv.py`

---

## Execution Order

```
WS1 (checkpoints, 30 min)
  |
WS2 (quality diagnostic, 20 min)
  |
[Decision gate: resolution adequate?]
  |-- No --> WS3a (resolution increase, 20 min) --> WS3b (reward tuning, 45 min)
  +-- Yes --> WS3b (reward tuning, 45 min)
  |
[If time remains]
  +-- WS4 (triangle detection, 30 min)
```

**Total estimated:** 2-2.5 hours including training runs. Leaves buffer for iteration.

## Key Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| Quality ceiling too low even at 24x8 | Try 36x12 or add refinement action type |
| Reward rebalancing breaks completion | Keep area_consumed at 1.0x (not 0.5x), test incrementally |
| Training takes longer than expected | Checkpoints (WS1) let us resume; start with 50k steps |
| Quadratic bonus under-incentivizes completion | Monitor completion rate; if <80%, increase base from 5.0 to 7.0 |

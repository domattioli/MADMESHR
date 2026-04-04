# Session 11 Plan: Type-2 DQN Integration

**Date:** Planned for session after 2026-04-04
**Status:** Final (adversarial-reviewed)
**Budget:** 3 hours

## Context

Session 10 completed multi-loop boundary architecture (`pending_loops`), H-shape domain (7Q, q=0.362), and boundary growth guard. The critical gap: type-2 actions exist only in `MeshEnvironment` methods and the oracle script -- they are NOT in `DiscreteActionEnv` or the DQN action space. The DQN agent literally cannot learn to use type-2 actions.

### Adversarial Review Summary

**Round 1 critiques (incorporated):**
- "_enumerate_for_vertex is architecturally mismatched for type-2." The current flow locks onto a single reference vertex by smallest angle. Type-2 depends on vertex PAIRS. If the chosen ref vertex has no proximity pairs, type-2 is invisible. **Fix:** Enumerate type-2 independently -- scan all boundary vertices for proximity pairs, not just the chosen ref vertex.
- "step() needs a complete type-2 branch." Three separate code paths (form, update, guard) must all work. The growth guard rollback doesn't save/restore `pending_loops`. **Fix:** Save/restore pending_loops on rollback; type-2 branch in step() bypasses growth guard (split always reduces active boundary).
- "30k steps on annulus is wildly insufficient. 132-233 min estimated." **Fix:** Reduce to 5k diagnostic steps; full training is session 12.
- "DQN may never explore type-2 (Q-values start noisy)." **Acknowledged.** epsilon-greedy will try type-2 occasionally; 5k diagnostic run verifies the pipeline, not learning.

**Round 2 critiques (incorporated):**
- "Action space 49→57 breaks all existing checkpoints." **Fix:** Add `n_type2` parameter to DiscreteActionEnv (default 0). Only annulus training passes `n_type2=8`. Existing domains unchanged.
- "Padding slot confirmed at index 43." Plan to use it for `type2_valid` is valid.
- "WS2 (annulus coverage) is independent of WS1 -- do it first to de-risk." **Accepted.** Reorder: WS1 is annulus coverage, WS2 is type-2 DQN integration.
- "Move test updates into WS2, not separate." **Accepted.**
- "Training time: 5k steps ≈ 22-39 min, feasible as diagnostic." **Accepted.**

## Workstreams (3, strict priority order)

---

### WS1: Improve Type-2 Coverage on Annulus (Priority 1, ~30 min)

**Problem:** At threshold=0.02, only 1 of 7 coincident pairs produces a valid type-2 quad. This means even with type-2 in the action space, the agent rarely sees valid type-2 options.

**Steps:**

1. **Increase proximity threshold to 0.15** (~10 min)
   - In `_find_proximity_pairs` default threshold → 0.15
   - In `annulus_oracle_type2.py`, change threshold from 0.02 to 0.15
   - Run: count how many pairs and valid type-2 quads exist now

2. **Re-run annulus oracle** (~10 min)
   - Compare to session 10 baseline: 23Q, q=0.420, stuck at 21v
   - Target: more type-2 quads placed, possibly further along

3. **If still stuck: relax centroid check for non-coincident** (~10 min)
   - The centroid of a bridging quad may lie outside current boundary in narrow strips
   - Try checking centroid against initial boundary instead of current
   - Only for non-coincident pairs (dist > 1e-6)

**Verification:**
- More proximity pairs found at threshold=0.15
- At least 2 valid type-2 quads (up from 1)
- Oracle gets further than 21v stall
- 28 existing tests still pass

**Decision gate at 30 min:** If threshold=0.15 produces >= 3 valid type-2 quads, proceed to WS2. If 0 improvement, investigate why and document.

---

### WS2: Type-2 Actions in DiscreteActionEnv (Priority 2, ~120 min)

**Problem:** DQN cannot select type-2 actions. This is the critical gap preventing RL-based meshing of multi-loop domains like annulus.

**Architecture decision:** Type-2 enumeration is INDEPENDENT of the reference vertex. Scan all boundary vertices for proximity pairs, separate from type-0/type-1 enumeration. This avoids the architectural mismatch of trying to fit type-2 into `_enumerate_for_vertex`.

**Steps:**

1. **Add `n_type2` parameter to DiscreteActionEnv** (~10 min)
   - Default `n_type2=0` (backward compatible, existing checkpoints work)
   - `max_actions = 1 + n_angle * n_dist + n_type2`
   - Update action_space and observation_space

2. **Add type-2 enumeration method** (~30 min)
   - New method `_enumerate_type2()` in DiscreteActionEnv
   - Scans all boundary vertices for proximity pairs via `_find_proximity_pairs`
   - For each pair, calls `_form_type2_element` to validate
   - Returns top-k candidates sorted by quality (k = n_type2)
   - Stores `(2, (ref_idx, far_idx, element, consumed))` in `_valid_actions[49:]`
   - Call from `_enumerate()` after existing enumeration

3. **Add type-2 branch in step()** (~30 min)
   - Branch on `action_type == 2`: extract `(ref_idx, far_idx, element, consumed)`
   - Re-validate with `_form_type2_element` (safety check for stale data)
   - Add element, call `_update_boundary_type2` instead of `_update_boundary`
   - Save/restore `pending_loops` on rollback (growth guard fix)
   - Skip growth guard for type-2 (split always reduces active boundary)
   - Use same per-step reward formula (eta_e + 0.3*eta_b + mu)

4. **Update enriched state** (~10 min)
   - Replace padding at index 43 with `type2_valid` (1.0 if any type-2 slot is valid, 0.0 otherwise)
   - State remains 44 dimensions, existing checkpoints unaffected (type2_valid=0 on non-annulus domains)

5. **Add unit tests** (~20 min)
   - `test_type2_enumeration_on_annulus`: at least 1 type-2 slot masked True
   - `test_type2_step_creates_pending_loop`: step a type-2 action, verify pending_loops
   - `test_existing_domains_unaffected`: square/octagon still complete with n_type2=0
   - Update `test_action_mask_shape` to use `env.max_actions` instead of hardcoded 49

6. **Update main.py** (~10 min)
   - Pass `n_type2=8` when domain is annulus-layer2 (or when a CLI flag like `--n-type2 8` is given)
   - DQN `num_actions` already uses `discrete_env.max_actions`, auto-adapts

**Verification:**
- Annulus enumeration shows type-2 candidates in mask
- Type-2 step produces boundary split with pending_loops
- Existing domains (square, octagon, star) complete identically (n_type2=0)
- All tests pass (28+ existing, target 32+)

**Files:** `src/DiscreteActionEnv.py`, `src/MeshEnvironment.py`, `main.py`, `tests/test_discrete_env.py`

---

### WS3: Diagnostic DQN Training on Annulus (Priority 3, ~30 min)

**Problem:** Need to verify the type-2 pipeline works end-to-end in training. Full convergence is session 12; this session validates the integration doesn't crash.

**Steps:**

1. **Run 5k step diagnostic** (~25 min wall clock)
   ```bash
   python main.py --domain annulus-layer2 --timesteps 5000 \
       --n-type2 8 --save-dir checkpoints/annulus-type2-s11
   ```

2. **Verify** (~5 min)
   - Training completes without crash
   - At least 1 type-2 action taken (check training log)
   - Generate mesh visualization for output/latest/

**Verification:**
- No crash during training
- Type-2 actions appear in training logs (at least 1 episode)
- Output image pushed to repo

**Decision gate:** If training crashes, debug and document. If type-2 is never selected (all slots masked), investigate enumeration threshold.

---

## Execution Order

```
WS1: Annulus Coverage (30 min)
  |
  +-- Increase threshold, re-run oracle
  +-- DECISION GATE: >= 3 valid type-2? Yes → WS2. No → investigate.
  |
WS2: Type-2 DQN Integration (120 min)
  |
  +-- Step 1: n_type2 parameter (10 min)
  +-- Step 2: Type-2 enumeration (30 min)
  +-- Step 3: Type-2 step() branch (30 min)
  +-- Step 4: Enriched state (10 min)
  +-- Step 5: Unit tests (20 min)
  +-- Step 6: main.py CLI (10 min)
  |
  DECISION GATE (150 min):
    If type-2 enumeration works on annulus → WS3
    If blocked → debug and document
  |
WS3: Diagnostic Training (30 min)
  |
  +-- 5k steps on annulus
  +-- Verify + push images
```

## What NOT to Do

- **Do not change the reward structure.** Type-2 gets the same per-step reward.
- **Do not change n_type2 default from 0.** Existing domains must be unaffected.
- **Do not try to train annulus for 30k steps.** It will take 2+ hours. 5k diagnostic only.
- **Do not load existing checkpoints into the 57-action network.** Retrain from scratch.
- **Do not run parallel TF training.** OOM on RTX 3060.

## Success Criteria

| Metric | Session 10 | Target | Stretch |
|--------|-----------|--------|---------|
| Type-2 in DQN action space | No | **Yes** | Yes |
| Annulus type-2 candidates ≥ 3 (threshold=0.15) | 1 | **≥ 3** | ≥ 5 |
| Type-2 step produces boundary split | N/A | **Working** | Working |
| Annulus diagnostic training (5k) no crash | N/A | **Yes** | Yes |
| Existing domains unaffected (n_type2=0) | 28 tests | **28+ pass** | 32+ pass |

## Risk / Mitigation Table

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| _enumerate_for_vertex mismatch | Eliminated | N/A | Type-2 enumerated independently |
| Action data stale between enumerate/step | Low | Medium | Re-validate with _form_type2_element in step() |
| Growth guard rollback doesn't restore pending_loops | Eliminated | N/A | Save/restore pending_loops; skip guard for type-2 |
| DQN never explores type-2 | Medium | Medium | epsilon-greedy will try; 5k diagnostic verifies pipeline |
| Existing checkpoints break | Eliminated | N/A | n_type2=0 default preserves 49-action space |
| WS2 overruns (>120 min) | Medium | High | Decision gate at 150 min: ship what works |

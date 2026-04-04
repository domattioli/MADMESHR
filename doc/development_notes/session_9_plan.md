# Session 9 Plan: Type-2 Prototype + Star Quality Push

**Date:** Planned for next session after 2026-04-04
**Status:** Final (adversarial-reviewed, two rounds of critique incorporated)
**Budget:** 3 hours

## Context

Session 8 results:
1. **Annulus-layer2 is DEAD END** with current type-0/type-1 formulation. Quad topology [ref, ref+1, candidate, ref-1] fails on narrow strips because all 3 boundary vertices are on the same side. Tested 12 grid/distance-range configs — zero viable configurations.
2. **Octagon 24×4 breakthrough:** q=0.579 (5Q), up from 0.478 (3Q) — 95% of 0.61 ceiling. Quality jump happened late (10k-15k) when epsilon dropped below 0.10.
3. **Star 24×4 at 0.405** (92% of 0.44 ceiling, session 7).
4. **Zero-shot transfer works:** star model gets 100% completion on all domains.
5. **Critical geometry finding:** The annulus-layer2 boundary has **7 pairs of exactly coincident vertices** (distance=0.000) that are non-adjacent in boundary order (gaps of 5-23 positions). This confirms the boundary folds back on itself and proximity detection WILL find candidates.

## Adversarial Review Summary

### Round 1 Critiques (Incorporated)
- **Boundary update splits loop into TWO loops:** A type-2 quad consuming 4 non-contiguous vertices from a single loop creates two disjoint arcs. The current `_update_boundary()` assumes contiguous consumed segments. This is a fundamental topology change, not a patch. **Action: Scope WS1 to prototype + oracle only. No DQN integration this session.**
- **State representation blind to type-2:** 44-float state has no cross-strip info. DQN can't learn when to use type-2. **Action: Defer DQN integration to session 10. Oracle-only this session.**
- **Quad orientation ambiguous:** [ref, ref+1, far+1, far] — `far+1` might be on the wrong side. **Action: Add winding check; try both far+1 and far-1, pick the one that forms a valid quad.**
- **120 min unrealistic for full DQN integration:** **Action: Split WS1 into geometric prototype (oracle test) only. DQN integration is session 10.**

### Round 2 Critiques (Incorporated)
- **Action space 49→57 breaks all existing checkpoints:** DQN output layer is Dense(49). Changing to 57 invalidates star/octagon/rectangle models. **Action: Keep DQN at 49 actions. Type-2 is oracle-only, implemented as a separate code path in the environment, not wired into DiscreteActionEnv.**
- **Proximity threshold (0.5*fan_radius ≈ 0.15-0.25) will find same-side candidates:** Edge lengths ~0.05-0.27, so vertices 3-4 positions apart are within 0.15-0.25. **Action: Use much smaller threshold (0.02) since the actual cross-strip distance is ~0.000 for 7 vertex pairs. Also filter by boundary gap (require gap >= 3 in boundary order).**
- **Star resume at epsilon=0.05 won't find new strategies:** Octagon's phase transition happened at epsilon ~0.15. Star at 0.05 has no exploration budget left. **Action: Train star from scratch with slower epsilon decay (epsilon_decay_steps doubled) instead of resuming. This keeps exploration in the critical 0.10-0.30 range longer.**

### Remaining Acknowledged Risks
1. The boundary update topology after type-2 quads (two-arc reconnection) may have subtle winding issues. Prototype will reveal these.
2. The coincident vertex pairs may not all produce valid quads — the quad [ref, ref+1, far+1, far] could still self-intersect depending on local geometry.
3. Even if the oracle works, DQN integration (session 10) requires: action space expansion, state enrichment, checkpoint migration, and boundary update in the training loop.

## Workstreams (3, strict priority order)

---

### WS1: Type-2 Geometric Prototype + Oracle Test (Priority 1, ~90 min)

**Problem:** Annulus-layer2 cannot be meshed with type-0/type-1. Need to prove that type-2 actions (quads spanning opposite boundary sides) can complete the mesh, before investing in DQN integration.

**Scope:** Standalone prototype. NOT integrated into DiscreteActionEnv or DQN. Implemented as helper functions in MeshEnvironment + an oracle script.

**Steps:**

1. **Implement `_find_proximity_pairs(self, ref_idx, threshold=0.02, min_gap=3)` in MeshEnvironment.py** (~20 min)
   - For ref vertex, find all boundary vertices where:
     - Euclidean distance < threshold
     - Boundary gap (min of forward/backward distance in loop) >= min_gap
   - Return list of (far_idx, distance) pairs sorted by distance
   - Vectorized: single distance computation `np.linalg.norm(boundary - ref_vertex, axis=1)`, mask adjacent indices

2. **Implement `_form_type2_element(self, ref_idx, far_idx)` in MeshEnvironment.py** (~20 min)
   - Try two quad orientations:
     - Option A: [ref, ref+1, far-1, far] (far-1 in boundary order)
     - Option B: [ref, ref+1, far+1, far] (far+1 in boundary order)
   - For each, check: non-self-intersecting (`_is_valid_quad`), all 4 vertices on boundary, reasonable area (> 0)
   - Return the first valid quad, or None

3. **Implement `_update_boundary_type2(self, element, ref_idx, far_idx)` in MeshEnvironment.py** (~30 min)
   - This is the hardest part. After consuming {ref, ref+1} and {far, far±1}:
     - Two disjoint arcs remain
     - The quad's interior edges reconnect them into a single loop
   - Algorithm:
     - Identify which quad vertices are consumed: the 4 boundary vertices
     - Identify the two interior edges of the quad (edges NOT on the boundary)
     - Arc 1: boundary vertices from ref+2 to far-1 (or far+1 depending on orientation)
     - Arc 2: boundary vertices from far+2 to ref-1 (or vice versa)
     - New boundary = Arc1 + [connector vertex from quad] + Arc2 + [connector vertex from quad]
     - Verify: new boundary has correct winding (CCW), forms a valid closed loop
   - **Fallback:** If winding is wrong, reverse one arc and retry.

4. **Unit tests** (~10 min)
   - Test proximity detection on annulus: at least 5 vertex pairs found with threshold=0.02
   - Test type-2 element formation: at least one valid quad from a coincident pair
   - Test boundary update: boundary vertex count decreases, single loop maintained

5. **Oracle test on annulus-layer2** (~10 min)
   - Greedy oracle that tries type-2 first (when proximity candidates exist), then falls back to type-0
   - Strategy: prefer type-2 actions that maximize boundary reduction
   - Run up to 200 steps
   - Report: completion status, element count, quality, boundary trajectory

**Verification criteria:**
- Proximity detection finds >= 5 candidate pairs on fresh annulus boundary: **PASS/FAIL**
- At least one type-2 quad is valid (non-self-intersecting, positive area): **PASS/FAIL**
- Boundary update after type-2 preserves single CCW loop: **PASS/FAIL**
- Oracle completes annulus-layer2 (boundary < 3 vertices): **STRETCH**
- All 21 existing tests still pass (type-2 is additive): **REQUIRED**

**Decision gate at 60 min:**
- If proximity detection + element formation work but boundary update is broken: document the topology issue, move to WS2
- If proximity detection finds nothing: re-examine geometry, try larger threshold
- If everything works: run oracle test

**Files changed:**
- `src/MeshEnvironment.py`: 3 new methods (~100 lines total)
- `tests/test_discrete_env.py`: 3 new tests
- `annulus_oracle_type2.py`: Standalone oracle script (diagnostic, not production)

**What NOT to change:**
- `DiscreteActionEnv.py` — no action space changes
- `DQN.py` — no network changes
- Existing checkpoints — fully compatible
- Reward structure — unchanged

---

### WS2: Star 24×4 from Scratch with Slow Epsilon (Priority 2, ~50 min)

**Problem:** Star at 0.405 quality (92% of 0.44 ceiling). The octagon's phase transition happened at epsilon ~0.15 — the agent discovered a 5Q strategy when it had enough exploration to try non-greedy actions. Star at 30k steps has epsilon=0.05, which is too low to discover new strategies. Resuming won't help.

**Approach:** Train star 24×4 from scratch with doubled epsilon_decay_steps. This keeps epsilon in the 0.10-0.30 range for twice as long, giving the agent more time to discover quality-improving strategies (like the octagon's 3Q→5Q transition).

**Steps:**

1. **Check current epsilon schedule** in `src/trainer_dqn.py`: find `epsilon_decay_steps` and compute the schedule. The default is likely ~11k steps (epsilon decays from 1.0 to 0.05 over this range).

2. **Training Run A — 30k steps with slow epsilon** (~20 min):
   - Train star 24×4 from scratch
   - Set `epsilon_decay_steps` to 22k (doubled), keeping `epsilon_min=0.05`
   - This means epsilon reaches 0.10 at ~20k instead of ~10k
   - Eval at 5k, 10k, 15k, 20k, 25k, 30k

3. **Compare to session 7 baseline** (30k steps, default epsilon):
   - If q > 0.405: slow epsilon helped
   - If q ≈ 0.405: the ceiling is at current resolution, not epsilon-dependent
   - If q < 0.405: slower exploration hurts convergence

4. **If Run A shows improvement, extend to 50k** (~15 min additional):
   - Continue training for +20k steps
   - Eval at 35k, 40k, 45k, 50k

**Files:** `src/trainer_dqn.py` (configurable epsilon_decay_steps, may need CLI arg). No architecture changes.

**Verification criteria:**
- Star quality > 0.42 (96% of ceiling): **PASS**
- Star quality > 0.405 (any improvement): **PARTIAL**
- Star quality <= 0.405: **FAIL** — ceiling is saturated

**Time estimate:** 50 min (20 min training + 15 min extension + 15 min analysis)

---

### WS3: Circle 24×4 + Octagon Reproducibility (Priority 3, stretch, ~30 min)

**Problem:** Need to validate that 24×4 improvement generalizes beyond octagon.

**Steps:**

1. **Circle 24×4 training** (15 min):
   - Train circle 24×4, 15k steps from scratch
   - Circle has 16 vertices — more complex than octagon (8v)
   - Ceiling is 0.78
   - If quality approaches ceiling, 24×4 is validated across convex domains

2. **Octagon 24×4 reproducibility** (15 min):
   - Retrain octagon 24×4, 15k steps (different random seed from session 8)
   - Session 8 result: q=0.579, 5Q
   - If q > 0.55, the 5Q strategy is reproducible

**No code changes.** Training runs only. Sequential (never parallel TF).

**Verification criteria:**
- Circle 24×4 quality > 0.65: **PASS**
- Octagon reproducible (q > 0.55): **PASS**

---

## Execution Order with Decision Gates

```
WS1: Type-2 Prototype (90 min)
  |
  ├── Step 1: _find_proximity_pairs (20 min)
  │     └── Test: finds >= 5 pairs on annulus
  │
  ├── Step 2: _form_type2_element (20 min)
  │     └── Test: at least one valid quad
  │
  ├── Step 3: _update_boundary_type2 (30 min) ← HARDEST PART
  │     └── Test: single loop preserved
  │
  ├── DECISION GATE (60 min):
  │     ├── All working → Step 4-5 (oracle test)
  │     └── Boundary update broken → document, proceed to WS2
  │
  ��── Step 4: Unit tests (10 min)
  ├── Step 5: Oracle test (10 min)
  │
WS2: Star Slow Epsilon (50 min)
  |
  ├── Run A: 30k steps with 2× epsilon decay → eval
  └── If improved → Run A+: extend to 50k
  
WS3: Circle + Octagon (30 min, stretch)
  |
  ├── Circle 24x4 (15 min)
  └── Octagon repro (15 min)
```

## Risk/Mitigation Table

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Boundary update creates invalid winding after type-2 | HIGH | Blocks oracle test | Try both arc orderings. If neither works, manually construct the correct loop by tracing edges. Fall back to WS2 at 60 min gate. |
| Coincident vertex pairs produce self-intersecting quads | MEDIUM | Fewer usable type-2 actions | Try both far+1 and far-1 orientations. Even if only 3/7 pairs work, that may be enough for the oracle. |
| Type-2 quads have very low quality (thin spanning elements) | LOW | Cosmetic issue only | Expected. On the annulus, any mesh completion is a breakthrough. Quality optimization is session 10+. |
| Slow epsilon hurts star convergence | MEDIUM | WS2 produces worse result | Compare at 30k. If quality is worse, the default schedule was already near-optimal. This is still useful data. |
| Circle 24×4 doesn't improve (ceiling is vertex-count-limited) | LOW | WS3 neutral result | Informative regardless — tells us whether 24×4 helps only for small domains. |

## What NOT to Do

- **Don't integrate type-2 into DiscreteActionEnv or DQN.** Action space stays at 49. Checkpoints stay compatible. DQN integration is session 10 once the geometry is proven.
- **Don't change the reward structure.** Type-2 oracle uses the same reward formula for analysis.
- **Don't run parallel TF training.** OOM risk on RTX 3060.
- **Don't resume star from checkpoint.** Epsilon is too low. Train from scratch with slow schedule.
- **Don't over-engineer proximity detection.** The 7 coincident pairs at distance=0.000 are trivially findable with a tiny threshold (0.02). No need for KD-trees or angular filtering.

## Success Criteria

| Metric | Session 8 | Target | Stretch |
|--------|-----------|--------|---------|
| Type-2 proximity pairs found on annulus | N/A | >= 5 pairs | >= 7 pairs (all coincident) |
| Type-2 valid quads formed | N/A | >= 1 | >= 5 |
| Boundary update preserves single loop | N/A | Yes | Yes |
| Annulus oracle completion (type-0 + type-2) | 0% | Oracle tested | Oracle completes |
| Star quality (24×4, slow epsilon) | 0.405 | > 0.42 | > 0.43 |
| Circle quality (24×4) | N/A | > 0.65 | > 0.70 |
| Tests passing | 21 | 24+ | 24+ |

## Adversarial Review Process

**Round 1 findings:** Original plan attempted full DQN integration of type-2 (action space 49→57, state changes, DiscreteActionEnv wiring) within 120 minutes. The boundary update splitting a single loop into two arcs was identified as a fundamental topology change, not a minor extension. State representation lacks type-2 information. Revised to: oracle-only prototype, no DQN integration, no action space changes.

**Round 2 findings:** Action space change would break all existing checkpoints. Proximity threshold of 0.5*fan_radius would find same-side candidates (edge length ~0.16, many vertices within 0.25). Star resume at epsilon=0.05 has no exploration budget. Revised to: DQN stays at 49 actions, threshold reduced to 0.02 (matches coincident vertex pairs), star trained from scratch with slow epsilon.

**Key design decision:** Separating geometric proof-of-concept (session 9) from DQN integration (session 10) reduces risk and ensures we don't break existing working models while exploring new architecture.

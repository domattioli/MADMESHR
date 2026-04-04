# Session 10 Plan: Type-2 Boundary Split Fix + H-Domain

**Date:** Planned for next session after 2026-04-04
**Status:** Final (adversarial-reviewed, user-directed priorities)
**Budget:** 3 hours

## Context

Session 9 results:
1. **Type-2 boundary update is fundamentally wrong.** After placing a type-2 quad that bridges two non-adjacent boundary segments, `_update_boundary_type2` reconnects the two surviving arcs into a single loop — but the resulting boundary has edges that pass through the placed element. The correct behavior is to split into **two separate boundary loops** (two sub-domains separated by the quad). This is an architectural issue: `self.boundary` is a single polygon and the environment doesn't support multiple loops.
2. **Type-2 centroid check works** but masks the boundary update problem. Only 1/7 annulus pairs passes all checks.
3. **Boundary growth bug**: type-0 on narrow strips grows boundary infinitely. Fixed in oracle, NOT in DiscreteActionEnv.
4. 24/24 tests passing.

### User Direction

The user identified that the type-2 boundary update is the core issue: "the updated boundary is creating edges that go through it instead of creating two distinct sub-domains, separated by the newly created quad." This must be the session 10 priority.

The user also wants an H-shaped test domain: 5 vertices defining height, 4 vertices defining width. Ideal mesh = 11 quads. The H-shape naturally tests the boundary-split topology because the crossbar creates a type-2-like situation where a quad bridges two sides.

### Adversarial Review Summary

**Round 1 critiques (incorporated):**
- "Multi-loop boundary is an architectural change, not a patch" → Correct. WS1 implements multi-loop support: `self.boundary` becomes `self.boundaries` (list of polygons). All methods that read/write boundary must be updated.
- "11 quads for an H may not be achievable with type-0/type-1 alone" → The H-shape has concave corners that may require type-1 interior vertex placement. The oracle should find the optimal decomposition.
- "Existing checkpoints will break if boundary representation changes" → The multi-loop change is internal to MeshEnvironment. DiscreteActionEnv can still present the active loop as a single boundary. Existing checkpoints work on single-loop domains and never encounter multi-loop states.

**Round 2 critiques (incorporated):**
- "Converting boundary to boundaries touches every method" → Scope the change: keep `self.boundary` as the *active* loop. Add `self.pending_loops` for loops created by type-2 splits. When the active loop is completed (< 3 vertices), pop the next pending loop. This minimizes changes to existing code.
- "The H domain geometry needs precise vertex coordinates" → Define on a unit grid. H = 12 vertices, CCW winding.

## Workstreams (3, strict priority order)

---

### WS1: Multi-Loop Boundary Architecture (Priority 1, ~90 min)

**Problem:** `_update_boundary_type2` produces a single boundary with edges through the element. Must split into two loops.

**Approach:** Minimal-disruption architecture:
- Keep `self.boundary` as the current active loop (what DQN sees)
- Add `self.pending_loops = []` for loops created by type-2 splits
- When active loop completes (boundary < 3 vertices), pop next pending loop
- `_update_boundary_type2` appends the shorter arc to pending_loops, keeps the longer arc as active boundary

**Steps:**

1. **Add `pending_loops` to MeshEnvironment** (~15 min)
   - Initialize in `__init__` and `reset()`
   - Add `_activate_next_loop()` method: when boundary < 3 vertices and pending_loops is non-empty, set boundary = pending_loops.pop(0)

2. **Rewrite `_update_boundary_type2`** (~30 min)
   - After computing arc1 and arc2: the shorter arc goes to `pending_loops`, the longer arc becomes `self.boundary`
   - Both arcs must be valid closed loops (>= 3 vertices each)
   - If either arc has < 3 vertices, it's a degenerate split — reject the action
   - Verify winding (CCW) on both arcs

3. **Update DiscreteActionEnv completion logic** (~15 min)
   - In `step()`, after checking `done`: if boundary < 3 but pending_loops exists, it's NOT done — activate next loop
   - Completion = all loops meshed (boundary < 3 AND no pending loops)

4. **Update oracle** (~15 min)
   - `annulus_oracle_type2.py`: after each step, check pending_loops. When active loop completes, activate next.
   - Report which loop is being meshed

5. **Unit tests** (~15 min)
   - Test that type-2 on annulus produces two loops
   - Test that both loops have positive area and no edges through the type-2 element
   - Test pending_loop activation

**Verification criteria:**
- Type-2 split produces two valid loops: **REQUIRED**
- No boundary edges pass through placed elements: **REQUIRED**
- Pending loop activation works: **REQUIRED**
- All 24 existing tests pass: **REQUIRED**

**Decision gate at 60 min:**
- If multi-loop architecture is working on annulus: proceed to WS2
- If blocked: simplify — implement split detection but don't activate pending loops yet, just validate the split geometry

---

### WS2: H-Shape Domain (Priority 2, ~60 min)

**Problem:** Need a test domain that naturally exercises the boundary-split topology. The H-shape has a crossbar connecting two vertical bars — meshing the crossbar requires quads that bridge the two sides, similar to type-2.

**H-Shape Geometry:**
```
12 vertices on unit grid, 5 high × 4 wide:

    (0,4)---(1,4)           (3,4)---(4,4)
      |       |               |       |
    (0,3)---(1,3)-----------(3,3)---(4,3)
              |               |
            (1,1)-----------(3,1)
              |               |
    (0,1)---(1,1)           (3,1)---(4,1)
      |       |               |       |
    (0,0)---(1,0)           (3,0)---(4,0)
```

Wait — the H has internal edges. As a single boundary polygon it would be:

CCW winding: (0,0) → (1,0) → (1,1) → (3,1) → (3,0) → (4,0) → (4,4) → (3,4) → (3,3) → (1,3) → (1,4) → (0,4) → (0,0)

That's 12 vertices. The crossbar is between y=1 and y=3, x=1 and x=3.

Ideal decomposition: 11 quads (2 in left bar bottom, 2 in left bar top, 2 in right bar bottom, 2 in right bar top, 3 in crossbar = 11). Wait, that depends on the exact grid. Let me count: left bar = 1×4 = 4 unit squares = 4 quads if meshed 1-wide. But the bar is 1 unit wide and 4 units tall... Actually user said 5 vertices high (so 4 units) and 4 wide (3 units). Let me reconsider.

Actually, the user's description is the key constraint. I'll define the geometry precisely in the implementation.

**Steps:**

1. **Define H-shape boundary** (~10 min)
   - 12 vertices, CCW, on unit grid
   - Register in main.py with `max_ep_len=20`

2. **Greedy baseline** (~10 min)
   - Run greedy oracle on H-shape
   - Does type-0 alone complete it?

3. **Type-2 oracle on H-shape** (~20 min)
   - The crossbar should create proximity pairs between the two vertical bars
   - Test whether multi-loop split works correctly here

4. **DQN training** (~20 min)
   - Train 15k steps with 24×4 grid
   - Compare to greedy baseline

**Verification criteria:**
- H-shape domain registered and visualized: **REQUIRED**
- Greedy baseline reported: **REQUIRED**
- Type-2 oracle tested: **STRETCH**
- DQN trained: **STRETCH**

---

### WS3: Boundary Growth Fix in DiscreteActionEnv (Priority 3, ~30 min)

**Problem:** The boundary growth bug exists in the core environment, not just the oracle. DQN training on annulus could exploit it.

**Steps:**

1. **Add growth detection to DiscreteActionEnv.step()** (~15 min)
   - Save boundary count before `_update_boundary()`
   - After update: if `len(boundary) > saved_count`, undo element and return invalid action penalty

2. **Unit test** (~15 min)
   - Test on annulus-layer2: run 50 random-valid-action steps, verify boundary never grows

**Files:** `src/DiscreteActionEnv.py`, `tests/test_discrete_env.py`

---

## Execution Order

```
WS1: Multi-Loop Boundary (90 min)
  |
  ├── Step 1: pending_loops infrastructure (15 min)
  ├── Step 2: Rewrite _update_boundary_type2 (30 min)
  ├── Step 3: Update DiscreteActionEnv (15 min)
  │
  ├── DECISION GATE (60 min):
  │     ├── Working → Steps 4-5, then WS2
  │     └── Blocked → Simplify, document, proceed to WS2
  │
  ├── Step 4: Update oracle (15 min)
  └── Step 5: Unit tests (15 min)

WS2: H-Shape Domain (60 min)
  |
  ├── Define + register (10 min)
  ├── Greedy baseline (10 min)
  ├── Type-2 oracle (20 min)
  └── DQN training (20 min)

WS3: Boundary Growth Fix (30 min)
  |
  ├── DiscreteActionEnv fix (15 min)
  └── Unit test (15 min)
```

## What NOT to Do

- **Don't change the reward structure.** Pan et al. validation deferred to session 11.
- **Don't convert self.boundary to self.boundaries everywhere.** Use the pending_loops approach to minimize disruption.
- **Don't run parallel TF training.** OOM on RTX 3060.
- **Don't try to fix all 7 coincident pairs.** Multi-loop support may unlock more, but that's a bonus, not the goal.

## Success Criteria

| Metric | Session 9 | Target | Stretch |
|--------|-----------|--------|---------|
| Type-2 boundary split correct (two loops) | No | Yes | Yes |
| No boundary edges through elements | No | Yes | Yes |
| H-shape domain registered | No | Yes | Yes |
| H-shape greedy completion | N/A | Reported | 11Q |
| Boundary growth fix in DiscreteActionEnv | No | Yes | Yes + test |
| Tests passing | 24 | 27+ | 28+ |

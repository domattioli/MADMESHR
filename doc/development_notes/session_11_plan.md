# Session 11 Plan: Concave Domain Validity Fix

**Date:** Planned for next session after 2026-04-04
**Status:** Final (multi-agent reviewed)
**Budget:** 3 hours

## Context

Session 10 discovered that **elements extend outside the domain boundary on concave domains.** The H-shape DQN mesh (7Q, q=0.362) has quads whose edges shortcut across the rectangular cutouts. This affects all concave domains (H-shape, L-shape, star) and invalidates training results on those domains.

The current validity checks (self-intersection, centroid-in-polygon, positive area) are insufficient. A quad can pass all checks but have edges that cross through exterior concavities of the domain boundary.

### Multi-Agent Design Review (Session 10)

Three specialized agents collaboratively designed the fix:

1. **Robustness Planner** proposed 10 comprehensive checks covering every conceivable failure mode
2. **Minimalist Critic** reduced to 3 essential checks by proving redundancy (Jordan Curve Theorem: if endpoints are inside and edges don't cross boundary, midpoints must be inside)
3. **Computational Optimizer** analyzed costs, proposed vectorized implementations, and validated feasibility (<20% wall-clock overhead)

After 3 rounds of debate, all three converged on **4 checks** (see WS1 below).

## Workstreams (3, strict priority order)

---

### WS1: Implement Concave Domain Validity Checks (Priority 1, ~120 min)

**Problem:** Elements can extend outside concave domain boundaries. The mesh generator produces invalid meshes on H-shape, L-shape, and star.

**Agreed check plan (4 checks):**

#### Check A: Element edges vs original boundary (in enumerate)

For each candidate quad that passes existing PIP + self-intersection checks, verify that none of its 4 edges cross any segment of `self.initial_boundary`. Shared vertices are excluded (element edges that start/end at boundary vertices are fine).

**Implementation:** New method `_batch_edges_cross_original_boundary(quads)` in MeshEnvironment. Called in `_enumerate_for_vertex` after existing `_batch_is_valid_quad`, only on survivors.

**Files:** `src/MeshEnvironment.py`

**Precomputation in `reset()`:**
```python
self._orig_seg_starts = self.initial_boundary.copy()  # (M, 2)
self._orig_seg_ends = np.roll(self.initial_boundary, -1, axis=0).copy()  # (M, 2)
```

**Vectorized check (pseudocode):**
```python
def _batch_edges_cross_original_boundary(self, quads):
    """(N, 4, 2) array of quads -> (N,) bool, True if any edge crosses original boundary."""
    N = len(quads)
    crosses = np.zeros(N, dtype=bool)
    for edge_idx in range(4):
        e_start = quads[:, edge_idx, :]              # (N, 2)
        e_end = quads[:, (edge_idx + 1) % 4, :]      # (N, 2)
        for seg_idx in range(len(self._orig_seg_starts)):
            s_start = self._orig_seg_starts[seg_idx]  # (2,)
            s_end = self._orig_seg_ends[seg_idx]      # (2,)
            # Skip shared vertices
            shared = (dist(e_start, s_start) < tol) | (dist(e_start, s_end) < tol) |
                     (dist(e_end, s_start) < tol) | (dist(e_end, s_end) < tol)
            hit = batch_segments_intersect(e_start, e_end, s_start, s_end)
            crosses |= (hit & ~shared)
    return crosses
```

**Cost:** O(N_survivors * 4 * M_orig). For H-shape (M=12, ~10 survivors): 480 segment tests per step. Annulus (M=64, ~10 survivors): 2560. Both negligible.

#### Check B: Element edges vs current boundary, non-consumed (in enumerate)

Same logic as Check A but against current boundary edges, excluding edges where both endpoints are consumed by the element (they're part of the element itself). This catches elements crossing interior edges from previously placed elements.

**Implementation:** New method `_batch_edges_cross_current_boundary(quads, ref_idx)` in MeshEnvironment. Called in `_enumerate_for_vertex` after Check A.

**Cost:** O(N_survivors * 4 * B_current). Similar to Check A.

#### Check C: Post-update boundary self-intersection (in step, with rollback)

After `_update_boundary()` in `DiscreteActionEnv.step()`, check that the updated boundary polygon has no self-intersecting edges. If it does, roll back (same pattern as boundary growth guard).

**Implementation:** New method `_boundary_has_self_intersection()` in MeshEnvironment. Called in `DiscreteActionEnv.step()` after the boundary growth guard.

**Cost:** O(B^2) per step. B typically 10-30, so 100-900 segment pair tests. Runs once per step, not per candidate.

#### Check D: Element overlap with existing elements (in step, with rollback)

Before committing a new element, check that none of its edges cross edges of any previously placed element (excluding shared vertices), and that neither centroid is inside the other.

**Implementation:** New method `_element_overlaps_existing(element)` in MeshEnvironment. Called in `DiscreteActionEnv.step()` before `elements.append()`.

**Cost:** O(E * 16) per step. At E=20 elements, 320 tests. Runs once per step.

#### Integration into enumerate pipeline:

```
Existing: _batch_point_in_polygon(candidates, self.boundary)     # PIP filter
Existing: _batch_is_valid_quad(quads[pip_survivors])             # self-intersection
NEW:      _batch_edges_cross_original_boundary(quads[survivors]) # Check A
NEW:      _batch_edges_cross_current_boundary(quads[survivors])  # Check B
```

#### Integration into step():

```
Existing: form element + validate
NEW:      _element_overlaps_existing(new_element)                # Check D
Existing: _update_boundary(new_element)
Existing: boundary growth guard (rollback if grew)
NEW:      _boundary_has_self_intersection() guard (rollback)     # Check C
```

**Verification criteria:**
- H-shape greedy produces no out-of-domain elements: **REQUIRED**
- L-shape greedy produces no out-of-domain elements: **REQUIRED**
- 7-point validation passes on all domains: **REQUIRED**
- All 28 existing tests pass: **REQUIRED**
- New tests for each check: **REQUIRED**
- Enumerate performance < 2x slower: **REQUIRED**

**Decision gate at 90 min:** If checks A+B are working in enumerate and domains produce valid meshes, proceed to WS2. If blocked on vectorization, implement scalar versions and optimize later.

---

### WS2: Re-Train Concave Domains (Priority 2, ~50 min)

**Problem:** Existing H-shape and L-shape training results are invalid (elements outside domain). Need to retrain with validity checks active.

**Steps:**

1. **H-shape DQN 15k steps** (~22 min)
   - Compare to old result (7Q, q=0.362)
   - Expect: more elements (validity checks reject shortcuts), possibly lower quality

2. **L-shape greedy + DQN 10k steps** (~15 min)
   - Greedy baseline with new checks
   - DQN training to verify convergence

3. **7-point validation on all domains** (~10 min)
   - Run validation script on square, octagon, circle, star, L-shape, H-shape, rectangle
   - All should pass all 7 checks
   - Push result images to `output/latest/`

**Verification criteria:**
- H-shape mesh has zero elements outside domain: **REQUIRED**
- All domains pass 7-point validation: **REQUIRED**

---

### WS3: Update Test Suite + Documentation (Priority 3, ~10 min)

**Steps:**

1. Add tests for each new check:
   - `test_element_edges_reject_concave_shortcut`: H-shape type-0 that would shortcut across notch is masked out
   - `test_boundary_self_intersection_detected`: synthetic self-intersecting boundary is caught
   - `test_element_overlap_detected`: overlapping element is caught

2. Update CLAUDE.md known issues

3. Push all images to repo

---

## Execution Order

```
WS1: Validity Checks (120 min)
  |
  +-- Precomputation: _orig_seg_starts/ends in reset() (5 min)
  +-- Check A: _batch_edges_cross_original_boundary (30 min)
  +-- Check B: _batch_edges_cross_current_boundary (20 min)
  +-- Integrate into _enumerate_for_vertex (15 min)
  +-- Check D: _element_overlaps_existing in step() (15 min)
  +-- Check C: _boundary_has_self_intersection in step() (15 min)
  +-- Unit tests (20 min)
  |
  DECISION GATE (90 min):
    Checks working + domains valid → WS2
    Blocked → scalar fallback, skip WS2
  |
WS2: Re-Train Concave Domains (50 min)
  |
  +-- H-shape DQN 15k (22 min, run in background)
  +-- L-shape greedy + DQN 10k (15 min)
  +-- 7-point validation all domains (10 min)
  |
WS3: Tests + Docs (10 min)
```

## What NOT to Do

- **Do not attempt type-2 DQN integration.** That is session 12 — validity must be fixed first.
- **Do not change the reward structure.** The checks are in enumerate (masking) and step (rollback), not reward.
- **Do not over-optimize the vectorization.** A scalar loop that works is better than a broken vectorized version. Optimize in session 12 if needed.
- **Do not run parallel TF training.** OOM on RTX 3060.

## Success Criteria

| Metric | Session 10 | Target | Stretch |
|--------|-----------|--------|---------|
| Elements outside domain (H-shape) | Yes (broken) | **0** | 0 |
| Elements outside domain (L-shape) | Unknown | **0** | 0 |
| 7-point validation all domains | 6/7 | **7/7** | 7/7 |
| H-shape DQN quality (valid mesh) | 0.362 (invalid) | **Reported** | > 0.30 |
| Enumerate performance overhead | N/A | **< 2x** | < 1.5x |
| Tests passing | 28 | **31+** | 34+ |

## Risk / Mitigation Table

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Vectorized segment intersection has edge cases | Medium | Medium | Test with known-bad geometries from H-shape; fall back to scalar |
| Too many candidates masked → no valid actions | Medium | High | Monitor mask count; if all candidates masked, existing truncation logic handles it |
| Performance > 2x overhead | Low | Medium | Only run expensive checks on PIP+self-intersection survivors (~10 of 48) |
| Boundary self-intersection check false positives | Low | Medium | Use tolerance in shared-vertex exclusion (1e-8) |

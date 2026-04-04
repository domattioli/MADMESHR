# Session 7 Report: Transfer Diagnostic + Enumeration Speedup + Curriculum

**Date:** 2026-04-03/04

## Summary

Four workstreams completed with significant results: (1) Zero-shot transfer diagnostic revealed the star 12×4 model achieves 100% completion on all other domains without retraining — a strong positive signal for feature generalization. (2) Vectorized action enumeration eliminated 493k `np.allclose` calls per 500 steps, achieving a 9.2× speedup on annulus-layer2. (3) Curriculum learning (octagon→star→annulus-layer2) confirmed that learned features transfer for faster convergence on similar-sized domains, but annulus-layer2 (64v) still fails at 0% — diagnosed as an action space coverage problem, not a feature quality problem. (4) 24×4 star ablation ran to convergence at q=0.405 (vs 0.371 for 12×4), confirming angular resolution helps modestly. Code changes: vectorized geometry in MeshEnvironment.py. 21/21 tests passing.

## What Was Completed

### WS1: Zero-Shot Transfer Diagnostic (STRONG POSITIVE)

**Setup:** Loaded best star 12×4 checkpoint, evaluated zero-shot on octagon/l-shape/rectangle (10 episodes each, greedy policy). Compared to random baseline (10 episodes, uniform random valid action).

**Results:**

| Domain | Metric | Star Model (zero-shot) | Random Baseline | Previously Trained |
|--------|--------|----------------------|-----------------|-------------------|
| Octagon | Quality | 0.404 | 0.311 | 0.478 |
| | Completion | 100% | 10% | 100% |
| | Elements | 4 | 9.9 | 3 |
| L-shape | Quality | 0.459 | 0.267 | 0.459 |
| | Completion | 100% | 60% | 100% |
| | Elements | 2 | 7.8 | 2 |
| Rectangle | Quality | 0.460 | 0.271 | 0.464 |
| | Completion | 100% | 0% | 100% |
| | Elements | 12 | 25 | 9 |
| Star | Quality | 0.371 | 0.338 | 0.371 |
| | Completion | 100% | 20% | 100% |

**Result: STRONG POSITIVE.** Zero-shot transfer achieves 100% completion on all domains. L-shape quality matches the trained model exactly (0.459). Rectangle quality nearly matches (0.460 vs 0.464) but uses more elements (12 vs 9). Octagon quality (0.404) below trained (0.478) — room for domain-specific fine-tuning but the features generalize well.

**Key insight:** The 44-dim state representation and learned policy generalize across domains of similar vertex count (6-20 vertices). The features learned on star (10v) are not star-specific.

---

### WS2: Action Enumeration Speedup (9.2× on annulus-layer2)

**Problem:** Action enumeration was the primary training bottleneck, caused by per-vertex `np.allclose` loops.

**Profiling results (before optimization):**

| Function | Time (500 steps) | % of Total | Calls |
|----------|-------------------|-----------|-------|
| `np.allclose` | 22.7s | 67% | 493,187 |
| `_update_boundary` | 14.0s | 41% | 500 |
| `_get_vertices_by_angle` | 6.4s | 19% | 508 |
| `compute_min_boundary_angle` | 1.9s | 6% | 500 |
| Total training | 34.0s | 100% | - |

**Changes made (all in `src/MeshEnvironment.py`):**
1. Added `_find_vertex_index()`: Vectorized vertex lookup using `np.sum((boundary - vertex)**2, axis=1)` instead of `allclose` loop
2. Added `_find_vertex_indices()`: Batch vertex lookup using distance matrix `(V, B)` for `_update_boundary`
3. Vectorized `_update_boundary`: Replaced O(n×4) allclose loop with single matrix distance computation
4. Vectorized `_get_vertices_by_angle`: Replaced per-vertex angle loop with array operations
5. Vectorized `compute_min_boundary_angle`: Replaced per-vertex loop with array operations
6. Replaced all remaining `allclose` calls in `_get_state`, `_enumerate_for_vertex`, `_calculate_fan_shape_radius`, `_get_fan_shape_points`, `_form_element`
7. Added `_cached_ref_idx` alongside `_cached_ref_vertex` to avoid redundant lookups

**Benchmark results:**

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Annulus-layer2 (steps/min) | ~1,000 | ~9,248 | **9.2×** |
| Star (steps/min) | ~5,000+ | ~11,793 | ~2.4× |
| `allclose` calls (500 steps) | 493,187 | 0 | **Eliminated** |
| Function calls (500 steps) | 37.3M | 8.9M | 4.2× |
| Training time (500 annulus steps) | 34.0s | 4.3s | 7.9× |

**Tests:** 21/21 passing. Star 12×4 checkpoint produces identical results (q=0.371, 5Q, 100%).

---

### WS3: Curriculum Learning + Annulus Diagnosis

#### Curriculum Training Results

**Step 1: Octagon from scratch (8k steps)**
- EVAL at 4k: q=0.478, 3Q, 100% (already converged)
- EVAL at 8k: q=0.478, 3Q, 100%
- Saved to `checkpoints/curriculum_octagon/`

**Step 2: Load octagon → train on star (15k steps)**

| Eval Step | Curriculum MeanQ | From-Scratch MeanQ | Curriculum Elements | From-Scratch Elements |
|-----------|-----------------|-------------------|--------------------|--------------------|
| 5k | **0.392** | 0.275 | 6 | 5 |
| 10k | 0.364 | 0.364 | 5 | 5 |
| 15k | 0.364 | 0.371 | 5 | 5 |

**Result:** Curriculum provides faster convergence (0.392 vs 0.275 at 5k) but reaches a similar ceiling (0.364 vs 0.371). The slight quality gap is within noise.

**Step 3: Load star curriculum → train on annulus-layer2 (20k steps, killed at 10k)**

| Eval Step | Return | Completion | MeanQ | Elements |
|-----------|--------|------------|-------|----------|
| 5k | -9.05 | 0% | 0.356 | 70.0 |
| 10k | -9.05 | 0% | 0.356 | 70.0 |

**Result: FAIL.** 0% completion, identical to from-scratch session 6 result. Curriculum learning does NOT help annulus-layer2.

#### Annulus-Layer2 Root Cause Diagnosis

Ran a diagnostic rollout logging per-step state, valid actions, and boundary size:

**Critical findings:**
1. **Only 1 valid action per step:** Type-0 only (connect adjacent boundary vertices). All 48 type-1 actions (interior vertex placement) fail validation — candidates are either outside the domain or produce self-intersecting quads.
2. **Boundary grows instead of shrinking:** Started at 64 vertices, grew to 344 after 70 type-0 steps. Type-0 consumes 2 boundary vertices but adds replacement vertices from the element's non-boundary edges, causing net growth on this geometry.
3. **State is invariant:** The first 5 observation components (0.51, 0.55, 0.00, 0.57, 0.77) were identical across all 70 steps. The 44-dim state representation captures no meaningful variation as the boundary grows.
4. **The agent has zero choice:** With only 1 valid action per step, the DQN cannot learn anything — the policy is deterministic regardless of weights.

**Diagnosis:** The annulus-layer2 failure is an **action space coverage problem**, not a feature quality, training speed, or curriculum problem. The 12×4 grid cannot place valid interior vertices on the narrow, curved annulus geometry. This makes the domain impossible for the current architecture.

---

### WS4: 24×4 Star Ablation to Convergence (NEW BEST q=0.405)

**Setup:** 30k steps, 24×4 grid (97 actions), eval every 5k.

**Results:**

| Eval Step | Return | MeanQ | Elements | Completion | Epsilon |
|-----------|--------|-------|----------|------------|---------|
| 5k | 7.18 | 0.223 | 4 | 100% | ~0.76 |
| 10k | 7.12 | 0.355 | 9 | 100% | ~0.52 |
| 15k | 9.23 | 0.398 | 6 | 100% | ~0.29 |
| 20k | 9.28 | 0.402 | 6 | 100% | ~0.05 |
| 25k | 9.05 | 0.388 | 6 | 100% | 0.05 |
| **30k** | **9.33** | **0.405** | **6** | **100%** | **0.05** |

**Comparison with 12×4:**

| Metric | 12×4 (15k) | 24×4 (30k) | Delta |
|--------|-----------|-----------|-------|
| Quality | 0.371 | **0.405** | **+9.2%** |
| Elements | 5 | 6 | +1 |
| Training time | ~15k steps | ~30k steps | 2× |
| Ceiling | ~0.44 | ~0.44 | Same |

**Result: PARTIAL PASS.** 24×4 achieves q=0.405 — better than 12×4's 0.371, confirming angular resolution improves quality. However, 0.405 is still below the 0.44 ceiling and uses 6 elements vs 5. The extra resolution helps placement precision but doesn't change the fundamental geometry ceiling.

---

## Key Metrics Comparison

| Metric | Session 6 | Session 7 | Change |
|--------|-----------|-----------|--------|
| Star quality (12×4) | 0.371 | 0.371 | Same |
| Star quality (24×4) | 0.308 (10k, undertrained) | **0.405** (30k, converged) | +31% |
| Star elements (24×4) | 6 (10k) | 6 (converged) | Same |
| Annulus-layer2 completion | 0% | 0% (curriculum also fails) | Same |
| Annulus-layer2 diagnosis | "too hard" | **Root cause identified** | New |
| Training speed (annulus) | ~1k steps/min | **~9.2k steps/min** | **9.2×** |
| Zero-shot transfer | Not measured | **100% all domains** | New |
| Tests passing | 21 | 21 | Stable |

## What Didn't Work

### Curriculum Learning for Annulus-Layer2
The curriculum (octagon→star→annulus-layer2) completely failed — 0% completion at 10k steps, identical to from-scratch training. The root cause is that the 12×4 action grid has zero valid type-1 actions on annulus geometry, making the agent's policy irrelevant. No amount of pre-training or training budget can overcome having only 1 valid action per step.

## What Went Well

- **9.2× speedup** through vectorized geometry — unblocked both 24×4 convergence and fast annulus experiments
- **Zero-shot transfer** was a surprising result that validates the architecture for similar-sized domains
- **Clear root cause** for annulus failure — actionable for architectural planning
- **24×4 ablation conclusive** — angular resolution helps (0.405 vs 0.371), answering an open question from sessions 5-6
- **Killed runs early** — annulus curriculum killed at 10k (0% completion), saving time for 24×4

## What Didn't Go Well

- Curriculum for annulus-layer2 was doomed by the action space constraint. The WS1 transfer diagnostic (positive on 6-20v domains) was misleading about whether curriculum would help on 64v — the bottleneck wasn't features but action space coverage.

## Observations

1. **The architecture generalizes well within its vertex range (6-20v).** Zero-shot transfer to octagon, l-shape, and rectangle all succeed. The 44-dim state and learned policy capture domain-general meshing strategies.

2. **Annulus-layer2 is fundamentally action-space-limited.** Only type-0 actions are valid, boundary grows instead of shrinking, and the agent has zero choice. This is not solvable with better training, curriculum, or reward tuning — it requires architectural changes:
   - Finer grid (higher n_angle, n_dist) to reach valid interior points in narrow geometry
   - Adaptive grid that respects local boundary curvature
   - Continuous action space (revisit SAC) with learned placement
   - Hierarchical decomposition into sub-domains

3. **24×4 angular resolution provides diminishing returns.** 0.405 vs 0.371 (+9%) at 2× training cost. Still below the 0.44 geometry ceiling. Going to 48×8 would be expensive and likely yield marginal additional improvement.

4. **Curriculum accelerates early training but doesn't raise the ceiling.** Star curriculum reached 0.392 at 5k (vs 0.275 from-scratch) but converged to 0.364 (vs 0.371). The features transfer but the final quality is geometry-limited.

5. **The speedup changes what's practical.** At 9.2k steps/min, a 100k-step annulus run takes ~11 minutes instead of ~100 minutes. This makes experimentation with larger grid sizes feasible.

## Files Changed

| File | Changes |
|------|---------|
| `src/MeshEnvironment.py` | Added `_find_vertex_index()`, `_find_vertex_indices()` vectorized helpers. Replaced all `np.allclose` loops with vectorized distance computations. Vectorized `_get_vertices_by_angle()` and `compute_min_boundary_angle()`. Added `_cached_ref_idx`. |
| `checkpoints/curriculum_octagon/` | New: octagon curriculum model (8k steps) |
| `checkpoints/curriculum_star/` | New: star curriculum model (octagon→star, 15k steps) |
| `checkpoints/curriculum_annulus/` | New: annulus curriculum model (star→annulus, 10k steps, 0% completion) |
| `checkpoints/star-24x4-s7/` | New: star 24×4 converged model (30k steps, q=0.405) |
| `output/latest/star_24x4.png` | New: 24×4 star visualization |
| `output/latest/star_curriculum.png` | New: curriculum star visualization |
| `output/latest/octagon.png` | Updated: curriculum octagon visualization |
| `output/latest/star.png` | Updated: 12×4 star visualization |

## Short-term Next Steps (Session 8)

1. **Finer grid for annulus-layer2:** Test 24×8 or 48×8 grid to see if more angle/distance bins produce valid type-1 actions on annulus geometry. With the 9× speedup, this is now feasible. If even high-resolution grids produce zero valid type-1 actions, the advancing-front approach itself may be unsuited for this geometry.

2. **Multi-domain training:** Train a single agent on octagon+star+l-shape+rectangle simultaneously. Zero-shot transfer shows the architecture generalizes — multi-domain training should produce a robust generalist agent.

3. **Quality improvement on octagon:** Currently 0.478 vs 0.61 ceiling. With 24×4 showing +9% on star, try 24×4 on octagon to approach the ceiling.

## Medium-term Next Steps (Sessions 9-10)

4. **Continuous action space (revisit SAC):** The annulus failure is fundamentally about discrete grid coverage. SAC with continuous placement could reach valid interior points that no grid captures. The improved reward structure (quality-gated completion) should help SAC learn better than early sessions.

5. **Variable-size state representation:** GNN or attention over boundary graph to handle variable vertex counts (4-64+). This is the prerequisite for any domain >20 vertices working well.

6. **Adaptive action grid:** Grid resolution proportional to local boundary curvature — dense where the boundary curves sharply, sparse where it's straight.

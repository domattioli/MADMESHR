# Session Report: Training Experiments & Bug Fixes

## Summary

Starting from a codebase where the DQN trivially solved all domains using type-0 actions only (existing boundary vertices), we discovered and fixed three critical bugs that were preventing type-1 actions (placing new interior vertices) from working. After fixes, the DQN successfully learns to mesh a 10-vertex star domain — a geometry where type-0 is completely unavailable — achieving **100% completion rate** with all-quad meshes.

## What Went Well

### 1. Pipeline Validation (Stages 1-3)
- 4-vertex square: 100% completion, 1 step, return=10.0
- 8-vertex octagon: 100% completion, 2 steps, return=10.43
- 16-vertex circle: 100% completion, 6 steps, return=10.78
- All with stable Q-values and converged loss

### 2. Bug Discovery & Fixes

**Bug 1: Fan sweep direction inverted**
- Type-1 candidate vertices were placed *outside* the domain (exterior arc instead of interior)
- Root cause: cross-product winding logic in `_get_fan_shape_points` and `_enumerate_for_vertex` was wrong
- Fix: Simplified to always ensure `left_angle > right_angle`, sweep `linspace(right_angle, left_angle)`
- Impact: On 16-vertex circle, went from 0 valid type-1 actions to 46

**Bug 2: No inside-domain check for type-1 vertices**
- Even after fix #1, some candidates at large fan radius landed outside the boundary
- Fix: Added `_point_in_polygon` ray-casting check in `_form_element`

**Bug 3: enumerate/step ref_vertex mismatch (CRITICAL)**
- `enumerate_valid_actions` has fallback logic that tries different reference vertices when the min-angle vertex has no valid actions
- `DiscreteActionEnv.step()` independently called `_select_reference_vertex()` which could return a *different* vertex
- Pre-validated actions were executed against the wrong reference vertex, causing 90%+ failure rate
- Fix: Store `_cached_ref_vertex` during enumeration, reuse in step()

### 3. Performance Optimizations
- Vectorized `enumerate_valid_actions` with batch point-in-polygon, batch quad validation, and vectorized trig
- `_form_element_fast` accepts pre-computed ref_idx to avoid redundant boundary scans

### 4. Star Domain Training Success
- 10-vertex star (alternating radius 1.0/0.4) — type-0 unavailable at start
- **Final eval: Return=11.40, Completion=100% (50 episodes)**
- 8/10 eval checkpoints at 100% completion
- Q-values stable at ~5.3, loss 0.3-1.1
- Greedy policy produces 5 quads, 0 triangles

## What Didn't Work

### 1. Sparse Reward (+10 terminal only)
- On hard domains, random exploration rarely completed → no reward signal
- Q-values diverged (reaching 15+ with loss spiking to 39)
- The agent learned nothing useful

### 2. Dense Reward Without Ref Vertex Fix
- Before bug #3 was fixed, dense reward led to "reward farming" — agent placed many elements (30+) to accumulate intermediate rewards without ever completing
- Returns of 10+ from farming looked like success but masked 0% completion

### 3. High Gamma (0.99)
- With sparse rewards and 10-20 step episodes, gamma=0.99 caused Q-value overestimation
- Switching to gamma=0.95 helped stability

## Reward Structure (Final)

| Outcome | Reward Formula | Typical Value |
|---------|---------------|---------------|
| All-quad completion (best) | `5.0 + 10.0 * mean_quality` | ~12-15 |
| Boundary triangle | `2.0 + 4.0 * mean_quality` | ~4-5 |
| Self-intersecting → 2 triangles | `0.5 + 2.0 * mean_quality` | ~1.5-2 |
| Per-step (intermediate) | `quality + 2.0 * area_consumed - 0.01` | ~0.3-0.5 |
| Invalid action | `-0.1` | -0.1 |
| No valid actions (truncation) | `-2.0` | -2.0 |

## What to Investigate Next

### 1. Element Quality
- Current mean quality is 0.37 on the star — below the 0.7 target
- The agent optimizes for completion speed (5 elements) over quality
- Need: increase quality weight in intermediate reward, or add quality-gated completion bonus

### 2. Harder Domains
- Elongated rectangle (20 vertices, 4:1 aspect ratio) — type-0 unavailable, 31 type-1 actions
- L-shaped domain with concave vertex
- Grid with square hole (requires multi-boundary support or slit encoding)

### 3. SAC Migration
- The motivating paper (Pan et al.) used SAC, not DQN
- Switching back is feasible: the key fixes (fan direction, point-in-polygon, ref_vertex caching) are in `MeshEnvironment` and `DiscreteActionEnv`, not the agent
- Would need: continuous action space wrapper that maps SAC's continuous output to the discrete action grid
- Alternatively: keep discrete actions but use SAC with Gumbel-Softmax or categorical policy

### 4. Triangle Handling
- Currently triangles are allowed as fallback final elements
- Interior triangles (surrounded by quads) should be heavily penalized — they break the broader algorithm
- Boundary triangles (sharing edge with domain boundary) are acceptable but not ideal

### 5. Curriculum Learning
- Train on easy → hard domains progressively
- Could significantly speed up learning on complex geometries
- Natural progression: square → octagon → circle → star → L-shape → rectangle → hole domains

## GPU Usage Note
The RTX 3060 GPU provides minimal benefit for this workload. The DQN network is tiny (~100k params), batch sizes are small (64), and the bottleneck is CPU-bound geometry computation in `enumerate_valid_actions`. GPU would help with larger networks or vectorized environments.

## Files Changed
- `src/MeshEnvironment.py` — fan direction fix, point-in-polygon, ref_vertex caching, vectorized enumeration, _has_self_intersection
- `src/DiscreteActionEnv.py` — cached ref_vertex usage, dense reward, tiered completion reward, triangle/degenerate quad support
- `tests/test_discrete_env.py` — updated completion reward threshold
- `test_domains.py` — 4 test domain geometries (new file)
- `star_mesh_result.png` — visualization of DQN mesh result (new file)

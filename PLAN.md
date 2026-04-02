# MADMESHR: Unified Development Plan

## Status Summary

The MADMESHR quad mesh generator has a working advancing-front environment (`MeshEnvironment`) but the SAC RL agent fails to learn. After 19k steps, eval returns are stuck at -14 to -20. Two root causes are identified:

1. **Unlearnable action space**: Continuous `[-1,1]^3` where ~33% maps to unimplemented type-2 (always fails), type-0 ignores 2 of 3 dims, valid actions are ~1% of the space.
2. **Broken `_update_boundary()`**: After placing a quad, boundary vertices are duplicated and reordered incorrectly, corrupting all subsequent steps. This blocks ALL multi-step episodes.

The fix is a three-track approach: (A) fix critical mesh infrastructure bugs, (B) build a comprehensive test suite, (C) reformulate the action space and switch from SAC to DQN.

## Dependency Graph

```
Phase 0 (tests + bugfixes)
├─► Phase 1 (action enumeration)
│   └─► Phase 2 (discrete wrapper)
│       └─► Phase 3 (DQN agent)
│           └─► Phase 4 (trainer + CLI)
│               └─► Training Stages 1-4
```

**Phase 0 must come first** — the test suite catches existing bugs and prevents regressions.

---

## Phase 0: Test Suite + Critical Bugfixes

### Phase 0A: Test Infrastructure

Create `tests/` directory with pytest framework. **174 tests** across 14 files:

| File | Tests | Priority |
|------|-------|----------|
| `tests/conftest.py` | — | Fixtures: square, triangle, pentagon, hexagon, octagon, circle-16/32, L-shape |
| `tests/test_boundary_update.py` | 17 | **CRITICAL** — validates the most bug-prone code |
| `tests/test_form_element.py` | 15 | **CRITICAL** — element formation + action type mapping |
| `tests/test_geometry.py` | 28 | Tier 2 — polygon area, segment intersection, convexity |
| `tests/test_element_quality.py` | 12 | Tier 2 — quality metric correctness + invariance |
| `tests/test_reward.py` | 11 | Tier 2 — reward function (range is [-1.2, 0.8], not [-1,1]) |
| `tests/test_state.py` | 13 | Tier 2 — 22-float state vector, truncation/padding |
| `tests/test_reference_vertex.py` | 8 | Tier 2 — min-angle vertex selection |
| `tests/test_fan_shape.py` | 10 | Tier 2 — fan-shaped area points |
| `tests/test_environment.py` | 20 | Tier 3 — gym env integration, full episodes |
| `tests/test_domains.py` | 10 | Tier 3 — domain-specific mesh generation |
| `tests/test_replay_buffer.py` | 8 | Tier 3 — replay buffer FIFO, capacity |
| `tests/test_sac.py` | 8 | Tier 3 — SAC agent forward/backward pass |
| `tests/test_numerical.py` | 14 | Tier 3 — numerical stability, NaN/inf guards |

#### Critical Test Cases (Tier 1)

**Boundary update (`_update_boundary`, lines 514-558):**
- `test_boundary_update_float_equality` — type-1 step with trig-computed vertex; float list `==` may fail
- `test_boundary_update_multi_edge_removal` — type-0 on pentagon removes 2+ adjacent edges; pop-index shift may remove wrong vertex
- `test_boundary_vertex_ordering` — vertices in correct cyclic order after update
- `test_boundary_update_wraparound_edges` — element edges cross the 0→N-1 boundary index
- `test_boundary_area_conservation` — element_area + remaining ≈ original (catches silent corruption)

**Element formation (`_form_element`, lines 408-449):**
- `test_action_type_mapping_boundaries` — `int((action[0]+1)*1.5)`: verify exact boundaries at -1, -0.34, -0.33, 0.33, 0.34 → types 0, 0, 1, 1, 2

**Known bugs these tests will expose:**

| Test | Expected Bug |
|------|-------------|
| `test_quality_NOT_cyclic_invariant` | Quality changes when vertex order rotates (diags use fixed indices 0,2 and 1,3) |
| `test_completion_on_degenerate_4vert_boundary` | +10 reward even if 4-vert boundary is self-intersecting |
| `test_reward_completion_ignores_quality` | Flat +10 regardless of element quality |
| `test_boundary_update_float_equality` | Float `==` fails for trig-computed vertices |
| `test_boundary_update_multi_edge_removal` | Pop-index shift removes wrong vertex |

### Phase 0B: Fix `_update_boundary()` (BLOCKER)

**Severity**: Critical — prevents ALL multi-step episodes.

**Evidence** (from `implement-plan-md-zipMz` branch): After placing a type-0 quad on a 16-vertex circle, boundary stays at 16 vertices (should become 14). Vertices are duplicated and reordered.

**Root cause**: Edge-matching logic uses list-based comparison with sorting (lines 514-558):
1. Fails to remove consumed vertices (only removes one endpoint per matched edge)
2. Inserts new edges at wrong positions  
3. Creates duplicate vertices

**Fix**: Rewrite `_update_boundary()` with proper advancing-front update:
- Identify which boundary vertices become interior (fully surrounded by elements)
- Replace consumed boundary segment with new exterior edges of the element
- Maintain consistent winding order
- Use `np.allclose()` instead of list `==` for vertex/edge matching

**Verification**:
- Type-0 on 16-vert circle → boundary has 14 vertices
- Type-1 adding one vertex → boundary has N-1+1 vertices
- All 17 `test_boundary_update.py` tests pass

### Phase 0C: Fix Completion Check

Line 102: `len(self.boundary) <= 4 and self._is_quadrilateral(self.boundary)` — `_is_quadrilateral` only checks `len == 4`, so a self-intersecting 4-vertex boundary gets +10 reward.

**Fix**: Change completion to `len(self.boundary) == 4 and self._is_valid_quad(self.boundary)`

---

## Phase 1: Action Enumeration

**File**: `src/MeshEnvironment.py` (modify)

Add `enumerate_valid_actions(n_angle=12, n_dist=4)` method:
- **Type-0**: Check the single deterministic action (ref, ref+1, ref+2, ref-1). 0 or 1 result.
- **Type-1**: Grid over `n_angle` x `n_dist` bins within the fan-shaped interior angle (not full 2π).
- Returns: `list[(action_type, new_vertex_or_None)]` + boolean mask of size `1 + n_angle*n_dist = 49`
- Cache result (invalidated on `step()` or `reset()`)
- **Zero valid actions fallback**: Try next-smallest-angle vertex. If none work, return all-False mask.

**Verify**: 4-vertex square → 1 valid action (type-0). 8-vertex octagon → multiple. Zero-valid path tested.

Add `_get_enriched_state()` for Phase 2:
- Existing 22 floats + boundary vertex count (1) + elements placed (1) + num valid actions (1) + 8 boundary samples (16) + action-type availability (2) = **44 floats**

**New tests**: `tests/test_discrete_env.py`
- 4-vertex square: exactly 1 valid action
- 16-vertex circle: > 0 valid actions per step
- Zero-valid fallback with contrived degenerate boundary
- Enriched state shape == 44

## Phase 2: Discrete Action Wrapper

**File**: `src/DiscreteActionEnv.py` (create)

Gymnasium wrapper:
- `action_space = spaces.Discrete(49)`
- On `reset()`: enumerate valid actions, build mask
- On `step(action_index)`: look up pre-computed action, delegate, re-enumerate
- Mask in `info["action_mask"]` (not in observation — Huang & Ontanon 2020)
- No valid actions → truncated, penalty = -5.0

**Verify**: Random agent achieves 100% valid action rate. Mask correctly in info dict.

## Phase 3: DQN Agent

**File**: `src/DQN.py` (create)

Dueling Double DQN with post-hoc action masking:
- **Architecture**: `44 → 256 → 256 → 128 (ReLU)`, split to value `128 → 1` and advantage `128 → 49`. `Q = V + (A - mean_valid(A))`.
- **Post-hoc masking**: `Q[invalid] = -inf` using mask from info
- **Target network**: Polyak averaging, tau=0.005
- **Double DQN**: action selection from online net, value from target net (both masked)
- **Epsilon-greedy**: 1.0 → 0.05 over first 50% of training, among valid actions only
- **Replay buffer**: `MaskedReplayBuffer` storing `(state, action, reward, next_state, next_mask, done)`

```python
class DQN:
    def select_action(self, state, valid_mask, evaluate=False) -> int
    def train_step(self, batch) -> dict  # {'loss': float}
```

## Phase 4: DQN Trainer + CLI

**File**: `src/trainer_dqn.py` (create)

Based on existing `trainer.py`: Uses `DiscreteActionEnv`, passes mask, epsilon decay, logs loss/Q-value/return/elements/epsilon/completion rate.

**File**: `main.py` (modify)
- Add `--algorithm {dqn,sac}` flag (default: `dqn`)

---

## Training Stages (After All Phases Complete)

### Stage 1: 4-vertex square (trivial smoke test)
- **Goal**: Agent always picks type-0, completes in 1 step
- **Config**: 1,000 timesteps, epsilon 1.0→0.05 over 500
- **Pass**: Return > 10 within 200 episodes

### Stage 2: 8-vertex octagon (easy)
- **Goal**: Complete mesh in ~2-4 steps
- **Config**: 10,000 timesteps, batch=64, eval/2,000
- **Pass**: Completion rate > 80% by 8k steps

### Stage 3: 16-vertex circle (target)
- **Goal**: Mesh circle to completion
- **Config**: 50,000 timesteps, batch=64, eval/5,000
- **Pass**: Completion > 50% by 30k steps, beats type-0-only baseline

### Stage 4: Quality optimization
- **Goal**: High-quality meshes
- **Config**: Tuned rewards: completion = `2.0 + 8.0 * mean(qualities)`, step penalty = -0.01
- **Pass**: Mean element quality > 0.7

### Hyperparameters

| Parameter | Stage 1 | Stage 2 | Stage 3 | Stage 4 |
|-----------|---------|---------|---------|---------|
| Timesteps | 1,000 | 10,000 | 50,000 | 100,000 |
| Batch size | 32 | 64 | 64 | 64 |
| Buffer capacity | 10,000 | 50,000 | 100,000 | 100,000 |
| Epsilon start/end | 1.0/0.05 | 1.0/0.05 | 1.0/0.05 | 1.0/0.02 |
| Epsilon decay | 50% | 50% | 50% | 40% |
| Learning rate | 3e-4 | 3e-4 | 3e-4 | 1e-4 |
| Max ep length | 10 | 20 | 50 | 50 |
| Random steps | 100 | 1,000 | 2,000 | 2,000 |

### Red Flags (stop and diagnose)
- Q-values diverging (> 100 or < -100)
- Loss increasing after initial decrease
- Completion rate stuck at 0% after 50% of training
- All episodes hitting max_ep_len

---

## MVP Scope: All Files

| File | Action | Phase |
|------|--------|-------|
| `tests/conftest.py` | Create: shared fixtures | 0A |
| `tests/test_boundary_update.py` | Create: 17 boundary topology tests | 0A |
| `tests/test_form_element.py` | Create: 15 element formation tests | 0A |
| `tests/test_geometry.py` | Create: 28 geometry tests | 0A |
| `tests/test_element_quality.py` | Create: 12 quality metric tests | 0A |
| `tests/test_reward.py` | Create: 11 reward function tests | 0A |
| `tests/test_state.py` | Create: 13 state representation tests | 0A |
| `tests/test_reference_vertex.py` | Create: 8 reference vertex tests | 0A |
| `tests/test_fan_shape.py` | Create: 10 fan shape tests | 0A |
| `tests/test_environment.py` | Create: 20 gym integration tests | 0A |
| `tests/test_domains.py` | Create: 10 domain-specific tests | 0A |
| `tests/test_replay_buffer.py` | Create: 8 replay buffer tests | 0A |
| `tests/test_sac.py` | Create: 8 SAC agent tests | 0A |
| `tests/test_numerical.py` | Create: 14 numerical stability tests | 0A |
| `src/MeshEnvironment.py` | Fix: `_update_boundary()`, completion check; Add: `enumerate_valid_actions()`, `_get_enriched_state()` | 0B, 0C, 1 |
| `src/DiscreteActionEnv.py` | Create: Gym wrapper with discrete actions + mask | 2 |
| `src/DQN.py` | Create: Dueling Double DQN | 3 |
| `src/trainer_dqn.py` | Create: DQN training loop | 4 |
| `tests/test_discrete_env.py` | Create: action enum + wrapper tests | 1-2 |
| `main.py` | Modify: `--algorithm {dqn,sac}` flag | 4 |
| `src/SAC.py` | No change | — |
| `src/trainer.py` | No change | — |

## Deferred Work (post-MVP)
- Type-2 actions (adds another angle*dist block)
- Curriculum learning (automatic stage progression)
- N-step returns (n=3 or n=5)
- Prioritized experience replay
- Finer discretization (24x8=192 if 12x4 too coarse)

## Performance Notes
- Action enumeration: 49 calls to `_is_valid_quad` per step — cheaper than one DQN forward pass
- DQN training faster than SAC (1 Q-network + 1 target vs. actor + 2 critics + 2 targets)
- Episodes shrink from ~200 steps (99% invalid) to ~6-16 steps (100% valid)

# MADMESHR: RL Reformulation Plan

## Context

The MADMESHR quad mesh generator has a working advancing-front environment (`MeshEnvironment`) but the SAC RL agent fails to learn. After 19k steps, eval returns are stuck at -14 to -20. The root cause is a **fundamentally unlearnable action space**: continuous `[-1,1]^3` where ~33% maps to unimplemented type-2 (always fails), type-0 actions ignore 2 of 3 dimensions entirely, and valid actions are ~1% of the space. The fix is to reformulate the problem with discrete, pre-computed valid actions and switch from SAC to DQN.

## Implementation Phases

### Phase 1: Action Enumeration (MeshEnvironment addition)
**File**: `src/MeshEnvironment.py` (modify)

Add `enumerate_valid_actions(n_angle=12, n_dist=4)` method:
- Reuses existing `_select_reference_vertex()`, `_calculate_fan_shape_radius()`, `_form_element()`, `_is_valid_quad()`
- **Type-0**: Check the single deterministic type-0 action (ref, ref+1, ref+2, ref-1). 0 or 1 result.
- **Type-1**: Grid over `n_angle` angle bins x `n_dist` distance bins within the **fan-shaped interior angle** (not full 2π). The angular range spans from the direction toward ref-1 to the direction toward ref+1, computed the same way `_get_fan_shape_points()` does at lines 415-420. This ensures all candidate vertices point into the unmeshed domain interior.
- Returns: list of `(action_type, new_vertex_or_None)` tuples + boolean mask of size `1 + n_angle*n_dist = 49`
- Cache result between calls (invalidated on `step()` or `reset()`)
- **Type-2 (deferred)**: Not included in MVP. Type-0 + type-1 are sufficient for circle/L-shape domains.
- **Zero valid actions fallback**: If the min-angle reference vertex has no valid actions, try the next-smallest-angle vertex, and so on. If no vertex yields valid actions, return an all-False mask (wrapper will terminate the episode).

**Verify**: Unit test on 4-vertex square returns exactly 1 valid action (type-0). 8-vertex octagon returns multiple. Zero-valid-actions path tested with a contrived degenerate boundary.

### Phase 2: Discrete Action Wrapper
**File**: `src/DiscreteActionEnv.py` (create)

Gymnasium wrapper around `MeshEnvironment`:
- `action_space = spaces.Discrete(max_actions)` where `max_actions = 49`
- On `reset()`: enumerate valid actions, build mask
- On `step(action_index)`: look up pre-computed action, delegate to underlying env, re-enumerate for next state
- **Observation**: enriched state only (44 floats). The valid-action mask is stored separately as `info["action_mask"]` and passed to the agent outside the observation. This follows the standard masked-DQN pattern (Huang & Ontanon 2020) — the network learns state value from state features only; the mask is applied post-hoc to Q-values.
- If no valid actions exist (all-False mask): episode terminates (truncated, penalty = -5.0)

**Enriched state** (added via `_get_enriched_state()` in MeshEnvironment):
- Existing 22 floats (neighbors + fan points + area ratio)
- Boundary vertex count (1, normalized by initial count)
- Elements placed (1, normalized)
- Num valid actions (1, normalized by max_actions)
- Boundary shape: 8 evenly-spaced boundary samples as (x, y) = 16 floats
- Action-type availability: 2 booleans (is type-0 valid, is any type-1 valid)
- Total: **44 floats**

### Phase 3: DQN Agent
**File**: `src/DQN.py` (create)

Dueling Double DQN with action masking:
- **Architecture**: shared trunk `44 -> 256 -> 256 -> 128 (ReLU)`, then split into value stream `128 -> 1` and advantage stream `128 -> 49`. Q = V + (A - mean_valid(A)).
- **Action masking applied post-hoc**: Network receives only the 44-float state. After computing Q-values for all 49 actions, set `Q[invalid] = -inf` using the mask from `info["action_mask"]`. This prevents the shortcut of learning "pick first valid action" from mask features.
- **Target network**: Polyak averaging, tau=0.005
- **Double DQN**: action selection from online net, value from target net (both masked)
- **Epsilon-greedy**: 1.0 -> 0.05 over first 50% of training, among valid actions only
- **Replay buffer**: `MaskedReplayBuffer` extending existing `ReplayBuffer` from `SAC.py` to store `(state, action, reward, next_state, next_mask, done)` — need next_mask for target Q computation

Interface:
```python
class DQN:
    def select_action(self, state, valid_mask, evaluate=False) -> int
    def train_step(self, batch) -> dict  # {'loss': float}
```

### Phase 4: DQN Trainer + CLI
**File**: `src/trainer_dqn.py` (create)

Based on existing `trainer.py` structure:
- Uses `DiscreteActionEnv` wrapper
- Passes valid mask (from `info["action_mask"]`) to `agent.select_action(state, mask)`
- Stores 6-tuples in `MaskedReplayBuffer`
- Epsilon linear decay schedule
- Logs: loss, mean Q-value, episode return, elements placed, epsilon, completion rate
- Eval: deterministic (epsilon=0), reports completion rate + mean return

**File**: `main.py` (modify)
- Add `--algorithm {dqn,sac}` flag (default: `dqn`)
- When `dqn`: instantiate DiscreteActionEnv, DQN, DQNTrainer
- When `sac`: existing code path, unchanged

### Deferred Work (post-MVP validation)
- **Type-2 actions**: Extends naturally — add another angle*dist block to the action space
- **Reward tuning**: Scale completion bonus to `2.0 + 8.0 * mean(element_qualities)` so quality throughout the episode matters (currently +10 dominates 6-step episodes). Add `-0.01` step penalty for efficiency.
- **Curriculum learning**: 4-vertex square → 8-vertex → 16-vertex circle
- **N-step returns**: n=3 or n=5 for faster credit propagation from terminal reward
- **Prioritized replay**: Emphasize rare successful completions
- **Finer discretization**: 24x8=192 if 12x4 proves too coarse for quality

## MVP Scope: Files Modified/Created

| File | Action | Phase |
|------|--------|-------|
| `src/MeshEnvironment.py` | Modify: add `enumerate_valid_actions()`, `_get_enriched_state()` | 1, 2 |
| `src/DiscreteActionEnv.py` | Create: Gym wrapper with discrete actions + mask in info | 2 |
| `src/DQN.py` | Create: Dueling Double DQN with post-hoc action masking | 3 |
| `src/trainer_dqn.py` | Create: DQN training loop with epsilon schedule | 4 |
| `main.py` | Modify: add `--algorithm dqn` flag, wire new pipeline | 4 |
| `tests/test_discrete_env.py` | Create: tests for action enum, wrapper, zero-valid fallback | 1-2 |
| `src/SAC.py` | No change | — |
| `src/trainer.py` | No change | — |
| `tests/test_domains.py` | No change (must still pass) | — |

## Design Decisions
- **Reference vertex**: Fixed (deterministic min-angle selection) with fallback to next-best vertex if no valid actions. Keeps action space small (49).
- **Action mask**: Separate from observation (in `info` dict). Network sees only 44-float state.
- **Angular grid**: Spans fan-shaped interior angle (not full 2π) to maximize useful coverage.
- **Type-2**: Deferred — type-0 + type-1 sufficient for target domains.
- **Scope**: MVP first (Phases 1-4), validate learning, then iterate.

## Dependency Order
```
Phase 1 (enumerate actions) → Phase 2 (discrete wrapper) → Phase 3 (DQN) → Phase 4 (trainer + CLI)
```

## Verification Plan

1. **After Phase 1**: `enumerate_valid_actions()` on 4-vertex square returns 1 valid action (type-0). On 16-vertex circle returns > 0 per step. Zero-valid fallback tested.
2. **After Phase 2**: Random agent with `DiscreteActionEnv` achieves 100% valid action rate. Existing `test_domains.py` still passes. Mask correctly passed via `info["action_mask"]`.
3. **After Phase 4**: Train on 4-vertex square for 1k steps — agent achieves return > 10 (completion). Train on 16-vertex circle for 50k steps — agent beats the type-0 greedy baseline (6 quads).

## Performance Notes
- Action enumeration: 49 calls to `_is_valid_quad` per step, O(49 * N_elements * 4). Fine for < 30 elements; cheaper than one DQN forward pass.
- DQN training faster than SAC (1 Q-network + 1 target vs. actor + 2 critics + 2 targets).
- Episodes shrink from ~200 steps (99% invalid) to ~6-16 steps (100% valid). Replay buffer fills with complete trajectories much faster.

# Training Plan: Getting DQN to Learn

## Current Status
- Phases 1-4 implemented (action enumeration, discrete wrapper, DQN agent, trainer)
- 17 unit tests passing
- **Cannot train yet** — critical bugs in mesh infrastructure prevent multi-step episodes

## Blockers (Must Fix Before Training)

### Blocker 1: `_update_boundary()` is broken
**Severity**: Critical — prevents ALL multi-step episodes from working
**Evidence**: After placing a type-0 quad on a 16-vertex circle, boundary stays at 16 vertices (should become 14). Vertices are duplicated and reordered incorrectly. Subsequent element placement fails because the boundary is corrupt.

**Root cause**: The edge-matching and insertion logic in `_update_boundary()` (lines 514-558) uses list-based edge comparison with sorting, which:
1. Fails to remove consumed vertices (only removes one endpoint per matched edge)
2. Inserts new edges at wrong positions
3. Creates duplicate vertices

**Fix**: Rewrite `_update_boundary()` with a proper advancing-front boundary update:
- Identify which boundary vertices become interior (fully surrounded by elements)
- Replace the consumed boundary segment with the new exterior edges of the element
- Maintain consistent winding order

**Verification**: After type-0 on 16-vertex circle, boundary should have 14 vertices. After type-1 adding one vertex, boundary should have N-1 vertices (one consumed vertex replaced by new vertex path).

### Blocker 2: Boundary-corrupted states cascade
**Evidence**: After blocker 1 corrupts the boundary, `_form_element()` returns False for actions that were pre-validated (because the boundary indices shift). Every subsequent step returns -0.1.

**Fix**: This resolves automatically when blocker 1 is fixed.

## Training Stages (After Blockers Fixed)

### Stage 1: Validate on 4-vertex square (trivial)
**Goal**: Agent learns to always pick type-0 (the only valid action) and complete in 1 step.
**Config**: 1,000 timesteps, epsilon 1.0→0.05 over 500 steps
**Expected**: Return > 10 (completion reward) within 200 episodes
**Why this matters**: Smoke test that the full pipeline works end-to-end

### Stage 2: 8-vertex octagon (easy)
**Goal**: Agent learns to complete the mesh in ~2-4 steps
**Config**: 10,000 timesteps, batch_size=64, eval every 2,000
**Expected**: Completion rate > 80% by 8k steps
**Why this matters**: Tests that the agent can learn a short sequence of valid actions

### Stage 3: 16-vertex circle (target)
**Goal**: Agent learns to mesh a circle domain to completion
**Config**: 50,000 timesteps, batch_size=64, eval every 5,000
**Expected**: Completion rate > 50% by 30k steps, beats type-0-only greedy baseline (6 quads)
**Baseline**: Type-0-only policy — always pick action 0. Measure how many elements it places.

### Stage 4: Quality optimization
**Goal**: Not just complete, but produce high-quality meshes
**Config**: Same as Stage 3 but with tuned rewards:
- Completion bonus: `2.0 + 8.0 * mean(element_qualities)` (replaces flat +10)
- Step penalty: `-0.01` per step (encourages efficiency)
**Expected**: Mean element quality > 0.7

## Hyperparameters

| Parameter | Stage 1 | Stage 2 | Stage 3 | Stage 4 |
|-----------|---------|---------|---------|---------|
| Timesteps | 1,000 | 10,000 | 50,000 | 100,000 |
| Batch size | 32 | 64 | 64 | 64 |
| Buffer capacity | 10,000 | 50,000 | 100,000 | 100,000 |
| Epsilon start | 1.0 | 1.0 | 1.0 | 1.0 |
| Epsilon end | 0.05 | 0.05 | 0.05 | 0.02 |
| Epsilon decay | 50% | 50% | 50% | 40% |
| Learning rate | 3e-4 | 3e-4 | 3e-4 | 1e-4 |
| Gamma | 0.99 | 0.99 | 0.99 | 0.99 |
| Tau | 0.005 | 0.005 | 0.005 | 0.005 |
| Max ep length | 10 | 20 | 50 | 50 |
| Random steps | 100 | 1,000 | 2,000 | 2,000 |

## Monitoring & Success Criteria

Track per stage:
1. **Episode return** (moving average over 100 episodes)
2. **Completion rate** (fraction of eval episodes that terminate with done=True)
3. **Elements placed** (mean per episode — should increase as agent learns)
4. **Mean Q-value** (should stabilize, not diverge)
5. **Loss** (should decrease then stabilize)

**Red flags** (stop and diagnose):
- Q-values diverging (> 100 or < -100)
- Loss increasing after initial decrease
- Completion rate stuck at 0% after 50% of training
- All episodes hitting max_ep_len (agent not learning to complete)

## Future Improvements (Post-MVP)
- N-step returns (n=3) for faster credit propagation from terminal reward
- Prioritized experience replay (emphasize rare completions)
- Curriculum learning (automatic progression through stages)
- Finer action discretization (24x8=192) if quality is poor
- Type-2 actions for more complex domains

# Session 8 Plan: Annulus Feasibility + Distance-Range Fix + Quality Push

**Date:** Planned for next session after 2026-04-04
**Status:** Final (adversarial-reviewed, two rounds of critique incorporated)

## Context

Session 7 completed four workstreams with significant results:
1. **Zero-shot transfer:** Star 12×4 model achieves 100% completion on all other domains (6-20v) without retraining
2. **9.2× speedup:** Vectorized geometry eliminated 493k `np.allclose` calls, annulus-layer2 training now ~9.2k steps/min
3. **Curriculum learning:** Faster convergence (0.392 vs 0.275 at 5k on star) but same ceiling (0.364 vs 0.371). Failed completely on annulus-layer2 (0% completion)
4. **24×4 star converged:** q=0.405, 6Q — a 9% improvement over 12×4 (0.371, 5Q), confirming angular resolution helps modestly

**The annulus-layer2 problem (session 7 diagnosis):**
- Only 1 valid action per step (type-0 only — connect adjacent vertices)
- All 48 type-1 actions (interior vertex) fail: candidates outside domain or produce self-intersecting quads
- Boundary grows from 64→344 vertices over 70 steps instead of shrinking
- The agent has zero choice — policy is deterministic regardless of weights
- Curriculum, speedup, and training budget cannot overcome this — it's an action space coverage problem

## Adversarial Review Summary

### Round 1 Critiques (Incorporated)
- **Grid diagnostic too shallow:** Testing fresh boundary only misses the mid-rollout failure (boundary grows, type-1 viability collapses). Must test across multiple rollout steps.
- **No oracle/feasibility test:** We don't know if ANY action sequence can complete annulus-layer2. Need to prove solvability before training.
- **Multi-domain training is misallocated:** 90 minutes on generalist experiment while annulus unsolved for 4 sessions. Demote to stretch goal.
- **Boundary coarsening is incoherent:** Decimating 64v to 16v creates a different domain, doesn't solve the real problem.
- **Quality ceiling not challenged:** Worth understanding whether the 0.44/0.61/0.78 ceilings can be raised.

### Round 2 Critiques (Incorporated)
- **Distance range is the likely root cause:** `dist_bins = np.linspace(0.2, 1.0, n_dist) * radius` starts at 0.2×radius. If the valid annulus strip is < 0.2×radius wide, no grid resolution increase will help. Check this FIRST before grid sweeps.
- **Oracle must use boundary-reduction heuristic**, not quality-greedy. Quality-greedy selects type-1 actions that complicate the boundary.
- **30k training steps is grossly insufficient** for 1537-action space. Budget 100k minimum (~11 min at 9.2k steps/min).
- **Curriculum on star already disproven:** Converges to 0.364 (worse than from-scratch 0.371). Drop curriculum, train octagon 24×4 from scratch instead.
- **Plan is over-scoped:** WS3 should be explicitly stretch-only.

## Workstreams (3, strict priority order)

### WS1: Annulus-Layer2 Feasibility Analysis (Priority 1, ~60 min)

**Problem:** We don't know if annulus-layer2 is solvable in the current advancing-front formulation. Before any more training, we need to prove (or disprove) that a valid completion exists.

**Steps:**

1. **Distance-range diagnostic (10 min):** For 5 reference vertices on the initial annulus boundary, compute:
   - Fan-shape radius
   - Distance from reference vertex to the nearest valid interior point (inside the annulus strip)
   - Compare: if nearest_valid_distance / fan_radius < 0.2, the current distance bins cannot reach the valid region
   
   This is the likely root cause — the annulus strip is narrow relative to fan radius, so all candidates at 0.2-1.0× radius overshoot into the inner region or miss the strip entirely.

2. **Grid enumeration across rollout (20 min):** For grids 12×4, 24×8, 48×8, enumerate valid actions at steps 0, 5, 10, 20 during random type-0 rollouts. Also test a modified distance range: `np.linspace(0.05, 0.5, n_dist)` instead of `np.linspace(0.2, 1.0, n_dist)`.

3. **Oracle test (30 min, only if any grid produces type-1 actions):**
   - Greedy-by-boundary-reduction: at each step, pick the action that minimizes boundary vertex count (prefer type-0 that removes 2 vertices over type-1 that adds 2)
   - Greedy-by-quality: at each step, pick highest-quality element from all valid actions
   - Run both oracles on the best-performing grid. Target: complete the mesh (boundary < 3 vertices).

**Verification:**
- Oracle completes annulus-layer2: **BREAKTHROUGH** — the formulation works, proceed to WS2
- Distance-range fix enables type-1 actions but oracle doesn't complete: **PARTIAL** — need further investigation but path forward exists
- No grid or distance range produces type-1 actions: **DEAD END** — advancing-front with discrete grid cannot mesh this geometry. Document for session 9 architectural rethink.

**Decision gate:**
- If oracle completes → WS2 (training)
- If distance-range fix enables type-1 but no completion → WS2 with extended training
- If nothing works → Skip WS2, move to WS3 (quality push), document findings

**Files:** `src/MeshEnvironment.py` (possible distance-range tweak), diagnostic scripts only.

**What NOT to change:** Reward structure, network architecture, training loop. Only modify action grid parameters.

---

### WS2: Annulus-Layer2 Training with Working Configuration (Priority 2, ~60 min, depends on WS1)

**Prerequisite:** WS1 must produce an oracle completion or at least confirmed type-1 action viability.

**Steps:**
1. Configure action grid to whatever worked in WS1 (grid size + distance range)
2. If distance-range change is needed, implement as a configurable parameter in DiscreteActionEnv
3. Train DQN for 100k steps (budget ~11 min at 9.2k steps/min)
4. Eval at 10k, 30k, 50k, 100k

**Verification:**
- Completion > 0% at any checkpoint: **SUCCESS** — first ever completion on annulus-layer2
- Completion 0% but returns improving: **PARTIAL** — agent learning but needs more training or capacity
- No improvement over random: **FAIL** — DQN cannot exploit the expanded action space

**Training time estimate:** 100k steps / 9.2k steps/min ≈ 11 minutes wall-clock (will be longer with larger action space, budget 20 min). Plus eval, analysis, debugging: budget 60 min total.

**Files:** `src/DiscreteActionEnv.py` (configurable distance range), `main.py` (new CLI args if needed)

---

### WS3: Quality Push on Octagon (Priority 3, stretch goal, ~30 min)

**Problem:** Octagon quality is 0.478 vs 0.61 ceiling (78%). Star improved 9% with 24×4 (0.371→0.405). Apply same approach to octagon.

**Steps:**
1. Train octagon with 24×4 grid, 15k steps from scratch
2. Compare to 12×4 baseline (0.478)
3. If improved, this establishes 24×4 as the standard grid for quality-sensitive training

**Verification:**
- q > 0.50: **PASS** — 24×4 helps octagon
- q ≈ 0.478: **NEUTRAL** — no benefit from finer grid on this geometry
- q < 0.478: **FAIL** — larger action space hurts (worse exploration)

**Files:** No code changes. Training run only.

---

## Execution Order with Decision Gates

```
WS1: Annulus Feasibility Analysis (60 min)
  |
  ├── Step 1: Distance-range diagnostic (10 min)
  │     └── If strip_width / fan_radius < 0.2: test modified distance range
  │
  ├── Step 2: Grid enumeration across rollout (20 min)
  │     └── Test original and modified distance ranges
  │
  ├── Step 3: Oracle test (30 min, only if type-1 actions found)
  │     └── Two strategies: boundary-reduction and quality-greedy
  │
  ├── ORACLE COMPLETES → WS2 (training with working config)
  ├── TYPE-1 FOUND BUT NO COMPLETION → WS2 (training with extended budget)
  └── NOTHING WORKS → Skip WS2, document, proceed to WS3
  
WS2: Annulus Training (60 min, if WS1 positive)
  |
  └── 100k steps → eval at 10k/30k/50k/100k
  
WS3: Octagon 24×4 Quality Push (30 min, stretch goal)
  |
  └── 15k steps → compare to 12×4 baseline
```

## Risk/Mitigation Table

| Risk | Mitigation |
|------|-----------|
| Distance range is NOT the root cause — even 0.05×radius candidates fail PIP | Then the annulus geometry is inherently unsuited for grid-based interior placement. Document and propose continuous-action approach for session 9. |
| Grid diagnostic takes too long with 96×16 (1537 actions × 64 vertices) | The 9.2× speedup makes this feasible. Budget 5 min per grid config. If still slow, test 48×8 as the largest grid. |
| Oracle produces a completion sequence but it's ≥ 50 steps | Long episodes are fine if the agent can learn them. Set max_ep_len higher for annulus-layer2. |
| 100k training steps insufficient for large action space | Monitor learning curve. If returns improve steadily at 100k, extend to 200k. At 9.2k steps/min, 200k = 22 min. |
| WS1 takes entire session | WS1 is all diagnostic — even if WS2-3 don't run, the feasibility data is the highest-value output of the session. |

## What NOT to Do

- **Don't run curriculum on annulus.** Session 7 proved it doesn't help (0% completion with curriculum star model).
- **Don't increase network size** without evidence that the bottleneck is representation capacity.
- **Don't change the reward structure.** It's stable across 3 sessions.
- **Don't attempt boundary coarsening.** Decimating the boundary creates a different, easier domain — it doesn't solve the annulus problem.
- **Don't run multi-domain training.** Zero-shot transfer already shows generalization. Multi-domain is a polish experiment, not a blocker.

## Success Criteria

| Metric | Session 7 | Target | Stretch |
|--------|-----------|--------|---------|
| Annulus feasibility determined | No | **Yes** | Oracle completion |
| Distance-range root cause confirmed/rejected | No | **Yes** | Fix enables type-1 actions |
| Annulus-layer2 completion | 0% | Any > 0% | > 50% |
| Octagon quality (24×4) | 0.478 (12×4) | > 0.50 | > 0.55 |
| Tests passing | 21 | 21 | 21 |

## Adversarial Review Process

**Round 1 findings:** Original plan proposed finer grids without checking if the distance range is the bottleneck, a greedy oracle using the wrong heuristic (quality instead of boundary reduction), and 90 minutes on multi-domain training while annulus remained unsolved for 4 sessions. Revised to: distance-range diagnostic first, two oracle strategies, demoted multi-domain to cut.

**Round 2 findings:** Grid search is pointless if `0.2*radius` floor prevents candidates from reaching the valid region. Oracle must run only on grids with confirmed type-1 actions. 30k training steps is 31× too few for 1537-action space. Curriculum on star already disproven. Revised to: distance-range check as first sub-step, 100k training budget, dropped curriculum, octagon 24×4 from scratch.

**Remaining acknowledged risks:**
1. The annulus strip geometry may reject even very close candidates due to self-intersection of the resulting quad (not just PIP failure). The distance-range fix addresses PIP but not intersection validation.
2. If the oracle requires > 70 steps, the current max_ep_len truncates episodes before completion is possible. May need to raise max_ep_len (but this makes credit assignment harder).
3. Octagon 24×4 may not improve over 12×4 if the octagon ceiling is constrained by vertex count (8v) rather than angular resolution.

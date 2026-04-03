# Session 4 Plan: Scaling, Baselines, and Mesh Quality Definition

**Date:** Planned for next session after 2026-04-03
**Status:** Final (adversarial-reviewed)

## Context

Session 3 solved reward farming and enabled concave domains. All tested domains achieve 100% completion. The obvious next step — reward tuning for quality — was rejected by adversarial review for two reasons:

1. **Quality gap is geometric, not motivational.** The difference between agent quality and ceiling is small in absolute terms (0.314 vs 0.44 on star). The 12x4 action grid physically cannot express higher-quality vertex placements. Reward tuning adds noise-level signal (0.007-0.07 per step) that DQN's value estimates cannot distinguish.

2. **"Quality below ceiling" may be the wrong framing.** The L-shape agent found a 2-element solution (q=0.459) that is strictly better than the greedy 5-element solution (q=0.457) — fewer elements with equal quality. Our quality metric doesn't account for parsimony. Before optimizing quality, we need to define what "better mesh" means.

## Adversarial Review Summary

Two adversarial agents agreed:
- **Drop reward tuning** — quantitative analysis shows both proposed options change per-step reward by <0.07, which is noise for DQN
- **Drop mean_quality_so_far in state** — unactionable (agent can't undo past placements), agent already implicitly knows quality from element history
- **Prioritize rectangle domain** — first real test of scaling beyond trivial domains
- **Add greedy baseline** — can't evaluate agent without comparison
- **Define mesh quality properly** — element count matters, not just per-element quality

## Workstreams

### WS1: Greedy Baseline + Agent vs Greedy Comparison (Priority 1, ~30 min)

**Problem:** We don't know if the RL agent is better than a greedy heuristic. On L-shape, the agent's 2Q solution appears better than greedy's 5Q, but we have no systematic comparison.

**Fix:**
1. Add `--greedy` flag to main.py that runs greedy-by-quality rollout (no training)
2. Save greedy mesh visualization alongside agent mesh for each domain
3. Compare: element count, mean quality, total area quality (mean_q * element_count), completion

**Proposed mesh score:** `completion_bonus + mean_quality * (min_elements / actual_elements)`
- Rewards both quality and parsimony
- `min_elements` estimated as `ceil(n_vertices / 2)` (theoretical minimum quads to consume all vertices)

**Verification:** Run greedy + agent on star, octagon, L-shape. Document which is better and why.

**Files:** `main.py`, `quality_diagnostic.py`, `src/utils/visualization.py`

### WS2: Rectangle Domain — First Scaling Test (Priority 2, ~45 min)

**Problem:** All tested domains have 6-10 vertices and converge in 2-10k steps. Rectangle (20 vertices, max_ep_len=25) is the first real scaling test — will the agent learn multi-step planning over longer horizons?

**Steps:**
1. Run quality diagnostic on rectangle at 12x4 and 24x8
2. Train DQN on rectangle, 15-20k steps
3. If completion <50% at 15k, diagnose: is it action space (try 24x8) or learning (try more steps)?

**Decision gate:** If rectangle fails completely at both resolutions, the advancing-front + DQN approach may not scale beyond ~10 vertices. This is a critical finding either way.

**Verification:** >50% completion, compare element count to vertex count, save visualization.

**Files:** Uses existing `main.py --domain rectangle`

### WS3: Transfer Learning Experiment (Priority 3, ~30 min)

**Problem:** 2-10k convergence on individual domains suggests the agent memorizes rather than generalizes. If it can't transfer, this is an expensive lookup table.

**Experiment:**
1. Train on star (10v) for 10k steps
2. Evaluate (zero-shot) on octagon (8v) — similar vertex count, different geometry
3. Compare zero-shot performance to randomly initialized agent

**Interpretation:**
- Zero-shot > random: agent learned transferable features (good sign for generalization)
- Zero-shot ≈ random: agent memorized star geometry (not necessarily bad — may need multi-domain training)

**Files:** Uses existing CLI, just needs different eval domain.

### WS4: Action Space Resolution Experiment (Priority 4, if time, ~20 min)

**Problem:** Quality ceiling on star is 0.44 at both 12x4 and 24x8 (session 2 diagnostic). But we haven't tested whether 24x8 helps on L-shape or octagon where ceilings are higher.

**Fix:** Re-run quality diagnostic at 24x8 on all domains. If any domain shows >15% quality ceiling improvement, train at 24x8 and compare.

**Verification:** Quality ceiling comparison table, 12x4 vs 24x8 per domain.

## Execution Order

```
WS1 (greedy baseline, 30 min)
  |
WS2 (rectangle training, 45 min) — start training while analyzing WS1 results
  |
WS3 (transfer experiment, 30 min) — can overlap with WS2 training
  |
[If time]
  +-- WS4 (resolution experiment, 20 min)
```

## Key Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| Rectangle fails completely | This is a valid finding — document scaling limit |
| Greedy baseline beats agent everywhere | Would redirect project toward heuristic-guided RL or different formulation |
| Transfer test shows zero transfer | Expected for single-domain training; motivates multi-domain curriculum |
| Quality metric definition is contentious | Keep it simple (parsimony-weighted mean quality), iterate later |

## What NOT to Do Next Session

- **Don't tune rewards** — quality gap is geometric, not motivational (adversarial agents proved this)
- **Don't add features to state** — mean_quality_so_far is unactionable
- **Don't increase training budgets beyond 20k** — if it doesn't converge in 20k, the problem is structure, not steps
- **Don't train on circle** — still too easy, proves nothing

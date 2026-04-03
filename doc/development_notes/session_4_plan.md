# Session 4 Plan: Return to Pan et al. Reward Structure

**Date:** Planned for next session after 2026-04-03
**Status:** Final (adversarial-reviewed, then revised with post-session reward analysis)

## Context

Session 3 solved reward farming and enabled concave domains. All tested domains achieve 100% completion. But post-session analysis revealed that the reward function has drifted from Pan et al.'s (FreeMeshRL) formulation in two structurally harmful ways that explain the degenerate quads (e.g., 0.04-quality quad on star):

1. **`area_consumed` inverts Pan's density incentive.** Pan uses a bounded negative penalty `mu` (-1 to 0) that discourages too-small elements. Our `area_consumed` is a positive reward that encourages covering area — rewarding big sloppy quads. The 0.3x quality weight was a blunt fix for farming, but the root cause is that positive area reward is structurally wrong.

2. **No boundary quality term `eta_b`.** Pan penalizes actions that leave sharp remaining angles on the boundary (-1 to 0). Without this, our agent has no incentive to preserve boundary quality for future steps.

**Pan et al. per-step reward (Eq. 5-9):**
```
r = eta_e + eta_b + mu

eta_e = sqrt(q_edge * q_angle)          # element quality, 0 to 1
eta_b = boundary angle penalty           # -1 to 0, penalizes sharp remaining angles
mu    = density penalty                  # -1 to 0, penalizes elements below A_min
        mu = -1                          if area < A_min
        mu = (area - A_min)/(A_max-A_min) if A_min <= area <= A_max
        mu = 0                           if area > A_max
Completion: +10 (flat)
Invalid action: -0.1
```

**Our current active reward:**
```
r = 0.3 * quality + area_consumed - 0.05
Completion: 5.0 + 10.0 * mean_q^2
```

The original `_calculate_reward()` in `MeshEnvironment.py` (lines 267-290) is structurally identical to Pan's but was never wired into the discrete wrapper. It has `eta_e + eta_b + mu` but `eta_b` is hardcoded to -0.2 instead of computed from remaining boundary angles.

## Previous Adversarial Review (Partially Superseded)

The session 3 adversarial review concluded:
- "Quality gap is geometric, not motivational — reward tuning cannot help"
- "Drop WS1 (reward tuning) and WS2 (quality in state)"

This was correct about the *ceiling* (max achievable quality is geometry-limited) but wrong about the *gap*. Star agent quality is 0.314 vs ceiling 0.44 — a 29% gap that is not explained by action resolution. The agent isn't approaching the ceiling because it has no forward-looking incentive (missing `eta_b`) and is positively rewarded for sloppy area coverage (inverted `mu`).

The review's recommendations for WS3 (rectangle) and WS4 (greedy baseline) remain valid and are kept.

## Workstreams

### WS1: Re-implement Pan et al. Reward in DiscreteActionEnv (Priority 1, ~60 min)

**Problem:** The active reward in `DiscreteActionEnv.step()` diverges from Pan et al. in structurally harmful ways.

**Changes:**

1. **Replace `area_consumed` with Pan's bounded density penalty `mu`:**
   ```python
   element_area = env._calculate_polygon_area(new_element)
   A_min = 0.01 * env.original_area
   A_max = 0.1 * env.original_area
   if element_area < A_min:
       mu = -1.0
   elif element_area < A_max:
       mu = (element_area - A_min) / (A_max - A_min)
   else:
       mu = 0.0
   ```

2. **Add boundary quality term `eta_b`:**
   - After placing an element, compute the minimum angle in the remaining boundary
   - `eta_b = min_remaining_angle / 180.0 - 1.0` (maps to -1 to 0; penalizes sharp angles)
   - This is a simplification of Pan's full boundary quality metric; can be refined later

3. **Use element quality at full weight:**
   - `eta_e = quality` (not `0.3 * quality`)
   - With `mu` as a bounded penalty instead of positive `area_consumed`, farming is prevented natively — no need to suppress quality signal

4. **Flat +10 completion bonus** (Pan's formulation):
   - Replace `5.0 + 10.0 * mean_q^2`
   - Simpler, proven to work in the paper

5. **Keep domain-specific `max_ep_len`** as a safety backstop.

**Combined per-step reward:** `r = eta_e + eta_b + mu`
- Range: roughly -2.2 to +1.0 (matches Pan)
- Farming analysis: 44-step farming episode with mu≈0, eta_b≈-0.5, eta_e≈0.3 → ~-0.2 per step → total -8.8, no completion bonus. Farming is severely punished.
- 5-step efficient episode: eta_e≈0.4, eta_b≈-0.3, mu≈0 → ~+0.1 per step → total +0.5 + 10 completion = 10.5.

**Verification:** Train star 10k, confirm:
- Completion rate >90%
- Mean quality improves toward 0.40+ (vs current 0.314)
- No degenerate quads (min quality >0.1)
- No reward farming (element count <10)

**Files:** `src/DiscreteActionEnv.py`, `src/MeshEnvironment.py` (for boundary angle computation helper)

---

### WS2: Greedy Baseline + Agent Comparison (Priority 2, ~30 min)

**Problem:** We can't evaluate the agent without a comparison baseline. On L-shape, the agent's 2Q solution appears better than greedy's 5Q, but we have no systematic comparison.

**Fix:**
1. Add `--greedy` flag to main.py that runs greedy-by-quality rollout (no training)
2. Save greedy mesh visualization alongside agent mesh for each domain
3. Compare: element count, mean quality, completion

**Verification:** Run greedy + agent on star, octagon, L-shape. Document which is better and why.

**Files:** `main.py`, `src/utils/visualization.py`

---

### WS3: Rectangle Domain — First Scaling Test (Priority 3, ~30 min)

**Problem:** All tested domains have 6-10 vertices. Rectangle (20 vertices, max_ep_len=25) tests whether the approach scales.

**Steps:**
1. Run quality diagnostic on rectangle at 12x4
2. Train DQN 15k steps with Pan-style reward from WS1
3. If completion <50%, diagnose: action space or learning?

**Decision gate:** If rectangle fails at both 12x4 and 24x8, the approach may not scale beyond ~10 vertices.

**Files:** Uses existing `main.py --domain rectangle`

---

### WS4: Transfer Learning Experiment (Priority 4, if time, ~20 min)

**Problem:** 2-10k convergence suggests memorization, not generalization.

**Experiment:**
1. Train on star (10v) for 10k steps
2. Evaluate zero-shot on octagon (8v)
3. Compare to randomly initialized agent

**Files:** Uses existing CLI, different eval domain.

## Execution Order

```
WS1 (Pan reward re-implementation, 60 min)
  |-- Train star 10k to verify
  |-- Re-train octagon, L-shape if star looks good
  |
WS2 (greedy baseline, 30 min) — can overlap with WS1 training
  |
WS3 (rectangle, 30 min) — uses WS1 reward
  |
[If time]
  +-- WS4 (transfer experiment, 20 min)
```

## Key Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| Pan's reward causes new failure mode (e.g., agent avoids placing elements to avoid eta_b penalty) | eta_b only penalizes sharp remaining angles; completing removes boundary entirely → maximum eta_b |
| A_min/A_max thresholds wrong for our domains | Start with Pan's 0.01/0.1 of original_area; adjust if diagnostic shows all mu=-1 |
| eta_b computation is expensive | Compute min boundary angle once per step; vectorize if needed |
| Flat +10 completion too weak relative to Pan reward scale | Pan's per-step range is -2.2 to +1.0; 5-step episode intermediates ≈ +0.5; +10 completion dominates by 20x |
| Completion rate drops below session 3 levels | Keep max_ep_len as backstop; if completion <50%, investigate whether eta_b penalty is too harsh |

## What NOT to Do Next Session

- **Don't keep `area_consumed` as a positive reward** — it's structurally wrong and the root cause of sloppy quads
- **Don't suppress quality signal below 1.0x** — with Pan's bounded penalties, farming is prevented natively
- **Don't train on circle** — still too easy
- **Don't increase training beyond 20k** — if it doesn't converge in 20k, the problem is structural

# Session 6 Plan: Annulus-Layer2 Scaling + Action Space Refinement

**Date:** Planned for next session after 2026-04-03
**Status:** Final (adversarial-reviewed, two rounds of critique incorporated)

## Context

Session 5 improved star quality from 0.223 to 0.371 via quality-gated completion bonus (`5 + 10*mean_q`) and eta_b scaling (0.3). All four domains complete at 100%. A 64-vertex non-convex domain (`annulus-layer2`) was extracted from CHILmesh's FEM-smoothed annulus mesh layer 2 and registered as a new domain. Greedy baseline on it failed to complete (30Q, q=0.340, truncated).

**Remaining quality gaps:**
- Star: 0.371 / 0.44 ceiling = 84%
- Octagon: 0.478 / 0.61 ceiling = 78%
- Annulus-layer2: unknown ceiling, greedy incomplete

**New priority:** The annulus-layer2 domain is the first real-world domain and a critical scaling test. Can the current architecture (49 actions, 44-dim state, Dueling DDQN) handle 64 vertices? This question takes priority over incremental quality improvements on star/octagon.

The ceilings on star/octagon were characterized as "geometry-limited, not discretization-limited" in session 3. Session 6 tests both the scaling question (annulus-layer2) and whether the 12×4 action grid is a contributing factor for star.

## Adversarial Review Summary

### Round 1 Critiques (Incorporated)
- **No evidence 12×4 is binding:** Quality ceilings declared geometry-limited. Plan must include a diagnostic step to confirm/reject this before committing to action space expansion.
- **DQN doesn't scale for free to 193 actions:** Advantage head becomes wider than feature extractor. Need to specify network changes and buffer sizing.
- **WS1 and WS2 epsilon schedules interact:** Larger action space needs proportionally more exploration. Can't isolate epsilon from action space.
- **No ablation baseline:** Must test angular resolution (24×4) before full expansion (24×8) to identify which axis matters.
- **Completion rate regression risk:** More actions = more ways to place bad elements. Must track completion, not just quality.
- **Transfer learning underspecified:** Deferred until action space experiments are done and state representation is verified domain-agnostic.

### Round 2 Critiques (Incorporated)
- **Wall-clock is the binding constraint:** Estimated ~80 min per 25k run. Session budget allows 3-4 runs max.
- **Decision gate threshold (>0.02) not anchored to noise:** Need repeated baseline runs or wider threshold.
- **Buffer 200k + slow epsilon decay = noisy early training:** Adjust initial_random_steps proportionally.
- **WS2 (octagon) is a rerun, not a workstream:** Folded into WS1 as a parameter sweep.
- **24x4 failure mode: can't distinguish angular resolution from network capacity:** Accept this ambiguity; full factorial is too expensive.
- **Should verify reward farming fix is stable before expanding action space:** Added as WS1 prerequisite diagnostic.

## Workstreams

### WS1: Annulus-Layer2 Scaling Test (Priority 1, ~120 min)

**Problem:** All domains trained so far are hand-crafted with ≤20 vertices. The annulus-layer2 domain has 64 vertices and is highly non-convex (derived from a real FEM mesh). Can the current architecture scale to this complexity?

**Key unknowns:**
- Will the agent learn to complete the domain at all? (Greedy couldn't.)
- What's the training budget needed? Rectangle (20v) took 10k steps. 64v may need 50-100k.
- Does the 44-dim state representation capture enough context for 64 boundary vertices?
- Will action masking produce enough valid actions? (Initial reset showed only 1 valid action.)

**Steps:**
1. Train annulus-layer2 50k steps (12×4 grid, default hyperparams, max_ep_len=70)
2. Eval at 10k, 20k, 30k, 40k, 50k — track completion rate, quality, element count
3. If 0% completion at 20k: increase to 100k steps. If still 0% at 50k: diagnose (check valid action counts per step, state representation adequacy)

**Verification:**
- Completion > 0% at any eval: **PARTIAL PASS** — agent can learn on this domain
- Completion ≥ 50% with quality > 0.25: **STRONG PASS**
- 0% completion at 50k: **FAIL** — architecture cannot handle 64v domains. Investigate: is the bottleneck action space, state representation, or training budget?

**Decision gate:** If FAIL, the rest of the session pivots to diagnosing why. If PASS, proceed to WS2.

**Files:** No code changes. Training runs only.

---

### WS2: Angular Resolution Ablation — 24×4 (Priority 2, ~90 min)

**Problem:** Star quality at 84% of ceiling may be limited by 12-angle resolution. Star tips are at 36° angles; with 12 angular bins (30° each), the agent can't precisely target inter-tip regions. Doubling to 24 bins (15° each) provides 2× finer angular control.

**Why 24×4 first (not 24×8):** Isolates angular resolution from radial resolution. If 24×4 doesn't help, 24×8 won't either (radial resolution adds interior point distance control, but angular positioning is the primary constraint for star tips).

**Changes:**
- No code changes needed — `DiscreteActionEnv` already accepts `n_angle` and `n_dist` as constructor params
- CLI already has `--n-angle` and `--n-dist` flags
- Total actions: 24×4 + 1 = 97

**Training adjustments for larger action space:**
- Increase timesteps to 25k (from 15k) — more actions need more exploration
- Increase initial_random_steps to 2000 (from 1000) — proportional to action space size
- Keep epsilon_decay_frac at 0.7 for now (isolate one variable)
- Keep network architecture unchanged (advantage head scales automatically with num_actions)
- Keep buffer at 100k (97 actions doesn't justify doubling)

**Runs:**
1. Star 25k with 24×4 grid
2. If quality improves: Octagon 15k with 24×4 grid

**Verification (star):**
- Quality > 0.38 (improvement over 0.371 baseline): **MARGINAL PASS** — proceed with caution
- Quality > 0.40 (crossing 90% of ceiling): **STRONG PASS** — proceed to octagon
- Quality ≤ 0.371: **FAIL** — angular resolution is not the bottleneck
- Completion must remain ≥ 90% (regression check)
- Element count 4-8 (no farming regression)

**Verification (octagon, if run):**
- Quality > 0.50: **PASS**
- Quality ≤ 0.478: **FAIL**

**Decision gate at octagon:** If both star and octagon improve, consider testing 24×8 in a follow-up run. If neither improves, conclude that the quality gap is intrinsic (geometry-limited) and pivot to other approaches.

**Files:** No code changes. CLI flags already support arbitrary grid sizes.

---

### WS3: Epsilon Schedule Experiment (Priority 3, ~45 min)

**Prerequisite:** WS1 and WS2 complete

**Problem:** Current linear epsilon decay (1.0 → 0.05 over 70% of training) may commit the agent to a suboptimal policy early. Session 5 showed the agent sometimes found better solutions at higher epsilon (10k eval with 7 elements) but converged to worse solutions at low epsilon (15k eval with 4 elements).

**Changes:**
```python
# In DQNTrainer.__init__, add two-phase epsilon schedule:
# Phase 1: fast decay 1.0 → 0.3 over first 40% of training
# Phase 2: slow decay 0.3 → 0.05 over remaining 60%
```

**Run on whichever grid performed best in WS2:**
- If 24×4 improved: train star 25k with two-phase epsilon on 24×4
- If 24×4 didn't help: train star 15k with two-phase epsilon on 12×4

**Verification:**
- Quality improves over WS2 baseline (whichever grid was used)
- Completion ≥ 90%

**Decision gate:** If two-phase epsilon doesn't improve quality by >0.01, revert to linear schedule and note that epsilon isn't the bottleneck.

**Files:** `src/trainer_dqn.py`

---

### WS4: Transfer Learning Diagnostic (Priority 4, deferred if time-constrained)

**Prerequisite:** WS2 complete, action space decided. Only attempt if session time remains.

**Problem:** Each domain trains from scratch in 3-10k steps. If the network learns generalizable boundary-processing features, transfer could reduce training time or improve quality.

**Diagnostic (not full experiment):**
1. Take trained star model (best from WS2/WS3)
2. Eval zero-shot on octagon (no training, just run eval)
3. Eval zero-shot on L-shape
4. If WS1 produced a trained annulus-layer2 model: eval zero-shot on star
5. Record: completion rate and quality. Compare to random baseline.

**Success criterion:** Zero-shot performance better than random (return > 0, completion > 0%). If yes, transfer learning is viable and should be a session 7 workstream. If no, features are domain-specific.

**Note:** State representation (44-float enriched vector) uses relative geometry (neighbor positions, angles, area ratio), which should be domain-agnostic. But the network may have learned domain-specific patterns.

**Files:** No code changes. Eval only.

---

## Execution Order with Decision Gates

```
WS1: Annulus-layer2 scaling test (120 min)
  |
  ├── PASS (completion > 0%) → Record results, proceed to WS2
  └── FAIL (0% at 50k) → Diagnose bottleneck (state repr? action space? budget?)
                           Still proceed to WS2 (independent)
  
WS2: 24×4 star training (90 min)
  |
  ├── STRONG PASS (q > 0.40) → WS2 octagon → WS3
  ├── MARGINAL PASS (q > 0.38) → WS3 (skip octagon)
  └── FAIL (q ≤ 0.371) → Skip WS3, conclude geometry-limited → WS4
  
WS3: Epsilon experiment (45 min)
  |
  └── Complete → WS4 (if time)

WS4: Transfer diagnostic (20 min, if time)
```

## Risk/Mitigation Table

| Risk | Mitigation |
|------|-----------|
| Annulus-layer2 never completes | Diagnose: log valid action counts per step. If consistently 1 (type-0 only), the action space can't reach interior points. May need larger n_dist or continuous actions |
| Annulus-layer2 training takes too long | Budget 50k, extend to 100k if completion rate rising. If flat at 0% after 50k, stop |
| 24×4 doesn't improve quality | Accept geometry ceiling. Pivot to curriculum learning or continuous action space |
| Training convergence slows with 97 actions | Budget 25k steps. If not converged at 15k, extend to 35k before declaring failure |
| Completion rate drops with more actions | Track completion explicitly. If < 90%, action masking may have bugs at new resolution |
| Two-phase epsilon destabilizes training | Revert to linear schedule. The linear schedule works reliably |
| Wall-clock budget exceeded | WS4 is optional. WS3 can be deferred. Minimum viable session = WS1 + WS2 star |

## What NOT to Do

- **Don't test 24×8 without 24×4 first.** Ablation matters. Two variables at once = confounded results.
- **Don't change reward structure.** Session 5's reward is working. Change one variable (action space or epsilon).
- **Don't attempt multi-domain training.** No stable cross-domain baseline exists yet.
- **Don't increase hidden layer width preemptively.** 97 actions (24×4) is a modest increase. Only change architecture if training metrics (loss convergence, Q-value distribution) indicate capacity issues.
- **Don't tune annulus-layer2 hyperparams before seeing baseline results.** First run is diagnostic — understand the failure mode before optimizing.

## Success Criteria

| Metric | Session 5 | Target | Stretch |
|--------|-----------|--------|---------|
| Annulus-layer2 completion | 0% (greedy) | > 0% | ≥ 50% |
| Annulus-layer2 quality | 0.340 (greedy) | any | > 0.30 |
| Star quality | 0.371 | ≥ 0.38 | ≥ 0.40 |
| Octagon quality | 0.478 | ≥ 0.50 | ≥ 0.55 |
| Star completion | 100% | ≥ 90% | 100% |
| Tests passing | 21 | 21 | 21 |
| Transfer signal | N/A | measured | positive |

## Adversarial Review Process

**Round 1 findings:** Original plan proposed 24×8 directly, no ablation, no network scaling analysis, underspecified transfer learning, epsilon and action space confounded. Revised to: 24×4 first, proportional training adjustments, transfer deferred.

**Round 2 findings:** Wall-clock not budgeted, decision gate threshold not noise-anchored, buffer/epsilon interaction unaddressed, octagon is a rerun not a workstream, should verify stability before expanding. Revised to: WS1 stability diagnostic, explicit wall-clock estimates, octagon folded under WS2, conservative scope.

**Remaining acknowledged risks:**
1. 24×4 may show marginal improvement that's within run-to-run noise. Without repeated baseline runs, we can't distinguish signal from noise. Mitigation: use 0.02 threshold (not 0.01), and inspect element configurations visually.
2. Two-phase epsilon is an additional variable on top of action space change. If WS2 changed the grid, WS3 results are partially confounded. Mitigation: run WS3 on whichever grid WS2 selected, so at most one variable changes.

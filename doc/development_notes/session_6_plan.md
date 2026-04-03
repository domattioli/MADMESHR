# Session 6 Plan: Action Space Refinement

**Date:** Planned for next session after 2026-04-03
**Status:** Final (adversarial-reviewed, two rounds of critique incorporated)

## Context

Session 5 improved star quality from 0.223 to 0.371 via quality-gated completion bonus (`5 + 10*mean_q`) and eta_b scaling (0.3). All four domains complete at 100%. Remaining quality gaps:
- Star: 0.371 / 0.44 ceiling = 84%
- Octagon: 0.478 / 0.61 ceiling = 78%

The ceilings were characterized as "geometry-limited, not discretization-limited" in session 3. Session 6 tests whether the 12×4 action grid is a contributing factor, or whether the ceilings are truly intrinsic.

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

### WS1: Reward Stability Diagnostic (Priority 1, ~20 min)

**Problem:** Before expanding action space, verify the reward farming fix (session 3) and eta_b rebalancing (session 5) are stable. If farming recurs at 49 actions, it will be worse at 97+ actions.

**Steps:**
1. Train star 10k with current reward (12×4). Log element counts.
2. Check: elements should be 4-8 (not 20+). If >10 average, reward farming has recurred.
3. Run 3 eval rollouts, record quality variance.

**Verification:** Element count 4-8, quality 0.35-0.40 (consistent with session 5), no farming.

**Decision gate:** If farming detected → diagnose before proceeding. If stable → proceed to WS2.

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

**Prerequisite:** WS2 star run complete (need to know whether 24×4 helps)

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
4. Record: completion rate and quality. Compare to random baseline.

**Success criterion:** Zero-shot performance better than random (return > 0, completion > 0%). If yes, transfer learning is viable and should be a session 7 workstream. If no, features are domain-specific.

**Note:** State representation (44-float enriched vector) uses relative geometry (neighbor positions, angles, area ratio), which should be domain-agnostic. But the network may have learned domain-specific patterns.

**Files:** No code changes. Eval only.

---

## Execution Order with Decision Gates

```
WS1: Reward stability diagnostic (20 min)
  |
  ├── STABLE → WS2
  └── FARMING → Diagnose, fix, re-verify (abort WS2-4)
  
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
| 24×4 doesn't improve quality | Accept geometry ceiling. Pivot to curriculum learning or continuous action space |
| Training convergence slows with 97 actions | Budget 25k steps. If not converged at 15k, extend to 35k before declaring failure |
| Completion rate drops with more actions | Track completion explicitly. If < 90%, action masking may have bugs at new resolution |
| Two-phase epsilon destabilizes training | Revert to linear schedule. The linear schedule works reliably |
| Wall-clock budget exceeded | WS4 is optional. WS3 can be deferred. Minimum viable session = WS1 + WS2 star |
| Session 5 reward farming fix not stable | WS1 catches this before committing to larger experiments |

## What NOT to Do

- **Don't test 24×8 without 24×4 first.** Ablation matters. Two variables at once = confounded results.
- **Don't change reward structure.** Session 5's reward is working. Change one variable (action space or epsilon).
- **Don't add new domains before understanding existing ones.** Pentagon can wait.
- **Don't attempt multi-domain training.** No stable cross-domain baseline exists yet.
- **Don't increase hidden layer width preemptively.** 97 actions (24×4) is a modest increase. Only change architecture if training metrics (loss convergence, Q-value distribution) indicate capacity issues.

## Success Criteria

| Metric | Session 5 | Target | Stretch |
|--------|-----------|--------|---------|
| Star quality | 0.371 | ≥ 0.38 | ≥ 0.40 |
| Octagon quality | 0.478 | ≥ 0.50 | ≥ 0.55 |
| Star completion | 100% | ≥ 90% | 100% |
| Reward farming | none | none | none |
| Tests passing | 21 | 21 | 21 |
| Transfer signal | N/A | measured | positive |

## Adversarial Review Process

**Round 1 findings:** Original plan proposed 24×8 directly, no ablation, no network scaling analysis, underspecified transfer learning, epsilon and action space confounded. Revised to: 24×4 first, proportional training adjustments, transfer deferred.

**Round 2 findings:** Wall-clock not budgeted, decision gate threshold not noise-anchored, buffer/epsilon interaction unaddressed, octagon is a rerun not a workstream, should verify stability before expanding. Revised to: WS1 stability diagnostic, explicit wall-clock estimates, octagon folded under WS2, conservative scope.

**Remaining acknowledged risks:**
1. 24×4 may show marginal improvement that's within run-to-run noise. Without repeated baseline runs, we can't distinguish signal from noise. Mitigation: use 0.02 threshold (not 0.01), and inspect element configurations visually.
2. Two-phase epsilon is an additional variable on top of action space change. If WS2 changed the grid, WS3 results are partially confounded. Mitigation: run WS3 on whichever grid WS2 selected, so at most one variable changes.

# Session 5 Plan: eta_b Rebalancing + Quality-Gated Completion

**Date:** Planned for next session after 2026-04-03
**Status:** Final (adversarial-reviewed, two rounds of critique incorporated)

## Context

Session 4 implemented Pan et al.'s reward structure (`eta_e + eta_b + mu`) and successfully scaled to rectangle (20v). However, quality regressed on star (0.314→0.223) and octagon (0.535→0.478) because the eta_b boundary angle penalty dominates per-step reward, making step-minimization the optimal strategy. The agent learned to place fewer, larger, lower-quality elements to avoid accumulating eta_b penalties.

**Key numbers:**
- Star tip angle: ~36° → eta_b ≈ -0.8 per step
- 4 steps at q≈0.22: `4*(0.22 - 0.8 + 0) + 10 = 7.68` (current optimum)
- 8 steps at q≈0.35: `8*(0.35 - 0.8 + 0) + 10 = 6.40` (worse return despite better quality)
- Quality ceilings: star≈0.44, octagon≈0.61, circle≈0.78

## Adversarial Review Summary

### Round 1 Critiques (Incorporated)
- **Ablation structure absent**: Original plan had 4 simultaneous reward changes → reduced to 1 change at a time
- **Coverage reward exploitable**: Dropped entirely — agent could exploit large-element coverage
- **Tier thresholds ungrounded**: Replaced with smooth function — no arbitrary thresholds
- **Transfer learning premature**: Dropped — no stable reward base to transfer from
- **No diagnostic instrumentation**: Moved logging to WS1 (before any training)
- **Triangle fallback unaddressed**: Added explicit check in WS1 verification

### Round 2 Critiques (Incorporated)
- **Math assumes the conclusion**: Added sensitivity analysis across quality ranges, not just one point
- **0.3 scalar ungrounded**: Added derivation from break-even analysis
- **Logging after experiments**: Fixed — logging is now WS1, before any training
- **10k steps insufficient**: Increased to 15k with explicit convergence check at 5k intervals
- **Rectangle is wrong verification target**: Changed to octagon (geometrically relevant)
- **No failure branch**: Added explicit fallback paths for each verification gate

## Workstreams

### WS1: Reward Component Logging (Priority 1, ~30 min)

**Problem:** Cannot diagnose reward issues without per-component breakdown. Session 4 required manual arithmetic to understand quality regression.

**Changes:**
1. In `DQNTrainer.train()`, after each episode ends, log:
   - Episode length (element count)
   - Mean element quality
   - Sum of eta_e, eta_b, mu across steps
   - Completion bonus received
   - Total return
2. Print summary at each eval checkpoint:
   ```
   EVAL at t=5000: Return=8.76 | Completion=100% | MeanQ=0.35 | AvgElements=6.2 | AvgEtaB=-2.1
   ```
3. Pass reward components through `info` dict from `DiscreteActionEnv.step()`

**Files:** `src/DiscreteActionEnv.py` (add eta_e, eta_b, mu to info), `src/trainer_dqn.py` (log components)

**Verification:** Run star 1k steps, confirm all components appear in output with plausible values.

---

### WS2: Scale eta_b by 0.3 (Priority 2, ~60 min)

**Problem:** eta_b at full weight makes step-minimization dominant. Need to reduce its influence so that eta_e (quality) matters more, without eliminating boundary feedback entirely.

**Derivation of 0.3 scalar:**
The break-even point where 8 steps matches 4 steps:
```
8*(eta_e_8 + w*eta_b + 0) + 10 = 4*(eta_e_4 + w*eta_b + 0) + 10
→ 8*eta_e_8 + 8*w*eta_b = 4*eta_e_4 + 4*w*eta_b
→ 8*eta_e_8 - 4*eta_e_4 = 4*w*eta_b * (1 - 2)  ... (not quite)
→ 4*w*eta_b = 8*eta_e_8 - 4*eta_e_4
```
With eta_b ≈ -0.8, eta_e_8 ≈ 0.30, eta_e_4 ≈ 0.22:
```
4*w*(-0.8) = 8*0.30 - 4*0.22 = 2.4 - 0.88 = 1.52
-3.2*w = 1.52 → w = -0.475
```
At w=0.475, 8 steps breaks even with 4 steps. We want 8 steps to clearly win, so w < 0.475. Setting w=0.3 gives 8 steps a margin:
```
4 steps: 4*(0.22 + 0.3*(-0.8)) + 10 = 4*(-0.02) + 10 = 9.92
8 steps: 8*(0.30 + 0.3*(-0.8)) + 10 = 8*(0.06) + 10 = 10.48
```
8 steps wins by 0.56. This margin increases with quality improvements, creating positive feedback.

**Sensitivity analysis** (varying actual per-element quality):

| Scenario | 4 steps | 8 steps | Winner |
|----------|---------|---------|--------|
| q4=0.22, q8=0.30 | 9.92 | 10.48 | 8 steps (+0.56) |
| q4=0.22, q8=0.25 | 9.92 | 10.08 | 8 steps (+0.16) |
| q4=0.22, q8=0.22 | 9.92 | 9.68 | 4 steps (+0.24) |
| q4=0.30, q8=0.35 | 10.24 | 10.88 | 8 steps (+0.64) |

**Key insight:** If quality doesn't improve with more elements (q8=q4=0.22), 4 steps still wins. This means the agent will only use more steps if doing so actually produces better elements. This is the correct incentive — more steps should only be taken when they help quality.

**Changes:**
```python
# In DiscreteActionEnv.step(), non-terminal reward:
reward = eta_e + 0.3 * eta_b + mu
```

**Verification (sequential, with decision gates):**

1. Train star 15k steps. At each 5k eval, check:
   - Quality > 0.28 (above session 4's 0.223) → continue
   - Quality > 0.30 at 10k → **PASS**, proceed to WS3
   - Quality still at 0.22 at 10k → **FAIL**, try w=0.15 (weaker eta_b)
   - Element count should be 5-8 (between session 3's 8 and session 4's 4)

2. If PASS: Train octagon 8k, verify quality ≥ 0.50 (between session 3's 0.535 and session 4's 0.478)

**Files:** `src/DiscreteActionEnv.py`

---

### WS3: Quality-Gated Completion Bonus (Priority 3, ~45 min)

**Prerequisite:** WS2 passes star verification (quality > 0.30)

**Problem:** Flat +10 completion gives no gradient signal for quality improvement once completion is achieved. The agent has no reason to prefer a 0.40 quality mesh over a 0.25 quality mesh if both complete.

**Changes:**
```python
# Replace flat +10 with:
completion_bonus = 5.0 + 10.0 * mean_q  # range: 5.0 to 15.0
```

Smooth function, no threshold tuning. Minimum bonus is 5.0 (even terrible meshes get some completion reward), maximum is 15.0 (mean_q=1.0, theoretical). At star ceiling quality (0.44): bonus = 9.4. At current agent quality (0.30): bonus = 8.0.

**Math check with WS2 reward:**
```
8 steps at q=0.30: 8*(0.30 + 0.3*(-0.8)) + 5 + 10*0.30 = 0.48 + 5 + 3.0 = 8.48
4 steps at q=0.22: 4*(0.22 + 0.3*(-0.8)) + 5 + 10*0.22 = -0.08 + 5 + 2.2 = 7.12
```
8 steps wins by 1.36 (larger margin than WS2 alone at 0.56). Quality-gated bonus amplifies the incentive.

**Apply to all terminal reward paths:** boundary consumed (bnd<3), auto-close quad (bnd=4), triangle fallback (bnd=3), self-intersecting quad split.

**Triangle fallback check:** Triangle completion currently gives fixed 6.0/3.0. With quality-gated: boundary triangle gets `5.0 + 10.0 * mean_q` (same formula). This is correct — triangle fallback is an acceptable endgame and shouldn't be penalized differently.

**Verification:** Train star 15k, verify quality improves vs WS2 alone (target: > 0.33).

**Files:** `src/DiscreteActionEnv.py`

---

### WS4: Octagon + Rectangle Verification (Priority 4, ~30 min)

**Prerequisite:** WS2 (or WS2+WS3) reward finalized

**Steps:**
1. Train octagon 8k steps, verify: completion ≥ 90%, quality ≥ 0.50
2. Train rectangle 15k steps, verify: completion ≥ 90%, quality ≥ 0.40
3. Run greedy baseline on octagon for comparison

**Decision gate:** If octagon quality drops below 0.45 (worse than session 4's 0.478), the reward change hurt non-star domains. Investigate whether eta_b weight should be domain-specific.

**Files:** No code changes, just training runs.

---

## Execution Order with Decision Gates

```
WS1: Logging instrumentation (30 min)
  |
WS2: eta_b * 0.3 + star training (60 min)
  |
  ├── PASS (star quality > 0.30, elements >= 5)
  |     |
  |     WS3: Quality-gated completion + star training (45 min)
  |     |
  |     WS4: Octagon + rectangle verification (30 min)
  |
  └── FAIL (star quality ≤ 0.22 at 10k)
        |
        Try eta_b * 0.15 and retrain
        |
        If still fails → conclude eta_b rebalancing alone insufficient
        → investigate action space resolution for star tips
```

## Risk/Mitigation Table

| Risk | Mitigation |
|------|-----------|
| w=0.3 too aggressive, completion drops | Check completion at 5k eval. If <80%, increase w to 0.4 |
| w=0.3 not enough, quality still ~0.22 | Try w=0.15 (failure branch). If still 0.22, issue is not eta_b weight |
| Quality-gated bonus re-enables farming | Math shows 20-step farm: `20*(0.3+0.3*(-0.5))+5+10*0.3 = 20*0.15+8 = 11`. 5-step efficient: `5*(0.4+0.3*(-0.4))+5+10*0.4 = 5*0.28+9 = 10.4`. Farming slightly wins. Verify in training — if elements >10, revert to flat +10 |
| Rectangle fails with new reward | Rectangle has mild boundary angles (90°), eta_b ≈ -0.5. Scaling by 0.3 gives -0.15 per step. Should still complete easily |
| Logging overhead slows training | Logging is per-episode, not per-step. Negligible overhead |

## What NOT to Do

- **Don't change mu.** It's working correctly (no elements below A_min observed).
- **Don't add coverage reward.** Exploitable and underspecified (devil's advocate #1).
- **Don't clip rewards.** Ambiguous interaction with completion bonus (devil's advocate #1).
- **Don't attempt transfer learning.** No stable reward base yet (devil's advocate #1+2).
- **Don't add new action space features.** One variable at a time.
- **Don't train on circle.** Still too easy, not informative.
- **Don't commit training results before verification gates pass.** (Lesson from memory.)

## Success Criteria

| Metric | Session 4 | Target | Stretch |
|--------|-----------|--------|---------|
| Star quality | 0.223 | ≥ 0.30 | ≥ 0.35 |
| Star elements | 4 | 5-8 | 6-8 |
| Octagon quality | 0.478 | ≥ 0.50 | ≥ 0.55 |
| Rectangle completion | 100% | ≥ 90% | 100% |
| Reward logging | absent | present | component breakdown per eval |
| Tests passing | 21 | 21 | 21 |

## Adversarial Review Process

This plan went through two rounds of adversarial critique:

**Round 1 findings:** Original plan had 4 simultaneous reward changes (eta_b terminal, quality-scaled completion, coverage reward, clipping), no ablation structure, ungrounded thresholds, premature transfer learning. All addressed by scope reduction and sequential gating.

**Round 2 findings:** Revised plan's break-even math assumed fixed quality values, 0.3 scalar was ungrounded, logging was scheduled after experiments, verification used wrong geometry (rectangle). Addressed by: sensitivity analysis showing 0.3 is below break-even threshold (0.475), logging moved to WS1, rectangle replaced with octagon as primary verification.

**Remaining acknowledged risk:** If per-element quality doesn't improve with more elements (q8 = q4), the agent will still prefer fewer steps. This is actually the correct behavior — more elements should only be placed when they help. The risk is that action space limitations prevent quality improvement regardless of count, in which case the problem is geometric, not reward-shaped.

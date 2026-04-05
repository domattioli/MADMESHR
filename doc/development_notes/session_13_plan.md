# Session 13 Plan: Retrain Existing Domains with Session 12 Fixes

**Date:** Next session after session 12
**Status:** Final (multi-agent reviewed)
**Budget:** 3 hours

## Context

Session 12 made three changes that could improve existing domains:
1. **Mu calibration**: A_min/A_max scaled by expected element count. On H-shape 24v, this eliminated the mu-avoidance trap.
2. **Type-0 priority vertex selection**: Scans all vertices for the best type-0 before angle-based selection. Reduced H-shape from 15Q to 11Q (optimal). Adds ~2x per-step overhead.
3. **Boundary distance filter**: Rejects interior vertices too close to boundary edges.

### Adversarial Review Summary

Three agents reviewed this plan:

**Devil's Advocate** attacked:
1. Type-0 priority validated on only H-shape — could hurt convex domains -> **Accepted**: WS1 eval must log per-step actions, not just final quality. Add ablation.
2. Plan's A_max arithmetic for rectangle was wrong (said 0.200, actual is 0.200 but double-check code) -> **Accepted**: verify in WS1
3. 15k/20k step budgets extrapolated from single data point -> **Accepted**: decision gates at 5k and 10k, not just 10k
4. Star retraining is waste — at ceiling, mu unchanged -> **Accepted**: star CUT from WS2
5. No ablation to separate mu vs type-0 effects -> **Accepted**: run one domain with type-0 disabled as control
6. Success criteria within noise without multiple seeds -> **Accepted**: raise targets or run 3 seeds
7. Rectangle per-step time makes 20k infeasible -> **Accepted**: cap at 7.5k (see scope realist)

**Scope Realist** computed:
- Rectangle at ~0.8s/step: 20k = 4.4 hours (impossible) -> **Accepted**: cap at 7.5k, treat as stretch goal
- Star cut saves 45 min -> **Accepted**
- Total realistic: ~200 min (3.3h) still tight -> **Accepted**: WS1 is hard decision gate; skip retraining if eval shows no change
- validate_mesh takes 2-3 min with type-0 scan -> **Accepted**: budget for it

**Reward Analyst** proved quantitatively:
- **Mu is 0 for typical elements in ALL four domains under both old and new calibration** -> Changes expectation: mu fix is NOT the driver for these domains
- Octagon gap (0.478 vs 0.61) is action-space limited, not mu -> **Accepted**: type-0 priority is the only lever; if it doesn't help, the gap requires action space changes
- Star: mu unchanged, quality at ceiling -> **Accepted**: confirms star is cut
- No regression risk from mu changes -> **Accepted**: removes concern about circle/rectangle

**Bottom line:** The mu fix is irrelevant for existing domains (mu=0 for all typical elements). The ONLY change that could improve quality is type-0 priority vertex selection. This was validated on one domain (H-shape). The plan is now a focused test of whether type-0 priority generalizes.

## Workstreams (3, strict priority order)

---

### WS1: Evaluate + Ablation Under New Code (Priority 1, ~35 min)

**Problem:** Type-0 priority changes action enumeration for all domains. Need to verify no regressions and measure the effect.

**Step 1: Eval all domains with new code (15 min)**
- Load best checkpoint for each non-trivial domain: octagon, star, circle, rectangle
- Run eval-only under new code
- Log: completion %, element count, quality, per-step action types
- Compare to session 11 baselines

**Step 2: Greedy baselines (10 min)**
- Run greedy-by-quality on octagon, rectangle with new code
- Compare to old greedy baselines
- Do greedy element counts change?

**Step 3: Type-0 priority ablation (10 min)**
- Temporarily disable type-0 priority (revert to angle-based selection)
- Re-eval octagon and rectangle
- Does the old vertex selection produce the same quality? If yes, type-0 priority is not helping these domains.

**Decision gate:** If eval shows:
- No quality change for any domain -> type-0 priority doesn't generalize. Retrain with larger action space (24x8) instead. Proceed to WS3 docs.
- Quality improved for >= 1 domain -> retrain that domain in WS2.
- Quality regressed for any domain -> investigate before retraining.

**Files:** `main.py` (eval-only), `src/MeshEnvironment.py` (ablation toggle)

---

### WS2: Retrain Priority Domains (Priority 2, ~120 min)

**Proceed only if WS1 shows potential improvement.**

**Training order (by expected impact):**
1. **Octagon** (8v): Largest quality gap (0.478 vs 0.61 ceiling). 15k steps (~50 min with type-0 scan).
   - Decision gate at 5k: if quality <= 0.48, kill.
   - Target: q >= 0.55 (well above noise)
2. **Rectangle** (20v): 7.5k steps (~100 min with type-0 scan). Stretch goal — only if octagon succeeds AND time remains.
   - Decision gate at 5k: if quality <= 0.47, kill.
   - Target: element count < 9 (currently 9Q)

**Star is CUT** (per all three reviewers: at ceiling, mu unchanged, pointless).

**Hyperparameters:**
- `--epsilon-decay-frac 0.5 --buffer-size 20000 --target-update-freq 500`

**Alternative if type-0 priority doesn't help:**
- Try larger action space (24 angles x 8 radial = 192 type-1 actions) on octagon
- This addresses the action-space limitation identified by the reward analyst

**Cannot run parallel training** (OOM on RTX 3060).

**Verification:**
- At least 1 domain shows quality improvement above noise (delta > 0.03)
- All retrained domains maintain 100% completion
- All domains pass 7-point validation

---

### WS3: Documentation + Next Steps (Priority 3, ~15 min)

**Steps:**
1. Write session_13_report.md
2. Run 7-point validation on all domains
3. Push images to output/latest/
4. Update CLAUDE.md with new baselines
5. Plan session 14 (adversarial review) — choose between:
   - **Option B**: Pan et al. benchmark domain comparison
   - **Option C**: Annulus type-2 curriculum training
   - **Option D**: Larger action space (24x8) for octagon if type-0 priority didn't help

---

## Execution Order

```
WS1: Eval + Ablation (35 min)
  |
  +-- Eval all domains under new code (15 min)
  +-- Greedy baselines (10 min)
  +-- Type-0 priority ablation (10 min)
  |
  DECISION GATE:
    Improvement found -> WS2 (retrain that domain)
    No improvement -> Skip WS2, explore larger action space or Pan et al.
    Regression found -> Investigate, potentially revert type-0 priority for that domain
  |
WS2: Retrain (120 min, conditional)
  |
  +-- Octagon 15k (50 min) with 5k decision gate
  +-- Rectangle 7.5k (100 min, stretch) with 5k decision gate
  |
WS3: Docs (15 min)
```

## What NOT to Do

- **Do not retrain star.** At geometry ceiling (0.44), mu unchanged. All three reviewers agree.
- **Do not retrain L-shape or square.** Already optimal.
- **Do not run parallel TF training.** OOM on RTX 3060.
- **Do not attempt rectangle 20k steps.** At 0.8s/step, that's 4.4 hours. Max 7.5k.
- **Do not skip the ablation.** Without it, we can't attribute improvements to type-0 priority vs random variance.

## Success Criteria

| Metric | Session 11 | Target | Stretch |
|--------|-----------|--------|---------|
| Octagon quality | 0.478 (3Q) | **>= 0.55** | >= 0.60 |
| Rectangle quality | 0.464 (9Q) | **>= 0.50** | element count < 9 |
| No regressions | All 100% complete | **All 100% complete** | Quality >= baseline for all |
| Ablation complete | N/A | **Type-0 priority effect measured** | Attributed to specific change |

## Risk / Mitigation Table

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Type-0 priority doesn't help on convex domains | Medium | Medium | WS1 ablation detects this; pivot to larger action space |
| Rectangle training exceeds time budget | High | Low | Cap at 7.5k; treat as stretch goal |
| Type-0 scan overhead makes training impractical | Medium | Medium | Monitor per-step time; can cache type-0 scan results |
| Octagon improvement within noise | Medium | Low | Run 3 eval seeds; raise target to 0.55 |
| Code changes cause subtle regression on circle | Low | Medium | WS1 eval catches this before any retraining |

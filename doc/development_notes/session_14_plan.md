# Session 14 Plan: Rectangle Filter Fix + Annulus Type-2 Feasibility

**Date:** Next session after session 13
**Status:** Final (multi-agent reviewed)
**Budget:** 3 hours

## Context

Session 13 proved type-0 priority vertex selection does not generalize to convex domains (regression on octagon, star, rectangle). Made it configurable per-domain (OFF by default, ON for h-shape/l-shape). Repo reorganized for library packaging (`src/` → `madmeshr/`).

Current quality baselines:
- Octagon 24×4: q=0.579, 5Q, 100% (ceiling 0.61)
- Star 24×4: q=0.405, 5Q, 100% (ceiling 0.44, CUT from training)
- Rectangle 12×4: q=0.464, 9Q, 100% (regressed to 0.426 under new code with type-0 OFF)
- H-shape 24v: q=0.677, 11Q, 100% (optimal)
- L-shape: q=0.459, 2Q, 100%

### Adversarial Review Summary

Two agents reviewed this plan:

**Devil's Advocate #1** attacked:
1. Rectangle regression cause is CONFOUNDED — mu calibration AND boundary filter changed in s12 → **Accepted**: add 5-min controlled test (disable filter, re-eval checkpoint) before any fix
2. Type-2 exploration near-impossible — only 1/31 valid type-2 actions at annulus initial state → **Accepted**: use sub-loop curriculum as primary approach, not full type-2 training
3. Annulus step time ~106ms (3× slower than expected) → **Accepted**: cap at 30k steps, treat as feasibility test
4. 24×8 octagon may not close the 0.031 quality gap → **Accepted**: run greedy eval at 24×8 before any training; moved to stretch goal
5. WS-B has no fallback → **Accepted**: restructured as two-phase (sub-loop curriculum first, full type-2 stretch)

**Scope Realist** computed:
1. WS-A training budget (25 min = 1875 steps) too small for rectangle at 0.8s/step → **Accepted**: eval-only first, retrain only if needed with 60+ min
2. Annulus has 64 vertices (not ~30), episodes will be long → **Accepted**: reframe as integration test
3. WS-C code changes underestimated (25 min for action space, tests, etc.) → **Accepted**: drop training, code-only if time permits
4. Existing checkpoint may work with relaxed filter — no retraining needed → **Accepted**: eval-first approach
5. 15-min buffer is fiction (35+ min invisible overhead) → **Accepted**: dropped WS-C as standalone, increased buffer
6. **Annulus oracle couldn't complete** (stuck at 21 boundary vertices) → **Critical**: debug oracle BEFORE any DQN training. If domain is geometrically dead-ended, DQN cannot succeed.

**Bottom line:** WS-A is likely a 15-minute eval-only fix. WS-B must start with oracle debugging — if the oracle can't complete, DQN training is premature. Octagon 24×8 is deprioritized to stretch goal (code only, no training).

## Workstreams (3, strict priority order)

---

### WS1: Rectangle Boundary Distance Filter Investigation (Priority 1, ~20 min)

**Problem:** Rectangle s5 checkpoint regressed from q=0.464 to q=0.426 under session 12 code changes (type-0 priority OFF). Session 13 attributed this to the boundary distance filter (3% fan radius threshold). But the mu calibration also changed — cause is not isolated.

**Step 1: Controlled experiment (5 min)**
- Disable boundary distance filter (`min_dist_threshold = 0`) and re-eval rectangle_s5 checkpoint
- If quality returns to 0.464: filter is the cause → proceed to Step 2
- If quality still 0.426: mu calibration or other change is the cause → investigate

**Step 2: Make threshold configurable (10 min)**
- Add `bnd_dist_threshold` parameter to `MeshEnvironment` (default 0.03)
- Add per-domain override in `main.py` domain registration
- Set rectangle to 0.01 (or 0.0 if experiment shows 0 is needed)

**Step 3: Verify (5 min)**
- Re-eval rectangle_s5 checkpoint with relaxed threshold
- Run 7-point validation on rectangle, octagon, h-shape
- If quality restored to 0.460+: done, no retraining needed
- If still below 0.455: defer retraining to WS1-extension (see below)

**Decision gate:** If controlled experiment shows filter is NOT the cause, skip Step 2 and investigate mu calibration instead.

**WS1-extension (CONDITIONAL, 60 min):** Only if eval doesn't recover quality AND time permits after WS2. Retrain rectangle 12×4 from scratch. At 0.8s/step, 60 min = 4500 steps with `--epsilon-decay-frac 0.5 --buffer-size 20000 --target-update-freq 500`.

**Files:** `madmeshr/mesh_environment.py` (threshold), `main.py` (per-domain config)

---

### WS2: Annulus Type-2 Feasibility (Priority 2, ~100 min)

**Problem:** Type-2 DQN architecture is implemented and tested but never trained. The annulus is the target domain, but the oracle couldn't complete it (stuck at 21 boundary vertices in session 9). DQN training on an incomplete domain will produce only truncated episodes with no completion reward — the "reward farming" failure mode.

**Phase 1: Oracle debugging (15 min)**
- Re-run `scripts/annulus_oracle_type2.py` with verbose output
- Diagnose WHY the oracle gets stuck at 21 boundary vertices
- Possible causes: (a) geometric dead-end (no valid actions), (b) type-2 threshold too tight, (c) auto-close bug on sub-loops
- **Decision gate:** If oracle still can't complete, identify the specific failure (which step, which vertex configuration, what actions were tried). Fix if possible.

**Phase 2: Sub-loop curriculum (45 min)**
- Pre-place type-2 elements using oracle (or manually) to split annulus into sub-loops
- Train DQN on the smaller (~20-30v) sub-loops only
- This sidesteps the type-2 exploration problem entirely
- Success criterion: DQN completes a sub-loop mesh with q > 0.35

**Phase 3: Full type-2 integration test (30 min, stretch)**
- Only if Phase 2 succeeds
- Run full annulus training with type-2 actions enabled
- 30k steps max (~53 min at 106ms/step — tight)
- Success criterion: agent places at least 1 type-2 action, no crashes
- Measure per-step timing for future budgeting

**Decision gates:**
- After Phase 1: If oracle is stuck due to fundamental geometry issue → fix domain or defer WS2 entirely
- After Phase 2: If sub-loop training fails → type-2 approach needs reward/architecture changes, document findings
- After Phase 3: Report step timing and whether type-2 actions were selected

**Files:** `scripts/annulus_oracle_type2.py`, `madmeshr/discrete_action_env.py` (type-2 threshold), `main.py` (curriculum setup)

---

### WS3: Documentation + Stretch Goals (Priority 3, ~20 min)

**Steps:**
1. Write session_14_report.md
2. Run 7-point validation on all domains
3. Push images to `tests/output/`
4. Update CLAUDE.md with any new baselines

**Stretch goal (code only, no training):**
- Octagon 24×8 action space: make `n_dist` configurable per-domain
- Run `--greedy` at 24×8 to check if greedy baseline exceeds 0.61 ceiling
- If greedy at 24×8 ≤ greedy at 24×4: radial resolution is not the bottleneck, defer
- No DQN training — just feasibility check

---

## Execution Order

```
WS1: Rectangle Filter Investigation (20 min)
  |
  +-- Controlled experiment: disable filter, re-eval (5 min)
  +-- DECISION GATE:
  |     Filter is cause → make configurable, re-eval (15 min)
  |     Filter NOT cause → investigate mu, document findings (10 min)
  |
WS2: Annulus Type-2 Feasibility (100 min)
  |
  +-- Phase 1: Oracle debugging (15 min)
  +-- DECISION GATE:
  |     Oracle fixed → Phase 2
  |     Oracle still stuck, cause understood → Fix and Phase 2
  |     Fundamental geometry dead-end → defer WS2, use time for WS1-extension
  |
  +-- Phase 2: Sub-loop curriculum training (45 min)
  +-- DECISION GATE:
  |     Sub-loop succeeds → Phase 3 (stretch)
  |     Sub-loop fails → Document, skip Phase 3
  |
  +-- Phase 3: Full type-2 integration test (30 min, stretch)
  |
WS3: Docs + Stretch (20 min)
  |
  +-- Report, validation, push
  +-- Octagon 24×8 greedy eval (stretch, 10 min)
  |
Buffer: ~40 min across session for debugging, testing, git operations
```

## What NOT to Do

- **Do not retrain rectangle without first confirming the cause.** Controlled experiment first.
- **Do not train DQN on annulus until oracle is debugged.** If the oracle can't complete, DQN can't either.
- **Do not attempt full type-2 training before sub-loop curriculum.** The exploration problem is too hard for direct training.
- **Do not train octagon 24×8.** Code-only + greedy eval as stretch goal.
- **Do not run parallel TF training.** OOM on RTX 3060.
- **Star is CUT.** At geometry ceiling (0.44), confirmed by session 12 adversarial review.

## Success Criteria

| Metric | Session 13 | Target | Stretch |
|--------|-----------|--------|---------|
| Rectangle quality | 0.426 (regressed) | **>= 0.460** (restored) | >= 0.464 (matches baseline) |
| Annulus oracle | Incomplete (21v stuck) | **Diagnosed and fixed** | Oracle completes |
| Sub-loop DQN | N/A | **Completes 1 sub-loop** | q > 0.35 |
| Type-2 integration | Untested | **No crashes in 1k steps** | Agent selects type-2 |
| No regressions | 8/8 validation | **8/8 validation** | All domains |

## Risk / Mitigation Table

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Rectangle regression is NOT from boundary filter | Medium | Low | 5-min controlled experiment catches this |
| Annulus oracle stuck due to fundamental geometry | Medium | High | Phase 1 diagnosis; fix domain if needed |
| Sub-loop curriculum environment setup complex | Medium | Medium | Reuse existing oracle code for pre-placement |
| Type-2 exploration too sparse for DQN | High | Medium | Sub-loop curriculum bypasses this |
| Annulus per-step time exceeds budget | Medium | Medium | Cap at 30k steps; reframe as feasibility test |
| Code changes introduce subtle bugs | Low | Medium | Run 44 tests after every change |

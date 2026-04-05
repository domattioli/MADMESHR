# Session 15 Plan: Annulus Sub-Loop Feasibility + Mu Recalibration

**Date:** Next session after session 14
**Status:** Final (multi-agent reviewed)
**Budget:** 3 hours

## Context

Session 14 found that (1) rectangle regression was an eval script bug (wrong domain definitions), not the boundary filter, and (2) annulus DQN training reaches 0% completion at 4k steps. Three bottlenecks were identified:
- Single-ref-vertex selection: env picks ONE vertex per step, wasting steps when that vertex has no productive actions
- Density penalty (`mu`): miscalibrated for annulus (n_expected=32, ideal_area=0.014, most elements get mu=-1.0)
- Exploration difficulty: 64-vertex domain with 57 actions requires ~30 valid placements to complete

Type-2 threshold was increased from 0.02 to 0.10, giving 5 valid type-2 actions (from 1). Oracle with threshold=0.10 places 4 type-2 elements, creates pending sub-loops of sizes 7, 3, 18, 3, but still dead-ends on 8-vertex active loop.

### Adversarial Review Summary

**Devil's Advocate #1** attacked:
1. Boundary growth guard already allows boundary-maintaining actions (`>` not `>=`) — relaxation is a no-op **Accepted**: removed from plan, verified code confirms `>` operator
2. SubLoopEnv state representation won't generalize across sub-loop sizes **Accepted**: train on single sub-loop size first, defer generalization
3. Multi-vertex selection requires architecture redesign (44-float state is local to one vertex) **Accepted**: cut from plan entirely, deferred to session 16+
4. Density penalty `mu` is miscalibrated for annulus — most elements get mu=-1.0 **Accepted**: made WS2
5. The minimum viable change is extracting a sub-loop as a standalone domain and testing DQN on it **Accepted**: restructured plan around this

**Scope Realist** computed:
1. WS2 (SubLoopEnv) at 90 min is underscoped — 50 min infrastructure + 67 min training doesn't fit **Accepted**: replaced with extract-and-register approach (20 min setup)
2. Mu recalibration won't fix the reward for narrow-section elements even with n_expected=10 **Accepted**: treat mu fix as improvement not solution, test after feasibility
3. Multi-vertex enumeration requires state representation change — not a "40-minute task" **Accepted**: cut entirely
4. The core question is binary: can DQN complete an 18v sub-loop? Answer before building infrastructure **Accepted**: Phase 1 is the feasibility gate
5. Training alone takes 27-67 min — budget must explicitly account for this **Accepted**: restructured timeline

**Bottom line:** Session 15 is a feasibility test. Extract an 18-vertex sub-loop from the oracle, train DQN on it as a standalone domain. If 0% completion → annulus needs fundamental approach change (multi-vertex architecture). If DQN can complete → proceed to mu recalibration and full pipeline in session 16.

## Workstreams (3, strict priority order)

---

### WS1: Extract and Register Sub-Loop Domain (Priority 1, ~25 min)

**Problem:** The full 64-vertex annulus is too large for DQN exploration. The oracle creates sub-loops of sizes 7, 3, 18, 3 via type-2 placements. An 18-vertex sub-loop is comparable to the rectangle (20v, 9 elements, q=0.464). Can DQN complete it?

**Step 1: Extract sub-loop boundary (10 min)**
- Run oracle with `type2_threshold=0.10` up to the point where the 18-vertex pending loop is created
- Extract the pending loop's boundary coordinates
- Save as `domains/annulus_subloop_18v.npy`
- Verify it's a valid polygon (non-self-intersecting, positive area)

**Step 2: Register as standalone domain (10 min)**
- Add `annulus-subloop-18` to `main.py` domain registry
- Set `max_ep_len=20`, `type2_threshold=0.02` (no type-2 needed within sub-loop), `bnd_dist_threshold=0.03`
- Run `python main.py --domain annulus-subloop-18 --greedy` to establish greedy baseline

**Step 3: Verify (5 min)**
- Run 7-point validation on new domain
- Run `pytest tests/ -v` — 44+ tests pass
- If greedy completes: record quality as baseline
- If greedy doesn't complete: investigate why (geometry too irregular for greedy)

**Decision gate:** If the 18v sub-loop polygon is degenerate (self-intersecting, zero area), try a different sub-loop (7v) or manually construct a clean sub-loop boundary from the annulus geometry.

**Files:** `scripts/annulus_oracle_type2.py`, `main.py`, `domains/annulus_subloop_18v.npy`

---

### WS2: Mu Penalty Recalibration (Priority 2, ~25 min)

**Problem:** The density penalty formula uses `n_expected = len(initial_boundary) / 2`. For a standalone 18v sub-loop, `n_expected = 9`, which gives `ideal_area = total_area / 9`. For the full 64v annulus, `n_expected = 32`, giving `ideal_area = 0.014` — far too small. Most elements on the annulus get mu=-1.0, making per-step rewards negative (~-0.55 typical).

**Step 1: Make n_expected configurable (15 min)**
- Add `n_expected_override` parameter to `DiscreteActionEnv.__init__()` (default `None` = use current formula)
- If `n_expected_override` is set, use it instead of `len(initial_boundary) / 2`
- Add to domain registry and wire through `main.py`
- Set annulus domains to appropriate value (compute from `original_area / mean_edge_length^2`)

**Step 2: Verify no regressions (10 min)**
- Run `pytest tests/ -v` — all tests pass
- Eval rectangle_s5 and octagon checkpoints — quality unchanged
- These domains use the default formula, so behavior is identical

**Decision gate:** If the sub-loop has area small enough that even with n_expected=9, mu is still harsh (all elements < A_min), then disable mu entirely for this domain (`mu_weight=0`) to test if the reward landscape is the bottleneck.

**Files:** `madmeshr/discrete_action_env.py`, `main.py`

---

### WS3: DQN Training on Sub-Loop (Priority 3, ~90 min)

**Problem:** The critical feasibility question: can DQN complete an 18v annulus sub-loop?

**Step 1: Training run (60 min)**
- Config: 12×4 (49 actions, no type-2 within sub-loop), ε-decay 50%, buffer 20k, target update 500, batch 64
- Timesteps: 7500 (at ~400ms/step = 50 min max)
- Save to `checkpoints/annulus_subloop_18v_s15/`

**Step 2: Monitor and kill if plateau (ongoing)**
- At t=2000 (eval #1): if completion=0%, continue (still early)
- At t=4000 (eval #2): if completion=0% AND avg elements < 5, **KILL** — domain is too hard
- At t=6000 (eval #3): if completion > 20%, great — let finish. If still 0%, kill.

**Step 3: Evaluate results (15 min)**
- Run best checkpoint eval: `python main.py --domain annulus-subloop-18 --eval-only --load-path checkpoints/annulus_subloop_18v_s15/best`
- Record: quality, element count, completion
- Run 7-point validation
- Write results to session 15 report

**Decision gates:**
- Completion > 0%: sub-loop approach is viable → plan WS for session 16 to build full pipeline (SubLoopEnv with existing_elements, train all sub-loops, assemble)
- Completion = 0% at 4k steps, avg elements < 5: reward landscape is hostile even for small domain → investigate mu, or try different sub-loop, or reconsider approach
- Completion = 0% but avg elements >= 8: agent progresses but can't close the last few vertices → likely fixable with reward tuning or longer training

**Files:** `main.py`, `checkpoints/annulus_subloop_18v_s15/`

---

### WS4: Documentation + Report (15 min, after training)

**Steps:**
1. Write session_15_report.md
2. Run 7-point validation on all domains
3. Push images to `tests/output/`
4. Create session_16_plan.md using adversarial methodology

---

## Execution Order

```
WS1: Extract Sub-Loop Domain (25 min)
  |
  +-- Run oracle, extract 18v pending loop (10 min)
  +-- Register as domain, run greedy (10 min)
  +-- Verify + validate (5 min)
  +-- DECISION GATE:
        Sub-loop is valid polygon → proceed
        Sub-loop is degenerate → try 7v or manual construction
  |
WS2: Mu Recalibration (25 min)
  |
  +-- Add n_expected_override parameter (15 min)
  +-- Verify no regressions (10 min)
  |
WS3: DQN Training on Sub-Loop (90 min)
  |
  +-- Start training (t=0)
  +-- Monitor at t=2000, t=4000, t=6000
  +-- DECISION GATE:
        0% completion at t=4000 + <5 elements → KILL
        Any completion → let finish to 7500
  +-- Evaluate best checkpoint (15 min)
  |
WS4: Documentation (15 min)
  |
  +-- Report, validation, push

Buffer: ~25 min across session
```

## What NOT to Do

- **Do not build SubLoopEnv with existing_elements parameter.** If the standalone sub-loop domain works, this infrastructure is session 16 work. If it doesn't work, the infrastructure would be wasted.
- **Do not implement multi-vertex selection.** This requires an architecture redesign (state representation is local to one vertex). Session 16+ scope at earliest.
- **Do not relax the boundary growth guard.** The guard already allows boundary-maintaining actions (uses `>`, not `>=`). Relaxation is a no-op.
- **Do not train on full 64-vertex annulus.** Session 14 proved this doesn't work.
- **Do not run parallel TF training.** OOM on RTX 3060.
- **Star is CUT.** At geometry ceiling (0.44).

## Success Criteria

| Metric | Session 14 | Target | Stretch |
|--------|-----------|--------|---------|
| Sub-loop greedy | N/A | **Completes** | q > 0.4 |
| Sub-loop DQN completion | N/A (full annulus 0%) | **> 0%** | > 50% |
| Sub-loop DQN quality | N/A | **> 0.25** | > 0.35 |
| Mu recalibration | Fixed at n_expected=N/2 | **Configurable** | Per-step reward > 0 |
| No regressions | 8/8 validation | **8/8 validation** | All domains |
| Tests | 44 | **44+** | New sub-loop tests |

## Risk / Mitigation Table

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| 18v sub-loop polygon is degenerate | Medium | High | Try 7v sub-loop or manually construct clean boundary |
| Sub-loop DQN hits 0% completion | Medium | High | Kill at t=4000, diagnose: reward or exploration? |
| Mu recalibration doesn't help sub-loop | Medium | Low | Mu is secondary to exploration; proceed regardless |
| Training wall-clock exceeds budget | Medium | Medium | Cap at 7500 steps, kill early if plateau |
| Sub-loop from oracle is too irregular for DQN | Medium | High | Run greedy first as sanity check; if greedy can't complete, DQN can't either |
| Code changes break existing domains | Low | High | All new params have backward-compatible defaults; run full test suite after each change |

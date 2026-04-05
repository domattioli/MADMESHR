# Session 12 Plan: Type-2 DQN Integration + H-shape Stability

**Date:** Planned for next session after 2026-04-05
**Status:** Final (multi-agent reviewed)
**Budget:** 3 hours

## Context

Session 11 fixed the critical concave domain validity bug (4 checks, all 8 domains pass 7-point validation). The H-shape domain was updated to 20 vertices with 1-unit spacing. DQN found a 10Q solution (q=0.533, best at 10k, regresses by 15k). L-shape is trivially solved (2Q, q=0.459). Type-2 DQN integration has been deferred for 2 sessions.

### Adversarial Review Summary

Three agents reviewed this plan over 1 round:

**Devil's Advocate** attacked:
1. ❌ WS1 epsilon-fix is a guess without diagnosing the regression cause → **Accepted**: added diagnostic step before re-training
2. ❌ q≥0.45 target may be above geometry ceiling for 20v H-shape → **Accepted**: lowered to q≥0.40, added ceiling diagnostic
3. ❌ Type-2 action-space Discrete(49+K) is broken — action K means different quads at different steps → **Accepted**: redesigned as fixed indexed slots with consistent mapping
4. ❌ Only 1 of 7 annulus pairs passes centroid check → **Accepted**: defer annulus training to session 13, focus on architecture
5. ❌ WS3 star retrain doesn't fit in 30 min → **Accepted**: cut star retrain from WS3

**Scope Realist** computed:
- Floor GPU time: 91 min (30k H + 25 annulus + 22 star)
- Total with coding: ~171 min minimum, 200+ if H needs 50k
- ❌ WS2 has 3 open design questions not budgeted → **Accepted**: resolve design questions IN this plan (below)
- ❌ WS3 is a casualty if WS1 runs long → **Accepted**: WS3 is docs only, no training

**Reward Analyst** found critical issues:
1. ❌ Type-2 per-step reward ≈ +0.225, much lower than type-0/1 (+0.35-0.50) — agent will avoid type-2 → **Accepted**: add type-2 split bonus (+0.3) 
2. ❌ mu=-1.0 on small type-2 quads (narrow strip) — catastrophic → **Accepted**: exempt type-2 from mu penalty
3. ❌ Completion bonus unreachable on annulus during exploration → **Accepted**: defer annulus DQN training; do architecture + unit tests only
4. ❌ No per-loop reward when pending_loop activates → **Accepted**: add sub-loop completion bonus (+2.0)
5. ❌ H-shape regression may be geometry-not-epsilon → **Accepted**: diagnostic first, then fix

## Design Decisions (resolved per scope realist critique)

### Type-2 Action Space Design

**Decision:** Fixed-slot indexed system. Add K_max=8 type-2 action slots after the existing 49. Total action space: Discrete(57).

**Mapping:** At each enumerate step:
1. Find all proximity pairs for current reference vertex
2. Sort by distance (closest first)
3. Map to slots 49..56 (up to 8 type-2 actions)
4. Mask unused slots as invalid

**Why 8:** Annulus has at most 7 pairs at threshold=0.02. 8 gives headroom. Domains with 0 type-2 pairs simply mask all 8 slots.

**Consistency:** The same reference vertex is used for all action types. Type-2 pairs are sorted deterministically (by distance, then by index). This means slot 49 = "closest type-2 pair" consistently within a step. Between steps, the mapping changes (as boundary changes), but this is the same as type-1 where candidates change with the reference vertex.

### Type-2 Reward Adjustments

1. **Type-2 split bonus:** When a type-2 action creates a valid split, add +0.3 to reward (compensates lower eta_e).
2. **Mu exemption:** Type-2 elements skip the density penalty (they are structurally required for narrow strips).
3. **Sub-loop completion bonus:** When a pending loop activates (active loop completed), award +2.0.

## Workstreams (3, strict priority order)

---

### WS1: H-shape DQN Stability Diagnostic + Fix (Priority 1, ~75 min)

**Problem:** 20v H-shape DQN regresses from 100% completion at 10k to 0% at 15k.

**Step 1: Diagnose (15 min)**
- Load best (10k) and final (15k) checkpoints
- Run 5 eval episodes each, print per-step actions, rewards, element types
- Compare: does the 15k model place different elements? Does it get stuck?
- Check replay buffer composition: is it contaminated with early random-policy experiences?

**Step 2: Fix based on diagnosis (15 min)**
- If epsilon-related: try epsilon_decay_frac=0.5 (faster convergence to exploitation)
- If catastrophic forgetting: try target_update_freq from every step to every 100 steps
- If replay contamination: try smaller buffer (10k instead of 100k) so old experiences age out

**Step 3: Retrain (45 min)**
- Train 30k steps with the chosen fix
- Monitor at 5k intervals, kill early if converged or clearly failing

**Verification:** Best checkpoint >= 80% completion AND q >= 0.40

**Files:** `src/trainer_dqn.py` (epsilon schedule), `src/DQN.py` (target update freq)

---

### WS2: Type-2 DQN Architecture (Priority 2, ~90 min)

**Problem:** DQN cannot use type-2 actions. This blocks training on annulus and other domains needing interior splits.

**Step 1: Extend action space (30 min)**
- `DiscreteActionEnv.__init__`: max_actions = 1 + n_angle*n_dist + K_max (57 total)
- `DiscreteActionEnv._enumerate`: after type-0/type-1, enumerate type-2 pairs for current reference vertex
- Sort pairs by distance, map to slots 49..56
- Apply Check A (original boundary) and Check B (current boundary) to type-2 candidates
- Store (action_type=2, (ref_idx, far_idx)) in `_valid_actions`

**Step 2: Handle type-2 in step() (20 min)**
- When action_index >= 49: look up (ref_idx, far_idx), call `_form_type2_element`
- Call `_update_boundary_type2` instead of `_update_boundary`
- Apply Check D (element overlap)
- Apply type-2 reward: eta_e + 0.3*eta_b + 0.3 (split bonus), skip mu

**Step 3: Sub-loop completion bonus (10 min)**
- In pending_loop activation block, add +2.0 to reward

**Step 4: Unit tests (20 min)**
- Test action space size is 57
- Test type-2 actions appear in mask on annulus
- Test type-2 actions masked on square (no proximity pairs)
- Test sub-loop bonus fires on synthetic pending_loop activation

**Step 5: Integration test (10 min)**
- Run 50 random steps on annulus with type-2 enabled
- Verify no crashes, boundary stays valid

**NO annulus DQN training in this session** (per scope realist + reward analyst: completion unreachable during exploration, need curriculum or pre-training first).

**Verification:**
- Action space is Discrete(57)
- Annulus enumerate shows type-2 actions in mask
- 50 random steps on annulus with type-2: no crashes
- All existing 37+ tests still pass

**Files:** `src/DiscreteActionEnv.py`, `src/MeshEnvironment.py` (minor), `tests/test_discrete_env.py`

---

### WS3: Documentation Only (Priority 3, ~15 min)

**Steps:**
1. Update CLAUDE.md with type-2 DQN architecture notes
2. Write session_12_report.md
3. Write session_13_plan.md (with adversarial review)
4. Push all images

**NO training in WS3** (per scope realist: insufficient time).

---

## Execution Order

```
WS1: H-shape Stability (75 min)
  |
  +-- Diagnostic: compare 10k vs 15k checkpoints (15 min)
  +-- Identify fix (15 min)
  +-- Retrain 30k with fix (45 min, run in background)
  |
  WHILE H-SHAPE TRAINS:
  |
WS2: Type-2 Architecture (90 min, coding while GPU busy)
  |
  +-- Extend action space (30 min)
  +-- Handle type-2 in step() (20 min)
  +-- Sub-loop completion bonus (10 min)
  +-- Unit tests (20 min)
  +-- Integration test (10 min)
  |
  DECISION GATE:
    H-shape converged + WS2 tests pass → WS3
    H-shape still failing → diagnose, iterate
  |
WS3: Docs (15 min)
```

## What NOT to Do

- **Do not train DQN on annulus.** The reward analyst showed completion is unreachable during exploration. Architecture + tests only in session 12; training in session 13 with curriculum.
- **Do not retrain star/octagon/circle.** These pass 7-point validation already. Retrain only if validity checks changed behavior.
- **Do not change the reward for type-0/type-1 actions.** Only add type-2-specific adjustments.
- **Do not run parallel TF training.** OOM on RTX 3060.

## Success Criteria

| Metric | Session 11 | Target | Stretch |
|--------|-----------|--------|---------|
| H-shape stability (20v) | Regresses at 15k | **Stable at 30k** | Stable at 50k |
| H-shape quality (20v) | 0.533 (best 10k) | **>= 0.40** | >= 0.50 |
| Type-2 in DQN action space | Not supported | **Working + tested** | Annulus random walk OK |
| Annulus type-2 in mask | N/A | **>= 1 type-2 action visible** | >= 3 |
| Tests passing | 37 | **41+** | 45+ |

## Risk / Mitigation Table

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| H-shape regression is fundamental, not epsilon | Medium | High | Diagnostic step first; if unfixable, reduce scope to 12v |
| Type-2 action space design doesn't work | Low | High | Conservative design (fixed slots, sorted); unit tests before integration |
| Type-2 enumerate too slow on 64v annulus | Medium | Medium | Profile; fall back to threshold increase if >2x overhead |
| Sub-loop bonus distorts type-0/type-1 learning | Low | Medium | Only fires on pending_loop activation, not on normal steps |
| 30k training exceeds time budget | Medium | Medium | Kill at 20k if converged; skip WS3 if needed |

# Session 13 Plan: Annulus DQN Training + Type-2 Threshold Tuning

**Date:** Next session after session 12
**Status:** Final (multi-agent reviewed)
**Budget:** 3 hours

## Context

Session 12 implemented type-2 DQN architecture (Discrete(57) = 49 type-0/1 + 8 type-2 slots), revised H-shape to 24v (crossbar y=1.5-2.5), added DQN stability fixes (hard target updates every 500 steps, 20k buffer, epsilon_decay_frac=0.5), reorganized project structure (scripts/, pyproject.toml, __init__.py), and grew tests from 37 to 44. H-shape 24v training results: TBD (see Step 0).

Type-2 actions are enumerated from all boundary proximity pairs (threshold=0.02), validated with Check A/B, sorted by distance. Annulus shows 1 type-2 action at initial state (only 1 of 7 pairs passes validation). Annulus oracle (greedy type-2 + type-0): 23Q, q=0.420, incomplete (stuck at 21 boundary vertices on active loop).

### Adversarial Review Summary

Three agents reviewed this plan over 1 round:

**Devil's Advocate** attacked:
1. CRITICAL: Oracle is incomplete — sub-loop from bad type-2 split may be unmeshable -> **Accepted**: added sub-loop completability verification before training
2. CRITICAL: WS1/WS2 have circular dependency -> **Accepted**: reordered threshold tuning before training
3. MAJOR: 30k steps likely insufficient for 30v+ sub-loop -> **Accepted**: 10k decision gate + kill early
4. MAJOR: Sub-loop bonus (+2.0) may be too weak for farming prevention -> **Partially accepted**: reward analyst confirmed +2.0 is adequate for 20v sub-loop; start with smaller sub-loop
5. MAJOR: H-shape 24v results unknown -> **Accepted**: Step 0 checks H-shape results
6. MINOR: Test targets vague -> **Accepted**: named specific tests
7. MINOR: Triangle fallback not mentioned -> **Accepted**: verify enabled in curriculum

**Scope Realist** computed:
- WS1 Step 1 (curriculum impl) estimated 20 min but realistic: 45-60 min -> **Accepted**: timeboxed at 45 min with hardcoded boundary fallback
- Total realistic: 165-225 min, borderline for 3h -> **Accepted**: cut WS2 optimization, only profile
- Hidden dependency: threshold tuning informs curriculum design -> **Accepted**: reordered
- Design question: fixed vs random sub-loop for training -> **Accepted**: resolved (fixed start, randomize later)

**Reward Analyst** found:
1. 40v sub-loop: completion/farming ratio ~1.0x (borderline). 20v sub-loop: ratio ~2.5x (healthy) -> **Accepted**: train on smaller sub-loop first
2. original_area MUST reset to sub-loop area or mu will be miscalibrated -> **Accepted**: added as critical implementation check
3. Sub-loop completion bonus (+2.0) is well-calibrated -> **Accepted**
4. max_ep_len should be ~25 for sub-loop (not 70) -> **Accepted**
5. Reward farming controlled by mu, but only if original_area is correct -> **Accepted**: added verification

## Design Decisions (resolved per reviews)

### Curriculum Strategy
- Train on the **smaller sub-loop** (~20v, not 40v) — reward ratio is 2.5x, healthy for learning
- Fixed start state (deterministic type-2 pre-placement) for initial training
- max_ep_len = 25 for sub-loop curriculum
- `original_area` MUST be set to the sub-loop polygon area, not the full annulus area

### Threshold Tuning Strategy
- Test thresholds: 0.01, 0.02, 0.05, 0.10 on annulus
- Pick threshold that gives >= 3 valid type-2 actions at initial state
- All thresholds must pass 7-point validation

## Workstreams (3, strict priority order)

---

### Step 0: H-shape 24v Results Check (10 min)

Review session 12 H-shape training results. If stability fixes (hard target updates, 20k buffer, eps_decay_frac=0.5) failed:
- **Do NOT assume same fixes work for annulus**
- Diagnose what went wrong before proceeding
- If H-shape succeeded: note parameters, apply same to annulus sub-loop

---

### WS1: Type-2 Threshold Tuning + Sub-Loop Verification (Priority 1, ~40 min)

**Problem:** Only 1 of 7 annulus proximity pairs passes validation at threshold=0.02. The resulting split may produce an unmeshable sub-loop.

**Step 1: Threshold sensitivity sweep (15 min)**
- Test thresholds: 0.01, 0.02, 0.05, 0.10 on annulus
- For each: count valid type-2 actions at initial state
- Run 7-point validation on any threshold that produces new valid actions
- Pick threshold with >= 3 valid type-2 actions AND clean validation

**Step 2: Verify sub-loop completability (15 min)**
- For each valid type-2 action at chosen threshold:
  - Execute type-2, get the two sub-loops
  - Run greedy oracle on each sub-loop independently
  - Record: completable? element count? quality?
- Select the type-2 split that produces the most completable sub-loops

**Step 3: Profile type-2 enumerate overhead (10 min)**
- Measure enumerate time on annulus with and without type-2 at chosen threshold
- If > 2x overhead: log it, defer optimization to session 14
- If <= 2x: no action needed

**Verification:**
- Threshold chosen with >= 3 valid type-2 actions on annulus
- At least 1 sub-loop from chosen split is completable by greedy
- 7-point validation passes at new threshold
- Enumerate overhead measured and logged

**Files:** `src/DiscreteActionEnv.py` (threshold param), `src/MeshEnvironment.py`

---

### WS2: Annulus Sub-Loop Curriculum Training (Priority 2, ~90 min)

**Problem:** Annulus completion is unreachable during random exploration. Curriculum approach: pre-place type-2 element to create smaller sub-loops, train DQN on the manageable sub-loop.

**Step 1: Implement curriculum reset (45 min, timeboxed)**
- Add curriculum domain option that:
  1. Resets annulus boundary
  2. Executes the pre-selected type-2 action deterministically
  3. Extracts the smaller sub-loop as the training boundary
  4. Sets `original_area` to the sub-loop polygon area (CRITICAL for mu calibration)
  5. Sets max_ep_len = 25
- **Fallback if timeboxed:** hardcode the post-split boundary vertices as a static domain

**Step 2: Verify curriculum (5 min)**
- Reset curriculum domain, check:
  - Boundary is the smaller sub-loop
  - Action mask has valid type-0/type-1 actions
  - original_area matches sub-loop area
  - Triangle fallback is enabled

**Step 3: Train DQN on sub-loop (35 min)**
- 30k steps, same hyperparams as H-shape stability fix (hard target updates/500, 20k buffer, eps_decay_frac=0.5)
- **Decision gate at 10k:** if 0% completion, investigate:
  - Is the sub-loop completable at all?
  - Is the episode too long?
  - Kill and diagnose rather than burning remaining time
- **Decision gate at 20k:** if < 30% completion, kill and reduce sub-loop size

**Step 4: Full annulus test (5 min)**
- If sub-loop DQN achieves >= 50% completion:
  - Test: oracle type-2 + DQN sub-loops on full annulus
  - Record: complete? quality?

**Verification:**
- Sub-loop DQN: >= 50% completion, q >= 0.30
- Full annulus with oracle type-2 + DQN: >= 1 complete mesh (stretch)

**Files:** `main.py` (curriculum domain), `src/DiscreteActionEnv.py`

---

### WS3: Documentation Only (Priority 3, ~15 min)

**Steps:**
1. Write session_13_report.md
2. Write session_14_plan.md (adversarial review)
3. Push all images
4. Name specific new tests:
   - test_curriculum_reset_produces_valid_sub_loop
   - test_threshold_sweep_valid_elements
   - test_sub_loop_original_area_correct

**NO training in WS3.**

---

## Execution Order

```
Step 0: Check H-shape 24v results (10 min)
  |
  DECISION GATE:
    H-shape stable → proceed with same hyperparams for annulus
    H-shape failed → diagnose, adjust before WS2
  |
WS1: Threshold Tuning + Sub-Loop Verification (40 min)
  |
  +-- Threshold sweep (15 min)
  +-- Sub-loop completability check (15 min)
  +-- Profile enumerate overhead (10 min)
  |
  DECISION GATE:
    >= 3 type-2 actions AND completable sub-loop → WS2
    No completable sub-loop → increase threshold, try again
    Still no → defer annulus to session 14, work on H-shape instead
  |
WS2: Annulus Sub-Loop Training (90 min)
  |
  +-- Curriculum implementation (45 min, timeboxed)
  +-- Verify curriculum (5 min)
  +-- Train 30k with 10k/20k decision gates (35 min)
  +-- Full annulus test if sub-loop works (5 min)
  |
WS3: Docs (15 min)
```

## What NOT to Do

- **Do not retrain star/octagon/circle/rectangle.** They pass validation already.
- **Do not run parallel TF training.** OOM on RTX 3060.
- **Do not change type-0/type-1 reward formulas.** Only tune type-2-specific parameters.
- **Do not train on the 40v sub-loop first.** Reward ratio is borderline (1.0x). Start with 20v.
- **Do not forget to reset original_area** to sub-loop area in curriculum. This will silently break mu calibration.

## Success Criteria

| Metric | Session 12 | Target | Stretch |
|--------|-----------|--------|---------|
| Type-2 actions on annulus | 1 | **>= 3** | >= 5 |
| Sub-loop completable by greedy | Unknown | **>= 1** | Both sub-loops |
| Sub-loop DQN completion | N/A | **>= 50%** | >= 80% |
| Sub-loop DQN quality | N/A | **>= 0.30** | >= 0.40 |
| Full annulus (oracle + DQN) | N/A | N/A | >= 1 complete |
| Type-2 enumerate overhead | Unknown | **< 2x** | < 1.5x |
| Tests passing | 44 | **47+** | 50+ |

## Risk / Mitigation Table

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| No threshold gives >= 3 valid type-2 actions | Medium | High | Try thresholds up to 0.20; if still < 3, consider widening min_gap |
| Sub-loop not completable by greedy | Medium | High | Try different type-2 splits; increase threshold; defer if no solution |
| Curriculum implementation exceeds 45 min | Medium | Medium | Fallback: hardcode post-split boundary as static domain |
| 30k training insufficient for 20v sub-loop | Low | Medium | Decision gates at 10k/20k; H-shape 24v took ~10k to reach 100% |
| original_area not reset in curriculum | Low | Critical | Explicit verification step before training |
| H-shape 24v training failed | Medium | Medium | Diagnose before applying same hyperparams to annulus |

# Session 10 Plan: Pan et al. Validation + Boundary Growth Fix

**Date:** Planned for next session after 2026-04-04
**Status:** Final (adversarial-reviewed)
**Budget:** 3 hours

## Context

Session 9 results:
1. **Type-2 prototype works** but only 1/7 coincident pairs produces valid quads after centroid-in-domain check. Oracle places 16Q on annulus (q=0.487, 34v remaining).
2. **Critical bug discovered in oracle**: type-0 quads on narrow strips cause infinite boundary growth (ref=4, bnd 38→∞). Fixed in oracle with save/restore guard but the **core MeshEnvironment does NOT have this guard** — the DQN training loop is vulnerable to the same issue.
3. **Star slow epsilon**: q=0.393, below 0.405 baseline. Not worth pursuing.
4. **User priority for session 10**: validate algorithm against Pan et al. paper, recreate a test domain.
5. 24/24 tests passing. 7-point mesh validation: 6/7 (1 false positive).

### Adversarial Review Summary

**Round 1 critiques (incorporated):**
- "WS1 is a reading exercise" → Restructured: WS1 now produces concrete code fixes, not a checklist. Paper comparison is focused on 3 specific, testable claims.
- "You don't know if Pan et al. publishes geometry" → WS2 uses a standard geometry (L-shape or similar) that can be independently constructed if paper doesn't give coordinates. The L-shape is already in main.py.
- "Comparison metrics undefined" → WS2 has specific pass/fail criteria based on completion rate and quality thresholds.
- "Centroid check is a symptom, not the disease" → WS3 dropped. The consumed-vertex-in-domain-polygon issue needs a cleaner architectural fix (tracked as known issue, not this session).
- "Hidden dependency between WS1 and WS2" → Decision gate added: if WS1 finds critical reward deviation, WS2 pivots to fixing it.
- "Time budget unrealistic" → WS3 dropped entirely. WS1 scoped to 3 specific comparisons. WS2 uses existing L-shape domain (no new implementation needed).

### Key architectural issue to address

The boundary growth bug (type-0 placing degenerate spanning quads that add +4 vertices/step) exists in the **core environment**, not just the oracle. The DQN training loop in `DiscreteActionEnv.step()` calls `_update_boundary()` without checking for boundary growth. On the annulus domain, this means the DQN agent could learn to exploit boundary growth for infinite intermediate rewards. This must be fixed in the environment, not just the oracle.

## Workstreams (2, strict priority order)

---

### WS1: Pan et al. Algorithm Validation (Priority 1, ~120 min)

**Problem:** The codebase was built referencing Pan et al. but has accumulated deviations across 9 sessions. The user wants to verify correctness against the paper. Rather than a reading exercise, this workstream tests 3 specific, falsifiable claims and fixes any deviations found.

**Scope:** 3 focused comparisons, each producing a concrete outcome (code fix or documented-as-intentional).

**Step 1: Reward formula verification (~30 min)**
- The paper defines per-step reward as `r = eta_e + eta_b + mu`
- Our implementation uses `r = eta_e + 0.3 * eta_b + mu` (session 5 change)
- **Task:** Read the paper's exact reward formula (Equations 5-9). Verify:
  - eta_e definition matches (element quality 0-1)
  - eta_b definition matches (boundary angle penalty)
  - mu definition matches (density penalty)
  - Completion bonus: paper likely uses area-based, we use `5 + 10 * mean_q`
- **Output:** For each component, document: paper formula → our formula → deviation → rationale (intentional or bug)

**Step 2: Element formation topology (~30 min)**
- Paper describes advancing-front element types. Our type-0 forms `[ref, ref+1, ref+2, ref-1]` and type-1 forms `[ref, ref+1, new_vertex, ref-1]`
- **Task:** Verify these match the paper's element formation rules. Check:
  - Are the vertex orderings correct (CCW)?
  - Does the paper allow concave quads? (We do, since session 4)
  - What element types does the paper define beyond type-0 and type-1?
  - Does the paper have a boundary-growth prevention mechanism?
- **Output:** Code fix if topology is wrong, or document as matching

**Step 3: State representation comparison (~30 min)**
- Paper defines state vector. We use 44-float enriched state (22 base + 22 enriched).
- **Task:** Compare our state components to the paper's. Key questions:
  - Does the paper include boundary sample points?
  - Does the paper normalize by fan radius?
  - What is the paper's state dimensionality?
- **Output:** Document any missing state features

**Step 4: Boundary growth fix (~30 min)**
- **This is the critical code change.** Add boundary-growth detection to `DiscreteActionEnv.step()`:
  - After `_update_boundary()`, check if boundary grew
  - If grew: undo element placement, treat as invalid action (return -0.1 penalty)
  - This prevents the infinite growth loop discovered in session 9
- **Test:** Add unit test that verifies boundary never grows during a full episode on annulus-layer2
- **Files:** `src/DiscreteActionEnv.py`, `tests/test_discrete_env.py`

**Verification criteria:**
- 3 paper comparisons documented with clear match/deviation/rationale: **REQUIRED**
- Boundary growth fix in DiscreteActionEnv: **REQUIRED**
- Unit test for boundary non-growth: **REQUIRED**
- All existing 24 tests still pass: **REQUIRED**

**Decision gate at 90 min:**
- If reward formula has critical deviation (wrong sign, missing term, wrong scale): pivot remaining time to fixing it. Skip WS2.
- If element topology is wrong: fix it. This may break existing checkpoints (acceptable if the current topology is incorrect per paper).
- If state representation differs: document but don't fix (too risky to change mid-project).

---

### WS2: Test on Pan et al.-style Domain (~60 min)

**Problem:** Need to validate the agent on a domain similar to what Pan et al. tested, to assess whether our results are in the right ballpark.

**Approach:** Use the existing L-shape domain (already in main.py, 6 vertices). This is a standard test domain in mesh generation literature. If Pan et al. uses a different domain, we construct the closest match from their published figures.

**Steps:**

1. **L-shape baseline with current best settings** (~20 min)
   - Train DQN on L-shape with 24×4 grid, 15k steps
   - The L-shape is concave (6 vertices) — tests the agent on non-convex geometry
   - Report: completion rate, mean quality, element count

2. **Compare to known domains** (~10 min)
   - Compare L-shape results to octagon (8v, convex) and star (10v, non-convex)
   - Is quality consistent? Does concavity hurt?

3. **Construct Pan et al. domain if identifiable** (~30 min)
   - If the paper uses a specific domain not in our registry, construct it
   - Common mesh generation test domains: stepped channel, notched rectangle, circular hole
   - Register as new domain in main.py, run greedy baseline

**Verification criteria:**
- L-shape 24×4 trained and evaluated: **REQUIRED**
- Quality comparison table across domains: **REQUIRED**
- Pan et al. domain constructed (if identifiable from paper): **STRETCH**

**Files:** `main.py` (possibly new domain), `checkpoints/l-shape-24x4-s10/`

---

## Execution Order

```
WS1: Pan et al. Validation (120 min)
  |
  ├── Step 1: Reward formula (30 min)
  ├── Step 2: Element topology (30 min)
  ├── Step 3: State representation (30 min)
  │
  ├── DECISION GATE (90 min):
  │     ├── Critical deviation found → Fix it, skip WS2
  │     └── All OK or minor deviations → Proceed to Step 4 + WS2
  │
  └── Step 4: Boundary growth fix (30 min)

WS2: L-shape + Pan et al. Domain (60 min)
  |
  ├── L-shape 24×4 training (20 min)
  ├── Cross-domain comparison (10 min)
  └── Pan et al. domain (30 min, if identifiable)
```

## Risk/Mitigation Table

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Paper not accessible or unclear on formulas | MEDIUM | Blocks WS1 steps 1-3 | Use the equations already referenced in CLAUDE.md as ground truth. Focus on step 4 (boundary growth fix). |
| Reward formula deviation is intentional (0.3 eta_b scaling) | HIGH | WS1 step 1 produces documentation, not a fix | Document why the deviation exists (session 5 eta_b dominance fix). This is fine — it's validation, not regression. |
| L-shape training produces poor results | MEDIUM | WS2 less informative | Still useful data — tells us which domain properties the agent struggles with. |
| Boundary growth fix breaks existing DQN training on other domains | LOW | Regression | The fix only triggers when boundary grows, which shouldn't happen on convex domains. Test on star, octagon, circle before committing. |
| Element topology doesn't match paper | MEDIUM | May require code changes that break checkpoints | Accept checkpoint breakage if paper topology is different — correctness > compatibility. |

## What NOT to Do

- **Don't attempt type-2 DQN integration.** That's session 11+ after geometry is fully proven.
- **Don't retrain star.** Slow epsilon experiment is done. Default epsilon is fine for star.
- **Don't run parallel TF training.** OOM risk on RTX 3060.
- **Don't change the state representation** even if it differs from paper. Too risky to change mid-project. Document and defer.
- **Don't try to improve annulus type-2 coverage.** The consumed-vertex-in-polygon issue needs a proper architectural fix, not threshold tuning.

## Success Criteria

| Metric | Session 9 | Target | Stretch |
|--------|-----------|--------|---------|
| Paper comparison documented (reward, topology, state) | N/A | 3/3 documented | 3/3 with code fixes |
| Boundary growth fix in DiscreteActionEnv | No | Yes | Yes + test |
| L-shape 24×4 quality | 0.459 (12×4, session 5) | > 0.459 | > 0.50 |
| Tests passing | 24 | 25+ | 26+ |
| Pan et al. domain constructed | N/A | Identified | Trained |

## Adversarial Review Process

**Round 1 findings:** Original 3-WS plan attempted paper reading (90 min) + domain creation (60 min) + type-2 improvement (30 min). Paper reading was flagged as a reading exercise producing no code. Domain creation assumed paper publishes geometry. Type-2 improvement was misdiagnosed (centroid check is symptom, not disease). Time budget was 6-8 hours compressed into 3.

**Revised plan:** Dropped WS3 entirely. Restructured WS1 from "read and document" to "test 3 claims and fix deviations." Added boundary growth fix as concrete engineering deliverable. WS2 reuses existing L-shape domain instead of creating new one. Decision gate prevents wasted work if critical deviation found.

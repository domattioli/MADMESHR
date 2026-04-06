# Session 16 Plan: Annulus Full Pipeline + Quality Push (12-Hour Extended Session)

**Date:** 2026-04-07
**Status:** Final (self-reviewed, adversarial agents explored codebase)
**Budget:** 12 hours

## Context

Session 15 confirmed DQN can complete annulus sub-loops:
- 7v sub-loop: 100% completion, q=0.417, 3Q+1T, trained in 50 min
- 9v sub-loop: 100% completion, q=0.368, 6Q+0T, trained in 50 min
- Greedy baselines: 7v q=0.460 (4Q+1T), 9v q=0.450 (13Q+1T)

The oracle with type2_threshold=0.10 creates 5 pending loops (7v, 3v, 18v*, 3v, 6v) + 29v active boundary. The 18v loop is a figure-8 (degenerate), splitting into 6v+9v+4v clean sub-loops.

**Key insight from session 15:** DQN trained on standalone sub-loops produces 100% completion with reasonable quality. The DQN is much more efficient than greedy (9v: 6 elements vs 13).

**Architecture constraints:**
- Single-ref-vertex selection per step (no multi-vertex yet)
- 44-float enriched state is local to one vertex
- RTX 3060: sequential training only (OOM if parallel)
- ~400ms/step → 7500 steps = 50 min, 15k steps = 100 min
- **Training budget: ~10-12 runs max in 12 hours** (with code/eval overhead)

## Strategy

The session has three tiers:

**Tier 1 (MUST DO, ~5 hours):** Complete the annulus sub-loop pipeline. Train DQN on ALL clean sub-loops from the oracle. Build the assembly script that combines type-2 elements + DQN sub-loop meshes into a complete annulus mesh.

**Tier 2 (SHOULD DO, ~4 hours):** Handle the 29v active boundary. This requires either (a) further type-2 splitting to create manageable sub-loops, or (b) training DQN directly on 29v, or (c) a hybrid greedy+DQN approach.

**Tier 3 (NICE TO HAVE, ~3 hours):** Quality optimization, longer training runs, octagon/rectangle quality push, documentation.

## Workstreams (Strict Priority Order)

---

### WS1: Complete Sub-Loop Coverage (Priority 1, ~90 min)

**Problem:** The oracle creates multiple sub-loops of different sizes. We've trained on 7v and 9v. The remaining clean sub-loops are: 6v (from pending loop 0 figure-8 split), 4v (from 18v split), 6v (pending loop 4, CW winding), and the two 3v loops. We need DQN models for each, or verify greedy handles the small ones.

**Step 1: Audit all sub-loops (15 min)**
- Re-run oracle extraction, catalog ALL sub-loops with:
  - Vertex count, area, edge lengths, winding direction
  - Whether greedy can complete them (3v and 4v should be trivial)
  - Whether they need CW→CCW reversal
- Decision: which sub-loops need DQN training vs greedy-only?
  - 3v loops: trivial (single triangle), no training needed
  - 4v loops: single quad, no training needed
  - 6v loops: register as domains, run greedy baseline
  - Already trained: 7v, 9v

**Step 2: Register and baseline remaining domains (15 min)**
- Register 6v sub-loops in main.py
- Fix CW winding to CCW where needed (reverse vertex order)
- Run greedy baselines on all new domains
- Run 7-point validation

**Step 3: Train DQN on 6v sub-loops (60 min, 2 sequential runs)**
- Config: 12x4, 5000 steps (smaller domain = faster convergence), eps-decay 0.5, buffer 15k, target update 400
- Kill at t=2500 if 0% completion
- These should be easy given 7v and 9v success

**Decision gate:** If ANY sub-loop fails greedy and DQN, that sub-loop needs investigation (geometry too irregular, or needs n_expected_override).

**Files:** `scripts/extract_subloop.py`, `main.py`, `domains/annulus_subloop_*.npy`

---

### WS2: Active Boundary Strategy (Priority 2, ~120 min)

**Problem:** After type-2 placements, the oracle leaves a 29v active boundary (area=0.357, 1 duplicate vertex). This is the largest remaining piece. Options:

**Option A: Further type-2 splitting (preferred)**
- Run oracle on just the 29v active boundary with type2_threshold=0.10
- If it creates manageable sub-loops (≤12v), train DQN on those
- This is the recursive application of the sub-loop strategy

**Option B: Direct DQN on 29v**
- Register 29v as standalone domain (clean the duplicate vertex first)
- max_ep_len=35, n_expected_override=14
- Train with 15000 steps (~100 min)
- High risk: session 14 showed 64v doesn't work, but 29v is much smaller

**Option C: Hybrid greedy+DQN**
- Use greedy to place first ~5 elements on easy vertices
- Then hand off to DQN for the remaining ~20v boundary
- Requires new infrastructure (partial initialization)

**Step 1: Try Option A first (30 min)**
- Extract 29v boundary, clean duplicate vertex
- Initialize MeshEnvironment with 29v as initial_boundary
- Run oracle-style type-2 scan: how many valid type-2 pairs at threshold=0.10?
- If ≥2 valid type-2: extract new sub-loops, register as domains
- If 0 valid type-2: fall through to Option B

**Step 2: Execute based on Option A result (90 min)**
- If Option A succeeds: train DQN on resulting sub-loops (2-3 runs @ 50 min each, overlap with evaluation)
- If Option A fails: try Option B (direct 29v training, 100 min, high risk)
  - Kill at t=5000 if 0% completion
  - If 0% at kill: accept that 29v needs multi-vertex selection (session 17 scope)

**Decision gate:** The 29v boundary completion is the session's biggest risk. If neither Option A nor B work, the annulus pipeline will be incomplete — we'll have a mesh with a 29v hole.

**Files:** `scripts/extract_subloop.py`, `main.py`

---

### WS3: Assembly Pipeline (Priority 3, ~90 min)

**Problem:** Need a script that combines oracle type-2 elements + DQN sub-loop meshes into a complete annulus mesh.

**Step 1: Build assembler (45 min)**
- Script: `scripts/assemble_annulus.py`
- Input: annulus-layer2 boundary
- Process:
  1. Run oracle type-2 pass → get type-2 elements + pending loops + active boundary
  2. For each pending loop: load best DQN checkpoint, run deterministic eval, collect elements
  3. For active boundary: run DQN or greedy (depending on WS2 outcome)
  4. Combine all elements
  5. Compute overall quality metrics
  6. Save visualization

**Step 2: End-to-end test (30 min)**
- Run assembler on annulus-layer2
- Validate assembled mesh (no overlapping elements, no gaps, all sub-regions covered)
- Compute coverage: what fraction of the annulus area is meshed?
- Save result to tests/output/

**Step 3: Handle gaps (15 min)**
- If assembly has gaps (unmeshed sub-loops): use greedy fallback for those
- If elements overlap: debug the sub-loop boundary extraction

**Decision gate:** If the assembled mesh covers ≥80% of the annulus with quality ≥0.3, this is a major milestone. If coverage <50%, the pipeline needs fundamental rework.

**Files:** `scripts/assemble_annulus.py`, `tests/output/annulus_assembled.png`

---

### WS4: Quality Optimization (Priority 4, ~90 min)

**Contingent on WS1-3 going well. Skip if behind schedule.**

**Step 1: Longer training on promising sub-loops (60 min)**
- 9v sub-loop: train 15000 steps (vs 7500 in session 15)
- 7v sub-loop: train 15000 steps
- Compare quality improvement: does 2x steps give meaningful quality gain?

**Step 2: Resolution experiment (30 min)**
- Try 24x4 (97 actions) on 9v sub-loop: 7500 steps
- Compare quality vs 12x4
- Decision: is higher resolution worth the slower training?

**Files:** `checkpoints/`, `main.py`

---

### WS5: Existing Domain Quality Push (Priority 5, ~60 min)

**Contingent on WS1-4 complete and time remaining.**

**Step 1: Octagon longer training (50 min)**
- Current best: q=0.579 (5Q, session 8), ceiling ~0.61
- Train 15000 steps from scratch with current hyperparams
- Is there a quality gap to close?

**Step 2: H-shape stability check (10 min)**
- H-shape 24v had instability issues in sessions 12-13
- Run eval on best checkpoint, verify quality

---

### WS6: Documentation + Session 17 Plan (Priority 6, ~60 min, ALWAYS DO)

**Step 1: Write session 16 report (30 min)**
- Results, metrics, failures for all workstreams
- Key decisions and their outcomes
- Updated metrics table

**Step 2: Create session 17 plan (30 min)**
- Use adversarial methodology
- Focus on: multi-vertex selection, gymnasium migration, Pan et al. benchmarks

---

## Execution Timeline (12 hours)

```
Hour 0-1.5:  WS1 — Audit sub-loops, register domains, train 6v (Run 1)
Hour 1.5-2:  WS1 — Train 6v (Run 2, if needed)
             WS2 Step 1 in parallel with eval
Hour 2-3.5:  WS2 — Active boundary strategy (Option A attempt)
Hour 3.5-5:  WS2 — Train on active boundary sub-loops OR Option B
Hour 5-6.5:  WS3 — Build assembly pipeline + end-to-end test
Hour 6.5-7:  CHECKPOINT — Evaluate progress, commit what works
Hour 7-8:    WS4 — Longer training on 9v (if WS1-3 done)
Hour 8-9:    WS4 — Resolution experiment on 9v
Hour 9-10:   WS5 — Octagon quality push (if time permits)
Hour 10-11:  WS5 — Additional optimization OR WS3 gap-filling
Hour 11-12:  WS6 — Report + session 17 plan (ALWAYS)
```

**Checkpoint at hour 7:** If WS1-3 are not complete by hour 7, skip WS4-5 and focus on completing the pipeline. WS6 (documentation) is mandatory regardless.

## Training Run Budget

At ~50-100 min per run (sequential only):

| Run | Domain | Steps | Est. Time | Priority |
|-----|--------|-------|-----------|----------|
| 1 | 6v sub-loop A | 5000 | 35 min | WS1 |
| 2 | 6v sub-loop B (if needed) | 5000 | 35 min | WS1 |
| 3 | Active boundary sub-loop 1 | 7500 | 50 min | WS2 |
| 4 | Active boundary sub-loop 2 | 7500 | 50 min | WS2 |
| 5 | Active boundary 29v direct (if needed) | 15000 | 100 min | WS2-B |
| 6 | 9v longer training | 15000 | 100 min | WS4 |
| 7 | 7v longer training | 15000 | 100 min | WS4 |
| 8 | 9v 24x4 resolution | 7500 | 50 min | WS4 |
| 9 | Octagon | 15000 | 100 min | WS5 |

**Total: 9 possible runs, ~620 min of GPU time = 10.3 hours.** This fits in 12 hours with code/eval overhead. Runs 5-9 are contingent and may be skipped.

## Backup Plans

### Backup A: Active boundary unsplittable (Option A fails)
If the 29v active boundary has 0 valid type-2 pairs:
1. Try threshold=0.15 (more permissive)
2. Try threshold=0.20
3. If still none: accept partial pipeline (sub-loops only, 29v remains unmeshed)
4. Session 17 builds multi-vertex selection for large boundaries

### Backup B: Direct 29v DQN fails (Option B fails)
If DQN can't complete 29v even with 15000 steps:
1. Use greedy on 29v (will likely produce poor quality but may complete)
2. Combine greedy mesh with DQN sub-loop meshes
3. Report quality difference between greedy-29v and DQN-sub-loops

### Backup C: Sub-loop DQN quality too low (<0.25)
If trained sub-loop quality is unacceptably low:
1. Increase n_angle to 24 (more action resolution)
2. Train for 15000 steps instead of 7500
3. Use n_expected_override to tune mu penalty
4. If still low: accept as geometry-limited ceiling

### Backup D: Assembly has gaps/overlaps
If the assembled mesh has geometric errors:
1. Debug boundary vertex matching between sub-loops (floating-point tolerance)
2. Check for shared edges between adjacent sub-loops
3. Fall back to greedy for problematic sub-loops

### Backup E: Behind schedule at hour 7
If WS1-3 not complete by hour 7:
1. Skip WS4 (quality optimization) entirely
2. Skip WS5 (other domains) entirely
3. Focus remaining 5 hours on completing WS1-3
4. WS6 (documentation) is mandatory

## Game Theory: What Might Happen

### Optimistic Path (30% probability)
- All sub-loops train quickly (hour 0-2)
- Active boundary splits cleanly into 2-3 sub-loops (hour 2-3)
- Assembly works first try (hour 5-6)
- Spend hours 7-12 on quality optimization + other domains
- **Outcome:** Complete annulus mesh, quality push on octagon, session 17 focuses on Pan et al.

### Expected Path (50% probability)
- Sub-loop training goes smoothly (hour 0-2)
- Active boundary needs multiple attempts (hour 2-5)
- Assembly has minor issues to debug (hour 5-7)
- Limited time for quality optimization (1-2 runs)
- **Outcome:** Mostly-complete annulus mesh (maybe 29v gap), modest quality improvement

### Pessimistic Path (20% probability)
- 6v sub-loops have unexpected issues (geometry, winding)
- Active boundary has 0 valid type-2 pairs AND direct DQN fails
- Assembly reveals fundamental boundary-matching bugs
- Spend entire session debugging infrastructure
- **Outcome:** Verified sub-loops work (session 15 confirmed), but full pipeline incomplete. Session 17 builds multi-vertex selection.

## What NOT to Do

- **Do not implement multi-vertex selection.** Architecture redesign, session 17+ scope.
- **Do not train on full 64v annulus.** Session 14 proved this doesn't work.
- **Do not run parallel TF training.** OOM on RTX 3060.
- **Do not build SubLoopEnv with existing_elements tracking.** If standalone sub-loops work (they do), assembly can be done geometrically without the env tracking existing elements.
- **Do not commit reports/plans until training results are in.**
- **Do not spend >2 hours on any single debugging issue.** If stuck, move to the next priority.

## Success Criteria

| Metric | Session 15 | Target | Stretch |
|--------|-----------|--------|---------|
| Sub-loops completed | 2 (7v, 9v) | **All clean sub-loops** | All + 29v |
| Annulus coverage | 0% (no assembly) | **≥50% area meshed** | 100% |
| Annulus quality | N/A | **≥0.30 mean** | ≥0.40 |
| Assembly pipeline | N/A | **Working script** | End-to-end automated |
| 9v quality (longer training) | 0.368 | **≥0.40** | ≥0.45 |
| No regressions | 10/10 validation | **10/10+** | All new domains pass |
| Tests | 44 | **44+** | New assembly tests |

## Risk / Mitigation Table

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| 29v active boundary too hard for DQN | High | High | Option A (further split) first; accept gap if both options fail |
| 6v sub-loops have CW winding issues | Medium | Low | Reverse vertex order before registering |
| Assembly boundary mismatch (float precision) | Medium | Medium | Use vertex-matching with tolerance 1e-8 |
| Training runs exceed time budget | Medium | Medium | Hard checkpoint at hour 7; skip WS4-5 |
| Sub-loop quality ceiling too low | Low | Medium | Accept as geometry-limited; higher resolution is WS4 |
| Code changes break existing domains | Low | High | All new params have defaults; run full test suite after each change |
| Duplicate vertices in active boundary | Known | Medium | Clean before registering as domain |

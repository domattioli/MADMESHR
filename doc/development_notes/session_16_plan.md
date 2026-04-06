# Session 16 Plan: DQN Completes Annulus-Layer2 (12-Hour Extended Session)

**Date:** 2026-04-07
**Status:** Final (4-agent reviewed, user-directed revision)
**Budget:** 12 hours

## Goal

**DQN produces a complete, validated quad mesh of annulus-layer2. The RL agent does the meshing, not a greedy script.**

The oracle handles type-2 decomposition (splitting the 64v boundary into sub-loops). DQN does ALL actual meshing within each sub-loop and the active boundary. Greedy is a scouting/diagnostic tool only — never the final mesh.

## Context

Session 15 proved DQN completes sub-loops (7v: q=0.417, 9v: q=0.368, both 100%). The oracle with type2_threshold=0.10 decomposes the annulus into:
- 5 type-2 quad elements (oracle places these — geometric decomposition, not RL)
- Pending sub-loops: 7v (DQN trained), 3v (trivial), 18v (figure-8→6v+9v+4v), 3v (trivial), 6v
- Active boundary: 29v (area=0.357, 1 duplicate vertex) — the hard problem

DQN checkpoints exist for 7v and 9v. Remaining work: train DQN on 6v sub-loops, crack the 29v boundary, assemble everything.

Training: ~400ms/step, 7500 steps ≈ 50 min, 15k steps ≈ 100 min. Sequential only (RTX 3060).

## Phase 1: DQN on All Sub-Loops (Hours 0-3)

**Objective:** Every sub-loop from the oracle has a DQN-trained checkpoint that achieves 100% completion.

### Step 1: Audit and register remaining sub-loops (30 min)
- Re-run oracle extraction, catalog ALL sub-loops
- 3v loops → handled by env automatically (triangle closure), no training needed
- 4v loops → handled by env automatically (quad closure), no training needed
- 6v loops → register as domains, fix CW→CCW winding where needed
- Greedy baseline on each new domain (scouting, not the deliverable)
- Run pytest + validation

### Step 2: Train DQN on 6v sub-loops (90 min, 2 runs)
- Config: 12x4 (49 actions), 5000 steps, eps-decay 0.5, buffer 15k, target update 400
- Kill at t=2500 if 0% completion
- Expected: fast convergence (7v took 1000 steps to reach 100%)

### Step 3: Verify full sub-loop coverage (30 min)
- For every pending sub-loop from the oracle, confirm a DQN checkpoint exists that achieves 100% completion
- Run eval on each checkpoint, record quality
- Any gaps → train or debug before moving to Phase 2

**Decision gate at hour 3:** All sub-loops have DQN checkpoints with 100% completion → Phase 2. If any sub-loop fails → debug (likely geometry issue, try n_expected_override or longer training).

### During training downtime:
- Implement a Pan et al. benchmark domain (pick one from the paper, define boundary, register, run greedy baseline)
- Sketch transformer architecture in `doc/development_notes/transformer_architecture.md` (attention over boundary vertices replacing 44-float local state — design notes, no code)

## Phase 2: Crack the 29v Active Boundary (Hours 3-8)

**Objective:** DQN completes the 29v active boundary, either directly or via further type-2 decomposition into DQN-sized sub-loops.

This is the hard problem. The 29v boundary is 3x larger than anything DQN has completed (9v). Strategy: decompose it further, then train DQN on each piece.

### Step 1: Scout with greedy (30 min)
- Extract the 29v boundary, clean the duplicate vertex
- Register as standalone domain
- Run greedy to understand the geometry: how far does it get? Where does it stall? What does the boundary look like at the stall point?
- Run oracle-style type-2 scan: how many valid type-2 pairs at threshold=0.10? 0.15? 0.20?
- This is scouting — greedy tells us what DQN will face

### Step 2: Decompose via type-2 (60 min)
- If valid type-2 pairs exist on 29v: place them, extract resulting sub-loops
- Check for figure-8 degeneracy (split at duplicate vertices if found)
- Register each new sub-loop as a domain
- Target: sub-loops of ≤12v each (proven DQN range)
- If NO valid type-2 pairs at any threshold: fall through to Step 3

### Step 3: Train DQN on 29v sub-loops (120 min, 2-3 runs)
- For each sub-loop from Step 2: train DQN, 7500 steps
- Kill at t=4000 if 0% completion AND <5 avg elements
- For any sub-loop >15v: try 15000 steps (100 min)

### Step 3-ALT: Direct DQN on 29v (if decomposition fails)
- If type-2 decomposition produces no useful sub-loops:
- Register cleaned 29v as domain, max_ep_len=35, n_expected_override=14
- Train DQN at 15000 steps (~100 min)
- Kill at t=5000 if 0% completion
- If DQN fails on 29v direct: this is the session's known risk. Record the failure, analyze why, plan for session 17 (multi-vertex selection)

**Decision gate at hour 8:**
- All 29v sub-loops completed by DQN → Phase 3 (assembly)
- Some sub-loops completed, some not → Phase 3 with partial coverage, note gaps
- 29v entirely unsolved → Phase 3 with 29v gap, document what was tried, session 17 tackles multi-vertex selection

## Phase 3: Assemble DQN Mesh (Hours 8-10)

**Objective:** Combine oracle type-2 elements + DQN sub-loop meshes into a single complete annulus mesh.

### Step 1: Build assembler (60 min)
- Script: `scripts/assemble_annulus.py`
- Process:
  1. Run oracle type-2 pass → type-2 elements + pending loops + active boundary
  2. For each pending sub-loop: load DQN checkpoint, run deterministic eval, collect elements
  3. For active boundary: load DQN checkpoint(s) for sub-loops from Phase 2
  4. Combine all elements into single mesh
  5. Validate: 7-point check on assembled mesh
  6. Save visualization

### Step 2: End-to-end test (30 min)
- Run assembler on annulus-layer2
- Check: all vertices consumed? No gaps? No overlapping elements?
- Compute: total element count, mean quality, triangle percentage
- Save result to `tests/output/annulus_assembled.png`

### Step 3: Fix gaps (30 min)
- If assembly has gaps from Phase 2 failures: document which sub-regions are missing
- If elements overlap at sub-loop boundaries: debug vertex matching (float tolerance)

**Decision gate at hour 10:**
- Complete mesh → celebrate, move to Phase 3B
- Mesh with gaps → document coverage percentage, move to Phase 4

### Phase 3B: Quality or Stretch Goals (if time, hours 10-11)

Pick one based on what's most valuable:

**Option A: Quality push** — retrain 9v sub-loop at 15k steps or 24x4 resolution. See if quality closes the gap with greedy (0.368 → 0.45?).

**Option B: Pan et al. benchmark** — if a domain was created during downtime, train DQN on it. First cross-paper comparison.

**Option C: Additional domain** — try the pipeline on a second multi-loop domain if available.

## Phase 4: Documentation (Hours 11-12, MANDATORY)

Regardless of outcome:
1. Write `session_16_report.md` — results, metrics, decisions, failures
2. Run 7-point validation on ALL domains
3. Push mesh images to `tests/output/`
4. Create `session_17_plan.md` using adversarial methodology
5. Commit and push everything
6. Update memory files with session learnings

## Training Run Budget

Sequential only. ~50-100 min per run.

| Run | Domain | Steps | Est. Time | Phase |
|-----|--------|-------|-----------|-------|
| 1 | 6v sub-loop A | 5000 | 35 min | Phase 1 |
| 2 | 6v sub-loop B (if needed) | 5000 | 35 min | Phase 1 |
| 3 | 29v sub-loop 1 (from decomposition) | 7500 | 50 min | Phase 2 |
| 4 | 29v sub-loop 2 | 7500 | 50 min | Phase 2 |
| 5 | 29v sub-loop 3 (or 29v direct) | 15000 | 100 min | Phase 2 |
| 6 | Quality push (9v 15k or 24x4) | 15000 | 100 min | Phase 3B |

**6 runs, ~370-470 min GPU time.** Fits in 12 hours with code/eval overhead. Runs 5-6 are contingent.

## Risk Table

| Risk | Likelihood | Impact | Plan |
|------|-----------|--------|------|
| 29v has no valid type-2 pairs | Medium | High | Try thresholds 0.10→0.15→0.20. If none: direct DQN on 29v. |
| Direct DQN on 29v fails (0% at 5k) | High | High | Accept gap. Document failure. Session 17: multi-vertex selection. |
| 6v sub-loop DQN fails | Low | Low | Debug geometry (winding, edge lengths). Should be easy given 7v/9v success. |
| Figure-8 from further type-2 splits | Medium | Medium | Split at duplicate vertices (session 15 code). Recurse. |
| Assembly has boundary gaps | Medium | Medium | Vertex-matching tolerance 1e-8, shared-edge verification. |
| Training exceeds time budget | Medium | Medium | Skip Phase 3B. Hard stop at hour 11 for Phase 4. |
| DQN quality below greedy | Known | Low | Accept for now. Quality push is Phase 3B / session 17 focus. |

## Constraints

- Never run parallel TF training (OOM on RTX 3060)
- Run `pytest tests/ -v` after code changes — maintain 44+ tests
- Always validate with `python scripts/validate_mesh.py`
- Do not spend >2 hours on any single debugging issue
- Do not implement multi-vertex selection (session 17+)
- Do not commit reports until results are in
- Greedy is for scouting/diagnostics only — DQN produces the final mesh

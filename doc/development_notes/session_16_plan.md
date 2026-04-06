# Session 16 Plan: Complete Annulus Mesh (12-Hour Extended Session)

**Date:** 2026-04-07
**Status:** Final (4-agent adversarial review)
**Budget:** 12 hours

## Goal

**Produce a complete, validated quad mesh of annulus-layer2. Every vertex consumed, no gaps.**

Quality is secondary to completion. A mesh with some triangles is acceptable in Phase 1. Quality improvement happens in Phase 2 once completion is proven.

## Context

Session 15 proved DQN completes sub-loops (7v: q=0.417, 9v: q=0.368). But four independent review agents converged on a surprise: **DQN is not the path to completing the full annulus.** The oracle (greedy script) with a relaxed boundary growth guard and triangle fallback is faster and more reliable. DQN is a quality optimizer for already-solvable domains, not a completion tool.

Key facts:
- Oracle gets the 29v active boundary down to ~5v before stalling (growth guard blocks remaining actions)
- Greedy beats DQN on quality for every tested sub-loop (0.460 vs 0.417 on 7v, 0.450 vs 0.368 on 9v)
- The 18v pending loop is a figure-8 (degenerate) — already handled by session 15 splitting code
- DQN checkpoints exist for 7v and 9v sub-loops — free to use in assembly, no training needed
- Training runs take 50-100 min each on the GPU — expensive for uncertain payoff

## Phase 1: Get a Complete Mesh (Hours 0-3)

**Objective:** A complete annulus mesh by any means. Triangles allowed. Quality floor: none.

### Hour 0-1: Build the oracle-assembler

Write `scripts/assemble_annulus.py` that operates directly on `MeshEnvironment` internals:

1. Load annulus-layer2 boundary
2. Run type-2 pass (threshold=0.10) to place type-2 elements and create sub-loops
3. For each pending sub-loop:
   - 3v → single triangle (hardcoded)
   - 4v → single quad (hardcoded)
   - Figure-8 loops → split at duplicate vertices, recurse on each piece
   - 6v-9v → use existing DQN checkpoint if available, else greedy
4. For the active boundary (29v): run greedy with **relaxed growth guard**
   - Change: allow boundary to grow by +1 (type-1 placements that add an interior vertex)
   - This unblocks the 5v stall the current oracle hits
5. Triangle fallback: if stuck at 5v, place a triangle (→4v), then a quad (→0v). If stuck at any N≤6, use triangle fan.
6. Combine all elements, save visualization, run 7-point validation

**The key code change** is in the assembler's growth guard: `len(boundary) > len(saved) + 1` instead of `len(boundary) > len(saved)`. This alone should unblock the stall.

### Hour 1-2: Debug and iterate

Run the assembler. It will probably stall somewhere unexpected. Fix it. Likely issues:
- Intersection checks rejecting valid placements near type-2 split boundaries
- Floating-point vertex matching between sub-loops
- CW winding on some sub-loops (reverse before meshing)

### Hour 2-3: First complete mesh or clear diagnosis

**Decision gate at hour 3:**
- Complete mesh achieved → move to Phase 2 (quality improvement)
- Stalled at a specific boundary (e.g., 12v irregular shape) → register as standalone domain, try greedy with different parameters, then DQN as last resort
- Fundamentally broken (assembly logic wrong) → spend hour 3-4 fixing, push Phase 2 to hour 4

### During GPU idle time (if any training runs needed)

Pre-queued tasks, in order:
1. Write assembly validation tests
2. Register next domain / prepare next training command
3. **Implement a Pan et al. benchmark domain** — pick one from the paper, define the boundary, register it, run greedy baseline. This has been deferred for 15 sessions; idle time is free time to finally do it.
4. **Sketch a transformer architecture** for MADMESHR — write a design doc exploring what a transformer-based policy would look like (attention over boundary vertices, replacing the 44-float local state with a sequence model that sees the full boundary). No code, just architecture notes in `doc/development_notes/transformer_architecture.md`.
5. Incremental documentation

Never poll training status. Use `run_in_background` and wait for notification.

## Phase 2: Improve the Mesh (Hours 3-8)

**Objective:** Replace triangles with quads. Improve quality. Make it publishable.

**Only enter Phase 2 if Phase 1 produced a complete mesh.**

### Step 1: Measure the baseline (30 min)
- Count: how many triangles? What's the mean quality? Where are the worst elements?
- Visualize: save annotated image showing element quality heatmap
- This tells us where to focus

### Step 2: Replace greedy sub-meshes with DQN (2 hours)
- For each sub-loop currently meshed by greedy, swap in the DQN checkpoint mesh
- We already have DQN checkpoints for 7v (q=0.417) and 9v (q=0.368)
- Compare quality: is the DQN mesh actually better or worse per sub-loop?
- For sub-loops without DQN checkpoints: train if the sub-loop has ≥6v and greedy quality <0.35 (otherwise not worth it)

### Step 3: Attack the 29v boundary quality (2 hours)
- If Phase 1 used triangle fallback on the 29v boundary, try to reduce triangle count:
  - Backtracking search: if greedy stalls at Nv, undo 2-3 steps, try different action sequence
  - DFS with depth limit (computationally trivial for ≤10v remaining boundary)
- If Phase 1 split the 29v into sub-loops, train DQN on the largest one
- **Hard time limit: 2 hours.** If no improvement, keep Phase 1 result.

### Step 4: Reassemble with improvements (30 min)
- Rebuild the mesh using best-available sub-mesh for each region
- Run 7-point validation on final assembly
- Save to tests/output/

**Decision gate at hour 8:**
- Quality ≥ 0.35 mean, ≤ 20% triangles → move to Phase 3
- Quality < 0.35 or > 20% triangles → continue Phase 2 improvements until hour 10, skip Phase 3

## Phase 3: Harden and Extend (Hours 8-11)

**Only enter Phase 3 if Phase 2 achieved acceptable quality.**

### Option A: Pan et al. benchmark domains (if mesh quality is good)
- This has been deferred for 15 sessions. With a working pipeline, now is the time.
- Implement 2-3 Pan et al. benchmark domains
- Run greedy + DQN on each
- Record quality comparison

### Option B: Pipeline hardening (if mesh needs more work)
- Make `assemble_annulus.py` a proper reusable tool (CLI args, configurable thresholds)
- Add tests for assembly logic
- Try the pipeline on a different multi-loop domain if one is available

### Option C: Quality push on existing domains (if everything else is done)
- Octagon: train 15k steps, try to close the 0.579→0.61 ceiling gap
- 9v sub-loop: train 15k steps, see if quality passes greedy (0.368→0.45?)

**Pick whichever option has the most project value at hour 8.** Don't pre-commit.

## Phase 4: Documentation (Hours 11-12, MANDATORY)

Regardless of what happened:
1. Write `session_16_report.md` with all results, metrics, decisions
2. Run 7-point validation on all domains (existing + new)
3. Push mesh images to `tests/output/`
4. Create `session_17_plan.md` with adversarial methodology
5. Commit and push everything

## What "Done" Looks Like

| Milestone | Definition | Acceptable? |
|-----------|-----------|-------------|
| Phase 1 complete | All 64 boundary vertices consumed, no gaps in mesh | Triangles OK, any quality |
| Phase 2 complete | ≤20% triangles, mean quality ≥0.35 | Publishable-quality mesh |
| Phase 3 complete | Pipeline reusable OR Pan et al. comparison started | Stretch goal |

## Constraints

- Never run parallel TF training (OOM on RTX 3060)
- Run `pytest tests/ -v` after any code changes (must maintain 44+ tests)
- Always validate mesh output with `python scripts/validate_mesh.py`
- Do not spend >2 hours on any single debugging issue — move on, use fallback
- Do not implement multi-vertex selection (architecture redesign, session 17+)
- Do not commit reports until results are in

## Risk Table

| Risk | Plan |
|------|------|
| Oracle-assembler stalls on 29v at 5v | Relax growth guard (+1), triangle fallback at ≤6v |
| Relaxed guard causes element intersections | Keep intersection checks, only relax growth guard |
| Figure-8 sub-loops from further type-2 splits | Split at duplicate vertices (session 15 code), recurse |
| Assembly has gaps at sub-loop boundaries | Vertex-matching with tolerance 1e-8, shared edges |
| Phase 1 takes >3 hours | Skip Phase 3, extend Phase 1-2 to hour 10 |
| DQN quality worse than greedy on sub-loops | Use greedy, DQN is optional quality experiment |
| Behind schedule at hour 8 | Keep Phase 1 mesh as-is, go directly to Phase 4 |

## What the Agents Said

**Cost Guardian:** Greedy-first is cheapest path. 29v is the token sinkhole — use greedy fallback, don't iterate. Write assembly code during GPU idle time.

**Goal Maximizer:** Build oracle-assembler with relaxed growth guard + triangle fallback. 2-4 hours to complete mesh. Skip all DQN training for completion.

**Big Picture:** The real deliverable is the pipeline proof, not just the mesh. DQN element efficiency (6Q vs 13Q) is a paper result. After annulus: quality parity with greedy, then Pan et al. benchmarks.

**Methods Optimizer:** No type2_threshold completes the oracle. Greedy beats DQN on quality everywhere. Implement backtracking search for stuck boundaries. Skip DQN for this session entirely.

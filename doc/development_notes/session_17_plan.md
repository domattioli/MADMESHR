# Session 17 Plan: Quality Diagnostics + Triangle Reduction + Pan Benchmark

**Date:** 2026-04-08
**Status:** Final (2-agent adversarial review)
**Budget:** 8 hours (conservative)

## Goal

Improve annulus mesh quality (q=0.427 → 0.50+) and reduce triangle rate (22% → <15%). Secondary: first Pan et al. benchmark comparison.

## Context

Session 16 achieved 100% annulus completion via sub-loop curriculum. The assembled mesh has 40 elements (31Q + 9T), mean quality 0.427. Quality bottlenecks are the 7v-b (q=0.315, 355° reflex angle) and 5v-b (q=0.264) sub-loops.

**Key critique from adversarial review:** The quality ceiling may be geometry-limited, not discretization-limited. Doubling the action space (24×4) without first diagnosing the bottleneck wastes training time. Must run quality diagnostics before committing to expensive retraining.

## WS1: Quality Diagnostic on Weak Sub-Loops (1.5 hours)

**Problem:** 7v-b (q=0.315) and 5v-b (q=0.264) drag down mean quality. Is this geometry-limited or training-limited?

### Step 1: Greedy quality ceiling (30 min)
- Run greedy on 7v-b and 5v-b — what quality does greedy achieve?
- If greedy also gets low quality → geometry-limited → action space changes won't help
- If greedy gets higher quality → training gap exists → longer training or hyperparameter tuning could close it
- Also run `quality_diagnostic.py` on both domains

### Step 2: Retrain 7v-b at 15k steps, same 12×4 (50 min)
- Double the training budget (7.5k → 15k), same action space
- If quality improves → training-limited → longer training is the fix
- If quality plateaus → ceiling is action-space or geometry
- Kill at 10k if quality unchanged from current 0.315

### Step 3: If training-limited, try 24×4 on 7v-b (50 min, contingent)
- Only if Step 2 shows improvement but not convergence
- 24×4 increases action space from 49 to 97 — more exploration needed
- 15k steps minimum

**Decision gate:** If greedy quality ≤ DQN quality → skip further quality work on these sub-loops (geometry-limited). Move to WS2.

## WS2: Triangle Reduction (2 hours)

**Problem:** 9 out of 40 elements are triangles (22%). Target: <15%.

### Step 1: Analyze triangle sources (30 min)
- Trace which sub-loops produce triangles in the assembly
- Check: are these triangle-fallback cases (3v boundary → forced triangle)?
- Or did DQN choose a triangle when a quad was available?
- Count: how many triangles are avoidable?

### Step 2: Targeted fixes (90 min)
- For avoidable triangles: try longer training (DQN learns quad solutions with more experience)
- For forced triangles: check if type-0 priority would help (only for concave sub-loops — NOT for convex ones, per prior regression)
- Retrain one sub-loop that produces avoidable triangles at 10k steps
- Reassemble and check triangle count

**Decision gate:** If all triangles are forced (3v boundary remnants) → triangle reduction requires architecture changes (multi-vertex selection). Document and defer.

## WS3: Pan et al. Benchmark Domain (2 hours)

**Problem:** No external comparison exists. Need a benchmark domain from Pan et al. 2023 to compare DQN quality with published results.

### Step 1: Define domain (45 min)
- Choose the L-shaped plate from Pan et al. (closest to our existing L-shape domain, but with their exact vertex placement)
- Read paper for exact coordinates or boundary specification
- Register domain, run validation

### Step 2: Greedy + DQN training (75 min)
- Greedy baseline
- DQN at 10k steps (budget allows one run)
- Compare quality metric to Pan et al. reported values
- Note: Pan et al. uses SAC with continuous action space — quality comparison is indicative, not apples-to-apples

## WS4: NOT INCLUDED — Deferred to Session 18+

**Transformer architecture** is a full-session project requiring new state representation, action space wrapper, and training loop changes. Not appropriate as a workstream in a mixed session. Plan a dedicated 2-session arc for implementation and testing.

## Training Run Budget

| Run | Domain | Steps | Est. Time | Phase |
|-----|--------|-------|-----------|-------|
| 1 | 7v-b quality diagnostic | 15000 | 100 min | WS1 |
| 2 | 7v-b 24×4 (contingent) | 15000 | 100 min | WS1 |
| 3 | Triangle-fix sub-loop | 10000 | 70 min | WS2 |
| 4 | Pan et al. benchmark | 10000 | 70 min | WS3 |

**4 runs max, ~340 min GPU time.** Fits in 8 hours with analysis overhead. Run 2 is contingent.

## Execution Order

1. WS1 Step 1 (greedy diagnostics) — no training, fast
2. WS1 Step 2 (retrain 7v-b) — start training, use downtime for WS2 Step 1 analysis
3. WS2 Step 1 (triangle analysis) — during WS1 training
4. WS1 Step 3 (24×4, contingent on Step 2 results)
5. WS2 Step 2 (triangle fix training)
6. WS3 (benchmark domain)
7. Reassemble annulus with updated checkpoints
8. Documentation (mandatory)

## Risk Table

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| 7v-b is geometry-limited (355° angle) | High | Medium | Accept q=0.315 as ceiling, focus on other sub-loops |
| All triangles are forced (3v remnants) | Medium | Low | Document and defer to multi-vertex selection |
| Pan et al. domain extraction is tedious | Medium | Low | Use our existing L-shape if exact extraction fails |
| 24×4 causes exploration problems | Medium | Medium | Only try if 12×4 at 15k shows improvement |
| Training exceeds time budget | Low | Medium | Skip WS3, hard stop at hour 7 for docs |

## Constraints

- Never run parallel TF training (OOM on RTX 3060)
- Run `pytest tests/ -v` after code changes — maintain 44+ tests
- Always validate with `python scripts/validate_mesh.py`
- Do not spend >2 hours on any single debugging issue
- Do not implement transformer architecture (session 18+)
- Do not commit reports until results are in
- Push mesh images to `tests/output/`

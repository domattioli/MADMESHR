# Session 7 Plan: Transfer Diagnostic + Action Enumeration Speedup

**Date:** Planned for next session after 2026-04-03
**Status:** Final (adversarial-reviewed, two rounds of critique incorporated)

## Context

Session 6 tested two key questions and got clear answers:
1. **Can the current architecture scale to 64-vertex domains?** No. Annulus-layer2 hit 0% completion at 7k steps. The agent places 70 elements (hitting max_ep_len) without completing the mesh.
2. **Does 24×4 angular resolution improve star quality?** Inconclusive. The 24×4 run reached q=0.308 at 10k (undertrained — epsilon still at 0.46). Training was ~2× slower than 12×4.
3. **Baseline reproducibility:** Star 12×4 confirmed at q=0.371, 5Q (identical to session 5).

**Key bottlenecks identified:**
- Training speed on large domains (1k steps/min on 64v vs 5k+ on 10v)
- 44-dim state representation has no global boundary awareness
- Long horizons (70 steps) make credit assignment near-impossible

## Adversarial Review Summary

### Round 1 Critiques (Incorporated)
- **Speedup assumption is optimistic:** Enumeration may not be the sole bottleneck. Need profiling data before committing to vectorization.
- **24×4 decision gate is statistically weak:** q=0.38 vs 0.371 is within noise. Need either multiple seeds or wider threshold.
- **Transfer learning is unvalidated:** No evidence star features generalize to annulus topology. Must diagnose before committing.
- **WS4 (epsilon) should be cut:** Star quality is geometry-limited at 0.44. Epsilon tuning won't breach the ceiling.

### Round 2 Critiques (Incorporated)
- **Plan solves wrong bottleneck:** Annulus-layer2 failure is a representation problem, not a compute problem. 5× faster training still won't help an agent that can't represent 64-vertex boundaries.
- **Transfer diagnostic must come first:** Zero-shot eval (30 min) determines whether curriculum learning has any theoretical basis.
- **Four workstreams is a known failure mode:** Session 6 planned 4, completed ~1. Cut to 2-3 max.
- **24×4 needs clearer success criteria:** What if it converges at 0.35? Plan must specify the action for marginal results.

## Workstreams (Reduced to 3, strict priority order)

### WS1: Zero-Shot Transfer Diagnostic (Priority 1, ~30 min)

**Problem:** Before investing in curriculum learning or speed optimizations, we need to know if learned features transfer across domains at all. This was deferred from session 5 and 6 and is now blocking all annulus-layer2 strategies.

**Steps:**
1. Load best star 12×4 checkpoint (session 6, checkpoints/star-12x4/best)
2. Eval zero-shot on octagon (5 episodes, no training)
3. Eval zero-shot on L-shape (5 episodes, no training)
4. Eval zero-shot on rectangle (5 episodes, no training)
5. Record: completion rate, quality, return. Compare to random baseline (run 5 episodes with random valid actions).

**Verification:**
- Zero-shot completion > 0% on any non-star domain: **POSITIVE SIGNAL** — features generalize, curriculum learning is viable
- Zero-shot completion = 0% on all domains: **NEGATIVE SIGNAL** — features are domain-specific, curriculum learning will not help annulus-layer2
- Zero-shot return > random baseline return: **WEAK POSITIVE** — some feature transfer even without completion

**Decision gate:** If POSITIVE, proceed to WS3 (curriculum). If NEGATIVE, pivot WS3 to diagnosing annulus-layer2 failure modes (state representation analysis).

**Files:** No code changes. Eval only.

**Implementation note:** Use `--load-path checkpoints/star-12x4/best --eval-only` for each domain. For random baseline, use short training with epsilon=1.0 or manually run random rollouts.

---

### WS2: Profile + Speed Up Action Enumeration (Priority 2, ~90 min)

**Problem:** Action enumeration in `DiscreteActionEnv._enumerate_valid_actions()` is the training speed bottleneck. On 64v domains, each step requires enumerating and validating up to 49 actions against a complex boundary. This makes annulus-layer2 training impractical (1k steps/min) and slows 24×4 runs (doubling action count roughly doubles enumeration time).

**Prerequisite:** Profile first to confirm enumeration is the bottleneck (not env stepping, numpy overhead, or DQN forward passes).

**Steps:**
1. **Profile:** Run `cProfile` on a 1k-step annulus-layer2 training run. Identify top-10 functions by cumulative time.
2. **Analyze:** What fraction of time is in `_enumerate_valid_actions()` vs `_form_element()` vs DQN `select_action()` vs other?
3. **Optimize based on profiling results:**
   - If enumeration dominates (>60%): Vectorize intersection checks, precompute boundary segments, batch point-in-polygon tests
   - If `_form_element` dominates: Optimize element formation geometry
   - If DQN forward pass dominates: Consider batch inference or model simplification
4. **Benchmark:** Re-run 1k-step annulus-layer2 training. Measure steps/min improvement.

**Verification:**
- Speedup > 3×: **STRONG PASS** — unlocks annulus-layer2 extended runs and 24×4 completion
- Speedup 1.5-3×: **PARTIAL PASS** — helps 24×4 but annulus-layer2 still slow
- Speedup < 1.5×: **FAIL** — bottleneck is elsewhere, need different approach

**Decision gate:** If speedup > 1.5×, use remaining session time to run the 24×4 star ablation to convergence. If < 1.5×, skip 24×4 and focus remaining time on state representation analysis for annulus-layer2.

**Files:** `src/DiscreteActionEnv.py`, `src/MeshEnvironment.py`

**What NOT to change:** Reward structure, network architecture, training loop. Only optimize geometry computations.

---

### WS3: Curriculum Learning OR Annulus Diagnosis (Priority 3, depends on WS1)

**Path A (WS1 positive — transfer works):**
1. Train octagon 8k → save model
2. Load octagon model → train on star 15k → save model
3. Load star model → train on annulus-layer2 20k (or more if WS2 speedup allows)
4. Compare annulus-layer2 completion rate vs from-scratch baseline (0%)

**Path B (WS1 negative — transfer doesn't work):**
1. Analyze annulus-layer2 failure: log valid action counts per step during a rollout
2. Check: does the agent have enough valid actions? (If consistently 1 = type-0 only, the action space can't reach interior points)
3. Check: what does the state vector look like at step 50+? Is the enriched state still informative?
4. Propose concrete architectural changes (variable-size state, attention, etc.) for session 8

**Verification (Path A):**
- Annulus-layer2 completion > 0%: **BREAKTHROUGH** — curriculum works
- Completion still 0% but elements < 70: **PARTIAL** — agent learned restraint but can't close
- No improvement: **FAIL** — the problem is not feature quality

**Verification (Path B):**
- Clear bottleneck identified (action coverage, state decay, etc.): **PASS** — actionable insight for session 8
- No clear bottleneck: **FAIL** — need fundamental rethink

**Files (Path A):** No code changes. Training runs only.
**Files (Path B):** Possibly `src/DiscreteActionEnv.py` (diagnostic logging).

---

## Execution Order with Decision Gates

```
WS1: Zero-shot transfer diagnostic (30 min)
  |
  ├── POSITIVE (completion > 0%) → WS3 Path A (curriculum)
  └── NEGATIVE (completion = 0%) → WS3 Path B (diagnose annulus)

WS2: Profile + optimize enumeration (90 min, parallel with WS1 analysis)
  |
  ├── Speedup > 1.5× → Run 24×4 star ablation to convergence
  └── Speedup < 1.5× → Skip 24×4, focus on WS3

WS3: Curriculum OR Diagnosis (depends on WS1)
```

## Risk/Mitigation Table

| Risk | Mitigation |
|------|-----------|
| Zero-shot transfer shows 0% on all domains | This is valuable data. Proceed to Path B diagnosis. |
| Profiling reveals bottleneck is NOT enumeration | Optimize whatever IS the bottleneck. The plan adapts. |
| Speedup insufficient for annulus-layer2 | Accept that 64v domains need architectural changes, not just optimization |
| Curriculum training doesn't improve annulus completion | The representation problem is fundamental. Document findings for session 8 architecture redesign. |
| Session runs out of time before WS3 | WS1 (30 min) and WS2 profiling (30 min) are both high-value regardless. Optimization can continue next session. |

## What NOT to Do

- **Don't run annulus-layer2 training without speedup or curriculum.** We proved it doesn't work from scratch at current speed. Repeating the experiment wastes time.
- **Don't change reward structure.** Session 5's reward is stable. Change one variable at a time.
- **Don't increase network size without profiling evidence.** The bottleneck may not be network capacity.
- **Don't plan more than 3 workstreams.** Sessions 5 and 6 both showed that 4 workstreams exceed the session budget.

## Success Criteria

| Metric | Session 6 | Target | Stretch |
|--------|-----------|--------|---------|
| Transfer signal measured | No | Yes | Positive signal |
| Bottleneck profiled | No | Yes | Actionable speedup |
| Enumeration speedup | 1× | ≥ 1.5× | ≥ 3× |
| Star 24×4 converged | No (killed at 12k) | Complete to convergence | q > 0.38 |
| Annulus-layer2 completion | 0% | Any improvement | > 0% |
| Tests passing | 21 | 21 | 21 |

## Adversarial Review Process

**Round 1 findings:** Original plan proposed speed optimization first without profiling, transfer learning without diagnostic, and 4 workstreams despite session 6 showing that's unrealistic. Revised to: diagnostic-first approach, mandatory profiling, cut to 3 workstreams.

**Round 2 findings:** Speed optimization doesn't fix the representation problem. Transfer diagnostic must come before any curriculum work. Success criteria for 24×4 too narrow. Revised to: WS1 is zero-shot diagnostic (cheap, high-information), WS2 is profile-driven optimization, WS3 branches based on WS1 results.

**Remaining acknowledged risks:**
1. Zero-shot transfer may show marginal results (e.g., completion=20% on octagon but 0% on rectangle). Threshold for "positive signal" is any completion > 0%, which may be too generous.
2. Profiling on annulus-layer2 (1k steps) may not be representative of bottleneck on smaller domains. Profile both 10v and 64v domains.
3. Curriculum learning (Path A) may not transfer across domain sizes even if it transfers across similar-sized domains. Star (10v) → annulus-layer2 (64v) is a 6× size jump.

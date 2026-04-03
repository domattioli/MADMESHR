# MADMESHR Planning Methodology

How we plan development sessions for MADMESHR. This methodology is designed to prevent wasted effort by stress-testing plans before execution.

## Inputs to Gather Before Planning

Before drafting any plan, read and synthesize these sources:

1. **SESSION_REPORT.md** -- Previous session writeup. Contains what worked, what failed, current metrics, and open questions. This is the primary context source.
2. **TRAINING_PLAN.md** -- Original staged training plan. May be partially outdated. Evaluate what's still relevant vs superseded by actual results.
3. **PLAN.md** -- Original DQN reformulation design doc. Architecture decisions, phase structure, deferred work list.
4. **Previous session plan** (in `doc/development_notes/`) -- What was planned vs what actually happened. Look for recurring failure patterns.
5. **Current codebase state** -- Read the actual reward functions, hyperparameters, network architecture, and test results. Don't rely on docs being up-to-date.
6. **test_domains.py** -- Available test geometries and their characteristics.

### Key Files to Always Inspect

| File | What to look for |
|------|-----------------|
| `src/DiscreteActionEnv.py` | Reward formulas, action space params, completion logic |
| `src/MeshEnvironment.py` | Action enumeration, quality metric, boundary tracking |
| `src/DQN.py` | Network architecture, hyperparams (gamma, tau, lr) |
| `src/trainer_dqn.py` | Training loop, eval logic, epsilon schedule |
| `main.py` | CLI args, how components are wired together |
| `tests/test_discrete_env.py` | What's tested, what's not |

## The Adversarial Planning Process

### Step 1: Draft Plan (v1)

Produce 3-4 workstreams prioritized by expected impact. For each workstream:
- **Problem statement**: What's wrong and why it matters
- **Proposed changes**: Specific code changes with file paths and line numbers
- **Files affected**: Explicit list
- **Verification**: How to confirm the change worked
- **Time estimate**: Wall-clock including training runs

### Step 2: Devil's Advocate Agent #1

Spawn an agent whose job is to attack the plan:
- What will fail?
- What's underspecified?
- What assumptions are wrong?
- What are we missing entirely?
- Is the quality ceiling achievable with the current action space?
- Are reward magnitudes calibrated correctly?

Feed it the full context: current metrics, reward formulas, action space details, training history. The more specific the context, the better the critique.

### Step 3: Revise Plan (v2)

Incorporate valid critiques. Common revisions:
- Adding a diagnostic step before a speculative change
- Replacing hard thresholds with smooth functions
- Reordering workstreams based on dependencies
- Cutting scope that won't fit

### Step 4: Devil's Advocate Agent #2

A second agent attacks from a different angle:
- Is the scope realistic for the time budget?
- Are there hidden dependencies between workstreams?
- Will the reward changes actually produce the learning signal we expect?
- Quantitative analysis: compute expected reward magnitudes under new formulas
- What's the training wall-clock time? How many iterations can we afford?

### Step 5: Finalize Plan (v3)

Synthesize both critiques into the final plan. Include:
- **Execution order** with dependency graph
- **Decision gates** (e.g., "if diagnostic shows X, do path A; else path B")
- **Risk/mitigation table**
- **Total time budget** with buffer for iteration

Present to user for approval before any code changes.

## Common Failure Patterns to Watch For

These have burned us before:

| Pattern | Example | Prevention |
|---------|---------|-----------|
| Reward tuning a ceiling problem | Tripling quality weight when max achievable quality is 0.45 | Always run quality diagnostic first |
| Reward discontinuities | Hard threshold at quality=0.5 creating 6+ reward cliff | Use smooth functions (quadratic, sigmoid) |
| Killing the progress signal | Reducing area_consumed weight to 0.5x | Keep area weight >= 1.0x |
| Optimistic scoping | 4 workstreams + training in 3 hours | Budget 30min per WS + training time |
| No checkpoints | Losing a good model to a bad hyperparameter change | Always implement save/load first |
| Confounded experiments | Changing reward AND domain simultaneously | Change one variable at a time |

## Session Wrap-Up

At the end of each session:
1. Update `SESSION_REPORT.md` with results, metrics, and what worked/didn't
2. Create next-session plan using this methodology
3. Save the plan to `doc/development_notes/session_N_plan.md`
4. Note any deferred work and why it was deferred

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MADMESHR (Mixed-element ADvanced MESH generator with Reinforcement-learning) uses deep RL to generate 2D quad meshes via an advancing-front method. A Dueling Double DQN agent iteratively selects actions to place quad (and fallback triangle) elements by consuming boundary vertices until the domain is fully meshed.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Train DQN on a domain
python main.py --domain star --timesteps 100000 --save-dir checkpoints/star

# Resume from checkpoint
python main.py --domain star --load-path checkpoints/star/best

# Evaluate only (no training)
python main.py --domain star --load-path checkpoints/star/best --eval-only

# Run tests
pytest tests/ -v

# Run a single test
pytest tests/test_discrete_env.py::TestActionEnumeration::test_type0_always_present -v

# Run greedy-by-quality baseline (no training)
python main.py --domain star --greedy

# Analyze quality ceiling for a domain
python quality_diagnostic.py
```

Available domains: `square`, `octagon`, `circle`, `star`, `l-shape`, `rectangle`

## Architecture

### Data Flow
```
main.py (CLI + domain registry)
  → MeshEnvironment (gym.Env, continuous action space)
    → DiscreteActionEnv (gym.Wrapper, 49 discrete actions + action masking)
      → DQN (Dueling Double DQN with masked Q-values)
        → DQNTrainer (epsilon-greedy training loop with eval + checkpointing)
```

### Key Components

- **`src/MeshEnvironment.py`** (~1080 lines): Core advancing-front environment. Manages polygon boundary, element formation (`_form_element`), boundary updates (`_update_boundary`), state computation, reward calculation, and geometry utilities (intersection, convexity, point-in-polygon). The continuous action space `[-1,1]^3` maps to element type + vertex placement.

- **`src/DiscreteActionEnv.py`**: Wraps MeshEnvironment with a Discrete(49) action space. Action 0 = type-0 (connect adjacent vertices). Actions 1-48 = type-1 on a 12-angle × 4-radial grid (interior vertex placement). Provides `info["action_mask"]` boolean array and enriched 44-float state vector.

- **`src/DQN.py`**: Dueling Double DQN. Architecture: shared trunk → value stream + advantage stream. Q = V + (A - mean_valid(A)). Invalid actions masked to -inf. `MaskedReplayBuffer` stores (state, action, reward, next_state, next_mask, done).

- **`src/trainer_dqn.py`**: `DQNTrainer` with linear epsilon decay, periodic eval, best-model checkpointing.

- **`main.py`**: Domain registry (decorator pattern), CLI argument parsing, orchestrates training/eval for both DQN and legacy SAC.

- **`src/utils/visualization.py`**: Mesh rendering and eval visualization, saves to `output/latest/`.

### State Representation
44-float enriched vector: boundary context (neighbor positions, angles) + fan-shape sample points + area ratio.

### Reward Structure (Pan et al.)
Per-step: `r = eta_e + eta_b + mu` where eta_e = element quality (0 to 1), eta_b = boundary angle penalty (-1 to 0), mu = density penalty (-1 to 0). Completion: flat +10. Quality metric is sqrt(q_edge * q_angle) (0-1 range). See `DiscreteActionEnv.step()` for implementation.

## Key Patterns

- **Action masking**: Invalid actions (self-intersecting elements, vertices outside domain) are masked at enumeration time. DQN sets Q[invalid] = -inf.
- **Domain registry**: `@register_domain(name, description)` decorator in main.py.
- **Triangle fallback**: When boundary has 3 vertices remaining and no valid quad action exists, a triangle element is allowed to complete the mesh.
- **Vectorized geometry**: Action enumeration and point-in-polygon checks are vectorized with NumPy for performance.

## Session Planning (Adversarial Methodology)

All development sessions must follow the adversarial planning process documented in `doc/development_notes/PLANNING_METHODOLOGY.md`. The key steps:

1. **Gather context** from session report, training plan, current code state (especially reward formulas and hyperparams)
2. **Draft plan** with 3-4 prioritized workstreams, each with problem statement, code changes, files, and verification criteria
3. **Devil's advocate #1**: Spawn an agent to attack assumptions — what will fail? what's underspecified? are reward magnitudes correct?
4. **Revise plan** incorporating valid critiques
5. **Devil's advocate #2**: Second agent attacks scope and dependencies — is it realistic? hidden dependencies? quantitative reward analysis?
6. **Finalize** with execution order, decision gates, risk/mitigation table

**Session wrap-up**: Write `doc/development_notes/session_N_report.md` with results/metrics/failures, then create `session_N+1_plan.md` using the adversarial process. Previous reports: `session_2_report.md`, `session_3_report.md`.

**Common failure patterns to avoid**: reward-tuning a ceiling problem, reward discontinuities, killing the area progress signal, optimistic scoping, confounded experiments (change one variable at a time).

## Known Issues

- Reward farming was fixed in session 3, and Pan et al. reward was implemented in session 4. Agent now uses very few elements (4 on star, 3 on octagon) but individual element quality is low (star=0.223). The eta_b boundary penalty may be driving the agent to minimize steps rather than place quality elements.
- Quality ceiling is geometry-limited (star≈0.44, circle≈0.78, octagon≈0.61), not discretization-limited.
- SAC agent (`src/SAC.py`, `src/trainer.py`) is legacy and does not learn effectively — DQN is the active approach.
- Rectangle (20v) scales: 100% completion with 9 elements, quality=0.464.

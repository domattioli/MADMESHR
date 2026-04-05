#!/usr/bin/env python3
"""WS1.3: Type-0 priority ablation.

Evaluates old checkpoints with type-0 priority DISABLED to confirm
it's the cause of regressions, then with it ENABLED to see if greedy
baselines improve.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from madmeshr.mesh_environment import MeshEnvironment
from madmeshr.discrete_action_env import DiscreteActionEnv
from madmeshr.dqn import DQN


from main import DOMAINS

def make_octagon():
    return DOMAINS['octagon']()

def make_star():
    return DOMAINS['star']()

def make_rectangle():
    return DOMAINS['rectangle']()


def eval_one(label, boundary_fn, ckpt_path, num_actions, n_angle, n_dist, type0_priority):
    if not os.path.exists(ckpt_path):
        print(f"  SKIP: {ckpt_path} not found")
        return None

    boundary = boundary_fn()
    env = MeshEnvironment(initial_boundary=boundary)
    env.type0_priority = type0_priority  # toggle
    discrete_env = DiscreteActionEnv(env, n_angle=n_angle, n_dist=n_dist)

    agent = DQN(state_dim=44, num_actions=num_actions)
    agent.load_weights(ckpt_path)

    state, info = discrete_env.reset()
    mask = info["action_mask"][:num_actions]

    ep_return = 0
    done = False
    steps = 0

    while not done and steps < 100:
        if not np.any(mask):
            break
        action = agent.select_action(state, mask, evaluate=True)
        state, reward, done, truncated, info = discrete_env.step(action)
        mask = info["action_mask"][:num_actions]
        ep_return += reward
        steps += 1
        if truncated:
            break

    completed = done and info.get("complete", False)
    mean_q = np.mean(env.element_qualities) if env.element_qualities else 0
    n_quads = sum(1 for e in env.elements if len(e) == 4)
    n_tri = sum(1 for e in env.elements if len(e) == 3)
    return {
        "label": label,
        "type0_priority": type0_priority,
        "mean_quality": mean_q,
        "n_quads": n_quads,
        "n_triangles": n_tri,
        "completed": completed,
        "steps": steps,
    }


def run_greedy(label, boundary_fn, n_angle, n_dist, type0_priority):
    """Run greedy-by-quality baseline."""
    boundary = boundary_fn()
    env = MeshEnvironment(initial_boundary=boundary)
    env.type0_priority = type0_priority
    discrete_env = DiscreteActionEnv(env, n_angle=n_angle, n_dist=n_dist)

    state, info = discrete_env.reset()
    mask = info["action_mask"]
    done = False
    steps = 0

    while not done and steps < 100:
        if not np.any(mask):
            break
        # Greedy: try each valid action, pick the one with best quality
        valid_indices = np.where(mask)[0]
        best_action = valid_indices[0]
        best_q = -999

        for ai in valid_indices:
            # Save state
            saved_bnd = env.boundary.copy()
            saved_elem = list(env.elements)
            saved_eq = list(env.element_qualities)
            saved_pending = [lp.copy() for lp in env.pending_loops] if env.pending_loops else []

            s2, r, d, t, i2 = discrete_env.step(ai)
            q = env.element_qualities[-1] if len(env.element_qualities) > len(saved_eq) else -1

            # Restore
            env.boundary = saved_bnd
            env.elements = saved_elem
            env.element_qualities = saved_eq
            env.pending_loops = saved_pending
            env._invalidate_action_cache()
            discrete_env._enumerate()

            if q > best_q:
                best_q = q
                best_action = ai

        state, reward, done, truncated, info = discrete_env.step(best_action)
        mask = info["action_mask"]
        steps += 1
        if truncated:
            break

    completed = done and info.get("complete", False)
    mean_q = np.mean(env.element_qualities) if env.element_qualities else 0
    n_quads = sum(1 for e in env.elements if len(e) == 4)
    n_tri = sum(1 for e in env.elements if len(e) == 3)
    return {
        "label": label,
        "type0_priority": type0_priority,
        "mean_quality": mean_q,
        "n_quads": n_quads,
        "n_triangles": n_tri,
        "completed": completed,
        "steps": steps,
    }


CKPT_EVALS = [
    ("octagon_s5",      make_octagon,   "checkpoints/octagon_s5/best",              49, 12, 4),
    ("octagon_24x4_s8", make_octagon,   "checkpoints/octagon-24x4-s8/best",         97, 24, 4),
    ("rectangle_s5",    make_rectangle, "checkpoints/rectangle_s5/best",            49, 12, 4),
]

GREEDY_EVALS = [
    ("octagon_greedy",   make_octagon,   12, 4),
    ("rectangle_greedy", make_rectangle, 12, 4),
]


if __name__ == "__main__":
    print("=" * 80)
    print("WS1.3: Type-0 Priority Ablation")
    print("=" * 80)

    # Part A: Checkpoint eval WITH and WITHOUT type-0 priority
    print("\n--- Part A: Checkpoint eval (type-0 ON vs OFF) ---")
    print(f"{'Label':<22} {'Type0':>6} {'Quality':>8} {'Elements':>10} {'Complete':>10}")
    print("-" * 80)

    for label, bfn, ckpt, nact, na, nd in CKPT_EVALS:
        for prio in [False, True]:
            r = eval_one(label, bfn, ckpt, nact, na, nd, type0_priority=prio)
            if r:
                prio_str = "ON" if prio else "OFF"
                elem_str = f"{r['n_quads']}Q+{r['n_triangles']}T"
                comp_str = "YES" if r['completed'] else "NO"
                print(f"{r['label']:<22} {prio_str:>6} {r['mean_quality']:>8.3f} {elem_str:>10} {comp_str:>10}")

    # Part B: Greedy baseline WITH and WITHOUT type-0 priority
    print("\n--- Part B: Greedy baseline (type-0 ON vs OFF) ---")
    print(f"{'Label':<22} {'Type0':>6} {'Quality':>8} {'Elements':>10} {'Complete':>10}")
    print("-" * 80)

    for label, bfn, na, nd in GREEDY_EVALS:
        for prio in [False, True]:
            r = run_greedy(label, bfn, na, nd, type0_priority=prio)
            if r:
                prio_str = "ON" if prio else "OFF"
                elem_str = f"{r['n_quads']}Q+{r['n_triangles']}T"
                comp_str = "YES" if r['completed'] else "NO"
                print(f"{r['label']:<22} {prio_str:>6} {r['mean_quality']:>8.3f} {elem_str:>10} {comp_str:>10}")

    print("\nBaselines (from reports):")
    print("  octagon_s5:       q=0.478, 3Q, 100%")
    print("  octagon_24x4_s8:  q=0.579, 5Q, 100%")
    print("  rectangle_s5:     q=0.464, 9Q, 100%")
    print("  octagon_greedy:   from session reports")
    print("  rectangle_greedy: from session reports")

#!/usr/bin/env python3
"""Evaluate existing checkpoints under current code.

Handles legacy checkpoints (pre-type-2) by creating DQN with matching action
count and truncating the action mask from the current (57 or 105 action) env.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from madmeshr.mesh_environment import MeshEnvironment
from madmeshr.discrete_action_env import DiscreteActionEnv
from madmeshr.dqn import DQN


# Domain boundary constructors (import from main.py to avoid duplication/drift)
from main import DOMAINS

def make_octagon():
    return DOMAINS['octagon']()

def make_star():
    return DOMAINS['star']()

def make_circle():
    return DOMAINS['circle']()

def make_rectangle():
    return DOMAINS['rectangle']()


# (boundary_fn, checkpoint_path, num_actions_at_training_time, n_angle, n_dist)
EVALS = [
    ("octagon_s5",      make_octagon,   "checkpoints/octagon_s5/best",              49, 12, 4),
    ("octagon_24x4_s8", make_octagon,   "checkpoints/octagon-24x4-s8/best",         97, 24, 4),
    ("star_s9",         make_star,      "checkpoints/star-24x4-slow-eps-s9/best",   97, 24, 4),
    ("rectangle_s5",    make_rectangle, "checkpoints/rectangle_s5/best",            49, 12, 4),
]


def eval_checkpoint(label, boundary_fn, ckpt_path, num_actions, n_angle, n_dist):
    """Eval a single checkpoint, handling action space mismatch."""
    if not os.path.exists(ckpt_path):
        print(f"  SKIP: {ckpt_path} not found")
        return None

    boundary = boundary_fn()
    env = MeshEnvironment(initial_boundary=boundary)
    discrete_env = DiscreteActionEnv(env, n_angle=n_angle, n_dist=n_dist)

    agent = DQN(state_dim=44, num_actions=num_actions)
    agent.load_weights(ckpt_path)

    state, info = discrete_env.reset()
    full_mask = info["action_mask"]
    mask = full_mask[:num_actions]  # truncate type-2 slots for old checkpoints

    ep_return = 0
    done = False
    steps = 0
    action_types = {"type0": 0, "type1": 0}

    while not done and steps < 100:
        if not np.any(mask):
            break
        action = agent.select_action(state, mask, evaluate=True)
        if action == 0:
            action_types["type0"] += 1
        else:
            action_types["type1"] += 1

        state, reward, done, truncated, info = discrete_env.step(action)
        full_mask = info["action_mask"]
        mask = full_mask[:num_actions]
        ep_return += reward
        steps += 1
        if truncated:
            break

    completed = done and info.get("complete", False)
    mean_q = np.mean(env.element_qualities) if env.element_qualities else 0
    n_elem = len(env.elements)
    n_quads = sum(1 for e in env.elements if len(e) == 4)
    n_tri = sum(1 for e in env.elements if len(e) == 3)

    return {
        "label": label,
        "return": ep_return,
        "mean_quality": mean_q,
        "n_elements": n_elem,
        "n_quads": n_quads,
        "n_triangles": n_tri,
        "completed": completed,
        "steps": steps,
        "action_types": action_types,
        "checkpoint": ckpt_path,
    }


if __name__ == "__main__":
    print("=" * 75)
    print("WS1.1: Eval existing checkpoints under new code (session 12 fixes)")
    print("=" * 75)

    results = []
    for label, bfn, ckpt, nact, na, nd in EVALS:
        print(f"\n--- {label} (ckpt: {ckpt}, actions: {nact}, grid: {na}x{nd}) ---")
        r = eval_checkpoint(label, bfn, ckpt, nact, na, nd)
        if r:
            results.append(r)
            print(f"  Return:   {r['return']:.2f}")
            print(f"  Quality:  {r['mean_quality']:.3f}")
            print(f"  Elements: {r['n_elements']} ({r['n_quads']}Q+{r['n_triangles']}T)")
            print(f"  Complete: {r['completed']}")
            print(f"  Steps:    {r['steps']}")
            print(f"  Actions:  {r['action_types']}")

    print("\n" + "=" * 75)
    print(f"{'Label':<20} {'Quality':>8} {'Elements':>10} {'Complete':>10}")
    print("-" * 75)
    for r in results:
        print(f"{r['label']:<20} {r['mean_quality']:>8.3f} "
              f"{r['n_quads']:>3}Q+{r['n_triangles']}T"
              f"      {'YES' if r['completed'] else 'NO':>4}")

    print("\nBaselines (from reports):")
    print("  octagon_s5:       q=0.478, 3Q, 100%")
    print("  octagon_24x4_s8:  q=0.579, 5Q, 100%")
    print("  star_s9:          q=0.405, 5Q, 100%")
    print("  rectangle_s5:     q=0.464, 9Q, 100%")

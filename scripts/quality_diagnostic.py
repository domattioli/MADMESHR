#!/usr/bin/env python3
"""Quality ceiling diagnostic: measures max achievable element quality per step.

For each step of a greedy rollout, enumerates all valid actions and computes
the element quality for each. Reports max, mean, and greedy-chosen quality.

Runs at both 12x4 (current) and 24x8 (proposed) resolutions.
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from madmeshr.mesh_environment import MeshEnvironment
from madmeshr.discrete_action_env import DiscreteActionEnv


def quality_for_all_valid_actions(env, discrete_env):
    """Compute element quality for every valid action at current state.

    Returns dict with action_index -> quality for all valid actions.
    """
    qualities = {}
    mask = discrete_env._action_mask
    actions = discrete_env._valid_actions

    ref_vertex = env._cached_ref_vertex
    if ref_vertex is None:
        return qualities

    ref_idx = -1
    for i, v in enumerate(env.boundary):
        if np.allclose(v, ref_vertex, atol=1e-10):
            ref_idx = i
            break
    if ref_idx == -1:
        return qualities

    for action_idx in range(len(mask)):
        if not mask[action_idx]:
            continue
        action_type, new_vertex = actions[action_idx]
        element, valid = env._form_element(ref_vertex, action_type, new_vertex)
        if valid and len(element) == 4:
            q = env._calculate_element_quality(element)
            qualities[action_idx] = q

    return qualities


def run_diagnostic(domain_name, boundary, n_angle, n_dist):
    """Run a greedy rollout (always pick highest-quality valid action)."""
    env = MeshEnvironment(initial_boundary=boundary)
    discrete_env = DiscreteActionEnv(env, n_angle=n_angle, n_dist=n_dist)

    state, info = discrete_env.reset()

    print(f"\n{'='*60}")
    print(f"  {domain_name} @ {n_angle}x{n_dist} = {n_angle*n_dist} type-1 actions")
    print(f"{'='*60}")
    print(f"{'Step':>4} | {'Valid':>5} | {'MaxQ':>6} | {'MeanQ':>6} | {'MinQ':>6} | {'Chosen':>6} | {'Action':>6}")
    print(f"{'-'*4:>4}-+-{'-'*5:>5}-+-{'-'*6:>6}-+-{'-'*6:>6}-+-{'-'*6:>6}-+-{'-'*6:>6}-+-{'-'*6:>6}")

    step = 0
    all_max_q = []
    all_chosen_q = []
    done = False

    while not done and step < 20:
        qualities = quality_for_all_valid_actions(env, discrete_env)

        if not qualities:
            print(f"{step:>4} | {'0':>5} | {'N/A':>6} | {'N/A':>6} | {'N/A':>6} | {'N/A':>6} | {'N/A':>6}")
            break

        q_values = list(qualities.values())
        max_q = max(q_values)
        mean_q = np.mean(q_values)
        min_q = min(q_values)
        all_max_q.append(max_q)

        # Pick the highest-quality action (greedy by quality)
        best_action = max(qualities, key=qualities.get)
        chosen_q = qualities[best_action]
        all_chosen_q.append(chosen_q)

        action_type = "T0" if best_action == 0 else f"T1[{best_action}]"

        print(f"{step:>4} | {len(qualities):>5} | {max_q:>6.3f} | {mean_q:>6.3f} | {min_q:>6.3f} | {chosen_q:>6.3f} | {action_type:>6}")

        state, reward, done, truncated, info = discrete_env.step(best_action)
        step += 1

        if truncated:
            print(f"  TRUNCATED (no valid actions)")
            break

    if all_max_q:
        print(f"\nSummary:")
        print(f"  Steps to complete: {step} (done={done})")
        print(f"  Max quality ceiling: {max(all_max_q):.3f}")
        print(f"  Mean max quality per step: {np.mean(all_max_q):.3f}")
        print(f"  Mean chosen quality: {np.mean(all_chosen_q):.3f}")
        print(f"  Elements placed: {len(env.elements)}")
        if env.element_qualities:
            print(f"  Mean element quality: {np.mean(env.element_qualities):.3f}")

    return all_max_q, all_chosen_q, done


def make_star_domain():
    n_points = 10
    angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    radii = np.where(np.arange(n_points) % 2 == 0, 1.0, 0.4)
    return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])


def make_octagon():
    n = 8
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([np.cos(angles), np.sin(angles)])


def make_circle_16():
    n = 16
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([np.cos(angles), np.sin(angles)])


if __name__ == "__main__":
    domains = [
        ("Star (10v)", make_star_domain()),
        ("Octagon (8v)", make_octagon()),
        ("Circle (16v)", make_circle_16()),
    ]

    resolutions = [
        (12, 4),   # current
        (24, 8),   # proposed
    ]

    print("QUALITY CEILING DIAGNOSTIC")
    print("=" * 60)
    print("Decision gate:")
    print("  max_q >= 0.6 at most steps -> keep 12x4, tune rewards")
    print("  max_q < 0.55 -> increase to 24x8 first")

    for domain_name, boundary in domains:
        for n_angle, n_dist in resolutions:
            run_diagnostic(domain_name, boundary, n_angle, n_dist)

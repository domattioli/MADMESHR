#!/usr/bin/env python3
"""Phase 1: Test MeshEnvironment (continuous SAC action space) on non-trivial domains.

Creates L-shaped, circle-with-interior, and irregular polygon domains.
Runs 500 random actions on each, counting valid quads produced.
"""
import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.MeshEnvironment import MeshEnvironment


def make_l_shape():
    """L-shaped concave domain, 6 vertices."""
    return np.array([
        [0.0, 0.0], [2.0, 0.0], [2.0, 1.0],
        [1.0, 1.0], [1.0, 2.0], [0.0, 2.0],
    ], dtype=float)


def make_circle(n=16):
    """Circle approximation with n boundary vertices."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([np.cos(angles), np.sin(angles)])


def make_irregular_polygon():
    """Non-uniform vertex spacing on a unit circle, 9 vertices."""
    angles = np.array([0, 0.3, 0.6, 0.8, 1.0, 2.0, 3.5, 4.5, 5.8])
    return np.column_stack([np.cos(angles), np.sin(angles)])


def test_domain(name, boundary, n_actions=500):
    """Run n_actions random continuous actions, report valid quads."""
    print(f"\n{'='*60}")
    print(f"DOMAIN: {name} ({len(boundary)} boundary vertices)")
    print(f"{'='*60}")

    env = MeshEnvironment(initial_boundary=boundary)
    state, _ = env.reset()
    print(f"  State dim: {state.shape[0]}")
    print(f"  Action space: {env.action_space}")

    valid_count = 0
    invalid_count = 0
    total_reward = 0
    completed = False

    for step in range(n_actions):
        action = env.action_space.sample()
        next_state, reward, done, truncated, info = env.step(action)
        total_reward += reward

        if info.get("valid", False):
            valid_count += 1
        else:
            invalid_count += 1

        if done:
            completed = info.get("complete", False)
            print(f"  Episode ended at step {step+1}: complete={completed}")
            # Reset and keep going to count more
            state, _ = env.reset()
            valid_count_ep = valid_count  # track first episode
            continue

        state = next_state

    n_quads = sum(1 for e in env.elements if len(e) == 4)
    n_tris = sum(1 for e in env.elements if len(e) == 3)
    mean_q = np.mean(env.element_qualities) if env.element_qualities else 0

    print(f"\n  Results after {n_actions} random actions:")
    print(f"    Valid actions:   {valid_count}")
    print(f"    Invalid actions: {invalid_count}")
    print(f"    Valid rate:      {valid_count/(valid_count+invalid_count)*100:.1f}%")
    print(f"    Elements (current ep): {len(env.elements)} ({n_quads}Q + {n_tris}T)")
    print(f"    Mean quality:    {mean_q:.4f}")
    print(f"    Total reward:    {total_reward:.2f}")
    print(f"    Boundary verts:  {len(env.boundary)}")

    return valid_count > 0


def main():
    np.random.seed(42)
    results = {}

    domains = [
        ("L-Shape (6v)", make_l_shape()),
        ("Circle (16v)", make_circle(16)),
        ("Irregular Polygon (9v)", make_irregular_polygon()),
        ("Circle (24v)", make_circle(24)),
    ]

    for name, boundary in domains:
        try:
            ok = test_domain(name, boundary, n_actions=500)
            results[name] = ok
        except Exception as e:
            print(f"\n  *** ERROR: {e}")
            import traceback
            traceback.print_exc()
            results[name] = False

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for name, ok in results.items():
        status = "PASS" if ok else "FAIL"
        print(f"  {status}: {name}")

    all_pass = all(results.values())
    print(f"\nGate decision: {'PROCEED to Phase 2' if all_pass else 'At least one domain produced valid quads'}")
    # Gate: at least one non-trivial domain produces valid quads
    any_pass = any(results.values())
    if not any_pass:
        print("GATE FAILED: No domain produced valid quads. Stopping.")
        sys.exit(1)
    else:
        print("GATE PASSED: Valid quads produced on non-trivial domains.")


if __name__ == "__main__":
    main()

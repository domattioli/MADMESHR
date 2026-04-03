#!/usr/bin/env python3
"""Test challenging domain geometries for the DQN mesh agent.

Creates four domain types and runs diagnostics:
1. Irregular polygon (non-uniform vertex spacing)
2. L-shaped domain (concave boundary)
3. Star-shaped domain (alternating long/short radii)
4. Elongated rectangle (4:1 aspect ratio)

For each domain, reports:
- Boundary vertices
- Type-0 vs type-1 valid actions at the start
- Whether type-0-only can complete the mesh
- What happens with random actions
"""

import sys
import os
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.MeshEnvironment import MeshEnvironment
from src.DiscreteActionEnv import DiscreteActionEnv


def make_irregular_polygon():
    """Non-uniform vertex spacing on a unit circle.
    Some edges are ~5x longer than others, creating bad quads for type-0."""
    # Angles with highly non-uniform spacing
    angles = np.array([0, 0.3, 0.6, 0.8, 1.0,   # 5 vertices crammed in ~57 deg
                        2.0, 3.5, 4.5, 5.8])      # 4 vertices spread over ~303 deg
    angles = angles * (2 * np.pi / angles[-1])     # normalize so last ~= 2*pi... no, keep raw
    # Actually just use radians directly, ensure CCW and closing
    angles_rad = np.sort(angles)
    r = 1.0
    verts = np.column_stack([r * np.cos(angles_rad), r * np.sin(angles_rad)])
    return verts


def make_l_shaped_domain():
    """L-shaped concave domain (CCW order).

         (0,2)-----(1,2)
           |         |
           |  (1,1)--(2,1)
           |  |        |
         (0,0)------(2,0)
    """
    verts = np.array([
        [0.0, 0.0],
        [2.0, 0.0],
        [2.0, 1.0],
        [1.0, 1.0],
        [1.0, 2.0],
        [0.0, 2.0],
    ])
    return verts


def make_star_domain():
    """Star-shaped domain: 10 vertices alternating between radius 1.0 and 0.4."""
    n_points = 10
    angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    radii = np.where(np.arange(n_points) % 2 == 0, 1.0, 0.4)
    verts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    return verts


def make_elongated_rectangle():
    """4:1 aspect ratio rectangle with many vertices on long sides.
    Width 4, height 1, with extra vertices along the long edges."""
    # Bottom edge: left to right
    bottom = [[x, 0.0] for x in np.linspace(0, 4, 9)]  # 9 points
    # Right edge: bottom to top (skip corners)
    right = [[4.0, y] for y in np.linspace(0, 1, 3)[1:]]  # 2 points
    # Top edge: right to left (skip right-top corner)
    top = [[x, 1.0] for x in np.linspace(4, 0, 9)[1:]]  # 8 points
    # Left edge: top to bottom (skip both corners)
    left = [[0.0, y] for y in np.linspace(1, 0, 3)[1:-1]]  # 1 point

    verts = np.array(bottom + right + top + left)
    return verts


def analyze_domain(name, boundary, interior_points=None):
    """Run full diagnostics on a domain geometry."""
    print("=" * 72)
    print(f"DOMAIN: {name}")
    print("=" * 72)

    if interior_points is None:
        centroid = boundary.mean(axis=0)
        interior_points = np.array([centroid])

    print(f"\nBoundary vertices ({len(boundary)}):")
    for i, v in enumerate(boundary):
        print(f"  [{i:2d}] ({v[0]:7.4f}, {v[1]:7.4f})")

    # Edge lengths
    edges = np.diff(np.vstack([boundary, boundary[0:1]]), axis=0)
    edge_lengths = np.linalg.norm(edges, axis=1)
    print(f"\nEdge lengths: min={edge_lengths.min():.4f}  max={edge_lengths.max():.4f}  "
          f"ratio={edge_lengths.max()/max(edge_lengths.min(), 1e-10):.2f}")

    # Create environment
    base_env = MeshEnvironment(initial_boundary=boundary, interior_points=interior_points)
    env = DiscreteActionEnv(base_env)

    # Reset and get initial action mask
    state, info = env.reset()
    mask = info["action_mask"]
    type0_valid = mask[0]
    type1_valid_count = int(np.sum(mask[1:]))
    total_valid = int(np.sum(mask))

    print(f"\nInitial action analysis:")
    print(f"  Type-0 valid: {bool(type0_valid)}")
    print(f"  Type-1 valid: {type1_valid_count} / {len(mask)-1}")
    print(f"  Total valid:  {total_valid} / {len(mask)}")

    # If type-0 is valid, check its quality
    if type0_valid:
        ref = base_env._select_reference_vertex()
        element, valid = base_env._form_element(ref, 0, None)
        if valid:
            q = base_env._calculate_element_quality(element)
            print(f"  Type-0 element quality: {q:.4f}")

    # --- Test 1: Type-0 only strategy ---
    print(f"\n--- Type-0-only strategy ---")
    base_env_t0 = MeshEnvironment(initial_boundary=boundary.copy(), interior_points=interior_points)
    env_t0 = DiscreteActionEnv(base_env_t0)
    state_t0, info_t0 = env_t0.reset()

    t0_steps = 0
    t0_complete = False
    t0_truncated = False
    t0_qualities = []
    t0_rewards = []

    for step in range(100):  # max 100 steps
        mask_t0 = info_t0["action_mask"]
        if not mask_t0[0]:
            # Type-0 not available
            print(f"  Step {step}: Type-0 not available. boundary={len(env_t0.env.boundary)} verts")
            # Check if any action is valid
            if not np.any(mask_t0):
                print(f"  STUCK: No valid actions at all!")
                t0_truncated = True
                break
            else:
                print(f"  (Type-1 actions available: {int(np.sum(mask_t0[1:]))})")
                break

        state_t0, reward, done, truncated, info_t0 = env_t0.step(0)
        t0_steps += 1
        t0_rewards.append(reward)
        if "element_quality" in info_t0:
            t0_qualities.append(info_t0["element_quality"])

        if done:
            t0_complete = True
            print(f"  Completed in {t0_steps} steps!")
            break
        if truncated:
            t0_truncated = True
            print(f"  Truncated at step {t0_steps} (no valid actions)")
            break

    if t0_complete:
        print(f"  Average quality: {np.mean(t0_qualities):.4f}")
        print(f"  Min quality:     {min(t0_qualities):.4f}")
        print(f"  Total reward:    {sum(t0_rewards):.4f}")
    elif not t0_truncated:
        print(f"  Could NOT complete with type-0 only after {t0_steps} steps")
        if t0_qualities:
            print(f"  Qualities so far: {[f'{q:.3f}' for q in t0_qualities]}")

    # --- Test 2: Random action strategy ---
    print(f"\n--- Random action strategy (5 trials) ---")
    for trial in range(5):
        base_env_r = MeshEnvironment(initial_boundary=boundary.copy(), interior_points=interior_points)
        env_r = DiscreteActionEnv(base_env_r)
        state_r, info_r = env_r.reset()

        r_steps = 0
        r_complete = False
        r_qualities = []
        r_type0_count = 0
        r_type1_count = 0

        for step in range(200):
            mask_r = info_r["action_mask"]
            valid_indices = np.where(mask_r)[0]
            if len(valid_indices) == 0:
                break

            chosen = np.random.choice(valid_indices)
            action_type = 0 if chosen == 0 else 1
            if action_type == 0:
                r_type0_count += 1
            else:
                r_type1_count += 1

            state_r, reward, done, truncated, info_r = env_r.step(chosen)
            r_steps += 1
            if "element_quality" in info_r:
                r_qualities.append(info_r["element_quality"])

            if done:
                r_complete = True
                break
            if truncated:
                break

        status = "COMPLETE" if r_complete else f"STUCK after {r_steps} steps"
        avg_q = np.mean(r_qualities) if r_qualities else 0
        print(f"  Trial {trial+1}: {status} | steps={r_steps} "
              f"t0={r_type0_count} t1={r_type1_count} "
              f"avg_q={avg_q:.4f}")

    print()


def main():
    np.random.seed(42)

    domains = [
        ("1. Irregular Polygon", make_irregular_polygon()),
        ("2. L-Shaped Domain", make_l_shaped_domain()),
        ("3. Star-Shaped Domain", make_star_domain()),
        ("4. Elongated Rectangle (4:1)", make_elongated_rectangle()),
    ]

    for name, boundary in domains:
        try:
            analyze_domain(name, boundary)
        except Exception as e:
            print(f"\n*** ERROR in {name}: {e}")
            import traceback
            traceback.print_exc()
            print()


if __name__ == "__main__":
    main()

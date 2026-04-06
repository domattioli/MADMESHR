"""Extract the 18-vertex pending sub-loop from annulus oracle type-2 run.

Runs the oracle with type2_threshold=0.10, captures the pending loops after
type-2 placements, and saves the 18v sub-loop as a standalone domain.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from madmeshr.mesh_environment import MeshEnvironment


def extract_subloops(type2_threshold=0.10, verbose=True):
    """Run oracle placing only type-2 elements, then return pending loops."""
    bnd = np.load(os.path.join(os.path.dirname(__file__), '..', "domains", "annulus_layer2.npy"))
    env = MeshEnvironment(initial_boundary=bnd)
    env.reset()

    orig_boundary = env.initial_boundary.copy()
    type2_placed = 0

    for step in range(50):
        n = len(env.boundary)
        if n < 5:
            break

        # Only try type-2 placements
        best_type2 = None
        best_q = -1

        for i in range(n):
            pairs = env._find_proximity_pairs(i, threshold=type2_threshold, min_gap=3)
            for far_idx, dist in pairs:
                for ref, far in [(i, far_idx), (far_idx, i)]:
                    elem, consumed = env._form_type2_element(ref, far)
                    if elem is not None:
                        q = env._calculate_element_quality(elem)
                        # Check centroid inside boundary
                        centroid = np.mean(elem, axis=0)
                        if not env._point_in_polygon(centroid, env.boundary):
                            continue
                        if q > best_q:
                            best_q = q
                            best_type2 = (elem, consumed, ref, far, q)

        if best_type2 is None:
            if verbose:
                print(f"Step {step}: No more valid type-2 actions, stopping")
            break

        elem, consumed, ref, far, q = best_type2

        # Place element
        saved_boundary = env.boundary.copy()
        env.elements.append(elem)
        env.element_qualities.append(q)
        ok = env._update_boundary_type2(elem, ref, far, consumed)

        if not ok:
            env.elements.pop()
            env.element_qualities.pop()
            env.boundary = saved_boundary
            if verbose:
                print(f"Step {step}: Type-2 boundary update failed, stopping")
            break

        # Boundary growth guard
        if len(env.boundary) > len(saved_boundary):
            env.elements.pop()
            env.element_qualities.pop()
            env.boundary = saved_boundary
            if verbose:
                print(f"Step {step}: Boundary grew, reverting")
            continue

        type2_placed += 1
        if verbose:
            print(f"Step {step}: type-2 ({ref},{far}), q={q:.4f}, "
                  f"bnd {len(saved_boundary)}->{len(env.boundary)}, "
                  f"pending={len(env.pending_loops)}")

    # Report
    print(f"\n{'='*60}")
    print(f"Type-2 elements placed: {type2_placed}")
    print(f"Active boundary: {len(env.boundary)} vertices")
    print(f"Pending loops: {len(env.pending_loops)}")
    for i, loop in enumerate(env.pending_loops):
        area = env._calculate_polygon_area(loop)
        print(f"  Loop {i}: {len(loop)} vertices, area={area:.6f}")
    print(f"{'='*60}")

    return env


def validate_polygon(vertices):
    """Check polygon is valid: non-self-intersecting, positive area, CCW."""
    n = len(vertices)
    if n < 3:
        return False, "Too few vertices"

    # Check area
    area = 0
    for i in range(n):
        j = (i + 1) % n
        area += vertices[i][0] * vertices[j][1]
        area -= vertices[j][0] * vertices[i][1]
    area /= 2
    if abs(area) < 1e-10:
        return False, f"Zero area ({area})"

    # Check for self-intersection
    def cross2d(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    def segments_cross(p1, p2, p3, p4):
        # Skip adjacent edges
        for a in [p1, p2]:
            for b in [p3, p4]:
                if np.linalg.norm(a - b) < 1e-8:
                    return False
        d1 = cross2d(p3, p4, p1)
        d2 = cross2d(p3, p4, p2)
        d3 = cross2d(p1, p2, p3)
        d4 = cross2d(p1, p2, p4)
        if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
           ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
            return True
        return False

    for i in range(n):
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue  # adjacent
            if segments_cross(vertices[i], vertices[(i+1) % n],
                              vertices[j], vertices[(j+1) % n]):
                return False, f"Self-intersection at edges {i} and {j}"

    return True, f"Valid polygon, area={abs(area):.6f}, winding={'CCW' if area > 0 else 'CW'}"


if __name__ == "__main__":
    env = extract_subloops()

    # Find the ~18v sub-loop
    target_sizes = sorted(
        [(i, len(lp)) for i, lp in enumerate(env.pending_loops)],
        key=lambda x: -x[1]
    )

    print(f"\nSub-loops by size (largest first):")
    for idx, size in target_sizes:
        loop = env.pending_loops[idx]
        valid, msg = validate_polygon(loop)
        print(f"  Loop {idx}: {size}v — {msg}")

    # Save the largest sub-loop (should be ~18v)
    if target_sizes:
        best_idx = target_sizes[0][0]
        best_loop = env.pending_loops[best_idx]
        valid, msg = validate_polygon(best_loop)
        print(f"\nSelected loop {best_idx} ({len(best_loop)}v): {msg}")

        if valid:
            outpath = os.path.join(os.path.dirname(__file__), '..', 'domains', 'annulus_subloop_18v.npy')
            np.save(outpath, best_loop)
            print(f"Saved to {outpath}")

            # Also save a plot
            try:
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(1, 1, figsize=(8, 8))
                closed = np.vstack([best_loop, best_loop[0]])
                ax.plot(closed[:, 0], closed[:, 1], 'b-o', markersize=4)
                for i, v in enumerate(best_loop):
                    ax.annotate(str(i), v, fontsize=7, ha='center', va='bottom')
                ax.set_aspect('equal')
                ax.set_title(f"Annulus Sub-Loop ({len(best_loop)}v)")
                ax.grid(True, alpha=0.3)
                figpath = os.path.join(os.path.dirname(__file__), '..', 'tests', 'output', 'annulus_subloop_18v_boundary.png')
                fig.savefig(figpath, dpi=100, bbox_inches='tight')
                print(f"Plot saved to {figpath}")
                plt.close()
            except Exception as e:
                print(f"Plot failed: {e}")
        else:
            print(f"WARNING: Largest sub-loop is invalid! {msg}")
            print("Trying next largest...")
            for idx, size in target_sizes[1:]:
                loop = env.pending_loops[idx]
                valid, msg = validate_polygon(loop)
                if valid:
                    print(f"  Loop {idx} ({size}v) is valid: {msg}")
                    outpath = os.path.join(os.path.dirname(__file__), '..', 'domains', 'annulus_subloop_18v.npy')
                    np.save(outpath, loop)
                    print(f"  Saved to {outpath}")
                    break

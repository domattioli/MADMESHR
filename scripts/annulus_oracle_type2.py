"""Oracle test for annulus-layer2 using type-2 + type-0 actions.

Greedy strategy: try type-2 first (when proximity candidates exist, prefer
highest quality), then fall back to type-0, then type-1. Rejects elements
that intersect existing elements, cross the boundary, have centroid outside
domain, or cause boundary growth. Reports completion status, element count,
quality, and boundary trajectory.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.MeshEnvironment import MeshEnvironment


def _element_edges(elem):
    """Return list of (p1, p2) edge tuples for an element."""
    n = len(elem)
    return [(elem[i], elem[(i + 1) % n]) for i in range(n)]


def _segments_intersect_strict(p1, p2, p3, p4):
    """Check if segments (p1,p2) and (p3,p4) properly intersect (cross, not just touch)."""
    for a in [p1, p2]:
        for b in [p3, p4]:
            if np.linalg.norm(a - b) < 1e-8:
                return False

    def cross2d(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    d1 = cross2d(p3, p4, p1)
    d2 = cross2d(p3, p4, p2)
    d3 = cross2d(p1, p2, p3)
    d4 = cross2d(p1, p2, p4)

    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True

    return False


def _intersects_existing(new_elem, existing_elements, original_boundary=None):
    """Check if new_elem's edges cross any edge of existing elements or original boundary."""
    new_edges = _element_edges(new_elem)
    for old_elem in existing_elements:
        old_edges = _element_edges(old_elem)
        for ne in new_edges:
            for oe in old_edges:
                if _segments_intersect_strict(ne[0], ne[1], oe[0], oe[1]):
                    return True
    if original_boundary is not None:
        n_bnd = len(original_boundary)
        for ne in new_edges:
            for i in range(n_bnd):
                bp1 = original_boundary[i]
                bp2 = original_boundary[(i + 1) % n_bnd]
                if _segments_intersect_strict(ne[0], ne[1], bp1, bp2):
                    return True
    return False


def _try_place_element(env, elem, update_fn, orig_boundary, verbose_tag="", verbose=True):
    """Try to place an element with all validity checks.

    Returns True if element was placed, False if rejected.
    """
    # Check centroid inside current boundary
    centroid = np.mean(elem, axis=0)
    if not env._point_in_polygon(centroid, env.boundary):
        return False

    # Check no intersection with existing elements or original boundary
    if _intersects_existing(elem, env.elements, orig_boundary):
        return False

    # Save state for rollback
    saved_boundary = env.boundary.copy()
    q = env._calculate_element_quality(elem) if len(elem) == 4 else 0.3
    env.elements.append(elem)
    env.element_qualities.append(q)

    # Apply boundary update
    update_fn()

    # Guard: reject if boundary grew
    if len(env.boundary) > len(saved_boundary):
        env.elements.pop()
        env.element_qualities.pop()
        env.boundary = saved_boundary
        return False

    return True


def run_oracle(max_steps=200, verbose=True):
    bnd = np.load(os.path.join(os.path.dirname(__file__), '..', "domains", "annulus_layer2.npy"))
    env = MeshEnvironment(initial_boundary=bnd)
    env.reset()

    boundary_trajectory = [len(env.boundary)]
    orig_boundary = env.initial_boundary.copy()
    type2_count = 0
    type0_count = 0
    type1_count = 0
    loop_count = 0  # Track which loop we're meshing

    for step in range(max_steps):
        n = len(env.boundary)

        # Activate pending loop if current boundary is complete
        if n < 3 and env.pending_loops:
            env._activate_next_loop()
            loop_count += 1
            n = len(env.boundary)
            if verbose:
                print(f"Step {step}: Activated pending loop {loop_count} "
                      f"({n} vertices, {len(env.pending_loops)} remaining)")
            boundary_trajectory.append(n)

        if n < 3:
            if verbose:
                print(f"Step {step}: Boundary < 3 vertices, mesh complete!")
            break

        if n == 3:
            tri = np.array(env.boundary)
            env.elements.append(tri)
            env.element_qualities.append(0.3)
            env.boundary = np.empty((0, 2))
            if verbose:
                print(f"Step {step}: Triangle remainder")
            boundary_trajectory.append(0)
            continue  # Check for pending loops at top of next iteration

        if n == 4:
            quad = np.array(env.boundary)
            if not env._has_self_intersection(quad):
                q = env._calculate_element_quality(quad)
                env.elements.append(quad)
                env.element_qualities.append(q)
                env.boundary = np.empty((0, 2))
                if verbose:
                    print(f"Step {step}: Final quad, q={q:.4f}")
                boundary_trajectory.append(0)
                continue  # Check for pending loops
            else:
                env.elements.append(np.array([quad[0], quad[1], quad[2]]))
                env.elements.append(np.array([quad[0], quad[2], quad[3]]))
                env.element_qualities.extend([0.2, 0.2])
                env.boundary = np.empty((0, 2))
                if verbose:
                    print(f"Step {step}: Final quad self-intersects, split into 2T")
                boundary_trajectory.append(0)
                continue  # Check for pending loops

        placed = False

        # --- Try type-2 first ---
        best_type2 = None
        best_type2_quality = -1

        for i in range(n):
            pairs = env._find_proximity_pairs(i, threshold=0.02, min_gap=3)
            for far_idx, dist in pairs:
                for ref, far in [(i, far_idx), (far_idx, i)]:
                    elem, consumed = env._form_type2_element(ref, far)
                    if elem is not None:
                        q = env._calculate_element_quality(elem)
                        if q > best_type2_quality:
                            best_type2_quality = q
                            best_type2 = (elem, consumed, ref, far, q)

        if best_type2 is not None:
            elem, consumed, ref, far, q = best_type2

            def update_type2():
                env._update_boundary_type2(elem, ref, far, consumed)

            if _try_place_element(env, elem, update_type2, orig_boundary, verbose=verbose):
                type2_count += 1
                if verbose:
                    print(f"Step {step}: type-2 ({ref},{far}), q={q:.4f}, "
                          f"bnd {n}->{len(env.boundary)}")
                boundary_trajectory.append(len(env.boundary))
                placed = True

        # --- Fall back to type-0 ---
        if not placed:
            # Collect ALL valid type-0 candidates sorted by quality
            type0_candidates = []
            vertices = env._get_vertices_by_angle()
            for ref_vertex in vertices:
                ref_idx = env._find_vertex_index(ref_vertex)
                if ref_idx == -1:
                    continue
                elem, valid = env._form_element_fast(ref_vertex, 0, None, ref_idx)
                if valid:
                    q = env._calculate_element_quality(elem)
                    type0_candidates.append((elem, q, ref_idx))

            type0_candidates.sort(key=lambda x: -x[1])  # best quality first

            for elem, q, ref_idx in type0_candidates:
                def update_type0(e=elem):
                    env._update_boundary(e)

                if _try_place_element(env, elem, update_type0, orig_boundary, verbose=verbose):
                    type0_count += 1
                    if verbose:
                        print(f"Step {step}: type-0 (ref={ref_idx}), q={q:.4f}, "
                              f"bnd {n}->{len(env.boundary)}")
                    boundary_trajectory.append(len(env.boundary))
                    placed = True
                    break

        # --- Fall back to type-1 ---
        if not placed:
            type1_candidates = []
            for ref_vertex in vertices[:5]:
                ref_idx = env._find_vertex_index(ref_vertex)
                if ref_idx == -1:
                    continue
                actions, mask = env._enumerate_for_vertex(ref_vertex, 24, 8, 1 + 24 * 8)
                for k in range(1, len(actions)):
                    if mask[k]:
                        action_type, new_vertex = actions[k]
                        if new_vertex is None:
                            continue
                        elem, valid = env._form_element_fast(ref_vertex, 1, new_vertex, ref_idx)
                        if valid:
                            q = env._calculate_element_quality(elem)
                            type1_candidates.append((elem, q, ref_idx))
                env._invalidate_action_cache()

            type1_candidates.sort(key=lambda x: -x[1])

            for elem, q, ref_idx in type1_candidates:
                def update_type1(e=elem):
                    env._update_boundary(e)

                if _try_place_element(env, elem, update_type1, orig_boundary, verbose=verbose):
                    type1_count += 1
                    if verbose:
                        print(f"Step {step}: type-1 (ref={ref_idx}), q={q:.4f}, "
                              f"bnd {n}->{len(env.boundary)}")
                    boundary_trajectory.append(len(env.boundary))
                    placed = True
                    break

        if not placed:
            if verbose:
                print(f"Step {step}: No valid action, stuck at bnd={n}")
            break

    # Report
    complete = len(env.boundary) < 3 and not env.pending_loops
    mean_q = np.mean(env.element_qualities) if env.element_qualities else 0.0
    n_elem = len(env.elements)
    n_quads = sum(1 for e in env.elements if len(e) == 4)
    n_tris = sum(1 for e in env.elements if len(e) == 3)

    print(f"\n{'='*60}")
    print(f"ORACLE RESULT")
    print(f"{'='*60}")
    print(f"Complete:        {complete}")
    print(f"Elements:        {n_elem} ({n_quads}Q + {n_tris}T)")
    print(f"Type-2 actions:  {type2_count}")
    print(f"Type-0 actions:  {type0_count}")
    print(f"Type-1 actions:  {type1_count}")
    print(f"Loops meshed:    {loop_count + 1}")
    print(f"Mean quality:    {mean_q:.4f}")
    print(f"Boundary:        {boundary_trajectory[0]} -> {len(env.boundary)} vertices")
    print(f"Trajectory:      {' -> '.join(str(x) for x in boundary_trajectory)}")
    print(f"{'='*60}")

    return {
        "complete": complete,
        "n_elements": n_elem,
        "n_quads": n_quads,
        "n_tris": n_tris,
        "type2_count": type2_count,
        "type0_count": type0_count,
        "type1_count": type1_count,
        "mean_quality": mean_q,
        "boundary_trajectory": boundary_trajectory,
        "env": env,
    }


if __name__ == "__main__":
    run_oracle()

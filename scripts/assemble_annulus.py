#!/usr/bin/env python3
"""Assemble a complete DQN mesh for annulus-layer2.

Pipeline:
1. Run oracle type-2 pass on full 64v annulus boundary → type-2 elements + pending loops + active boundary
2. Split figure-8 boundaries at duplicate vertices
3. Further type-2 decompose the active boundary (with validity checks)
4. For each sub-loop/fragment: load DQN checkpoint, run deterministic eval, collect elements
5. Combine all elements into a single mesh
6. Validate and visualize

The oracle handles geometric decomposition. DQN does ALL actual meshing.
"""

import numpy as np
import sys
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from madmeshr.mesh_environment import MeshEnvironment
from madmeshr.discrete_action_env import DiscreteActionEnv
from madmeshr.dqn import DQN


def find_duplicate_vertices(boundary, tol=1e-8):
    """Find pairs of duplicate vertices in a boundary."""
    n = len(boundary)
    dupes = []
    for a in range(n):
        for b in range(a + 1, n):
            if np.linalg.norm(boundary[a] - boundary[b]) < tol:
                dupes.append((a, b))
    return dupes


def split_figure8(boundary, dup_pair):
    """Split a figure-8 boundary at a duplicate vertex pair into two sub-loops."""
    a, b = dup_pair
    # Loop 1: vertices a to b (exclusive of b, since a==b)
    loop1 = boundary[a:b]
    # Loop 2: vertices b to end + 0 to a (exclusive of a, since b==a)
    loop2 = np.concatenate([boundary[b:], boundary[:a]])
    return loop1, loop2


def ensure_ccw(boundary):
    """Flip boundary to CCW if currently CW."""
    n = len(boundary)
    area = 0
    for i in range(n):
        j = (i + 1) % n
        area += boundary[i][0] * boundary[j][1] - boundary[j][0] * boundary[i][1]
    area /= 2
    if area < 0:
        return boundary[::-1]
    return boundary


def validate_polygon_simple(vertices):
    """Quick check: non-self-intersecting, positive area."""
    n = len(vertices)
    if n < 3:
        return False

    area = 0
    for i in range(n):
        j = (i + 1) % n
        area += vertices[i][0] * vertices[j][1] - vertices[j][0] * vertices[i][1]
    area /= 2
    if abs(area) < 1e-10:
        return False

    def cross2d(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    for i in range(n):
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue
            p1, p2 = vertices[i], vertices[(i + 1) % n]
            p3, p4 = vertices[j], vertices[(j + 1) % n]
            # Skip adjacent edges
            skip = False
            for aa in [p1, p2]:
                for bb in [p3, p4]:
                    if np.linalg.norm(aa - bb) < 1e-8:
                        skip = True
            if skip:
                continue
            d1 = cross2d(p3, p4, p1)
            d2 = cross2d(p3, p4, p2)
            d3 = cross2d(p1, p2, p3)
            d4 = cross2d(p1, p2, p4)
            if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
               ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
                return False
    return True


def oracle_type2_pass(env, threshold=0.10, min_quality=0.0, verbose=True):
    """Run oracle placing type-2 elements, return (elements, qualities)."""
    type2_elements = []
    type2_qualities = []

    for step in range(50):
        n = len(env.boundary)
        if n < 5:
            break

        best = None
        best_q = min_quality

        for i in range(n):
            pairs = env._find_proximity_pairs(i, threshold=threshold, min_gap=3)
            for far_idx, dist in pairs:
                for ref, far in [(i, far_idx), (far_idx, i)]:
                    elem, consumed = env._form_type2_element(ref, far)
                    if elem is not None:
                        q = env._calculate_element_quality(elem)
                        centroid = np.mean(elem, axis=0)
                        if not env._point_in_polygon(centroid, env.boundary):
                            continue
                        if q > best_q:
                            best = (elem, consumed, ref, far, q)
                            best_q = q

        if best is None:
            break

        elem, consumed, ref, far, q = best
        saved = env.boundary.copy()
        env.elements.append(elem)
        env.element_qualities.append(q)
        ok = env._update_boundary_type2(elem, ref, far, consumed)

        if not ok or len(env.boundary) > len(saved):
            env.elements.pop()
            env.element_qualities.pop()
            env.boundary = saved
            continue

        type2_elements.append(elem)
        type2_qualities.append(q)
        if verbose:
            print(f"  Oracle step {step}: type-2 ({ref},{far}), q={q:.4f}, "
                  f"bnd {len(saved)}->{len(env.boundary)}, pending={len(env.pending_loops)}")

    return type2_elements, type2_qualities


def further_type2_decompose(boundary, threshold=0.35, min_quality=0.15, verbose=True):
    """Further decompose a boundary via type-2, with validity checks.

    Returns (type2_elements, type2_qualities, pending_loops, active_boundary).
    """
    env = MeshEnvironment(initial_boundary=boundary)
    env.reset()

    type2_elements = []
    type2_qualities = []

    for step in range(20):
        n = len(env.boundary)
        if n < 5:
            break

        candidates = []
        for i in range(n):
            pairs = env._find_proximity_pairs(i, threshold=threshold, min_gap=3)
            for far_idx, dist in pairs:
                for ref, far in [(i, far_idx), (far_idx, i)]:
                    elem, consumed = env._form_type2_element(ref, far)
                    if elem is not None:
                        q = env._calculate_element_quality(elem)
                        if q < min_quality:
                            continue
                        centroid = np.mean(elem, axis=0)
                        if not env._point_in_polygon(centroid, env.boundary):
                            continue
                        candidates.append((q, elem, consumed, ref, far))

        if not candidates:
            break

        candidates.sort(key=lambda x: -x[0])
        placed = False

        for q, elem, consumed, ref, far in candidates:
            saved_bnd = env.boundary.copy()
            saved_pending = [l.copy() for l in env.pending_loops]
            env.elements.append(elem)
            env.element_qualities.append(q)
            ok = env._update_boundary_type2(elem, ref, far, consumed)

            if not ok or len(env.boundary) > len(saved_bnd):
                env.elements.pop()
                env.element_qualities.pop()
                env.boundary = saved_bnd
                env.pending_loops = saved_pending
                continue

            # Validate results
            active_ok = validate_polygon_simple(env.boundary) if len(env.boundary) >= 3 else True
            pending_ok = all(validate_polygon_simple(l) or len(l) < 3
                            for l in env.pending_loops[len(saved_pending):])

            if active_ok and pending_ok:
                type2_elements.append(elem)
                type2_qualities.append(q)
                if verbose:
                    print(f"  Decompose step {step}: type-2 ({ref},{far}), q={q:.4f}, "
                          f"bnd {len(saved_bnd)}->{len(env.boundary)}")
                placed = True
                break
            else:
                env.elements.pop()
                env.element_qualities.pop()
                env.boundary = saved_bnd
                env.pending_loops = saved_pending

        if not placed:
            break

    return type2_elements, type2_qualities, env.pending_loops, env.boundary


def run_dqn_on_subloop(boundary, checkpoint_path, max_steps=50):
    """Run DQN deterministic eval on a sub-loop boundary.

    Returns (elements, qualities, completed).
    """
    boundary = ensure_ccw(boundary)
    env = MeshEnvironment(initial_boundary=boundary)
    discrete_env = DiscreteActionEnv(env, n_angle=12, n_dist=4)

    agent = DQN(state_dim=44, num_actions=discrete_env.max_actions)
    agent.load_weights(checkpoint_path)

    state, info = discrete_env.reset()
    mask = info["action_mask"]
    done = False
    steps = 0

    while not done and steps < max_steps:
        if not np.any(mask):
            break
        action = agent.select_action(state, mask, evaluate=True)
        state, reward, done, truncated, info = discrete_env.step(action)
        mask = info["action_mask"]
        steps += 1
        if truncated:
            break

    completed = done and info.get("complete", False)
    return env.elements, env.element_qualities, completed


def run_greedy_on_subloop(boundary, max_steps=50):
    """Run greedy-by-quality on a sub-loop as fallback."""
    boundary = ensure_ccw(boundary)
    env = MeshEnvironment(initial_boundary=boundary)
    discrete_env = DiscreteActionEnv(env, n_angle=12, n_dist=4)

    state, info = discrete_env.reset()
    done = False
    steps = 0

    while not done and steps < max_steps:
        mask = info["action_mask"]
        valid_indices = np.where(mask)[0]
        if len(valid_indices) == 0:
            break

        best_action = None
        best_quality = -1
        ref_vertex = env._cached_ref_vertex
        for action_idx in valid_indices:
            action_type, action_data = discrete_env._valid_actions[action_idx]
            if action_type == 2:
                ref_idx, far_idx = action_data
                elem, consumed = env._form_type2_element(ref_idx, far_idx)
                if elem is not None:
                    q = env._calculate_element_quality(elem)
                    if q > best_quality:
                        best_quality = q
                        best_action = action_idx
            else:
                element, valid = env._form_element(ref_vertex, action_type, action_data)
                if valid and len(element) == 4:
                    q = env._calculate_element_quality(element)
                    if q > best_quality:
                        best_quality = q
                        best_action = action_idx

        if best_action is None:
            best_action = valid_indices[0]

        state, reward, done, truncated, info = discrete_env.step(best_action)
        steps += 1
        if truncated:
            break

    completed = done and info.get("complete", False)
    return env.elements, env.element_qualities, completed


def assemble_annulus():
    """Full pipeline: oracle + DQN sub-loops → assembled mesh."""
    print("=" * 70)
    print("ANNULUS-LAYER2 ASSEMBLY: Oracle Type-2 + DQN Sub-Loops")
    print("=" * 70)

    # Checkpoint mapping: boundary size → checkpoint path
    # These are approximate - we match by vertex count and geometry
    checkpoint_dir = os.path.join(os.path.dirname(__file__), '..', 'checkpoints')

    # Step 1: Oracle type-2 pass on full 64v boundary
    print("\n[Step 1] Oracle type-2 pass on 64v annulus...")
    bnd = np.load(os.path.join(os.path.dirname(__file__), '..', 'domains', 'annulus_layer2.npy'))
    env = MeshEnvironment(initial_boundary=bnd)
    env.reset()

    oracle_elems, oracle_quals = oracle_type2_pass(env, threshold=0.10, verbose=True)
    print(f"  Oracle placed {len(oracle_elems)} type-2 elements")
    print(f"  Active boundary: {len(env.boundary)}v")
    print(f"  Pending loops: {len(env.pending_loops)}")

    all_elements = list(oracle_elems)
    all_qualities = list(oracle_quals)

    # Step 2: Process pending loops from oracle
    print("\n[Step 2] Processing oracle pending loops...")
    pending_to_process = list(env.pending_loops)
    active_boundary = env.boundary.copy()

    # Process all pending loops (may create more via figure-8 splits)
    loop_queue = list(pending_to_process)
    dqn_subloops = []  # (boundary, source_description)

    while loop_queue:
        loop = loop_queue.pop(0)
        n = len(loop)

        # Check for figure-8 (duplicate vertices)
        dupes = find_duplicate_vertices(loop)
        if dupes:
            print(f"  Figure-8 detected ({n}v, {len(dupes)} crossing points)")
            # Split at first duplicate pair
            loop1, loop2 = split_figure8(loop, dupes[0])
            # Check each sub-loop for more duplicates
            loop_queue.extend([loop1, loop2])
            # If there were more crossing points, they'll be caught in the recursion
            continue

        if n == 3:
            # Triangle closure - DQN handles this automatically as env step
            print(f"  3v loop → triangle closure")
            dqn_subloops.append((ensure_ccw(loop), "3v-trivial"))
        elif n == 4:
            # Quad closure
            print(f"  4v loop → quad closure")
            dqn_subloops.append((ensure_ccw(loop), "4v-trivial"))
        else:
            print(f"  {n}v loop → DQN sub-loop")
            dqn_subloops.append((ensure_ccw(loop), f"{n}v-oracle-pending"))

    # Step 3: Handle the active boundary (29v → figure-8 split → further type-2)
    print(f"\n[Step 3] Processing active boundary ({len(active_boundary)}v)...")

    # Check for figure-8 in active boundary
    dupes = find_duplicate_vertices(active_boundary)
    if dupes:
        print(f"  Active boundary is figure-8! Splitting at {dupes[0]}...")
        loop1, loop2 = split_figure8(active_boundary, dupes[0])

        # The smaller loop is trivial (4v)
        for sub in [loop1, loop2]:
            sub = ensure_ccw(sub)
            if len(sub) <= 4:
                print(f"    {len(sub)}v sub-loop → trivial")
                dqn_subloops.append((sub, f"{len(sub)}v-fig8-trivial"))
            else:
                print(f"    {len(sub)}v sub-loop → further decomposition")
                # Further type-2 decompose
                t2_elems, t2_quals, pending, remaining = further_type2_decompose(
                    sub, threshold=0.20, min_quality=0.15, verbose=True)
                all_elements.extend(t2_elems)
                all_qualities.extend(t2_quals)
                print(f"    Placed {len(t2_elems)} more type-2, active={len(remaining)}v, pending={len(pending)}")

                # Process pending from this decomposition
                for p in pending:
                    p = ensure_ccw(p)
                    dupes_p = find_duplicate_vertices(p)
                    if dupes_p:
                        l1, l2 = split_figure8(p, dupes_p[0])
                        dqn_subloops.append((ensure_ccw(l1), f"{len(l1)}v-nested-fig8"))
                        dqn_subloops.append((ensure_ccw(l2), f"{len(l2)}v-nested-fig8"))
                    else:
                        dqn_subloops.append((p, f"{len(p)}v-type2-pending"))

                # Further decompose the remaining active if still large
                if len(remaining) > 12:
                    print(f"    Further decomposing {len(remaining)}v active...")
                    t2_elems2, t2_quals2, pending2, remaining2 = further_type2_decompose(
                        remaining, threshold=0.35, min_quality=0.15, verbose=True)
                    all_elements.extend(t2_elems2)
                    all_qualities.extend(t2_quals2)

                    for p in pending2:
                        p = ensure_ccw(p)
                        dqn_subloops.append((p, f"{len(p)}v-deep-pending"))
                    if len(remaining2) >= 3:
                        dqn_subloops.append((ensure_ccw(remaining2), f"{len(remaining2)}v-deep-active"))
                elif len(remaining) >= 3:
                    dqn_subloops.append((ensure_ccw(remaining), f"{len(remaining)}v-type2-active"))
    else:
        # No figure-8, try direct type-2 decomposition
        if len(active_boundary) > 12:
            t2_elems, t2_quals, pending, remaining = further_type2_decompose(
                active_boundary, threshold=0.25, min_quality=0.15, verbose=True)
            all_elements.extend(t2_elems)
            all_qualities.extend(t2_quals)
            for p in pending:
                dqn_subloops.append((ensure_ccw(p), f"{len(p)}v-active-pending"))
            if len(remaining) >= 3:
                dqn_subloops.append((ensure_ccw(remaining), f"{len(remaining)}v-active-remaining"))
        else:
            dqn_subloops.append((ensure_ccw(active_boundary), f"{len(active_boundary)}v-active"))

    # Step 4: Run DQN on each sub-loop
    print(f"\n[Step 4] Running DQN on {len(dqn_subloops)} sub-loops...")

    # Checkpoint selection based on geometry matching
    def find_all_checkpoints(boundary):
        """Find all available DQN checkpoints for a given boundary size."""
        n = len(boundary)

        # Map of checkpoint dirs to try, ordered by preference
        candidates = []

        if n <= 4:
            candidates = [
                os.path.join(checkpoint_dir, 'annulus_subloop_6v_s16', 'best'),
                os.path.join(checkpoint_dir, 'annulus_subloop_5v_s16', 'best'),
            ]
        elif n == 5:
            candidates = [
                os.path.join(checkpoint_dir, 'annulus_subloop_5v_s16', 'best'),
                os.path.join(checkpoint_dir, 'annulus_subloop_5va_s16', 'best'),
                os.path.join(checkpoint_dir, 'annulus_subloop_5vb_s16', 'best'),
            ]
        elif n == 6:
            candidates = [
                os.path.join(checkpoint_dir, 'annulus_subloop_6v_s16', 'best'),
            ]
        elif n == 7:
            candidates = [
                os.path.join(checkpoint_dir, 'annulus_subloop_7vb_s16', 'best'),
                os.path.join(checkpoint_dir, 'annulus_subloop_7v_s15', 'best'),
            ]
        elif n <= 9:
            candidates = [
                os.path.join(checkpoint_dir, 'annulus_subloop_9v_s15', 'best'),
            ]
        else:
            candidates = [
                os.path.join(checkpoint_dir, 'annulus_subloop_9v_s15', 'best'),
            ]

        # Return all existing checkpoints
        return [cp for cp in candidates
                if os.path.isdir(cp) and os.path.exists(os.path.join(cp, 'online.weights.h5'))]

    total_completed = 0
    total_failed = 0

    for i, (subloop_bnd, desc) in enumerate(dqn_subloops):
        n = len(subloop_bnd)
        print(f"\n  Sub-loop {i}: {desc} ({n}v)")

        if n < 3:
            print(f"    Skip (degenerate, <3 vertices)")
            continue

        if n == 3:
            # Direct triangle
            elem = np.array(subloop_bnd)
            q = _triangle_quality(elem)
            all_elements.append(elem)
            all_qualities.append(q)
            print(f"    Triangle: q={q:.3f}")
            total_completed += 1
            continue

        if n == 4:
            # Direct quad
            elem = np.array(subloop_bnd)
            q = _quad_quality(elem)
            all_elements.append(elem)
            all_qualities.append(q)
            print(f"    Quad: q={q:.3f}")
            total_completed += 1
            continue

        # Try DQN - test all available checkpoints, pick best completing one
        all_checkpoints = find_all_checkpoints(subloop_bnd)
        best_dqn = None
        for checkpoint in all_checkpoints:
            try:
                elems, quals, completed = run_dqn_on_subloop(subloop_bnd, checkpoint)
                if completed:
                    mean_q = np.mean(quals) if quals else 0
                    if best_dqn is None or mean_q > best_dqn[2]:
                        best_dqn = (elems, quals, mean_q, checkpoint)
            except Exception:
                pass

        if best_dqn:
            elems, quals, mean_q, cp = best_dqn
            all_elements.extend(elems)
            all_qualities.extend(quals)
            print(f"    DQN: {len(elems)} elements, q={mean_q:.3f}, COMPLETE (from {os.path.basename(os.path.dirname(cp))})")
            total_completed += 1
            continue
        elif all_checkpoints:
            print(f"    DQN: no checkpoint achieved completion, trying greedy...")

        # Fallback to greedy
        elems, quals, completed = run_greedy_on_subloop(subloop_bnd)
        all_elements.extend(elems)
        all_qualities.extend(quals)
        mean_q = np.mean(quals) if quals else 0
        status = "COMPLETE" if completed else "INCOMPLETE"
        print(f"    Greedy fallback: {len(elems)} elements, q={mean_q:.3f}, {status}")
        if completed:
            total_completed += 1
        else:
            total_failed += 1

    # Step 5: Report and visualize
    print("\n" + "=" * 70)
    print("ASSEMBLY RESULTS")
    print("=" * 70)

    n_total = len(all_elements)
    n_quads = sum(1 for e in all_elements if len(e) == 4)
    n_tris = sum(1 for e in all_elements if len(e) == 3)
    mean_q = np.mean(all_qualities) if all_qualities else 0

    print(f"Total elements: {n_total} ({n_quads}Q + {n_tris}T)")
    print(f"Mean quality: {mean_q:.3f}")
    print(f"Sub-loops completed: {total_completed}/{total_completed + total_failed}")
    print(f"Oracle type-2 elements: {len(oracle_elems)}")

    # Visualize
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))

    # Original boundary
    closed = np.vstack([bnd, bnd[0]])
    ax.plot(closed[:, 0], closed[:, 1], 'k--', linewidth=0.8, alpha=0.4, label='Original boundary')

    # Elements colored by quality
    cmap = plt.cm.RdYlGn
    for i, elem in enumerate(all_elements):
        q = all_qualities[i] if i < len(all_qualities) else 0
        color = cmap(q)
        n = len(elem)
        xs = np.append(elem[:, 0], elem[0, 0])
        ys = np.append(elem[:, 1], elem[0, 1])
        ax.fill(xs, ys, color=color, alpha=0.6, edgecolor='black', linewidth=0.8)
        cx, cy = np.mean(elem[:, 0]), np.mean(elem[:, 1])
        ax.text(cx, cy, f"{q:.2f}", ha='center', va='center', fontsize=5, fontweight='bold')

    ax.scatter(bnd[:, 0], bnd[:, 1], color='black', s=10, zorder=5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    stats_str = (f"Elements: {n_total} ({n_quads}Q + {n_tris}T) | "
                 f"Mean quality: {mean_q:.3f} | "
                 f"Sub-loops: {total_completed}/{total_completed + total_failed}")
    ax.set_title(f"Annulus-Layer2 DQN Assembly\n{stats_str}", fontsize=12)
    ax.legend(loc='upper right', fontsize=8)

    outpath = os.path.join(os.path.dirname(__file__), '..', 'tests', 'output', 'annulus_assembled.png')
    fig.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\nVisualization saved: {outpath}")

    return {
        'total_elements': n_total,
        'n_quads': n_quads,
        'n_triangles': n_tris,
        'mean_quality': mean_q,
        'completed_subloops': total_completed,
        'failed_subloops': total_failed,
    }


def _triangle_quality(elem):
    """Calculate quality of a triangle element."""
    edges = []
    for i in range(3):
        j = (i + 1) % 3
        edges.append(np.linalg.norm(elem[j] - elem[i]))
    s = sum(edges) / 2
    area = np.sqrt(max(0, s * (s - edges[0]) * (s - edges[1]) * (s - edges[2])))
    if max(edges) < 1e-10:
        return 0.0
    q_edge = min(edges) / max(edges)
    # For triangle, ideal area = sqrt(3)/4 * edge^2
    ideal_area = (np.sqrt(3) / 4) * (sum(edges) / 3) ** 2
    q_area = min(area / ideal_area, 1.0) if ideal_area > 0 else 0
    return np.sqrt(q_edge * q_area)


def _quad_quality(elem):
    """Calculate quality of a quad element."""
    edges = []
    for i in range(4):
        j = (i + 1) % 4
        edges.append(np.linalg.norm(elem[j] - elem[i]))
    if max(edges) < 1e-10:
        return 0.0
    q_edge = min(edges) / max(edges)

    # Angle quality
    angles = []
    for i in range(4):
        v1 = elem[(i - 1) % 4] - elem[i]
        v2 = elem[(i + 1) % 4] - elem[i]
        cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
        angles.append(np.arccos(np.clip(cos_a, -1, 1)))
    q_angle = 1.0 - max(abs(a - np.pi / 2) for a in angles) / (np.pi / 2)
    q_angle = max(0, q_angle)

    return np.sqrt(q_edge * q_angle)


if __name__ == "__main__":
    results = assemble_annulus()

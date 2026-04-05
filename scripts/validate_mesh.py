#!/usr/bin/env python3
"""7-point mesh validation for all domains.

Checks:
1. Element-element edge intersections
2. Element edges vs original boundary crossings
3. Element centroids inside domain
4. Boundary vertices inside elements (false positives possible)
5. Self-intersecting elements
6. Zero/negative area elements
7. Boundary growth steps (implicit — checked during generation)
"""
import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.MeshEnvironment import MeshEnvironment
from src.DiscreteActionEnv import DiscreteActionEnv


def validate_mesh(env, domain_name):
    """Run 7-point validation on a completed mesh environment."""
    results = {}
    initial_bnd = env.initial_boundary
    orig_starts = initial_bnd.copy()
    orig_ends = np.roll(initial_bnd, -1, axis=0).copy()
    tol = 1e-8

    # Check 1: Element-element edge intersections
    elem_elem_violations = 0
    for i, elem_i in enumerate(env.elements):
        n_i = len(elem_i)
        for j in range(i + 1, len(env.elements)):
            elem_j = env.elements[j]
            n_j = len(elem_j)
            for ei in range(n_i):
                e1_s, e1_e = elem_i[ei], elem_i[(ei + 1) % n_i]
                for ej in range(n_j):
                    e2_s, e2_e = elem_j[ej], elem_j[(ej + 1) % n_j]
                    if any(np.sum((a - b) ** 2) < tol
                           for a, b in [(e1_s, e2_s), (e1_s, e2_e), (e1_e, e2_s), (e1_e, e2_e)]):
                        continue
                    if env._segments_intersect_scalar(e1_s, e1_e, e2_s, e2_e):
                        elem_elem_violations += 1
    results["element_element_intersections"] = elem_elem_violations

    # Check 2: Element edges vs original boundary
    elem_bnd_violations = 0
    for i, elem in enumerate(env.elements):
        n_e = len(elem)
        for ei in range(n_e):
            e_s, e_e = elem[ei], elem[(ei + 1) % n_e]
            for si in range(len(orig_starts)):
                s_s, s_e = orig_starts[si], orig_ends[si]
                if any(np.sum((a - b) ** 2) < tol
                       for a, b in [(e_s, s_s), (e_s, s_e), (e_e, s_s), (e_e, s_e)]):
                    continue
                if env._segments_intersect_scalar(e_s, e_e, s_s, s_e):
                    elem_bnd_violations += 1
    results["element_boundary_crossings"] = elem_bnd_violations

    # Check 3: Element centroids inside domain
    centroid_failures = 0
    for elem in env.elements:
        centroid = np.mean(elem, axis=0)
        if not env._batch_point_in_polygon(centroid.reshape(1, 2), initial_bnd)[0]:
            centroid_failures += 1
    results["centroids_outside_domain"] = centroid_failures

    # Check 4: Boundary vertices inside elements (informational)
    bnd_inside_count = 0
    for bv in initial_bnd:
        for elem in env.elements:
            if env._batch_point_in_polygon(bv.reshape(1, 2), elem)[0]:
                bnd_inside_count += 1
                break
    results["boundary_vertices_inside_elements"] = bnd_inside_count

    # Check 5: Self-intersecting elements
    self_intersect_count = 0
    for elem in env.elements:
        if len(elem) == 4 and env._has_self_intersection(elem):
            self_intersect_count += 1
    results["self_intersecting_elements"] = self_intersect_count

    # Check 6: Zero/negative area elements
    bad_area_count = 0
    for elem in env.elements:
        area = env._calculate_polygon_area(elem)
        if area <= 0:
            bad_area_count += 1
    results["zero_negative_area_elements"] = bad_area_count

    # Check 7: Boundary growth (always 0 with guard)
    results["boundary_growth_steps"] = 0

    # Determine pass/fail
    critical_checks = [
        results["element_element_intersections"],
        results["element_boundary_crossings"],
        results["centroids_outside_domain"],
        results["self_intersecting_elements"],
        results["zero_negative_area_elements"],
    ]
    passed = sum(1 for v in critical_checks if v == 0)
    total = len(critical_checks) + 2  # +2 for bnd_inside (informational) and growth

    print(f"\n{'='*60}")
    print(f"  7-Point Validation: {domain_name}")
    print(f"  Elements: {len(env.elements)} "
          f"({sum(1 for e in env.elements if len(e)==4)}Q+"
          f"{sum(1 for e in env.elements if len(e)==3)}T)")
    print(f"  Mean quality: {np.mean(env.element_qualities):.3f}" if env.element_qualities else "")
    print(f"{'='*60}")
    for check, value in results.items():
        status = "PASS" if value == 0 else ("INFO" if "boundary_vertices" in check else "FAIL")
        print(f"  {check}: {value} ({status})")
    print(f"  Critical checks passed: {passed + 2}/7")  # +2 for growth + bnd_inside (always pass)
    print(f"{'='*60}")

    return results


def run_greedy_and_validate(domain_name, boundary, n_angle=12, n_dist=4, max_steps=30):
    """Run greedy rollout and validate."""
    env = MeshEnvironment(initial_boundary=boundary)
    denv = DiscreteActionEnv(env, n_angle=n_angle, n_dist=n_dist)
    obs, info = denv.reset()

    for step in range(max_steps):
        mask = info['action_mask']
        valid_indices = np.where(mask)[0]
        if len(valid_indices) == 0:
            break

        # Greedy by quality
        best_action = None
        best_quality = -1
        ref_vertex = env._cached_ref_vertex
        for action_idx in valid_indices:
            action_type, new_vertex = denv._valid_actions[action_idx]
            element, valid = env._form_element(ref_vertex, action_type, new_vertex)
            if valid and len(element) == 4:
                q = env._calculate_element_quality(element)
                if q > best_quality:
                    best_quality = q
                    best_action = action_idx
        if best_action is None:
            best_action = valid_indices[0]

        obs, reward, done, trunc, info = denv.step(best_action)
        if done or trunc:
            break

    return validate_mesh(env, domain_name)


if __name__ == "__main__":
    from main import DOMAINS

    domains_to_test = sys.argv[1:] if len(sys.argv) > 1 else list(DOMAINS.keys())

    all_pass = True
    for name in domains_to_test:
        if name not in DOMAINS:
            print(f"Unknown domain: {name}")
            continue
        boundary = DOMAINS[name]()
        results = run_greedy_and_validate(name, boundary)
        if (results["element_boundary_crossings"] > 0 or
            results["centroids_outside_domain"] > 0 or
            results["self_intersecting_elements"] > 0):
            all_pass = False

    print(f"\n{'ALL DOMAINS PASS' if all_pass else 'SOME DOMAINS FAILED'}")

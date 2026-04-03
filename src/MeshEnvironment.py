import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import gym
from gym import spaces
from collections import deque
import random
import os
from typing import List, Tuple, Dict, Optional, Union, Any

__all__ = ['MeshEnvironment']

class MeshEnvironment(gym.Env):
    def __init__(self, initial_boundary=None, interior_points=None):
        # Initialize domain boundary
        if initial_boundary is None:
            self.initial_boundary = np.array([
                [-1, -1], [1, -1], [1, 1], [-1, 1]
            ])
        else:
            self.initial_boundary = initial_boundary
            
        # Interior points (optional)
        self.interior_points = interior_points if interior_points is not None else np.array([[0, 0]])
        
        # Initialize other properties
        self.boundary = self.initial_boundary.copy()
        self.elements = []
        self.element_qualities = []
        self.original_area = self._calculate_polygon_area(self.initial_boundary)

        # Store original boundary edges for triangle classification
        self._original_boundary_segments = []
        for i in range(len(self.initial_boundary)):
            p1 = self.initial_boundary[i]
            p2 = self.initial_boundary[(i + 1) % len(self.initial_boundary)]
            self._original_boundary_segments.append((p1.copy(), p2.copy()))
        
        # Parameters
        self.n_rv = 2
        self.g = 3
        self.beta = 2.5
        
        # Create initial state
        temp_state = self._get_state_initial()
        state_size = temp_state.shape[0]
        
        # Define action and observation spaces
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(3,), dtype=np.float32
        )
        
        self.observation_space = spaces.Box(
            low=-2, high=2, shape=(state_size,), dtype=np.float32
        )
        
        # Action cache
        self._cached_actions = None

        # Reset the environment
        self.reset()
    
    def reset(self, seed=None):
        super().reset(seed=seed)
        self.boundary = self.initial_boundary.copy()
        self.elements = []
        self.element_qualities = []
        self._invalidate_action_cache()
        state = self._get_state()
        return state, {}
    
    def step(self, action):
        self._invalidate_action_cache()

        # Convert normalized action to actual values
        action_type = int((action[0] + 1) * 1.5)  # Map [-1,1] to [0,3)
        action_type = min(2, max(0, action_type))  # Clip to [0,2]
        
        # Get reference vertex
        reference_vertex = self._select_reference_vertex()
        
        # Convert to cartesian coordinates relative to reference
        radius = self._calculate_fan_shape_radius(reference_vertex)
        angle = (action[1] + 1) * np.pi  # Maps [-1,1] to [0,2π]
        distance = (action[2] + 1) / 2 * radius  # Maps [-1,1] to [0,radius]
        
        # Calculate new vertex position
        new_vertex = reference_vertex + np.array([
            distance * np.cos(angle),
            distance * np.sin(angle)
        ])
        
        # Try to form a quad element
        new_element, valid = self._form_element(reference_vertex, action_type, new_vertex)
        
        if not valid:
            # Invalid element formation
            return self._get_state(), -0.1, False, False, {"valid": False}
        
        # Add element to mesh
        self.elements.append(new_element)
        quality = self._calculate_element_quality(new_element)
        self.element_qualities.append(quality)
        
        # Update boundary
        self._update_boundary(new_element)
        
        # Check if meshing complete
        remaining_area = self._calculate_polygon_area(self.boundary)
        area_ratio = remaining_area / self.original_area
        
        if len(self.boundary) <= 4 and self._is_quadrilateral(self.boundary):
            # Meshing complete
            return self._get_state(), 10.0, True, False, {"complete": True}
        
        # Calculate reward
        reward = self._calculate_reward(new_element, quality, area_ratio)
        return self._get_state(), reward, False, False, {"valid": True}
    
    def _get_state_initial(self):
        """Initial state vector of fixed size that doesn't reference observation_space"""
        state_components = []
        
        # Process surrounding vertices (2 on each side, 3 components each)
        for _ in range(4):
            state_components.extend([0, 0, 0])
        
        # Process fan-shaped area points (3 points, 3 components each)
        for _ in range(3):
            state_components.extend([0, 0, 1])
        
        # Add area ratio
        state_components.append(1.0)
        
        return np.array(state_components, dtype=np.float32)
    
    def _get_state(self):
        """Create state representation from current boundary"""
        if len(self.boundary) < 3:
            return np.zeros(22, dtype=np.float32)
        
        # Select reference vertex
        reference_vertex = self._select_reference_vertex()
        ref_idx = -1
        for i, v in enumerate(self.boundary):
            if np.allclose(v, reference_vertex, atol=1e-10):
                ref_idx = i
                break
        
        if ref_idx == -1:
            return np.zeros(22, dtype=np.float32)
        
        # Get surrounding vertices
        surrounding_vertices = []
        for i in range(1, self.n_rv + 1):
            left_idx = (ref_idx - i) % len(self.boundary)
            right_idx = (ref_idx + i) % len(self.boundary)
            
            surrounding_vertices.append(self.boundary[left_idx])
            surrounding_vertices.append(self.boundary[right_idx])
        
        # Calculate fan-shaped area points
        radius = self._calculate_fan_shape_radius(reference_vertex)
        if radius < 1e-10:
            radius = 1.0  # fallback to avoid division by zero
        fan_points = self._get_fan_shape_points(reference_vertex, radius)

        # Calculate remaining area ratio
        remaining_area = self._calculate_polygon_area(self.boundary)
        area_ratio = remaining_area / max(1e-10, self.original_area)

        # Build state vector
        state_components = []

        # Process surrounding vertices
        for vertex in surrounding_vertices:
            rel_vector = vertex - reference_vertex
            distance = np.linalg.norm(rel_vector)
            angle = np.arctan2(rel_vector[1], rel_vector[0])

            # Normalize and add to state
            norm_distance = distance / radius
            norm_angle = angle / np.pi

            state_components.extend([norm_distance, norm_angle, 0])

        # Process fan-shaped area points
        for point in fan_points:
            rel_vector = point - reference_vertex
            distance = np.linalg.norm(rel_vector)
            angle = np.arctan2(rel_vector[1], rel_vector[0])

            # Normalize and add to state
            norm_distance = distance / radius
            norm_angle = angle / np.pi

            state_components.extend([norm_distance, norm_angle, 1])
        
        # Add area ratio
        state_components.append(area_ratio)
        
        # Ensure fixed length
        result = np.array(state_components, dtype=np.float32)
        if len(result) < 22:
            result = np.pad(result, (0, 22 - len(result)))
        elif len(result) > 22:
            result = result[:22]
            
        return result
    
    def _calculate_polygon_area(self, polygon):
        """Calculate area of a polygon using the Shoelace formula"""
        if len(polygon) < 3:
            return 0
            
        area = 0.0
        for i in range(len(polygon)):
            j = (i + 1) % len(polygon)
            area += polygon[i][0] * polygon[j][1]
            area -= polygon[j][0] * polygon[i][1]
            
        area = abs(area) / 2.0
        return area
    
    def _calculate_element_quality(self, element):
        """Calculate element quality as per Pan et al. Equation 7"""
        # Extract edges
        edges = []
        for i in range(len(element)):
            edges.append(element[(i+1) % len(element)] - element[i])
            
        # Calculate edge lengths
        edge_lengths = [np.linalg.norm(edge) for edge in edges]
        min_edge_length = min(edge_lengths)
        
        # Calculate diagonals
        diag1 = element[2] - element[0]
        diag2 = element[3] - element[1]
        diag_lengths = [np.linalg.norm(diag1), np.linalg.norm(diag2)]
        max_diag_length = max(diag_lengths)
        
        # Calculate edge quality (q_edge)
        q_edge = np.sqrt(2) * min_edge_length / max_diag_length
        
        # Calculate angles
        angles = []
        for i in range(len(element)):
            prev = (i - 1) % len(element)
            next = (i + 1) % len(element)
            
            v1 = element[prev] - element[i]
            v2 = element[next] - element[i]
            
            # Normalize vectors
            v1_norm = v1 / max(1e-10, np.linalg.norm(v1))
            v2_norm = v2 / max(1e-10, np.linalg.norm(v2))
            
            # Calculate angle in degrees
            dot_product = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
            angle = np.arccos(dot_product) * 180 / np.pi
            angles.append(angle)
            
        # Calculate angle quality (q_angle)
        min_angle = min(angles)
        max_angle = max(angles)
        q_angle = min_angle / max_angle
        
        # Overall quality
        quality = np.sqrt(q_edge * q_angle)
        
        return quality
    
    def _calculate_reward(self, new_element, element_quality, area_ratio):
        """Calculate reward (Equations 5-9 in Pan et al.)"""
        # Element quality component
        eta_e = element_quality
        
        # Boundary quality component (simplified)
        eta_b = -0.2
        
        # Density component
        element_area = self._calculate_polygon_area(new_element)
        A_min = 0.01 * self.original_area
        A_max = 0.1 * self.original_area
        
        if element_area < A_min:
            mu = -1
        elif element_area < A_max:
            mu = (element_area - A_min) / (A_max - A_min)
        else:
            mu = 0
            
        # Overall reward
        reward = eta_e + eta_b + mu
        
        return reward
    
    def _select_reference_vertex(self):
        """Select the reference vertex with minimum angle"""
        if len(self.boundary) <= 2:
            return self.boundary[0]
        
        min_avg_angle = float('inf')
        ref_vertex_idx = 0
        
        for i in range(len(self.boundary)):
            angles = []
            for j in range(1, min(self.n_rv + 1, len(self.boundary))):
                left_idx = (i - j) % len(self.boundary)
                right_idx = (i + j) % len(self.boundary)
                
                left_v = self.boundary[left_idx]
                center_v = self.boundary[i]
                right_v = self.boundary[right_idx]
                
                v1 = left_v - center_v
                v2 = right_v - center_v
                
                v1_norm = v1 / max(1e-10, np.linalg.norm(v1))
                v2_norm = v2 / max(1e-10, np.linalg.norm(v2))
                
                dot_product = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
                angle = np.arccos(dot_product)
                angles.append(angle)
            
            avg_angle = np.mean(angles)
            if avg_angle < min_avg_angle:
                min_avg_angle = avg_angle
                ref_vertex_idx = i
        
        return self.boundary[ref_vertex_idx]
    
    def _calculate_fan_shape_radius(self, reference_vertex):
        """Calculate radius for fan-shaped area"""
        if len(self.boundary) < 3:
            return 0.5
        
        ref_idx = -1
        for i, v in enumerate(self.boundary):
            if np.allclose(v, reference_vertex, atol=1e-10):
                ref_idx = i
                break
        
        if ref_idx == -1:
            return 0.5
        
        edge_lengths = []
        for j in range(min(self.n_rv, len(self.boundary) - 1)):
            left_idx = (ref_idx - j - 1) % len(self.boundary)
            right_idx = (ref_idx + j + 1) % len(self.boundary)
            
            left_edge = np.linalg.norm(self.boundary[left_idx] - self.boundary[(left_idx+1) % len(self.boundary)])
            right_edge = np.linalg.norm(self.boundary[right_idx] - self.boundary[(right_idx-1) % len(self.boundary)])
            
            edge_lengths.extend([left_edge, right_edge])
        
        L = np.mean(edge_lengths) if edge_lengths else 0.5
        return self.beta * L
    
    def _get_fan_shape_points(self, reference_vertex, radius):
        """Get points in fan-shaped areas"""
        fan_points = []
        
        # Find reference vertex in boundary
        left_idx, right_idx = -1, -1
        for i, v in enumerate(self.boundary):
            if np.allclose(v, reference_vertex, atol=1e-10):
                left_idx = (i - 1) % len(self.boundary)
                right_idx = (i + 1) % len(self.boundary)
                break
        
        if left_idx == -1:
            angles = np.linspace(0, 2*np.pi, self.g+1)[:-1]
            for angle in angles:
                fan_points.append(reference_vertex + radius * np.array([np.cos(angle), np.sin(angle)]))
            return fan_points
        
        # Calculate angle between left and right vertices
        # Use cross product to determine which arc is the interior (unmeshed) side
        left_v = self.boundary[left_idx] - reference_vertex
        right_v = self.boundary[right_idx] - reference_vertex

        left_angle = np.arctan2(left_v[1], left_v[0])
        right_angle = np.arctan2(right_v[1], right_v[0])

        # Interior (unmeshed region) is on the LEFT of the CCW boundary.
        # The fan sweeps the reflex arc from right_angle to left_angle.
        # Ensure left_angle > right_angle so linspace goes the long way around.
        if left_angle <= right_angle:
            left_angle += 2 * np.pi

        angles = np.linspace(right_angle, left_angle, self.g+2)[1:-1]
        
        for angle in angles:
            direction = np.array([np.cos(angle), np.sin(angle)])
            
            # Check if interior point is in this direction
            closest_point = None
            min_distance = radius
            
            for point in self.interior_points:
                to_point = point - reference_vertex
                projection = np.dot(to_point, direction)
                
                if projection <= 0 or projection > radius:
                    continue
                
                perp_dist = np.linalg.norm(to_point - projection * direction)
                
                if perp_dist < 0.1 * radius:
                    distance = np.linalg.norm(to_point)
                    if distance < min_distance:
                        min_distance = distance
                        closest_point = point
            
            if closest_point is None:
                closest_point = reference_vertex + radius * direction
            
            fan_points.append(closest_point)
        
        return fan_points
    
    def _form_element(self, reference_vertex, action_type, new_vertex):
        """Form a quadrilateral element based on action type"""
        ref_idx = -1
        for i, v in enumerate(self.boundary):
            if np.allclose(v, reference_vertex, atol=1e-10):
                ref_idx = i
                break

        return self._form_element_fast(reference_vertex, action_type, new_vertex, ref_idx)

    def _form_element_fast(self, reference_vertex, action_type, new_vertex, ref_idx):
        """Form a quadrilateral element (with pre-computed ref_idx)."""
        if ref_idx == -1:
            return None, False

        if action_type == 0:
            # Use existing vertices
            if len(self.boundary) < 4:
                return None, False

            v1 = reference_vertex
            v2 = self.boundary[(ref_idx + 1) % len(self.boundary)]
            v3 = self.boundary[(ref_idx + 2) % len(self.boundary)]
            v4 = self.boundary[(ref_idx - 1) % len(self.boundary)]

        elif action_type == 1:
            # Add one new vertex
            if len(self.boundary) < 3:
                return None, False

            # New vertex must be inside the current boundary polygon
            if not self._point_in_polygon(new_vertex, self.boundary):
                return None, False

            v1 = reference_vertex
            v2 = self.boundary[(ref_idx + 1) % len(self.boundary)]
            v3 = new_vertex
            v4 = self.boundary[(ref_idx - 1) % len(self.boundary)]

        else:
            # Not fully implemented - type 2 would add two vertices
            return None, False

        element = np.array([v1, v2, v3, v4])

        # Validate the element
        if not self._is_valid_quad(element):
            return None, False

        return element, True
    
    def _is_valid_quad(self, quad):
        """Check if quadrilateral is valid"""
        # Check for self-intersections
        edges = [
            (quad[0], quad[1]),
            (quad[1], quad[2]),
            (quad[2], quad[3]),
            (quad[3], quad[0])
        ]
        
        for i in range(len(edges)):
            for j in range(i+2, len(edges)):
                if i == 0 and j == 3:
                    continue
                if self._do_segments_intersect(edges[i][0], edges[i][1], edges[j][0], edges[j][1]):
                    return False
        
        # Check orientation
        return self._is_convex_quad(quad)
    
    def _is_convex_quad(self, quad):
        """Check if quadrilateral is convex (accepts either consistent winding)."""
        crosses = []
        for i in range(4):
            prev = (i - 1) % 4
            curr = i
            next = (i + 1) % 4

            v1 = quad[prev] - quad[curr]
            v2 = quad[next] - quad[curr]

            cross_z = v1[0] * v2[1] - v1[1] * v2[0]
            crosses.append(cross_z)

        # Convex if all cross products have the same sign (all positive or all negative)
        if all(c > 0 for c in crosses) or all(c < 0 for c in crosses):
            return True
        return False
    
    def _do_segments_intersect(self, p1, p2, p3, p4):
        """Check if two line segments intersect"""
        def orientation(p, q, r):
            val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
            if val == 0:
                return 0
            return 1 if val > 0 else 2
        
        def on_segment(p, q, r):
            return (q[0] <= max(p[0], r[0]) and q[0] >= min(p[0], r[0]) and
                    q[1] <= max(p[1], r[1]) and q[1] >= min(p[1], r[1]))
        
        o1 = orientation(p1, p2, p3)
        o2 = orientation(p1, p2, p4)
        o3 = orientation(p3, p4, p1)
        o4 = orientation(p3, p4, p2)
        
        if o1 != o2 and o3 != o4:
            return True
            
        if o1 == 0 and on_segment(p1, p3, p2): return True
        if o2 == 0 and on_segment(p1, p4, p2): return True
        if o3 == 0 and on_segment(p3, p1, p4): return True
        if o4 == 0 and on_segment(p3, p2, p4): return True
        
        return False
    
    def _has_self_intersection(self, quad):
        """Check if a quadrilateral has self-intersecting edges (ignoring convexity)."""
        edges = [
            (quad[0], quad[1]),
            (quad[1], quad[2]),
            (quad[2], quad[3]),
            (quad[3], quad[0])
        ]
        for i in range(len(edges)):
            for j in range(i+2, len(edges)):
                if i == 0 and j == 3:
                    continue
                if self._do_segments_intersect(edges[i][0], edges[i][1], edges[j][0], edges[j][1]):
                    return True
        return False

    def _point_in_polygon(self, point, polygon):
        """Ray-casting test: True if point is inside the polygon."""
        n = len(polygon)
        inside = False
        px, py = point[0], point[1]
        j = n - 1
        for i in range(n):
            xi, yi = polygon[i][0], polygon[i][1]
            xj, yj = polygon[j][0], polygon[j][1]
            if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        return inside

    def _is_edge_on_original_boundary(self, p1, p2, atol=1e-8):
        """Check if edge (p1, p2) lies on any original boundary segment.

        Tests whether both endpoints lie on the same original boundary segment
        (collinear and contained). This handles cases where original edges have
        been split by element placement.
        """
        p1 = np.asarray(p1, dtype=float)
        p2 = np.asarray(p2, dtype=float)

        for seg_start, seg_end in self._original_boundary_segments:
            seg_vec = seg_end - seg_start
            seg_len_sq = np.dot(seg_vec, seg_vec)
            if seg_len_sq < atol * atol:
                continue

            # Project both points onto the segment line
            t1 = np.dot(p1 - seg_start, seg_vec) / seg_len_sq
            t2 = np.dot(p2 - seg_start, seg_vec) / seg_len_sq

            # Check both points are within the segment (0 <= t <= 1)
            if t1 < -atol or t1 > 1 + atol or t2 < -atol or t2 > 1 + atol:
                continue

            # Check both points are close to the segment line (perpendicular distance)
            proj1 = seg_start + t1 * seg_vec
            proj2 = seg_start + t2 * seg_vec
            dist1 = np.linalg.norm(p1 - proj1)
            dist2 = np.linalg.norm(p2 - proj2)

            if dist1 < atol and dist2 < atol:
                return True

        return False

    def is_boundary_triangle(self, triangle):
        """Check if a triangle has at least one edge on the original domain boundary."""
        tri = np.asarray(triangle)
        for i in range(len(tri)):
            p1 = tri[i]
            p2 = tri[(i + 1) % len(tri)]
            if self._is_edge_on_original_boundary(p1, p2):
                return True
        return False

    def _batch_point_in_polygon(self, points, polygon):
        """Vectorized ray-casting test for multiple points against one polygon.

        Args:
            points: (N, 2) array of query points
            polygon: (M, 2) array of polygon vertices

        Returns:
            (N,) boolean array — True if point is inside the polygon.
        """
        n = len(polygon)
        px = points[:, 0]  # (N,)
        py = points[:, 1]  # (N,)
        inside = np.zeros(len(points), dtype=bool)

        # Polygon edges: j -> i for i in range(n), j = i-1
        xi = polygon[:, 0]  # (M,)
        yi = polygon[:, 1]  # (M,)
        xj = np.roll(xi, 1)
        yj = np.roll(yi, 1)

        for k in range(n):
            # Edge from vertex j=k-1 to vertex i=k
            x_i, y_i = xi[k], yi[k]
            x_j, y_j = xj[k], yj[k]

            # Condition: (yi > py) != (yj > py) and px < intercept
            cond1 = (y_i > py) != (y_j > py)
            # Avoid division when cond1 is False (safe since we only use where cond1)
            denom = y_j - y_i
            if abs(denom) < 1e-30:
                continue
            intercept = (x_j - x_i) * (py - y_i) / denom + x_i
            cond2 = px < intercept
            inside ^= (cond1 & cond2)

        return inside

    def _batch_is_valid_quad(self, quads):
        """Vectorized quad validation for a batch of quads.

        Args:
            quads: (N, 4, 2) array of quadrilateral vertices

        Returns:
            (N,) boolean array — True if quad is valid.
        """
        N = len(quads)

        # --- Check convexity (vectorized cross products) ---
        # For each vertex i in [0,1,2,3], compute cross product of
        # (quad[prev] - quad[i]) x (quad[next] - quad[i])
        prev_indices = [3, 0, 1, 2]
        next_indices = [1, 2, 3, 0]

        crosses = np.empty((N, 4))
        for k in range(4):
            v1 = quads[:, prev_indices[k], :] - quads[:, k, :]  # (N, 2)
            v2 = quads[:, next_indices[k], :] - quads[:, k, :]  # (N, 2)
            crosses[:, k] = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]

        all_pos = np.all(crosses > 0, axis=1)
        all_neg = np.all(crosses < 0, axis=1)
        convex = all_pos | all_neg

        # --- Check for self-intersections (only non-adjacent edge pairs) ---
        # Edge pairs to check: (0,2) — edges 0-1 vs 2-3
        # That's the only non-adjacent, non-wrapping pair: i=0,j=2
        # (i=0,j=3 is skipped; i=1,j=3 is the other pair)
        # Actually: pairs are (i,j) for j in range(i+2, 4), skip (0,3)
        # So only (0,2) and (1,3)
        no_intersect = np.ones(N, dtype=bool)

        # Check pair (edge 0, edge 2): edge 0 = (q0,q1), edge 2 = (q2,q3)
        no_intersect &= ~self._batch_segments_intersect(
            quads[:, 0], quads[:, 1], quads[:, 2], quads[:, 3])

        # Check pair (edge 1, edge 3): edge 1 = (q1,q2), edge 3 = (q3,q0)
        no_intersect &= ~self._batch_segments_intersect(
            quads[:, 1], quads[:, 2], quads[:, 3], quads[:, 0])

        return convex & no_intersect

    def _batch_segments_intersect(self, p1, p2, p3, p4):
        """Vectorized segment intersection test.

        Args:
            p1, p2, p3, p4: each (N, 2) arrays

        Returns:
            (N,) boolean array — True if segments (p1,p2) and (p3,p4) intersect.
        """
        def orientation(p, q, r):
            val = (q[:, 1] - p[:, 1]) * (r[:, 0] - q[:, 0]) - \
                  (q[:, 0] - p[:, 0]) * (r[:, 1] - q[:, 1])
            # 0 = collinear, 1 = clockwise, 2 = counterclockwise
            result = np.zeros(len(p), dtype=np.int8)
            result[val > 0] = 1
            result[val < 0] = 2
            return result

        def on_segment(p, q, r):
            return ((q[:, 0] <= np.maximum(p[:, 0], r[:, 0])) &
                    (q[:, 0] >= np.minimum(p[:, 0], r[:, 0])) &
                    (q[:, 1] <= np.maximum(p[:, 1], r[:, 1])) &
                    (q[:, 1] >= np.minimum(p[:, 1], r[:, 1])))

        o1 = orientation(p1, p2, p3)
        o2 = orientation(p1, p2, p4)
        o3 = orientation(p3, p4, p1)
        o4 = orientation(p3, p4, p2)

        intersect = (o1 != o2) & (o3 != o4)
        intersect |= (o1 == 0) & on_segment(p1, p3, p2)
        intersect |= (o2 == 0) & on_segment(p1, p4, p2)
        intersect |= (o3 == 0) & on_segment(p3, p1, p4)
        intersect |= (o4 == 0) & on_segment(p3, p2, p4)

        return intersect

    def _update_boundary(self, new_element):
        """Update boundary after adding a new element using splice-based algorithm.

        Identifies the contiguous boundary segment consumed by the element,
        then replaces it with the element's non-boundary edges, preserving
        winding order.
        """
        n = len(self.boundary)
        if n == 0:
            return

        # 1. Find which boundary vertex indices are in the element
        elem_boundary_indices = []
        for i in range(n):
            for ev in new_element:
                if np.allclose(self.boundary[i], ev, atol=1e-10):
                    elem_boundary_indices.append(i)
                    break

        if len(elem_boundary_indices) == 0:
            return

        idx_set = set(elem_boundary_indices)

        # 2. Find start of the contiguous consumed segment:
        #    a boundary index IN the element whose predecessor is NOT.
        start = None
        for idx in elem_boundary_indices:
            if (idx - 1) % n not in idx_set:
                start = idx
                break

        if start is None:
            # All boundary vertices are in the element → fully consumed
            self.boundary = np.array([]).reshape(0, 2)
            return

        # 3. Walk forward to get the full consumed segment in boundary order
        consumed = []
        current = start
        while current in idx_set:
            consumed.append(current)
            idx_set.discard(current)
            current = (current + 1) % n

        # 4. Find the replacement vertices: the element path from consumed[-1]
        #    back to consumed[0] that goes through NON-boundary element vertices.
        elem_list = list(new_element)

        last_consumed_coord = self.boundary[consumed[-1]]
        first_consumed_coord = self.boundary[consumed[0]]

        end_elem_idx = None
        start_elem_idx = None
        for i, ev in enumerate(elem_list):
            if np.allclose(ev, last_consumed_coord, atol=1e-10):
                end_elem_idx = i
            if np.allclose(ev, first_consumed_coord, atol=1e-10):
                start_elem_idx = i

        if end_elem_idx is None or start_elem_idx is None:
            return

        # Walk element cycle from consumed[-1] to consumed[0] (the non-shared path)
        replacement_verts = []
        curr = (end_elem_idx + 1) % 4
        while curr != start_elem_idx:
            replacement_verts.append(elem_list[curr])
            curr = (curr + 1) % 4

        # 5. Build new boundary:
        #    consumed[-1] → non-consumed boundary → consumed[0] → replacement → (back to consumed[-1])
        new_boundary = []

        # consumed[-1] stays (end of consumed segment)
        new_boundary.append(self.boundary[consumed[-1]])

        # Walk non-consumed boundary: consumed[-1]+1 → ... → consumed[0]-1
        idx = (consumed[-1] + 1) % n
        while idx != consumed[0]:
            new_boundary.append(self.boundary[idx])
            idx = (idx + 1) % n

        # consumed[0] stays (start of consumed segment)
        new_boundary.append(self.boundary[consumed[0]])

        # Replacement vertices (non-boundary element path from consumed[0] → consumed[-1])
        new_boundary.extend(replacement_verts)

        if len(new_boundary) > 0:
            self.boundary = np.array(new_boundary)
        else:
            self.boundary = np.array([]).reshape(0, 2)
    
    def _is_quadrilateral(self, polygon):
        """Check if polygon is a quadrilateral"""
        return len(polygon) == 4
    
    def enumerate_valid_actions(self, n_angle=12, n_dist=4):
        """Enumerate all valid discrete actions for the current boundary state.

        Returns:
            actions: list of (action_type, new_vertex_or_None) tuples
            mask: boolean np.array of shape (1 + n_angle*n_dist,)
        """
        if hasattr(self, '_cached_actions') and self._cached_actions is not None:
            return self._cached_actions

        max_actions = 1 + n_angle * n_dist

        # Try vertices in order of increasing interior angle (fallback logic)
        vertex_order = self._get_vertices_by_angle()

        for ref_vertex in vertex_order:
            actions, mask = self._enumerate_for_vertex(ref_vertex, n_angle, n_dist, max_actions)
            if np.any(mask):
                self._cached_ref_vertex = ref_vertex.copy()
                self._cached_actions = (actions, mask)
                return actions, mask

        # No vertex yields valid actions — return all-False mask
        actions = [(0, None)] + [(1, None)] * (n_angle * n_dist)
        mask = np.zeros(max_actions, dtype=bool)
        self._cached_ref_vertex = vertex_order[0].copy() if vertex_order else None
        self._cached_actions = (actions, mask)
        return actions, mask

    def _get_vertices_by_angle(self):
        """Return boundary vertices sorted by interior angle (smallest first)."""
        if len(self.boundary) == 0:
            return []
        if len(self.boundary) <= 2:
            return [self.boundary[0]]

        angles_and_verts = []
        for i in range(len(self.boundary)):
            angles = []
            for j in range(1, min(self.n_rv + 1, len(self.boundary))):
                left_idx = (i - j) % len(self.boundary)
                right_idx = (i + j) % len(self.boundary)

                v1 = self.boundary[left_idx] - self.boundary[i]
                v2 = self.boundary[right_idx] - self.boundary[i]

                v1_norm = v1 / max(1e-10, np.linalg.norm(v1))
                v2_norm = v2 / max(1e-10, np.linalg.norm(v2))

                dot_product = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
                angle = np.arccos(dot_product)
                angles.append(angle)

            avg_angle = np.mean(angles)
            angles_and_verts.append((avg_angle, i))

        angles_and_verts.sort(key=lambda x: x[0])
        return [self.boundary[idx] for _, idx in angles_and_verts]

    def _enumerate_for_vertex(self, ref_vertex, n_angle, n_dist, max_actions):
        """Enumerate valid actions for a specific reference vertex (vectorized)."""
        actions = []
        mask = np.zeros(max_actions, dtype=bool)

        # --- Find ref_idx once for the whole method ---
        ref_idx = -1
        for i, v in enumerate(self.boundary):
            if np.allclose(v, ref_vertex, atol=1e-10):
                ref_idx = i
                break

        # --- Type-0: deterministic quad from existing boundary vertices ---
        element, valid = self._form_element_fast(ref_vertex, 0, None, ref_idx)
        actions.append((0, None))
        mask[0] = valid

        # --- Type-1: grid over fan-shaped interior angle ---
        if ref_idx == -1:
            # Pad remaining slots
            for _ in range(n_angle * n_dist):
                actions.append((1, None))
            return actions, mask

        radius = self._calculate_fan_shape_radius(ref_vertex)

        # Compute angular range (same logic as _get_fan_shape_points)
        n_bnd = len(self.boundary)
        left_idx = (ref_idx - 1) % n_bnd
        right_idx = (ref_idx + 1) % n_bnd

        left_v = self.boundary[left_idx] - ref_vertex
        right_v = self.boundary[right_idx] - ref_vertex

        left_angle = np.arctan2(left_v[1], left_v[0])
        right_angle = np.arctan2(right_v[1], right_v[0])

        # Interior is on the LEFT of the CCW boundary — sweep the reflex arc.
        if left_angle <= right_angle:
            left_angle += 2 * np.pi

        # Grid: n_angle angles x n_dist distances — vectorized
        angle_bins = np.linspace(right_angle, left_angle, n_angle + 2)[1:-1]
        dist_bins = np.linspace(0.2, 1.0, n_dist) * radius

        # Build all candidate vertices at once: shape (n_angle * n_dist, 2)
        angle_grid, dist_grid = np.meshgrid(angle_bins, dist_bins, indexing='ij')
        angles_flat = angle_grid.ravel()
        dists_flat = dist_grid.ravel()
        offsets = np.column_stack([dists_flat * np.cos(angles_flat),
                                   dists_flat * np.sin(angles_flat)])
        candidates = ref_vertex + offsets  # (N, 2)
        n_candidates = len(candidates)

        # --- Batch point-in-polygon for all candidates ---
        pip_mask = self._batch_point_in_polygon(candidates, self.boundary)

        # --- For PIP-passing candidates, batch form quads and validate ---
        v2 = self.boundary[(ref_idx + 1) % n_bnd]
        v4 = self.boundary[(ref_idx - 1) % n_bnd]

        # Build all quads: shape (n_candidates, 4, 2)
        # quad[k] = [ref_vertex, v2, candidates[k], v4]
        all_quads = np.empty((n_candidates, 4, 2))
        all_quads[:, 0, :] = ref_vertex
        all_quads[:, 1, :] = v2
        all_quads[:, 2, :] = candidates
        all_quads[:, 3, :] = v4

        # Batch validate quads (only where pip passed)
        valid_mask = np.zeros(n_candidates, dtype=bool)
        pip_indices = np.where(pip_mask)[0]
        if len(pip_indices) > 0:
            valid_mask[pip_indices] = self._batch_is_valid_quad(all_quads[pip_indices])

        # Build actions list
        for k in range(n_candidates):
            v = bool(valid_mask[k])
            actions.append((1, candidates[k] if v else None))
            mask[1 + k] = v

        return actions, mask

    def _get_enriched_state(self):
        """Create enriched 44-float state representation."""
        base_state = self._get_state()  # 22 floats

        # Boundary vertex count (normalized by initial count)
        boundary_count = len(self.boundary) / max(1, len(self.initial_boundary))

        # Elements placed (normalized — assume max ~50 elements)
        elements_placed = len(self.elements) / max(1, len(self.initial_boundary) * 4)

        # Num valid actions (normalized by max_actions=49)
        if hasattr(self, '_cached_actions') and self._cached_actions is not None:
            _, mask = self._cached_actions
            num_valid = np.sum(mask) / len(mask)
        else:
            num_valid = 0.0

        # Boundary shape: 8 evenly-spaced samples, RELATIVE to reference vertex
        boundary_samples = self._sample_boundary_points(8)
        if len(self.boundary) >= 3:
            ref = self._select_reference_vertex()
            boundary_samples = boundary_samples - ref  # make reference-relative
            # Normalize by fan radius for scale invariance
            radius = self._calculate_fan_shape_radius(ref)
            if radius > 1e-10:
                boundary_samples = boundary_samples / radius

        # Action-type availability: 2 booleans
        if hasattr(self, '_cached_actions') and self._cached_actions is not None:
            _, mask = self._cached_actions
            type0_valid = float(mask[0])
            type1_valid = float(np.any(mask[1:]))
        else:
            type0_valid = 0.0
            type1_valid = 0.0

        enriched = np.concatenate([
            base_state,                          # 22
            [boundary_count],                    # 1
            [elements_placed],                   # 1
            [num_valid],                         # 1
            boundary_samples.flatten(),          # 16
            [type0_valid, type1_valid],          # 2
        ]).astype(np.float32)                    # total = 43... need 44

        # Pad to exactly 44 if needed
        if len(enriched) < 44:
            enriched = np.pad(enriched, (0, 44 - len(enriched)))
        elif len(enriched) > 44:
            enriched = enriched[:44]

        return enriched

    def _sample_boundary_points(self, n_samples):
        """Sample n evenly-spaced points along the boundary polygon."""
        if len(self.boundary) < 2:
            return np.zeros((n_samples, 2))

        # Compute cumulative edge lengths
        edges = np.diff(np.vstack([self.boundary, self.boundary[0:1]]), axis=0)
        edge_lengths = np.linalg.norm(edges, axis=1)
        cum_lengths = np.concatenate([[0], np.cumsum(edge_lengths)])
        total_length = cum_lengths[-1]

        if total_length < 1e-10:
            return np.zeros((n_samples, 2))

        # Sample at evenly-spaced arc lengths
        sample_lengths = np.linspace(0, total_length, n_samples, endpoint=False)
        points = []

        for s in sample_lengths:
            # Find which edge this sample falls on
            edge_idx = np.searchsorted(cum_lengths[1:], s, side='right')
            edge_idx = min(edge_idx, len(self.boundary) - 1)

            # Interpolate along that edge
            local_s = s - cum_lengths[edge_idx]
            edge_len = edge_lengths[edge_idx]
            t = local_s / edge_len if edge_len > 1e-10 else 0.0

            p1 = self.boundary[edge_idx]
            p2 = self.boundary[(edge_idx + 1) % len(self.boundary)]
            points.append(p1 + t * (p2 - p1))

        return np.array(points)

    def _invalidate_action_cache(self):
        """Clear the cached valid actions."""
        self._cached_actions = None
        self._cached_ref_vertex = None

    def plot_domain(self):
        """Plot the initial domain with interior points"""
        plt.figure(figsize=(8, 8))
        
        # Plot boundary
        boundary = self.initial_boundary
        x, y = boundary[:, 0], boundary[:, 1]
        plt.plot(np.append(x, x[0]), np.append(y, y[0]), 'k-', linewidth=2)
        
        # Plot interior points
        if self.interior_points is not None and len(self.interior_points) > 0:
            plt.scatter(self.interior_points[:, 0], self.interior_points[:, 1], 
                       color='blue', marker='o', s=100)
        
        plt.grid(True)
        plt.axis('equal')
        plt.title('Initial Domain with Interior Point')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.show()

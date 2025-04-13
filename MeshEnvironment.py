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
        
        # Parameters
        self.n_rv = 2
        self.g = 3
        self.beta = 6
        
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
        
        # Reset the environment
        self.reset()
    
    def reset(self, seed=None):
        super().reset(seed=seed)
        self.boundary = self.initial_boundary.copy()
        self.elements = []
        self.element_qualities = []
        state = self._get_state()
        return state, {}
    
    def step(self, action):
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
            if np.array_equal(v, reference_vertex):
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
        fan_points = self._get_fan_shape_points(reference_vertex, radius)
        
        # Calculate remaining area ratio
        remaining_area = self._calculate_polygon_area(self.boundary)
        area_ratio = remaining_area / self.original_area
        
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
            if np.array_equal(v, reference_vertex):
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
            if np.array_equal(v, reference_vertex):
                left_idx = (i - 1) % len(self.boundary)
                right_idx = (i + 1) % len(self.boundary)
                break
        
        if left_idx == -1:
            angles = np.linspace(0, 2*np.pi, self.g+1)[:-1]
            for angle in angles:
                fan_points.append(reference_vertex + radius * np.array([np.cos(angle), np.sin(angle)]))
            return fan_points
        
        # Calculate angle between left and right vertices
        left_v = self.boundary[left_idx] - reference_vertex
        right_v = self.boundary[right_idx] - reference_vertex
        
        left_angle = np.arctan2(left_v[1], left_v[0])
        right_angle = np.arctan2(right_v[1], right_v[0])
        
        # Ensure right angle is ahead of left angle
        if right_angle < left_angle:
            right_angle += 2 * np.pi
            
        angles = np.linspace(left_angle, right_angle, self.g+2)[1:-1]
        
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
            if np.array_equal(v, reference_vertex):
                ref_idx = i
                break
        
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
        """Check if quadrilateral is convex"""
        for i in range(4):
            prev = (i - 1) % 4
            curr = i
            next = (i + 1) % 4
            
            v1 = quad[prev] - quad[curr]
            v2 = quad[next] - quad[curr]
            
            cross_z = v1[0] * v2[1] - v1[1] * v2[0]
            if cross_z <= 0:
                return False
                
        return True
    
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
    
    def _update_boundary(self, new_element):
        """Update boundary after adding a new element"""
        boundary_list = self.boundary.tolist()
        element_points = new_element.tolist()
        
        # Create list of element edges
        element_edges = []
        for i in range(len(element_points)):
            edge = [element_points[i], element_points[(i+1) % len(element_points)]]
            edge.sort(key=lambda p: (p[0], p[1]))
            element_edges.append(edge)
        
        # Find edges to remove
        edges_to_remove = []
        for i in range(len(boundary_list)):
            edge = [boundary_list[i], boundary_list[(i+1) % len(boundary_list)]]
            edge.sort(key=lambda p: (p[0], p[1]))
            
            if edge in element_edges:
                edges_to_remove.append(i)
        
        edges_to_remove.sort(reverse=True)
        
        # Remove edges
        for idx in edges_to_remove:
            boundary_list.pop((idx + 1) % len(boundary_list))
        
        # Add new edges
        for edge in element_edges:
            edge_in_boundary = False
            for i in range(len(boundary_list)):
                b_edge = [boundary_list[i], boundary_list[(i+1) % len(boundary_list)]]
                b_edge.sort(key=lambda p: (p[0], p[1]))
                
                if edge == b_edge:
                    edge_in_boundary = True
                    break
            
            if not edge_in_boundary:
                for i in range(len(boundary_list)):
                    if boundary_list[i] == edge[0] or boundary_list[i] == edge[1]:
                        boundary_list.insert(i+1, edge[1] if boundary_list[i] == edge[0] else edge[0])
                        break
        
        self.boundary = np.array(boundary_list)
    
    def _is_quadrilateral(self, polygon):
        """Check if polygon is a quadrilateral"""
        return len(polygon) == 4
    
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

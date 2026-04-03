import numpy as np
import gym
from gym import spaces

from src.MeshEnvironment import MeshEnvironment

__all__ = ['DiscreteActionEnv']


class DiscreteActionEnv(gym.Wrapper):
    """Gymnasium wrapper that exposes MeshEnvironment with discrete, pre-computed valid actions.

    Action space: Discrete(max_actions) where max_actions = 1 + n_angle * n_dist = 49
    Observation: enriched 44-float state (from _get_enriched_state)
    Valid-action mask: stored in info["action_mask"] as boolean array
    """

    def __init__(self, env: MeshEnvironment, n_angle: int = 12, n_dist: int = 4,
                 no_valid_penalty: float = -2.0):
        super().__init__(env)
        self.n_angle = n_angle
        self.n_dist = n_dist
        self.max_actions = 1 + n_angle * n_dist  # 49
        self.no_valid_penalty = no_valid_penalty
        self._prev_boundary_count = None
        self._prev_area_ratio = None

        self.action_space = spaces.Discrete(self.max_actions)
        self.observation_space = spaces.Box(
            low=-2, high=2, shape=(44,), dtype=np.float32
        )

        self._valid_actions = None
        self._action_mask = None

    def reset(self, **kwargs):
        state, info = self.env.reset(**kwargs)
        self._prev_boundary_count = len(self.env.boundary)
        self._prev_area_ratio = 1.0
        self._enumerate()
        enriched = self.env._get_enriched_state()
        info["action_mask"] = self._action_mask.copy()
        return enriched, info

    def step(self, action_index: int):
        # Check if the chosen action is valid
        if not self._action_mask[action_index]:
            # Invalid action chosen — treat as failed step
            self.env._invalidate_action_cache()
            self._enumerate()
            enriched = self.env._get_enriched_state()
            info = {"valid": False, "action_mask": self._action_mask.copy()}
            return enriched, -0.1, False, False, info

        # Look up pre-computed action
        action_type, new_vertex = self._valid_actions[action_index]

        # Use the same reference vertex that enumerate used (critical for consistency)
        ref_vertex = self.env._cached_ref_vertex
        if ref_vertex is None:
            ref_vertex = self.env._select_reference_vertex()
        ref_idx = -1
        for i, v in enumerate(self.env.boundary):
            if np.allclose(v, ref_vertex, atol=1e-10):
                ref_idx = i
                break

        # Form element directly (bypassing the continuous action mapping)
        new_element, valid = self.env._form_element(ref_vertex, action_type, new_vertex)

        if not valid:
            # Shouldn't happen since we pre-validated, but handle gracefully
            self.env._invalidate_action_cache()
            self._enumerate()
            enriched = self.env._get_enriched_state()
            info = {"valid": False, "action_mask": self._action_mask.copy()}
            return enriched, -0.1, False, False, info

        # Add element to mesh
        self.env.elements.append(new_element)
        quality = self.env._calculate_element_quality(new_element)
        self.env.element_qualities.append(quality)

        # Update boundary
        self.env._update_boundary(new_element)
        self.env._invalidate_action_cache()

        # Check if meshing complete
        remaining_area = self.env._calculate_polygon_area(self.env.boundary)
        area_ratio = remaining_area / max(1e-10, self.env.original_area)

        done = False
        eta_e, eta_b, mu = 0.0, 0.0, 0.0
        bnd_len = len(self.env.boundary)
        if bnd_len < 3:
            # Boundary fully consumed — all quads, best possible outcome
            mean_q = np.mean(self.env.element_qualities) if self.env.element_qualities else 0.0
            reward = 5.0 + 10.0 * mean_q  # quality-gated completion bonus
            done = True
        elif bnd_len == 3:
            # Triangle remainder — check if boundary or interior
            tri = np.array(self.env.boundary)
            self.env.elements.append(tri)
            self.env.element_qualities.append(0.3)
            self.env.boundary = np.empty((0, 2))
            mean_q = np.mean(self.env.element_qualities) if self.env.element_qualities else 0.0
            if self.env.is_boundary_triangle(tri):
                reward = 5.0 + 10.0 * mean_q  # boundary triangle: quality-gated
            else:
                reward = 3.0  # interior triangle: heavy penalty
            done = True
        elif bnd_len == 4:
            bnd_quad = np.array(self.env.boundary)
            # Accept any non-self-intersecting quad (convex or concave)
            if not self.env._has_self_intersection(bnd_quad):
                quality_final = self.env._calculate_element_quality(bnd_quad)
                self.env.elements.append(bnd_quad)
                self.env.element_qualities.append(quality_final)
                self.env.boundary = np.empty((0, 2))
                mean_q = np.mean(self.env.element_qualities) if self.env.element_qualities else 0.0
                reward = 5.0 + 10.0 * mean_q  # quality-gated completion bonus
                done = True
            else:
                # Self-intersecting quad → 2 triangles (worst outcome)
                bnd = self.env.boundary
                self.env.elements.append(np.array([bnd[0], bnd[1], bnd[2]]))
                self.env.elements.append(np.array([bnd[0], bnd[2], bnd[3]]))
                self.env.element_qualities.extend([0.2, 0.2])
                self.env.boundary = np.empty((0, 2))
                reward = 2.0  # worst completion
                done = True
        else:
            # --- Pan et al. per-step reward: r = eta_e + eta_b + mu ---

            # eta_e: element quality at full weight (0 to 1)
            eta_e = quality

            # eta_b: boundary quality penalty (-1 to 0)
            # Penalizes actions that leave sharp remaining angles
            min_boundary_angle = self.env.compute_min_boundary_angle()
            eta_b = min_boundary_angle / 180.0 - 1.0  # maps [0,180] → [-1, 0]

            # mu: density penalty (-1 to 0)
            # Penalizes elements below minimum area threshold
            element_area = self.env._calculate_polygon_area(new_element)
            A_min = 0.01 * self.env.original_area
            A_max = 0.1 * self.env.original_area
            if element_area < A_min:
                mu = -1.0
            elif element_area < A_max:
                mu = (element_area - A_min) / (A_max - A_min) - 1.0
            else:
                mu = 0.0

            reward = eta_e + 0.3 * eta_b + mu

        self._prev_area_ratio = area_ratio
        # Track boundary for info
        self._prev_boundary_count = len(self.env.boundary)

        # Re-enumerate for next state (skip if done — boundary may be degenerate)
        if not done:
            self._enumerate()
        else:
            self._valid_actions = [(0, None)] * self.max_actions
            self._action_mask = np.zeros(self.max_actions, dtype=bool)

        enriched = self.env._get_enriched_state()

        # If no valid actions for next step, truncate
        truncated = False
        if not done and not np.any(self._action_mask):
            truncated = True
            reward += self.no_valid_penalty

        info = {
            "valid": True,
            "action_mask": self._action_mask.copy(),
            "complete": done,
            "element_quality": quality,
            "eta_e": eta_e if not done else 0.0,
            "eta_b": eta_b if not done else 0.0,
            "mu": mu if not done else 0.0,
            "completion_bonus": reward if done else 0.0,
        }

        return enriched, reward, done, truncated, info

    def _enumerate(self):
        """Enumerate valid actions and store mask."""
        self._valid_actions, self._action_mask = self.env.enumerate_valid_actions(
            n_angle=self.n_angle, n_dist=self.n_dist
        )

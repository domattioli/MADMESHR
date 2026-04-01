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
                 no_valid_penalty: float = -5.0):
        super().__init__(env)
        self.n_angle = n_angle
        self.n_dist = n_dist
        self.max_actions = 1 + n_angle * n_dist  # 49
        self.no_valid_penalty = no_valid_penalty

        self.action_space = spaces.Discrete(self.max_actions)
        self.observation_space = spaces.Box(
            low=-2, high=2, shape=(44,), dtype=np.float32
        )

        self._valid_actions = None
        self._action_mask = None

    def reset(self, **kwargs):
        state, info = self.env.reset(**kwargs)
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

        # Delegate to underlying environment
        ref_vertex = self.env._select_reference_vertex()
        ref_idx = -1
        for i, v in enumerate(self.env.boundary):
            if np.array_equal(v, ref_vertex):
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
        area_ratio = remaining_area / self.env.original_area

        done = False
        if len(self.env.boundary) < 3:
            # Boundary fully consumed — meshing complete
            reward = 10.0
            done = True
        elif len(self.env.boundary) <= 4 and self.env._is_quadrilateral(self.env.boundary):
            reward = 10.0
            done = True
        else:
            reward = self.env._calculate_reward(new_element, quality, area_ratio)

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
        }

        return enriched, reward, done, truncated, info

    def _enumerate(self):
        """Enumerate valid actions and store mask."""
        self._valid_actions, self._action_mask = self.env.enumerate_valid_actions(
            n_angle=self.n_angle, n_dist=self.n_dist
        )

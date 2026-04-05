import numpy as np
import gym
from gym import spaces

from src.MeshEnvironment import MeshEnvironment

__all__ = ['DiscreteActionEnv']


class DiscreteActionEnv(gym.Wrapper):
    """Gymnasium wrapper that exposes MeshEnvironment with discrete, pre-computed valid actions.

    Action space: Discrete(max_actions) where max_actions = 1 + n_angle * n_dist + K_type2
    - Slot 0: type-0 (connect adjacent boundary vertices)
    - Slots 1..n_angle*n_dist: type-1 (interior vertex placement on angle/dist grid)
    - Slots n_angle*n_dist+1..max_actions-1: type-2 (proximity split actions)
    Observation: enriched 44-float state (from _get_enriched_state)
    Valid-action mask: stored in info["action_mask"] as boolean array
    """

    K_TYPE2 = 8  # max type-2 action slots

    def __init__(self, env: MeshEnvironment, n_angle: int = 12, n_dist: int = 4,
                 no_valid_penalty: float = -2.0):
        super().__init__(env)
        self.n_angle = n_angle
        self.n_dist = n_dist
        self._type01_actions = 1 + n_angle * n_dist  # 49
        self.max_actions = self._type01_actions + self.K_TYPE2  # 57
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

    def _fail_step(self):
        """Return a failed step result (invalid action or rollback)."""
        self.env._invalidate_action_cache()
        self._enumerate()
        enriched = self.env._get_enriched_state()
        info = {"valid": False, "action_mask": self._action_mask.copy()}
        return enriched, -0.1, False, False, info

    def step(self, action_index: int):
        # Check if the chosen action is valid
        if not self._action_mask[action_index]:
            return self._fail_step()

        # Look up pre-computed action
        action_type, action_data = self._valid_actions[action_index]

        # Branch: type-2 actions have a completely different path
        is_type2 = (action_type == 2)

        if is_type2:
            ref_idx, far_idx = action_data
            new_element, consumed = self.env._form_type2_element(ref_idx, far_idx)
            if new_element is None:
                return self._fail_step()
        else:
            new_vertex = action_data
            ref_vertex = self.env._cached_ref_vertex
            if ref_vertex is None:
                ref_vertex = self.env._select_reference_vertex()

            new_element, valid = self.env._form_element(ref_vertex, action_type, new_vertex)
            if not valid:
                return self._fail_step()

        # Check D: element overlap with existing elements (before committing)
        if self.env._element_overlaps_existing(new_element):
            return self._fail_step()

        # Add element to mesh
        self.env.elements.append(new_element)
        quality = self.env._calculate_element_quality(new_element)
        self.env.element_qualities.append(quality)

        # Update boundary
        saved_boundary = self.env.boundary.copy()
        saved_pending = [lp.copy() for lp in self.env.pending_loops] if self.env.pending_loops else []

        if is_type2:
            ok = self.env._update_boundary_type2(new_element, ref_idx, far_idx, consumed)
            if not ok:
                self.env.elements.pop()
                self.env.element_qualities.pop()
                self.env.boundary = saved_boundary
                self.env.pending_loops = saved_pending
                return self._fail_step()
        else:
            saved_boundary_count = len(saved_boundary)
            self.env._update_boundary(new_element)

            # Boundary growth guard: undo if boundary grew
            if len(self.env.boundary) > saved_boundary_count:
                self.env.elements.pop()
                self.env.element_qualities.pop()
                self.env.boundary = saved_boundary
                return self._fail_step()

        # Check C: boundary self-intersection guard (rollback if detected)
        if len(self.env.boundary) >= 3 and self.env._boundary_has_self_intersection():
            self.env.elements.pop()
            self.env.element_qualities.pop()
            self.env.boundary = saved_boundary
            self.env.pending_loops = saved_pending
            return self._fail_step()

        self.env._invalidate_action_cache()

        # Check if meshing complete
        remaining_area = self.env._calculate_polygon_area(self.env.boundary)
        area_ratio = remaining_area / max(1e-10, self.env.original_area)

        # Helper: check and activate pending loops
        has_pending = hasattr(self.env, 'pending_loops') and len(self.env.pending_loops) > 0
        sub_loop_activated = False

        # Activate pending loop if current boundary is complete
        if len(self.env.boundary) < 3 and has_pending:
            self.env._activate_next_loop()
            sub_loop_activated = True

        done = False
        eta_e, eta_b, mu = 0.0, 0.0, 0.0
        split_bonus = 0.0
        sub_loop_bonus = 0.0
        bnd_len = len(self.env.boundary)
        if bnd_len < 3:
            # Boundary fully consumed and no pending loops — mesh complete
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
            # Check for pending loops before marking done
            if hasattr(self.env, 'pending_loops') and self.env.pending_loops:
                self.env._activate_next_loop()
                sub_loop_activated = True
                done = False
            else:
                done = True
        elif bnd_len == 4:
            bnd_quad = np.array(self.env.boundary)
            centroid = np.mean(bnd_quad, axis=0)
            centroid_ok = self.env._point_in_polygon(centroid, self.env.initial_boundary)
            # Accept non-self-intersecting quad with centroid inside domain
            if not self.env._has_self_intersection(bnd_quad) and centroid_ok:
                quality_final = self.env._calculate_element_quality(bnd_quad)
                self.env.elements.append(bnd_quad)
                self.env.element_qualities.append(quality_final)
                self.env.boundary = np.empty((0, 2))
                mean_q = np.mean(self.env.element_qualities) if self.env.element_qualities else 0.0
                reward = 5.0 + 10.0 * mean_q  # quality-gated completion bonus
                # Check for pending loops before marking done
                if hasattr(self.env, 'pending_loops') and self.env.pending_loops:
                    self.env._activate_next_loop()
                    sub_loop_activated = True
                    done = False
                    # Adjust reward: not a completion, use per-step reward instead
                    reward = quality_final + 0.3 * (self.env.compute_min_boundary_angle() / 180.0 - 1.0)
                else:
                    done = True
            else:
                # Self-intersecting quad → 2 triangles (worst outcome)
                bnd = self.env.boundary
                self.env.elements.append(np.array([bnd[0], bnd[1], bnd[2]]))
                self.env.elements.append(np.array([bnd[0], bnd[2], bnd[3]]))
                self.env.element_qualities.extend([0.2, 0.2])
                self.env.boundary = np.empty((0, 2))
                reward = 2.0  # worst completion
                if hasattr(self.env, 'pending_loops') and self.env.pending_loops:
                    self.env._activate_next_loop()
                    sub_loop_activated = True
                    done = False
                else:
                    done = True
        else:
            # --- Per-step reward ---
            eta_e = quality

            # eta_b: boundary quality penalty (-1 to 0)
            min_boundary_angle = self.env.compute_min_boundary_angle()
            eta_b = min_boundary_angle / 180.0 - 1.0  # maps [0,180] → [-1, 0]

            if is_type2:
                # Type-2 reward: eta_e + 0.3*eta_b + 0.3 (split bonus), skip mu
                split_bonus = 0.3
                reward = eta_e + 0.3 * eta_b + split_bonus
            else:
                # Type-0/1 reward: eta_e + 0.3*eta_b + mu
                # Scale density thresholds by expected element count
                element_area = self.env._calculate_polygon_area(new_element)
                n_expected = max(1, len(self.env.initial_boundary) / 2)
                ideal_area = self.env.original_area / n_expected
                A_min = 0.1 * ideal_area
                A_max = 0.5 * ideal_area
                if element_area < A_min:
                    mu = -1.0
                elif element_area < A_max:
                    mu = (element_area - A_min) / (A_max - A_min) - 1.0
                else:
                    mu = 0.0
                reward = eta_e + 0.3 * eta_b + mu

        # Sub-loop completion bonus: +2.0 when pending loop activates
        if sub_loop_activated and not done:
            sub_loop_bonus = 2.0
            reward += sub_loop_bonus

        self._prev_area_ratio = area_ratio
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
            "split_bonus": split_bonus,
            "sub_loop_bonus": sub_loop_bonus,
            "completion_bonus": reward if done else 0.0,
        }

        return enriched, reward, done, truncated, info

    def _enumerate(self):
        """Enumerate valid actions (type-0, type-1, type-2) and store mask."""
        # Type-0 and type-1 actions (slots 0..48)
        actions_01, mask_01 = self.env.enumerate_valid_actions(
            n_angle=self.n_angle, n_dist=self.n_dist
        )

        # Extend to full action space
        self._valid_actions = list(actions_01) + [(2, None)] * self.K_TYPE2
        self._action_mask = np.zeros(self.max_actions, dtype=bool)
        self._action_mask[:self._type01_actions] = mask_01

        # Type-2 actions (slots 49..56): all proximity pairs on boundary, sorted by distance
        seen_pairs = set()
        all_type2 = []
        for i in range(len(self.env.boundary)):
            for far_idx, dist in self.env._find_proximity_pairs(i, threshold=0.02, min_gap=3):
                pair_key = (min(i, far_idx), max(i, far_idx))
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)
                all_type2.append((i, far_idx, dist))
        all_type2.sort(key=lambda x: x[2])

        slot = self._type01_actions
        for ref_i, far_i, dist in all_type2:
            if slot >= self.max_actions:
                break
            # Validate: can we form a type-2 element from this pair?
            elem, consumed = self.env._form_type2_element(ref_i, far_i)
            if elem is None:
                continue
            # Check A: element edges vs original boundary
            elem_batch = elem.reshape(1, 4, 2)
            if self.env._batch_edges_cross_original_boundary(elem_batch)[0]:
                continue
            # Check B: element edges vs current boundary
            if self.env._batch_edges_cross_current_boundary(elem_batch, ref_i)[0]:
                continue
            self._valid_actions[slot] = (2, (ref_i, far_i))
            self._action_mask[slot] = True
            slot += 1

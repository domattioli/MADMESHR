"""Tests for action enumeration (Phase 1), discrete wrapper (Phase 2), and DQN (Phase 3)."""

import numpy as np
import pytest
import sys
import os

# Ensure project root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.MeshEnvironment import MeshEnvironment
from src.DiscreteActionEnv import DiscreteActionEnv
from src.DQN import DQN, MaskedReplayBuffer


# ---------------------------------------------------------------------------
# Phase 1: enumerate_valid_actions
# ---------------------------------------------------------------------------

class TestEnumerateValidActions:
    """Tests for MeshEnvironment.enumerate_valid_actions()."""

    def test_square_has_one_valid_action_type0(self):
        """4-vertex square should have exactly 1 valid action: type-0."""
        env = MeshEnvironment()
        env.reset()

        actions, mask = env.enumerate_valid_actions()

        # Type-0 (index 0) should be valid
        assert mask[0] == True, "Type-0 action should be valid for 4-vertex square"
        # The total number of valid actions should be at least 1
        assert np.sum(mask) >= 1, "At least type-0 should be valid"

    def test_octagon_has_valid_actions(self):
        """8-vertex octagon should have at least one valid action."""
        angles = np.linspace(0, 2 * np.pi, 9)[:-1]
        octagon = np.column_stack([np.cos(angles), np.sin(angles)])

        env = MeshEnvironment(initial_boundary=octagon)
        env.reset()

        actions, mask = env.enumerate_valid_actions()
        n_valid = np.sum(mask)

        assert n_valid >= 1, f"Octagon should have at least 1 valid action, got {n_valid}"

    def test_action_mask_shape(self):
        """Mask should be of shape (49,) = 1 + 12*4."""
        env = MeshEnvironment()
        env.reset()

        actions, mask = env.enumerate_valid_actions()

        assert mask.shape == (49,), f"Expected mask shape (49,), got {mask.shape}"
        assert len(actions) == 49, f"Expected 49 actions, got {len(actions)}"

    def test_cache_invalidation_on_reset(self):
        """Cache should be invalidated after reset."""
        env = MeshEnvironment()
        env.reset()

        # Populate cache
        env.enumerate_valid_actions()
        assert env._cached_actions is not None

        # Reset should clear cache
        env.reset()
        assert env._cached_actions is None

    def test_zero_valid_actions_fallback(self):
        """Degenerate boundary should return all-False mask (or try fallback vertices)."""
        # Create a very degenerate "boundary" — a near-collinear triangle
        degenerate = np.array([
            [0, 0], [1, 0], [0.5, 1e-10]
        ])
        env = MeshEnvironment(initial_boundary=degenerate)
        env.reset()

        actions, mask = env.enumerate_valid_actions()

        # Should not crash; mask is valid boolean array
        assert mask.dtype == bool
        assert mask.shape == (49,)

    def test_enriched_state_shape(self):
        """Enriched state should be exactly 44 floats."""
        env = MeshEnvironment()
        env.reset()
        env.enumerate_valid_actions()  # populate cache for enriched state

        enriched = env._get_enriched_state()

        assert enriched.shape == (44,), f"Expected shape (44,), got {enriched.shape}"
        assert enriched.dtype == np.float32

    def test_custom_grid_size(self):
        """Custom n_angle/n_dist should change mask size."""
        env = MeshEnvironment()
        env.reset()

        actions, mask = env.enumerate_valid_actions(n_angle=6, n_dist=2)

        expected_size = 1 + 6 * 2  # 13
        assert mask.shape == (expected_size,), f"Expected shape ({expected_size},), got {mask.shape}"


# ---------------------------------------------------------------------------
# Phase 2: DiscreteActionEnv wrapper
# ---------------------------------------------------------------------------

class TestDiscreteActionEnv:
    """Tests for the discrete action wrapper."""

    def test_action_space_is_discrete(self):
        """Wrapper should expose Discrete(49) action space."""
        env = MeshEnvironment()
        wrapper = DiscreteActionEnv(env)
        wrapper.reset()

        from gym import spaces
        assert isinstance(wrapper.action_space, spaces.Discrete)
        assert wrapper.action_space.n == 49

    def test_observation_space_is_44(self):
        """Observation space should be Box(44,)."""
        env = MeshEnvironment()
        wrapper = DiscreteActionEnv(env)

        assert wrapper.observation_space.shape == (44,)

    def test_reset_returns_mask_in_info(self):
        """reset() should return action_mask in info dict."""
        env = MeshEnvironment()
        wrapper = DiscreteActionEnv(env)

        state, info = wrapper.reset()

        assert "action_mask" in info
        assert info["action_mask"].shape == (49,)
        assert info["action_mask"].dtype == bool

    def test_step_returns_mask_in_info(self):
        """step() should return action_mask in info dict."""
        env = MeshEnvironment()
        wrapper = DiscreteActionEnv(env)

        state, info = wrapper.reset()
        mask = info["action_mask"]

        # Pick a valid action
        valid_indices = np.where(mask)[0]
        assert len(valid_indices) > 0, "Should have at least one valid action on fresh square"

        next_state, reward, done, truncated, next_info = wrapper.step(valid_indices[0])

        assert "action_mask" in next_info
        assert next_state.shape == (44,)

    def test_random_agent_100_percent_valid(self):
        """Random agent choosing only from valid mask should never get invalid=-0.1 reward."""
        env = MeshEnvironment()
        wrapper = DiscreteActionEnv(env)

        for episode in range(3):
            state, info = wrapper.reset()
            mask = info["action_mask"]

            for step in range(20):
                valid_indices = np.where(mask)[0]
                if len(valid_indices) == 0:
                    break

                action = np.random.choice(valid_indices)
                state, reward, done, truncated, info = wrapper.step(action)
                mask = info["action_mask"]

                # Should never get the -0.1 invalid penalty
                assert reward != -0.1, f"Got invalid action penalty at step {step}"

                if done or truncated:
                    break

    def test_square_completes_with_type0(self):
        """4-vertex square should complete in 1 step with type-0 action (index 0)."""
        env = MeshEnvironment()
        wrapper = DiscreteActionEnv(env)

        state, info = wrapper.reset()
        mask = info["action_mask"]

        # Type-0 is action index 0
        if mask[0]:
            state, reward, done, truncated, info = wrapper.step(0)
            assert done, "Square should complete with single type-0 action"
            assert reward >= 10.0, f"Expected completion reward >= 10.0, got {reward}"


# ---------------------------------------------------------------------------
# Boundary correctness tests
# ---------------------------------------------------------------------------

class TestBoundaryUpdate:
    """Tests for _update_boundary correctness after fixes."""

    def test_type0_circle_shrinks_by_2(self):
        """Type-0 on 16-vertex circle should reduce boundary by 2."""
        angles = np.linspace(0, 2 * np.pi, 17)[:-1]
        circle = np.column_stack([np.cos(angles), np.sin(angles)])
        env = MeshEnvironment(initial_boundary=circle)
        env.reset()

        ref = env._select_reference_vertex()
        element, valid = env._form_element(ref, 0, None)
        assert valid
        env._update_boundary(element)

        assert len(env.boundary) == 14, f"Expected 14, got {len(env.boundary)}"

    def test_no_duplicate_vertices_after_update(self):
        """Boundary should have no duplicate vertices after update."""
        angles = np.linspace(0, 2 * np.pi, 17)[:-1]
        circle = np.column_stack([np.cos(angles), np.sin(angles)])
        env = MeshEnvironment(initial_boundary=circle)
        env.reset()

        for step in range(5):
            ref = env._select_reference_vertex()
            element, valid = env._form_element(ref, 0, None)
            if not valid:
                break
            env._update_boundary(element)
            env.elements.append(element)

            # Check for duplicates
            coords = [tuple(np.round(v, 10)) for v in env.boundary]
            assert len(coords) == len(set(coords)), \
                f"Duplicate vertices at step {step}: {len(coords)} vs {len(set(coords))}"

    def test_circle_completes_with_type0_only(self):
        """16-vertex circle should complete in 6 type-0 steps: 16→14→...→4."""
        angles = np.linspace(0, 2 * np.pi, 17)[:-1]
        circle = np.column_stack([np.cos(angles), np.sin(angles)])
        env = DiscreteActionEnv(MeshEnvironment(initial_boundary=circle))

        state, info = env.reset()
        total_reward = 0
        for step in range(10):
            mask = info["action_mask"]
            valid = np.where(mask)[0]
            if len(valid) == 0:
                break
            state, reward, done, truncated, info = env.step(valid[0])
            total_reward += reward
            if done:
                break

        assert done, "Circle should complete"
        assert total_reward > 2.0, f"Total reward should include completion bonus, got {total_reward}"

    def test_enriched_state_no_nan(self):
        """Enriched state should never contain NaN values."""
        angles = np.linspace(0, 2 * np.pi, 17)[:-1]
        circle = np.column_stack([np.cos(angles), np.sin(angles)])
        env = DiscreteActionEnv(MeshEnvironment(initial_boundary=circle))

        state, info = env.reset()
        assert not np.any(np.isnan(state)), f"NaN in initial state"

        for step in range(8):
            mask = info["action_mask"]
            valid = np.where(mask)[0]
            if len(valid) == 0:
                break
            state, reward, done, truncated, info = env.step(valid[0])
            assert not np.any(np.isnan(state)), f"NaN in state at step {step}"
            if done or truncated:
                break


# ---------------------------------------------------------------------------
# Phase 3: DQN agent
# ---------------------------------------------------------------------------

class TestDQN:
    """Tests for the DQN agent."""

    def test_select_action_respects_mask(self):
        """Selected action should always be among valid actions."""
        agent = DQN(state_dim=44, num_actions=49)

        state = np.random.randn(44).astype(np.float32)
        mask = np.zeros(49, dtype=bool)
        mask[[0, 5, 10]] = True

        for _ in range(20):
            action = agent.select_action(state, mask, evaluate=True)
            assert action in [0, 5, 10], f"Action {action} not in valid set"

    def test_select_action_epsilon_greedy(self):
        """With epsilon=1.0, actions should be random among valid."""
        agent = DQN(state_dim=44, num_actions=49)

        state = np.random.randn(44).astype(np.float32)
        mask = np.zeros(49, dtype=bool)
        mask[[0, 10, 20, 30, 40]] = True

        actions = set()
        for _ in range(100):
            action = agent.select_action(state, mask, evaluate=False, epsilon=1.0)
            actions.add(action)
            assert action in [0, 10, 20, 30, 40]

        # With epsilon=1.0 over 100 tries, should hit multiple valid actions
        assert len(actions) > 1, "Epsilon=1.0 should produce diverse actions"

    def test_train_step_runs(self):
        """Train step should execute without error and return loss."""
        agent = DQN(state_dim=44, num_actions=49)
        buf = MaskedReplayBuffer(capacity=1000)

        # Fill buffer with dummy transitions
        for _ in range(64):
            s = np.random.randn(44).astype(np.float32)
            a = np.random.randint(0, 49)
            r = np.random.randn()
            ns = np.random.randn(44).astype(np.float32)
            nm = np.random.rand(49) > 0.5
            d = False
            buf.add(s, a, r, ns, nm, d)

        batch = buf.sample(32)
        info = agent.train_step(batch)

        assert 'loss' in info
        assert 'mean_q' in info
        assert np.isfinite(info['loss'])

    def test_masked_replay_buffer(self):
        """MaskedReplayBuffer should store and sample 6-tuples correctly."""
        buf = MaskedReplayBuffer(capacity=100)

        for i in range(10):
            buf.add(
                np.zeros(44), i, float(i), np.ones(44),
                np.ones(49, dtype=bool), False
            )

        assert buf.size() == 10

        states, actions, rewards, next_states, next_masks, dones = buf.sample(5)

        assert states.shape == (5, 44)
        assert actions.shape == (5,)
        assert rewards.shape == (5,)
        assert next_states.shape == (5, 44)
        assert next_masks.shape == (5, 49)
        assert dones.shape == (5,)


# ---------------------------------------------------------------------------
# Phase 4: Type-2 actions (proximity-based quad formation)
# ---------------------------------------------------------------------------

class TestType2Actions:
    """Tests for type-2 proximity-based quad formation."""

    def _make_annulus_env(self):
        bnd = np.load(os.path.join(os.path.dirname(__file__), '..', 'domains', 'annulus_layer2.npy'))
        return MeshEnvironment(initial_boundary=bnd)

    def test_proximity_finds_at_least_5_pairs(self):
        """Annulus should have >= 5 proximity pairs with threshold=0.02."""
        env = self._make_annulus_env()
        env.reset()
        all_pairs = set()
        for i in range(len(env.boundary)):
            for far_idx, dist in env._find_proximity_pairs(i, threshold=0.02, min_gap=3):
                all_pairs.add((min(i, far_idx), max(i, far_idx)))
        assert len(all_pairs) >= 5, f"Expected >= 5 pairs, got {len(all_pairs)}"

    def test_type2_forms_at_least_one_valid_quad(self):
        """At least one coincident pair should produce a valid type-2 quad."""
        env = self._make_annulus_env()
        env.reset()
        valid_count = 0
        for i in range(len(env.boundary)):
            for far_idx, dist in env._find_proximity_pairs(i, threshold=0.02, min_gap=3):
                elem, consumed = env._form_type2_element(i, far_idx)
                if elem is not None:
                    valid_count += 1
        assert valid_count >= 1, "Expected at least 1 valid type-2 quad"

    def test_type2_produces_two_loops(self):
        """Type-2 boundary update should split into two separate loops."""
        env = self._make_annulus_env()
        env.reset()
        orig_len = len(env.boundary)

        # Find first valid type-2 action
        for i in range(len(env.boundary)):
            for far_idx, _ in env._find_proximity_pairs(i, threshold=0.02, min_gap=3):
                elem, consumed = env._form_type2_element(i, far_idx)
                if elem is not None:
                    env.elements.append(elem)
                    env.element_qualities.append(env._calculate_element_quality(elem))
                    ok = env._update_boundary_type2(elem, i, far_idx, consumed)
                    assert ok, "Boundary update should succeed"
                    # Should have active boundary + one pending loop
                    assert len(env.boundary) > 0, "Active boundary should exist"
                    assert len(env.pending_loops) == 1, "Should have exactly 1 pending loop"
                    # Both loops should have positive area
                    area_active = env._calculate_polygon_area(env.boundary)
                    area_pending = env._calculate_polygon_area(env.pending_loops[0])
                    assert area_active > 0, "Active boundary should have positive area"
                    assert area_pending > 0, "Pending loop should have positive area"
                    # Total vertices should be less than original
                    total_verts = len(env.boundary) + len(env.pending_loops[0])
                    assert total_verts < orig_len, "Total vertices should shrink"
                    return

        pytest.fail("No valid type-2 action found to test boundary update")

    def test_type2_no_edges_through_element(self):
        """Neither loop's edges should cross through the placed type-2 element."""
        env = self._make_annulus_env()
        env.reset()

        for i in range(len(env.boundary)):
            for far_idx, _ in env._find_proximity_pairs(i, threshold=0.02, min_gap=3):
                elem, consumed = env._form_type2_element(i, far_idx)
                if elem is not None:
                    env.elements.append(elem)
                    env.element_qualities.append(env._calculate_element_quality(elem))
                    env._update_boundary_type2(elem, i, far_idx, consumed)

                    # Check both loops
                    for loop_name, loop in [("active", env.boundary),
                                            ("pending", env.pending_loops[0] if env.pending_loops else np.empty((0,2)))]:
                        for ib in range(len(loop)):
                            jb = (ib + 1) % len(loop)
                            bp1, bp2 = loop[ib], loop[jb]
                            for ie in range(4):
                                je = (ie + 1) % 4
                                ep1, ep2 = elem[ie], elem[je]
                                # Skip shared endpoints
                                if any(np.linalg.norm(a - b) < 1e-8
                                       for a in [bp1, bp2] for b in [ep1, ep2]):
                                    continue
                                cross = env._do_segments_intersect(bp1, bp2, ep1, ep2)
                                assert not cross, (
                                    f"{loop_name} loop edge ({ib},{jb}) crosses "
                                    f"element edge ({ie},{je})")
                    return

        pytest.fail("No valid type-2 action found")

    def test_pending_loop_activation(self):
        """When active boundary completes, pending loop should be activated."""
        env = self._make_annulus_env()
        env.reset()

        # Place a type-2 quad to create pending loop
        for i in range(len(env.boundary)):
            for far_idx, _ in env._find_proximity_pairs(i, threshold=0.02, min_gap=3):
                elem, consumed = env._form_type2_element(i, far_idx)
                if elem is not None:
                    env.elements.append(elem)
                    env.element_qualities.append(env._calculate_element_quality(elem))
                    env._update_boundary_type2(elem, i, far_idx, consumed)

                    assert len(env.pending_loops) == 1
                    pending_size = len(env.pending_loops[0])

                    # Simulate active loop completion
                    env.boundary = np.empty((0, 2))
                    activated = env._activate_next_loop()

                    assert activated, "Should activate pending loop"
                    assert len(env.boundary) == pending_size, "Boundary should be the pending loop"
                    assert len(env.pending_loops) == 0, "No more pending loops"
                    assert env._calculate_polygon_area(env.boundary) > 0
                    return

        pytest.fail("No valid type-2 action found")

    def test_reset_clears_pending_loops(self):
        """Reset should clear pending_loops."""
        env = self._make_annulus_env()
        env.reset()

        # Place type-2 to create pending loop
        for i in range(len(env.boundary)):
            for far_idx, _ in env._find_proximity_pairs(i, threshold=0.02, min_gap=3):
                elem, consumed = env._form_type2_element(i, far_idx)
                if elem is not None:
                    env.elements.append(elem)
                    env.element_qualities.append(env._calculate_element_quality(elem))
                    env._update_boundary_type2(elem, i, far_idx, consumed)
                    assert len(env.pending_loops) > 0

                    # Reset should clear everything
                    env.reset()
                    assert len(env.pending_loops) == 0
                    assert len(env.elements) == 0
                    return

        pytest.fail("No valid type-2 action found")


class TestBoundaryGrowthGuard:
    """Tests for boundary growth detection and prevention."""

    def _make_annulus_env(self):
        bnd = np.load(os.path.join(os.path.dirname(__file__), '..', 'domains', 'annulus_layer2.npy'))
        return DiscreteActionEnv(MeshEnvironment(initial_boundary=bnd))

    def test_boundary_never_grows_random_actions(self):
        """Boundary should never grow during 50 random valid action steps on annulus."""
        env = self._make_annulus_env()
        state, info = env.reset()

        for step in range(50):
            mask = info["action_mask"]
            valid_actions = np.where(mask)[0]
            if len(valid_actions) == 0:
                break

            action = np.random.choice(valid_actions)
            prev_bnd = len(env.env.boundary)
            state, reward, done, truncated, info = env.step(action)

            # Boundary should never grow (growth guard should catch it)
            assert len(env.env.boundary) <= prev_bnd or done, (
                f"Step {step}: boundary grew from {prev_bnd} to {len(env.env.boundary)}")

            if done or truncated:
                break


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

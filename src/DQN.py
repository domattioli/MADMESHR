import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from collections import deque
import random

__all__ = ['MaskedReplayBuffer', 'DQN']


class MaskedReplayBuffer:
    """Replay buffer storing (state, action, reward, next_state, next_mask, done) tuples."""

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, next_mask, done):
        self.buffer.append((state, action, reward, next_state, next_mask, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, next_masks, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(next_masks),
            np.array(dones, dtype=np.float32),
        )

    def size(self):
        return len(self.buffer)


class DQN:
    """Dueling Double DQN with post-hoc action masking.

    Architecture: shared trunk -> value stream + advantage stream.
    Q = V + (A - mean_valid(A))
    Action masking: Q[invalid] = -inf after Q computation.
    """

    def __init__(self, state_dim=44, num_actions=49, gamma=0.95, tau=0.005, lr=3e-4):
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.gamma = gamma
        self.tau = tau

        self.online_net = self._build_network()
        self.target_net = self._build_network()
        self.target_net.set_weights(self.online_net.get_weights())

        self.optimizer = keras.optimizers.Adam(learning_rate=lr)

    def _build_network(self):
        """Build dueling DQN network: state -> (value, advantage) -> Q-values."""
        inputs = layers.Input(shape=(self.state_dim,))

        # Shared trunk
        x = layers.Dense(256, activation='relu')(inputs)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dense(128, activation='relu')(x)

        # Value stream
        value = layers.Dense(1)(x)

        # Advantage stream
        advantage = layers.Dense(self.num_actions)(x)

        # Combine: Q = V + (A - mean(A))
        # Note: mean_valid(A) is applied post-hoc with masking, so here we use
        # standard mean for the network output. Masking happens in select_action/train.
        q_values = layers.Lambda(
            lambda va: va[0] + (va[1] - tf.reduce_mean(va[1], axis=1, keepdims=True))
        )([value, advantage])

        return keras.Model(inputs=inputs, outputs=q_values)

    def select_action(self, state, valid_mask, evaluate=False, epsilon=0.0):
        """Select action using epsilon-greedy among valid actions.

        Args:
            state: (state_dim,) array
            valid_mask: (num_actions,) boolean array
            evaluate: if True, always greedy
            epsilon: exploration rate (ignored if evaluate=True)

        Returns:
            action index (int)
        """
        valid_indices = np.where(valid_mask)[0]
        if len(valid_indices) == 0:
            return 0  # fallback; caller should handle all-False mask

        if not evaluate and np.random.random() < epsilon:
            return np.random.choice(valid_indices)

        # Greedy: pick valid action with highest Q-value
        state_batch = np.expand_dims(state, axis=0)
        q_values = self.online_net(state_batch, training=False).numpy()[0]

        # Mask invalid actions
        q_values[~valid_mask] = -np.inf

        return int(np.argmax(q_values))

    def train_step(self, batch):
        """One gradient step on a batch from MaskedReplayBuffer.

        Uses Double DQN: action selection from online net, value from target net.
        Both masked by next_mask.

        Returns:
            dict with 'loss' and 'mean_q'
        """
        states, actions, rewards, next_states, next_masks, dones = batch

        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.int32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
        next_masks = tf.convert_to_tensor(next_masks, dtype=tf.bool)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)

        # Compute target Q-values using Double DQN
        # 1. Online net selects best valid action for next state
        next_q_online = self.online_net(next_states, training=False)
        # Mask invalid actions for selection
        masked_next_q = tf.where(next_masks, next_q_online, tf.fill(tf.shape(next_q_online), -1e9))
        best_next_actions = tf.argmax(masked_next_q, axis=1)

        # 2. Target net evaluates Q at those actions
        next_q_target = self.target_net(next_states, training=False)
        best_next_q = tf.gather(next_q_target, best_next_actions, axis=1, batch_dims=1)

        # TD target
        targets = rewards + (1.0 - dones) * self.gamma * best_next_q

        with tf.GradientTape() as tape:
            q_values = self.online_net(states, training=True)
            # Gather Q-values for the actions that were taken
            action_q = tf.gather(q_values, actions, axis=1, batch_dims=1)
            loss = tf.reduce_mean((action_q - tf.stop_gradient(targets)) ** 2)

        gradients = tape.gradient(loss, self.online_net.trainable_variables)
        gradients, _ = tf.clip_by_global_norm(gradients, 10.0)
        self.optimizer.apply_gradients(zip(gradients, self.online_net.trainable_variables))

        # Polyak update target network
        for target_var, online_var in zip(self.target_net.variables, self.online_net.variables):
            target_var.assign(self.tau * online_var + (1.0 - self.tau) * target_var)

        return {
            'loss': loss.numpy(),
            'mean_q': tf.reduce_mean(q_values).numpy(),
        }

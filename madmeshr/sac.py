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


__all__ = ['ReplayBuffer', 'SAC']


class ReplayBuffer:
    """Experience replay buffer for SAC."""
    
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        return states, actions, rewards, next_states, dones
    
    def size(self):
        return len(self.buffer)


class SAC:
    def __init__(self, state_dim, action_dim, hidden_dim=128, gamma=0.99, tau=0.005):
        self.gamma = gamma
        self.tau = tau
        self.alpha = 1.0
        self.action_dim = action_dim
        
        # Build networks
        self.actor = self._build_actor(state_dim, action_dim, hidden_dim)
        self.critic_1 = self._build_critic(state_dim, action_dim, hidden_dim)
        self.critic_2 = self._build_critic(state_dim, action_dim, hidden_dim)
        
        # Target networks
        self.target_critic_1 = self._build_critic(state_dim, action_dim, hidden_dim)
        self.target_critic_2 = self._build_critic(state_dim, action_dim, hidden_dim)
        self.target_critic_1.set_weights(self.critic_1.get_weights())
        self.target_critic_2.set_weights(self.critic_2.get_weights())
        
        # Optimizers
        self.actor_optimizer = keras.optimizers.Adam(learning_rate=3e-4)
        self.critic_1_optimizer = keras.optimizers.Adam(learning_rate=3e-4)
        self.critic_2_optimizer = keras.optimizers.Adam(learning_rate=3e-4)
        
        # Alpha
        self.target_entropy = -action_dim
        self.log_alpha = tf.Variable(tf.math.log(self.alpha), dtype=tf.float32)
        self.alpha_optimizer = keras.optimizers.Adam(learning_rate=3e-4)
    
    def _build_actor(self, state_dim, action_dim, hidden_dim):
        inputs = layers.Input(shape=(state_dim,))
        x = layers.Dense(hidden_dim, activation='relu')(inputs)
        x = layers.Dense(hidden_dim, activation='relu')(x)
        
        # Output mean and log_std separately
        mean = layers.Dense(action_dim, activation='tanh')(x)
        log_std = layers.Dense(action_dim)(x)
        log_std = layers.Lambda(lambda x: tf.clip_by_value(x, -20, 2))(log_std)
        
        model = keras.Model(inputs=inputs, outputs=[mean, log_std])
        return model
    
    def _build_critic(self, state_dim, action_dim, hidden_dim):
        state_input = layers.Input(shape=(state_dim,))
        action_input = layers.Input(shape=(action_dim,))
        
        x = layers.Concatenate()([state_input, action_input])
        x = layers.Dense(hidden_dim, activation='relu')(x)
        x = layers.Dense(hidden_dim, activation='relu')(x)
        q_value = layers.Dense(1)(x)
        
        return keras.Model(inputs=[state_input, action_input], outputs=q_value)
    
    def _sample_action(self, state):
        """Sample action and compute log probability"""
        mean, log_std = self.actor(state)
        std = tf.exp(log_std)
        normal_dist = tfp.distributions.Normal(mean, std)
        
        # Sample from normal distribution
        z = normal_dist.sample()
        action = tf.tanh(z)
        
        # Calculate log probability
        log_prob = normal_dist.log_prob(z)
        # Apply tanh squashing correction
        log_prob = tf.reduce_sum(log_prob - tf.math.log(1 - action**2 + 1e-6), axis=1, keepdims=True)
        return action, log_prob
    
    def select_action(self, state):
        """Select action for evaluation"""
        state = np.expand_dims(state, axis=0) if state.ndim == 1 else state
        
        # For inference, use mean action (no sampling)
        mean, _ = self.actor(state)
        action = tf.tanh(mean)
        return action.numpy()[0]
    
    def train_step(self, batch):
        states, actions, rewards, next_states, dones = batch
        
        # Convert to tensors
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)
        
        # Get current alpha
        alpha = tf.exp(self.log_alpha)
        
        # Update critics
        with tf.GradientTape(persistent=True) as tape:
            # Sample actions from actor for next states
            next_actions, next_log_probs = self._sample_action(next_states)
            
            # Compute target Q-values
            target_q1 = self.target_critic_1([next_states, next_actions])
            target_q2 = self.target_critic_2([next_states, next_actions])
            target_q = tf.minimum(target_q1, target_q2) - alpha * next_log_probs
            target_q = rewards + (1 - dones) * self.gamma * target_q
            
            # Compute current Q-values
            current_q1 = self.critic_1([states, actions])
            current_q2 = self.critic_2([states, actions])
            
            # Compute critic losses
            critic_1_loss = tf.reduce_mean((current_q1 - target_q) ** 2)
            critic_2_loss = tf.reduce_mean((current_q2 - target_q) ** 2)
        
        # Update critics
        critic_1_gradients = tape.gradient(critic_1_loss, self.critic_1.trainable_variables)
        critic_2_gradients = tape.gradient(critic_2_loss, self.critic_2.trainable_variables)
        self.critic_1_optimizer.apply_gradients(zip(critic_1_gradients, self.critic_1.trainable_variables))
        self.critic_2_optimizer.apply_gradients(zip(critic_2_gradients, self.critic_2.trainable_variables))
        
        # Update actor
        with tf.GradientTape() as tape:
            # Sample actions and log probs
            actions, log_probs = self._sample_action(states)
            
            # Compute Q-values
            q1 = self.critic_1([states, actions])
            q2 = self.critic_2([states, actions])
            q = tf.minimum(q1, q2)
            
            # Actor loss
            actor_loss = tf.reduce_mean(alpha * log_probs - q)
        
        actor_gradients = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_gradients, self.actor.trainable_variables))
        
        # Update alpha
        with tf.GradientTape() as tape:
            _, log_probs = self._sample_action(states)
            alpha_loss = -tf.reduce_mean(self.log_alpha * (log_probs + self.target_entropy))
        
        alpha_gradients = tape.gradient(alpha_loss, [self.log_alpha])
        self.alpha_optimizer.apply_gradients(zip(alpha_gradients, [self.log_alpha]))
        
        # Update target networks
        for target_var, source_var in zip(self.target_critic_1.variables, self.critic_1.variables):
            target_var.assign(self.tau * source_var + (1 - self.tau) * target_var)
        for target_var, source_var in zip(self.target_critic_2.variables, self.critic_2.variables):
            target_var.assign(self.tau * source_var + (1 - self.tau) * target_var)
        
        return {
            'actor_loss': actor_loss.numpy(),
            'critic_1_loss': critic_1_loss.numpy(),
            'critic_2_loss': critic_2_loss.numpy(),
            'alpha': alpha.numpy()
        }
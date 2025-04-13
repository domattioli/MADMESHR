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
        # Parameters
        self.gamma = gamma
        self.tau = tau
        self.alpha = 1.0
        
        # Build networks
        self.actor = self._build_actor(state_dim, action_dim, hidden_dim)
        self.critic_1 = self._build_critic(state_dim, action_dim, hidden_dim)
        self.critic_2 = self._build_critic(state_dim, action_dim, hidden_dim)
        
        # Build target networks
        self.target_critic_1 = self._build_critic(state_dim, action_dim, hidden_dim)
        self.target_critic_2 = self._build_critic(state_dim, action_dim, hidden_dim)
        
        # Copy weights
        self.target_critic_1.set_weights(self.critic_1.get_weights())
        self.target_critic_2.set_weights(self.critic_2.get_weights())
        
        # Optimizers
        self.actor_optimizer = keras.optimizers.Adam(learning_rate=3e-4)
        self.critic_1_optimizer = keras.optimizers.Adam(learning_rate=3e-4)
        self.critic_2_optimizer = keras.optimizers.Adam(learning_rate=3e-4)
        
        # Alpha optimization
        self.target_entropy = -action_dim
        self.log_alpha = tf.Variable(tf.math.log(self.alpha), dtype=tf.float32)
        self.alpha_optimizer = keras.optimizers.Adam(learning_rate=3e-4)
    
    def _build_actor(self, state_dim, action_dim, hidden_dim):
        """Build actor network with proper scalar log probability"""
        inputs = layers.Input(shape=(state_dim,))
        x = layers.Dense(hidden_dim, activation='relu')(inputs)
        x = layers.Dense(hidden_dim, activation='relu')(x)
        
        action_mean = layers.Dense(action_dim, activation='tanh')(x)
        log_std = layers.Dense(action_dim)(x)
        log_std = layers.Lambda(lambda x: tf.clip_by_value(x, -20, 2))(log_std)
        
        class SamplingLayer(layers.Layer):
            def call(self, inputs):
                mean, log_std = inputs
                std = tf.exp(log_std)
                normal_dist = tfp.distributions.Normal(mean, std)
                action = normal_dist.sample()
                tanh_action = tf.tanh(action)
                
                # Sum log probs to scalar per batch item
                log_prob = tf.reduce_sum(normal_dist.log_prob(action), axis=1, keepdims=True)
                log_prob -= tf.reduce_sum(tf.math.log(1 - tanh_action**2 + 1e-6), axis=1, keepdims=True)
                
                return tanh_action, log_prob
        
        action, log_prob = SamplingLayer()([action_mean, log_std])
        return keras.Model(inputs=inputs, outputs=[action, log_prob])
    
    def _build_critic(self, state_dim, action_dim, hidden_dim):
        state_input = layers.Input(shape=(state_dim,))
        action_input = layers.Input(shape=(action_dim,))
        
        x = layers.Concatenate()([state_input, action_input])
        x = layers.Dense(hidden_dim, activation='relu')(x)
        x = layers.Dense(hidden_dim, activation='relu')(x)
        q_value = layers.Dense(1)(x)
        
        return keras.Model(inputs=[state_input, action_input], outputs=q_value)
    
    def select_action(self, state, evaluate=False):
        state = np.expand_dims(state, axis=0) if state.ndim == 1 else state
        action, _ = self.actor(state)
        return action.numpy()[0]

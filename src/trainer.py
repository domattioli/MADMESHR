import numpy as np
from typing import Optional, Dict

from tqdm import trange
import matplotlib.pyplot as plt

from SAC import *             
from MeshEnvironment import * 

__all__ = ['SACTrainer']

class SACTrainer:
    def __init__(self, env, agent, replay_buffer,
                 total_timesteps=100_000,
                 batch_size=256,
                 initial_random_steps=10_000,
                 eval_interval=5_000,
                 max_ep_len=100):
        self.env = env
        self.agent = agent
        self.replay_buffer = replay_buffer
        
        self.total_timesteps = total_timesteps
        self.batch_size = batch_size
        self.initial_random_steps = initial_random_steps
        self.eval_interval = eval_interval
        self.max_ep_len = max_ep_len

        self.results = {
            'timesteps': [],
            'episode_returns': [],
            'actor_losses': [],
            'critic_losses': [],
            'alpha_values': []
        }

    def train(self):
        state, _ = self.env.reset()
        episode_return, episode_length = 0, 0

        for t in range(1, self.total_timesteps + 1):
            if t < self.initial_random_steps:
                action = self.env.action_space.sample()
            else:
                action = self.agent.select_action(state)

            next_state, reward, done, truncated, _ = self.env.step(action)
            self.replay_buffer.add(state, action, reward, next_state, done)

            episode_return += reward
            episode_length += 1
            state = next_state

            if done or truncated or episode_length >= self.max_ep_len:
                print(f"Episode Return: {episode_return:.2f} | Elements: {len(self.env.elements)}")
                self.results['timesteps'].append(t)
                self.results['episode_returns'].append(episode_return)
                state, _ = self.env.reset()
                episode_return, episode_length = 0, 0

            if self.replay_buffer.size() >= self.batch_size and t >= self.initial_random_steps:
                batch = self.replay_buffer.sample(self.batch_size)
                train_info = self.agent.train_step(batch)

                if t % 1000 == 0:
                    self.results['actor_losses'].append(train_info['actor_loss'])
                    self.results['critic_losses'].append(
                        (train_info['critic_1_loss'] + train_info['critic_2_loss']) / 2
                    )
                    self.results['alpha_values'].append(train_info['alpha'])
                    print(f"Timestep: {t} | Actor Loss: {train_info['actor_loss']:.4f} | "
                          f"Critic Loss: {(train_info['critic_1_loss'] + train_info['critic_2_loss']) / 2:.4f} | "
                          f"Alpha: {train_info['alpha']:.4f}")

            if t % self.eval_interval == 0:
                eval_return = self.evaluate()
                print(f"Evaluation at timestep {t}: {eval_return:.2f}")

        return self.results

    def evaluate(self, num_episodes=5, render=False):
        returns = []
        for _ in range(num_episodes):
            state, _ = self.env.reset()
            done, ep_len, total = False, 0, 0
            while not done and ep_len < self.max_ep_len:
                action = self.agent.select_action(state, evaluate=True)
                state, reward, done, truncated, _ = self.env.step(action)
                total += reward
                ep_len += 1
                if render: self.env.render()
                if truncated: break
            returns.append(total)
        return np.mean(returns)

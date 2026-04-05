import os
import numpy as np
from typing import Optional

from madmeshr.dqn import DQN, MaskedReplayBuffer
from madmeshr.discrete_action_env import DiscreteActionEnv

__all__ = ['DQNTrainer']


class DQNTrainer:
    """Training loop for Dueling Double DQN with action masking."""

    def __init__(self, env: DiscreteActionEnv, agent: DQN,
                 replay_buffer: MaskedReplayBuffer,
                 total_timesteps: int = 100_000,
                 batch_size: int = 64,
                 initial_random_steps: int = 1_000,
                 eval_interval: int = 5_000,
                 max_ep_len: int = 100,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.05,
                 epsilon_decay_frac: float = 0.7,
                 save_dir: Optional[str] = None):
        self.env = env
        self.agent = agent
        self.replay_buffer = replay_buffer

        self.total_timesteps = total_timesteps
        self.batch_size = batch_size
        self.initial_random_steps = initial_random_steps
        self.eval_interval = eval_interval
        self.max_ep_len = max_ep_len
        self.save_dir = save_dir
        self.best_eval_return = -np.inf

        # Epsilon schedule: linear decay over first epsilon_decay_frac of training
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = int(total_timesteps * epsilon_decay_frac)

        self.results = {
            'timesteps': [],
            'episode_returns': [],
            'losses': [],
            'mean_q_values': [],
            'epsilons': [],
            'elements_placed': [],
            'completion_rates': [],
        }

    def _get_epsilon(self, t):
        """Linear epsilon decay."""
        if t >= self.epsilon_decay_steps:
            return self.epsilon_end
        frac = t / self.epsilon_decay_steps
        return self.epsilon_start + frac * (self.epsilon_end - self.epsilon_start)

    def train(self):
        state, info = self.env.reset()
        mask = info["action_mask"]
        episode_return, episode_length = 0.0, 0
        completions, episodes = 0, 0
        ep_eta_e, ep_eta_b, ep_mu, ep_bonus = 0.0, 0.0, 0.0, 0.0

        for t in range(1, self.total_timesteps + 1):
            epsilon = self._get_epsilon(t)

            # Select action
            if t < self.initial_random_steps:
                valid_indices = np.where(mask)[0]
                if len(valid_indices) > 0:
                    action = np.random.choice(valid_indices)
                else:
                    action = 0
            else:
                action = self.agent.select_action(state, mask, evaluate=False, epsilon=epsilon)

            # Take step
            next_state, reward, done, truncated, info = self.env.step(action)
            next_mask = info["action_mask"]

            episode_return += reward
            episode_length += 1
            ep_eta_e += info.get("eta_e", 0.0)
            ep_eta_b += info.get("eta_b", 0.0)
            ep_mu += info.get("mu", 0.0)
            ep_bonus += info.get("completion_bonus", 0.0)

            # Mark terminal: either env signals done/truncated, or we hit max_ep_len
            is_terminal = done or truncated or (episode_length >= self.max_ep_len)

            # Store transition (6-tuple) — mark as done if terminal to prevent bootstrapping
            self.replay_buffer.add(state, action, reward, next_state, next_mask, is_terminal)

            state = next_state
            mask = next_mask

            # Episode end
            if is_terminal:
                episodes += 1
                if done and info.get("complete", False):
                    completions += 1

                self.results['timesteps'].append(t)
                self.results['episode_returns'].append(episode_return)
                self.results['elements_placed'].append(len(self.env.env.elements))
                self.results['epsilons'].append(epsilon)

                if t % 1000 < 50:
                    print(f"t={t} | Return: {episode_return:.2f} | "
                          f"Elements: {len(self.env.env.elements)} | "
                          f"Eps: {epsilon:.3f} | "
                          f"Completions: {completions}/{episodes} | "
                          f"EtaE: {ep_eta_e:.2f} EtaB: {ep_eta_b:.2f} Mu: {ep_mu:.2f} Bonus: {ep_bonus:.1f}")

                state, info = self.env.reset()
                mask = info["action_mask"]
                episode_return, episode_length = 0.0, 0
                ep_eta_e, ep_eta_b, ep_mu, ep_bonus = 0.0, 0.0, 0.0, 0.0

            # Train
            if self.replay_buffer.size() >= self.batch_size and t >= self.initial_random_steps:
                batch = self.replay_buffer.sample(self.batch_size)
                train_info = self.agent.train_step(batch)

                if t % 1000 == 0:
                    self.results['losses'].append(train_info['loss'])
                    self.results['mean_q_values'].append(train_info['mean_q'])
                    print(f"  Loss: {train_info['loss']:.4f} | Mean Q: {train_info['mean_q']:.4f}")

            # Evaluate
            if t % self.eval_interval == 0:
                self._eval_t = t
                eval_return, eval_completion = self.evaluate()
                self.results['completion_rates'].append(eval_completion)

                if self.save_dir:
                    self.agent.save_weights(os.path.join(self.save_dir, "latest"))
                    if eval_return > self.best_eval_return:
                        self.best_eval_return = eval_return
                        self.agent.save_weights(os.path.join(self.save_dir, "best"))
                        print(f"  Saved new best model (return={eval_return:.2f})")

        if self.save_dir:
            self.agent.save_weights(os.path.join(self.save_dir, "final"))
            print(f"Saved final model to {self.save_dir}/final")

        return self.results

    def evaluate(self, num_episodes: int = 5):
        """Deterministic evaluation (epsilon=0)."""
        returns = []
        completions = 0
        all_qualities = []
        all_elements = []
        all_eta_b = []

        for _ in range(num_episodes):
            state, info = self.env.reset()
            mask = info["action_mask"]
            ep_return, ep_len = 0.0, 0
            ep_eta_b_sum = 0.0
            done = False

            while not done and ep_len < self.max_ep_len:
                if not np.any(mask):
                    break
                action = self.agent.select_action(state, mask, evaluate=True)
                state, reward, done, truncated, info = self.env.step(action)
                mask = info["action_mask"]
                ep_return += reward
                ep_len += 1
                ep_eta_b_sum += info.get("eta_b", 0.0)
                if truncated:
                    break

            if done and info.get("complete", False):
                completions += 1
            returns.append(ep_return)
            mean_q = np.mean(self.env.env.element_qualities) if self.env.env.element_qualities else 0
            all_qualities.append(mean_q)
            all_elements.append(len(self.env.env.elements))
            all_eta_b.append(ep_eta_b_sum)

        avg_q = np.mean(all_qualities)
        avg_elem = np.mean(all_elements)
        avg_eta_b = np.mean(all_eta_b)
        print(f"  EVAL at t={self._eval_t}: Return={np.mean(returns):.2f} | "
              f"Completion={completions / num_episodes:.0%} | "
              f"MeanQ={avg_q:.3f} | AvgElements={avg_elem:.1f} | AvgEtaB={avg_eta_b:.2f}")

        return np.mean(returns), completions / num_episodes

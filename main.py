import argparse
import numpy as np

from src.MeshEnvironment import MeshEnvironment
from src.SAC import SAC, ReplayBuffer


def train_sac(env, agent, replay_buffer,
              total_timesteps=100000,
              batch_size=256,
              initial_random_steps=10000,
              eval_interval=5000,
              max_ep_len=100):
    """Train the agent using SAC."""
    episode_return = 0
    episode_length = 0
    state, _ = env.reset()

    results = {
        'timesteps': [],
        'episode_returns': [],
        'actor_losses': [],
        'critic_losses': [],
        'alpha_values': []
    }

    for t in range(1, total_timesteps + 1):
        if t < initial_random_steps:
            action = env.action_space.sample()
        else:
            action = agent.select_action(state)

        next_state, reward, done, truncated, info = env.step(action)
        episode_return += reward
        episode_length += 1

        replay_buffer.add(state, action, reward, next_state, done)
        state = next_state

        if done or truncated or episode_length >= max_ep_len:
            print(f"Episode Return: {episode_return:.2f} | Elements: {len(env.elements)}")
            results['timesteps'].append(t)
            results['episode_returns'].append(episode_return)

            state, _ = env.reset()
            episode_return = 0
            episode_length = 0

        if replay_buffer.size() >= batch_size and t >= initial_random_steps:
            batch = replay_buffer.sample(batch_size)
            train_info = agent.train_step(batch)

            if t % 1000 == 0:
                results['actor_losses'].append(train_info['actor_loss'])
                results['critic_losses'].append((train_info['critic_1_loss'] + train_info['critic_2_loss'])/2)
                results['alpha_values'].append(train_info['alpha'])

                print(f"Timestep: {t}/{total_timesteps} | "
                      f"Actor Loss: {train_info['actor_loss']:.4f} | "
                      f"Critic Loss: {(train_info['critic_1_loss'] + train_info['critic_2_loss'])/2:.4f} | "
                      f"Alpha: {train_info['alpha']:.4f}")

        if t % eval_interval == 0:
            eval_return = evaluate_sac(env, agent, num_episodes=5)
            print(f"Evaluation at timestep {t}: {eval_return:.2f}")

    return results


def evaluate_sac(env, agent, num_episodes=5, max_ep_len=100):
    """Evaluate SAC agent."""
    returns = []
    for _ in range(num_episodes):
        state, _ = env.reset()
        episode_return = 0
        episode_length = 0
        done = False

        while not done and episode_length < max_ep_len:
            action = agent.select_action(state)
            next_state, reward, done, truncated, _ = env.step(action)
            episode_return += reward
            episode_length += 1
            state = next_state
            if truncated:
                break

        returns.append(episode_return)

    return np.mean(returns)


def main():
    parser = argparse.ArgumentParser(description='MADMESHR: RL Quad Mesh Generator')
    parser.add_argument('--algorithm', type=str, default='dqn', choices=['dqn', 'sac'],
                        help='RL algorithm to use (default: dqn)')
    parser.add_argument('--timesteps', type=int, default=50_000,
                        help='Total training timesteps')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Training batch size')
    parser.add_argument('--eval-interval', type=int, default=5_000,
                        help='Evaluation interval')
    parser.add_argument('--save-dir', type=str, default=None,
                        help='Directory to save model checkpoints (best, latest, final)')
    parser.add_argument('--load-path', type=str, default=None,
                        help='Path to load model weights from before training')
    args = parser.parse_args()

    env = MeshEnvironment()

    if args.algorithm == 'dqn':
        from src.DiscreteActionEnv import DiscreteActionEnv
        from src.DQN import DQN, MaskedReplayBuffer
        from src.trainer_dqn import DQNTrainer

        discrete_env = DiscreteActionEnv(env)
        agent = DQN(state_dim=44, num_actions=discrete_env.max_actions)
        replay_buffer = MaskedReplayBuffer(capacity=100_000)
        trainer = DQNTrainer(
            env=discrete_env,
            agent=agent,
            replay_buffer=replay_buffer,
            total_timesteps=args.timesteps,
            batch_size=args.batch_size,
            eval_interval=args.eval_interval,
            save_dir=args.save_dir,
        )

        if args.load_path:
            agent.load_weights(args.load_path)
            print(f"Loaded weights from {args.load_path}")

        print(f"Training DQN on {args.timesteps} timesteps...")
        print(f"Action space: Discrete({discrete_env.max_actions})")
        print(f"Observation space: {discrete_env.observation_space}")
        results = trainer.train()

    else:
        agent = SAC(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.shape[0],
            hidden_dim=128,
        )
        replay_buffer = ReplayBuffer(capacity=100_000)

        print(f"Training SAC on {args.timesteps} timesteps...")
        results = train_sac(
            env, agent, replay_buffer,
            total_timesteps=args.timesteps,
            batch_size=args.batch_size,
            eval_interval=args.eval_interval,
        )

    print("Training complete.")


if __name__ == "__main__":
    main()

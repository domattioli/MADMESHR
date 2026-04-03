#!/usr/bin/env python3
"""MADMESHR: RL-based quad mesh generator.

Usage examples:
    # Train DQN on star domain with checkpoints
    python main.py --domain star --timesteps 100000 --save-dir checkpoints/star

    # Resume training from checkpoint
    python main.py --domain star --load-path checkpoints/star/best

    # Train on circle with custom resolution
    python main.py --domain circle --n-angle 24 --n-dist 8

    # Evaluate only (no training)
    python main.py --domain star --load-path checkpoints/star/best --eval-only
"""
import argparse
import os
import numpy as np

from src.MeshEnvironment import MeshEnvironment


# ---------------------------------------------------------------------------
# Domain definitions
# ---------------------------------------------------------------------------

DOMAINS = {}


def register_domain(name, description):
    """Decorator to register a domain factory function."""
    def decorator(fn):
        fn.description = description
        DOMAINS[name] = fn
        return fn
    return decorator


@register_domain("square", "4-vertex unit square (trivial, 1-step)")
def _make_square():
    return np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=float)


@register_domain("octagon", "8-vertex regular octagon")
def _make_octagon():
    angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    return np.column_stack([np.cos(angles), np.sin(angles)])


@register_domain("circle", "16-vertex circle approximation")
def _make_circle():
    angles = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    return np.column_stack([np.cos(angles), np.sin(angles)])


@register_domain("star", "10-vertex star (alternating radii 1.0/0.4)")
def _make_star():
    n = 10
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    radii = np.where(np.arange(n) % 2 == 0, 1.0, 0.4)
    return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])


@register_domain("l-shape", "6-vertex L-shaped concave domain")
def _make_l_shape():
    return np.array([
        [0.0, 0.0], [2.0, 0.0], [2.0, 1.0],
        [1.0, 1.0], [1.0, 2.0], [0.0, 2.0],
    ], dtype=float)


@register_domain("rectangle", "20-vertex elongated rectangle (4:1 aspect)")
def _make_rectangle():
    bottom = [[x, 0.0] for x in np.linspace(0, 4, 9)]
    right = [[4.0, y] for y in np.linspace(0, 1, 3)[1:]]
    top = [[x, 1.0] for x in np.linspace(4, 0, 9)[1:]]
    left = [[0.0, y] for y in np.linspace(1, 0, 3)[1:-1]]
    return np.array(bottom + right + top + left, dtype=float)


# ---------------------------------------------------------------------------
# SAC training (legacy)
# ---------------------------------------------------------------------------

def train_sac(env, agent, replay_buffer,
              total_timesteps=100000, batch_size=256,
              initial_random_steps=10000, eval_interval=5000, max_ep_len=100):
    """Train the agent using SAC."""
    episode_return = 0
    episode_length = 0
    state, _ = env.reset()

    results = {
        'timesteps': [], 'episode_returns': [],
        'actor_losses': [], 'critic_losses': [], 'alpha_values': [],
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
                results['critic_losses'].append(
                    (train_info['critic_1_loss'] + train_info['critic_2_loss']) / 2)
                results['alpha_values'].append(train_info['alpha'])

        if t % eval_interval == 0:
            eval_return = evaluate_sac(env, agent)
            print(f"  EVAL at t={t}: Return={eval_return:.2f}")

    return results


def evaluate_sac(env, agent, num_episodes=5, max_ep_len=100):
    """Evaluate SAC agent."""
    returns = []
    for _ in range(num_episodes):
        state, _ = env.reset()
        ep_return, ep_len, done = 0, 0, False
        while not done and ep_len < max_ep_len:
            action = agent.select_action(state)
            state, reward, done, truncated, _ = env.step(action)
            ep_return += reward
            ep_len += 1
            if truncated:
                break
        returns.append(ep_return)
    return np.mean(returns)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description='MADMESHR: RL Quad Mesh Generator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Available domains: {', '.join(DOMAINS.keys())}")

    # Algorithm & domain
    parser.add_argument('--algorithm', type=str, default='dqn',
                        choices=['dqn', 'sac'], help='RL algorithm (default: dqn)')
    parser.add_argument('--domain', type=str, default='square',
                        choices=list(DOMAINS.keys()),
                        help='Domain geometry (default: square)')

    # Training
    parser.add_argument('--timesteps', type=int, default=50_000,
                        help='Total training timesteps (default: 50000)')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Training batch size (default: 64)')
    parser.add_argument('--eval-interval', type=int, default=5_000,
                        help='Evaluate every N steps (default: 5000)')
    parser.add_argument('--warmup', type=int, default=5_000,
                        help='Random warmup steps before training (default: 5000)')
    parser.add_argument('--eval-every', type=int, default=None,
                        help='Alias for --eval-interval')

    # Action space (DQN only)
    parser.add_argument('--n-angle', type=int, default=12,
                        help='Angular bins for type-1 actions (default: 12)')
    parser.add_argument('--n-dist', type=int, default=4,
                        help='Radial bins for type-1 actions (default: 4)')

    # Checkpoints
    parser.add_argument('--save-dir', type=str, default=None,
                        help='Save model checkpoints to this directory')
    parser.add_argument('--load-path', type=str, default=None,
                        help='Load model weights before training')

    # Eval only
    parser.add_argument('--eval-only', action='store_true',
                        help='Skip training, just evaluate loaded model')

    return parser.parse_args()


def main():
    args = parse_args()

    # Resolve eval-every alias
    if args.eval_every is not None:
        args.eval_interval = args.eval_every

    # Build domain
    boundary = DOMAINS[args.domain]()
    env = MeshEnvironment(initial_boundary=boundary)
    print(f"Domain: {args.domain} ({len(boundary)} vertices)")

    if args.algorithm == 'dqn':
        from src.DiscreteActionEnv import DiscreteActionEnv
        from src.DQN import DQN, MaskedReplayBuffer
        from src.trainer_dqn import DQNTrainer
        from src.utils.visualization import run_dqn_eval_and_save

        discrete_env = DiscreteActionEnv(env, n_angle=args.n_angle, n_dist=args.n_dist)
        agent = DQN(state_dim=44, num_actions=discrete_env.max_actions)

        if args.load_path:
            agent.load_weights(args.load_path)
            print(f"Loaded weights from {args.load_path}")

        if args.eval_only:
            stats = run_dqn_eval_and_save(
                agent, boundary, args.domain,
                n_angle=args.n_angle, n_dist=args.n_dist)
            print(f"Eval: return={stats['return']:.2f} | "
                  f"quality={stats['mean_quality']:.3f} | "
                  f"elements={stats['n_elements']} ({stats['n_quads']}Q+{stats['n_triangles']}T) | "
                  f"complete={stats['completed']}")
            return

        replay_buffer = MaskedReplayBuffer(capacity=100_000)
        trainer = DQNTrainer(
            env=discrete_env, agent=agent, replay_buffer=replay_buffer,
            total_timesteps=args.timesteps, batch_size=args.batch_size,
            eval_interval=args.eval_interval, save_dir=args.save_dir,
        )

        n_actions = discrete_env.max_actions
        print(f"Training DQN | {args.timesteps} steps | "
              f"Actions: {n_actions} ({args.n_angle}x{args.n_dist}+1)")
        results = trainer.train()

        # Save visualization of final model
        stats = run_dqn_eval_and_save(
            agent, boundary, args.domain,
            n_angle=args.n_angle, n_dist=args.n_dist)
        print(f"\nFinal eval: return={stats['return']:.2f} | "
              f"quality={stats['mean_quality']:.3f} | "
              f"elements={stats['n_elements']} ({stats['n_quads']}Q+{stats['n_triangles']}T) | "
              f"complete={stats['completed']}")

    else:
        from src.SAC import SAC, ReplayBuffer

        agent = SAC(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.shape[0],
            hidden_dim=128,
        )
        replay_buffer = ReplayBuffer(capacity=100_000)

        print(f"Training SAC | {args.timesteps} steps | warmup={args.warmup}")
        results = train_sac(
            env, agent, replay_buffer,
            total_timesteps=args.timesteps, batch_size=args.batch_size,
            initial_random_steps=args.warmup,
            eval_interval=args.eval_interval,
        )

        # Save training curves
        os.makedirs("results", exist_ok=True)
        from src.utils.visualization import plot_training_results, save_mesh_result
        plot_training_results(results)
        import shutil
        if os.path.exists("training_results.png"):
            shutil.move("training_results.png", "results/training_results.png")

        # Run final evaluation episode and save mesh visualization
        print("\nRunning final evaluation episode...")
        eval_env = MeshEnvironment(initial_boundary=boundary)
        state, _ = eval_env.reset()
        ep_return = 0
        for step in range(200):
            action = agent.select_action(state)
            state, reward, done, truncated, info = eval_env.step(action)
            ep_return += reward
            if done or truncated:
                break

        n_quads = sum(1 for e in eval_env.elements if len(e) == 4)
        n_tris = sum(1 for e in eval_env.elements if len(e) == 3)
        mean_q = np.mean(eval_env.element_qualities) if eval_env.element_qualities else 0
        completed = done and info.get("complete", False)
        print(f"Final eval: return={ep_return:.2f} | "
              f"quality={mean_q:.3f} | "
              f"elements={len(eval_env.elements)} ({n_quads}Q+{n_tris}T) | "
              f"complete={completed}")

        save_mesh_result(eval_env, f"{args.domain}_sac_final", output_dir="results")

        # Post-processing disabled — focusing on Layers × FreeMeshRL algorithm.

    print("Training complete.")


if __name__ == "__main__":
    main()

import numpy as np
from sac_agent import SAC
from replay_buffer import ReplayBuffer
from mesh_env import MeshEnvironment
from trainer import SACTrainer

def main():
    # --- Initialize Environment ---
    env = MeshEnvironment()
    print(f"Observation space: {env.observation_space.shape}")
    print(f"Action space: {env.action_space.shape}")
    env.plot_domain()  # Optional: visualize mesh

    # --- Initialize Agent ---
    agent = SAC(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        hidden_dim=128
    )

    # --- Initialize Replay Buffer ---
    replay_buffer = ReplayBuffer(capacity=100_000)

    # --- Train with SACTrainer ---
    trainer = SACTrainer(env, agent, replay_buffer)
    results = trainer.train(
        total_timesteps=50_000,
        batch_size=256,
        initial_random_steps=10_000,
        eval_interval=5_000,
        max_ep_len=100
    )

    # Optionally, save results
    np.savez("training_results.npz", **results)

if __name__ == "__main__":
    main()

# main.py
from src.SAC import *
from src.trainer import *
from src.MeshEnvironment import *
from src.utilsvisualization.visualization import *

def main():
    # --- Environment Setup ---
    env = MeshEnvironment()
    env.plot_domain()
    print(f"Observation space: {env.observation_space.shape}")
    print(f"Action space: {env.action_space.shape}")

    # --- Agent and Trainer Setup ---
    agent = SAC(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        hidden_dim=128
    )
    buffer = ReplayBuffer(capacity=100_000)
    trainer = SACTrainer(agent, env, buffer)

    # --- Train the Agent ---
    results = trainer.train(
        total_timesteps=50_000,
        batch_size=256,
        initial_random_steps=1_000,
        eval_interval=5_000
    )

    # --- Visualize Results ---
    plot_training_results(results)
    visualize_mesh_generation(env, agent, max_steps=100)

if __name__ == "__main__":
    main()

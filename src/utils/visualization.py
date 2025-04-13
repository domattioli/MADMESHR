import matplotlib.pyplot as plt
import numpy as np

__all__ =['plot_training_results', 'visualize_mesh_generation']

def plot_training_results(results):
    """Plot training metrics over time."""
    plt.figure(figsize=(15, 10))

    # Episode returns
    plt.subplot(2, 2, 1)
    plt.plot(results['timesteps'], results['episode_returns'], label="Episode Return")
    plt.xlabel('Timesteps')
    plt.ylabel('Return')
    plt.title('Episode Return Over Time')
    plt.grid(True)

    # Actor loss
    plt.subplot(2, 2, 2)
    plt.plot(results['actor_losses'], label="Actor Loss", color='orange')
    plt.xlabel('Updates (x1000)')
    plt.ylabel('Loss')
    plt.title('Actor Loss')
    plt.grid(True)

    # Critic loss
    plt.subplot(2, 2, 3)
    plt.plot(results['critic_losses'], label="Critic Loss", color='green')
    plt.xlabel('Updates (x1000)')
    plt.ylabel('Loss')
    plt.title('Critic Loss')
    plt.grid(True)

    # Alpha values
    plt.subplot(2, 2, 4)
    plt.plot(results['alpha_values'], label="Alpha", color='red')
    plt.xlabel('Updates (x1000)')
    plt.ylabel('Alpha')
    plt.title('Alpha Value')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('training_results.png')
    plt.show()


def visualize_mesh_generation(env, agent, max_steps=100):
    """
    Visualize mesh generation by the agent over time.
    Saves a 'mesh_generation.gif' of the process if imageio is installed.
    """
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)

    state, _ = env.reset()
    frames = []

    # Initial frame
    fig = env.render()
    frames.append(fig)

    for step in range(max_steps):
        action = agent.select_action(state, evaluate=True)
        next_state, reward, done, truncated, _ = env.step(action)

        fig = env.render()
        frames.append(fig)

        state = next_state
        if done or truncated:
            break

    try:
        import imageio
        from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

        images = []
        for fig in frames:
            canvas = FigureCanvas(fig)
            canvas.draw()
            img = np.array(canvas.renderer.buffer_rgba())
            images.append(img)
            plt.close(fig)

        imageio.mimsave('mesh_generation.gif', images, fps=2)
        print("✅ Saved: mesh_generation.gif")

    except ImportError:
        print("⚠️ imageio not installed — skipping animation export.")

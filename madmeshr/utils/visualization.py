import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np

__all__ = ['plot_training_results', 'visualize_mesh_generation', 'save_mesh_result']


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
    plt.close()


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
        print("Saved: mesh_generation.gif")

    except ImportError:
        print("imageio not installed -- skipping animation export.")


def save_mesh_result(env, domain_name, output_dir="output"):
    """Plot the final mesh state and save to output_dir/{domain_name}.png.

    Shows original boundary, placed elements (colored by quality), and remaining
    boundary. Overwrites any existing image for this domain.

    Args:
        env: MeshEnvironment after a completed (or truncated) episode
        domain_name: Used for filename and plot title (e.g., "star_10v")
        output_dir: Directory to save into (created if needed)
    """
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    # Draw original boundary (dashed gray)
    ib = env.initial_boundary
    ax.plot(np.append(ib[:, 0], ib[0, 0]), np.append(ib[:, 1], ib[0, 1]),
            'k--', linewidth=0.8, alpha=0.4, label='Original boundary')

    # Draw elements colored by quality
    cmap = plt.cm.RdYlGn  # red=bad, green=good
    for i, elem in enumerate(env.elements):
        q = env.element_qualities[i] if i < len(env.element_qualities) else 0
        color = cmap(q)
        n = len(elem)
        # Close the polygon
        xs = np.append(elem[:, 0], elem[0, 0])
        ys = np.append(elem[:, 1], elem[0, 1])
        ax.fill(xs, ys, color=color, alpha=0.6, edgecolor='black', linewidth=1.2)
        # Label with quality
        cx, cy = np.mean(elem[:, 0]), np.mean(elem[:, 1])
        label = f"{q:.2f}"
        ax.text(cx, cy, label, ha='center', va='center', fontsize=7, fontweight='bold')

    # Draw remaining boundary (if any)
    if len(env.boundary) >= 3:
        rb = env.boundary
        ax.plot(np.append(rb[:, 0], rb[0, 0]), np.append(rb[:, 1], rb[0, 1]),
                'r-', linewidth=2, label=f'Remaining ({len(rb)} verts)')

    # Mark original boundary vertices
    ax.scatter(ib[:, 0], ib[:, 1], color='black', s=20, zorder=5)

    # Stats
    n_quads = sum(1 for e in env.elements if len(e) == 4)
    n_tris = sum(1 for e in env.elements if len(e) == 3)
    mean_q = np.mean(env.element_qualities) if env.element_qualities else 0
    completed = len(env.boundary) < 5

    stats = (f"Elements: {len(env.elements)} ({n_quads}Q + {n_tris}T) | "
             f"Mean quality: {mean_q:.3f} | "
             f"{'Complete' if completed else 'Incomplete'}")

    ax.set_title(f"{domain_name}\n{stats}", fontsize=11)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=8)

    filepath = os.path.join(output_dir, f"{domain_name}.png")
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved mesh result: {filepath}")
    return filepath


def run_dqn_eval_and_save(agent, boundary, domain_name, n_angle=12, n_dist=4,
                          output_dir="output", type0_priority=False, bnd_dist_threshold=0.03,
                          type2_threshold=0.02, n_expected_override=None):
    """Run a single deterministic DQN episode and save the mesh visualization.

    Args:
        agent: DQN agent
        boundary: numpy array of boundary vertices
        domain_name: Name for the file (e.g., "star_10v")
        n_angle, n_dist: Action space resolution
        output_dir: Where to save
        type0_priority: Enable type-0 priority vertex selection (for concave domains)
        bnd_dist_threshold: Boundary distance filter threshold (fraction of fan radius)
        type2_threshold: Type-2 proximity threshold (fraction of edge length)
        n_expected_override: Override n_expected for mu penalty (None = use default formula)

    Returns:
        dict with episode stats (return, quality, completion, n_elements)
    """
    from madmeshr.mesh_environment import MeshEnvironment
    from madmeshr.discrete_action_env import DiscreteActionEnv

    env = MeshEnvironment(initial_boundary=boundary, type0_priority=type0_priority, bnd_dist_threshold=bnd_dist_threshold)
    discrete_env = DiscreteActionEnv(env, n_angle=n_angle, n_dist=n_dist, type2_threshold=type2_threshold, n_expected_override=n_expected_override)

    state, info = discrete_env.reset()
    mask = info["action_mask"]
    ep_return = 0
    done = False
    steps = 0

    while not done and steps < 100:
        if not np.any(mask):
            break
        action = agent.select_action(state, mask, evaluate=True)
        state, reward, done, truncated, info = discrete_env.step(action)
        mask = info["action_mask"]
        ep_return += reward
        steps += 1
        if truncated:
            break

    filepath = save_mesh_result(env, domain_name, output_dir)

    stats = {
        "domain": domain_name,
        "return": ep_return,
        "mean_quality": np.mean(env.element_qualities) if env.element_qualities else 0,
        "n_elements": len(env.elements),
        "n_quads": sum(1 for e in env.elements if len(e) == 4),
        "n_triangles": sum(1 for e in env.elements if len(e) == 3),
        "completed": done and info.get("complete", False),
        "steps": steps,
        "filepath": filepath,
    }
    return stats

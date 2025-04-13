def train(env, agent, replay_buffer, 
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
        # Sample action
        if t < initial_random_steps:
            action = env.action_space.sample()
        else:
            action = agent.select_action(state)
        
        # Take step
        next_state, reward, done, truncated, info = env.step(action)
        episode_return += reward
        episode_length += 1
        
        # Store transition
        replay_buffer.add(state, action, reward, next_state, done)
        
        # Move to next state
        state = next_state
        
        # Reset if episode ends
        if done or truncated or episode_length >= max_ep_len:
            print(f"Episode Return: {episode_return:.2f} | Elements: {len(env.elements)}")
            results['timesteps'].append(t)
            results['episode_returns'].append(episode_return)
            
            state, _ = env.reset()
            episode_return = 0
            episode_length = 0
        
        # Train agent
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
        
        # Evaluate
        if t % eval_interval == 0:
            eval_return = evaluate(env, agent, num_episodes=5)
            print(f"Evaluation at timestep {t}: {eval_return:.2f}")
            
    return results

def evaluate(env, agent, num_episodes=5, render=False, max_ep_len=100):
    """Evaluate the agent."""
    returns = []
    
    for _ in range(num_episodes):
        state, _ = env.reset()
        episode_return = 0
        episode_length = 0
        done = False
        
        while not done and episode_length < max_ep_len:
            action = agent.select_action(state, evaluate=True)
            next_state, reward, done, truncated, _ = env.step(action)
            episode_return += reward
            episode_length += 1
            
            if render:
                env.render()
                
            state = next_state
            if truncated:
                break
                
        returns.append(episode_return)
        
    return np.mean(returns)
    

def main():
    env = MeshEnvironment()
    env.plot_domain()
    
    print(f"Observation space: {env.observation_space.shape}")
    agent = SAC(state_dim=env.observation_space.shape[0], action_dim=env.action_space.shape[0], hidden_dim=128)
    replay_buffer = ReplayBuffer(capacity=100_000)
    
    state, _ = env.reset()
    for t in range(1, 5000):
        action = env.action_space.sample() if t < 1000 else agent.select_action(state)
        next_state, reward, done, _, info = env.step(action)
        replay_buffer.add(state, action, reward, next_state, done)
        state = next_state
        if done: state, _ = env.reset()
        
        if replay_buffer.size() >= 64 and t >= 1000 and t % 10 == 0:
            try:
                batch = replay_buffer.sample(64)
                train_info = agent.train_step(batch)
                if t % 500 == 0: print(f"Step {t}: Loss={train_info['actor_loss']:.4f}")
            except Exception as e:
                print(f"Training error at step {t}: {str(e)}")
                break


if __name__ == "__main__":
    main()

def train(env, agent, n_episodes, max_steps_per_episode):
    """
    Train the agent in the given environment.

    :param env: The gym environment.
    :param agent: The ACAgent instance.
    :param n_episodes: Number of episodes to train for.
    :param max_steps_per_episode: Maximum steps in each episode.
    """
    for episode in range(n_episodes):
        obs = env.reset()
        agent.rollout_storage.reset()

        for step in range(max_steps_per_episode):
            # Use the actor-critic model to select an action
            action, value, action_log_prob = agent.act(obs)
            
            # Apply the action to the environment
            next_obs, reward, done, _ = env.step(action)
            
            # Insert the results into the RolloutStorage
            agent.insert(obs, action, reward, done, value, action_log_prob)

            obs = next_obs

            if done:
                break

        # Compute the final return for the adversary agent
        # This should be based on your environment and agents' specifics
        env_return = compute_final_return(agent, env)

        # Replace the final return in the RolloutStorage
        with torch.no_grad():
            adversary_env_obs_id = agent.storage.get_obs(-1)
            next_value = agent.get_value(adversary_env_obs_id)
        agent.rollout_storage.replace_final_return(env_return)

        # After computing the final return, compute the rest of the returns
        agent.rollout_storage.compute_returns(next_value)

        # Perform PPO update
        agent.update()

        # Logging and any additional updates

    print("Training completed.")

# Assuming env, agent and other parameters are already defined
# train(env, agent, n_episodes, max_steps_per_episode)

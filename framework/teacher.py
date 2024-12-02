import torch

def sample_trajectories(replay_buffer_obs, replay_buffer_actions, replay_buffer_rewards):
    # Select two random trajectories from the replay buffer
    idx = torch.randint(0, replay_buffer_obs.size(0), (2,))  # Sample two random indices

    # Extract trajectories for the selected indices
    obs = replay_buffer_obs[idx].flatten(start_dim=-1) # Shape: (2, num_steps, obs_space_dim)
    actions = replay_buffer_actions[idx].unsqueeze(-1) # Shape: (2, num_steps, action_space_dim)
    rewards = replay_buffer_rewards[idx] # Shape: (2, num_steps)

    # Shape into (2, num_steps, obs_space_dim + action_space_dim)
    trajectory1 = torch.cat([obs[0], actions[0]], dim=-1) # Shape: (num_steps, obs_space_dim + action_space_dim)
    trajectory2 = torch.cat([obs[1], actions[1]], dim=-1) # Shape: (num_steps, obs_space_dim + action_space_dim)

    return trajectory1, trajectory2, rewards[0], rewards[1]

def give_preference(o1, o2, o1_reward, o2_reward):
    cumulative_o1_reward = torch.sum(o1_reward)
    cumulative_o2_reward = torch.sum(o2_reward)

    if cumulative_o1_reward > cumulative_o2_reward:
        return 2.0
    if cumulative_o1_reward < cumulative_o2_reward:
        return 1.0
    return 0.5


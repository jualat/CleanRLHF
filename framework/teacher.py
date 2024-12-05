from replay_buffer import Trajectory

def give_preference(first_trajectory: Trajectory, second_trajectory: Trajectory) -> float:
    first_trajectory_reward = first_trajectory.samples.rewards.sum()
    second_trajectory_reward = second_trajectory.samples.rewards.sum()

    if first_trajectory_reward > second_trajectory_reward:
        return 2.0
    if first_trajectory_reward < second_trajectory_reward:
        return 1.0

    return 0.5
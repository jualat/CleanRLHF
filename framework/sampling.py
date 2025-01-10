import numpy as np
import torch
from replay_buffer import ReplayBuffer
from reward_net import RewardNet


def uniform_sampling(rb: ReplayBuffer, traj_len: int):
    traj_mb1, traj_mb2 = rb.sample_trajectories(traj_len=traj_len)
    return traj_mb1[0], traj_mb2[0]


def disagreement_sampling(rb: ReplayBuffer, reward_net: RewardNet, traj_len: int):
    traj_mb1, traj_mb2 = rb.sample_trajectories(traj_len=traj_len)

    disagrees = []
    with torch.no_grad():
        for traj_1, traj_2 in zip(traj_mb1, traj_mb2):
            probs = []
            for member in range(len(reward_net.ensemble)):
                probs.append(
                    reward_net.preference_prob_hat_member(traj_1, traj_2, member)
                )
            if isinstance(probs, list):
                probs_numpy = np.array(
                    [
                        p.cpu().numpy() if isinstance(p, torch.Tensor) else p
                        for p in probs
                    ]
                )
            elif isinstance(probs, torch.Tensor):
                probs_numpy = probs.cpu().numpy()
            else:
                probs_numpy = np.array(probs)
            disagrees.append(np.std(probs_numpy, axis=0))

    disagrees_argmax = max(enumerate(disagrees), key=lambda x: x[1])[0]
    return traj_mb1[disagrees_argmax], traj_mb2[disagrees_argmax]


def entropy_sampling(rb: ReplayBuffer, reward_net: RewardNet, traj_len: int):
    traj_mb1, traj_mb2 = rb.sample_trajectories(traj_len=traj_len)

    entropies = []
    with torch.no_grad():
        for traj_1, traj_2 in zip(traj_mb1, traj_mb2):
            probs = []
            for member in range(len(reward_net.ensemble)):
                probs.append(
                    reward_net.preference_hat_entropy_member(traj_1, traj_2, member)
                )
            if isinstance(probs, list):
                probs_numpy = np.array(
                    [
                        p.cpu().numpy() if isinstance(p, torch.Tensor) else p
                        for p in probs
                    ]
                )
            elif isinstance(probs, torch.Tensor):
                probs_numpy = probs.cpu().numpy()
            else:
                probs_numpy = np.array(probs)
            entropies.append(np.mean(probs_numpy, axis=0))

    entropies_argmax = max(enumerate(entropies), key=lambda x: x[1])[0]
    return traj_mb1[entropies_argmax], traj_mb2[entropies_argmax]


def sample_pairs(
    size: int,
    rb: ReplayBuffer,
    sampling_strategy: str,
    reward_net: RewardNet,
    traj_len: int,
):
    pairs = []
    for i in range(size):
        traj1, traj2 = sample_trajectories(rb, sampling_strategy, reward_net, traj_len)
        traj1_tuple = (traj1.replay_buffer_start_idx, traj1.replay_buffer_end_idx)
        traj2_tuple = (traj2.replay_buffer_start_idx, traj2.replay_buffer_end_idx)
        pairs.append((traj1_tuple, traj2_tuple))

    return pairs


def sample_trajectories(
    rb: ReplayBuffer, sampling_strategy: str, reward_net: RewardNet, traj_len: int
):
    if sampling_strategy == "disagree":
        first_trajectory, second_trajectory = disagreement_sampling(
            rb, reward_net, traj_len
        )
    elif sampling_strategy == "entropy":
        first_trajectory, second_trajectory = entropy_sampling(rb, reward_net, traj_len)
    elif sampling_strategy == "uniform":
        first_trajectory, second_trajectory = uniform_sampling(rb, traj_len)
    else:
        assert False, "Invalid sampling strategy"
    return first_trajectory, second_trajectory

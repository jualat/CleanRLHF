import numpy as np
import torch
from replay_buffer import ReplayBuffer
from reward_net import RewardNet


def uniform_sampling(rb: ReplayBuffer, traj_len: int):
    """
    Sample two trajectories from the replay buffer, uniformly.
    :param rb: The replay buffer
    :param traj_len: The trajectory length
    :return:
    """
    traj_mb1, traj_mb2 = rb.sample_trajectories(traj_len=traj_len)
    return traj_mb1[0], traj_mb2[0]


def disagreement_sampling(rb: ReplayBuffer, reward_net: RewardNet, traj_len: int):
    """
    Sample two trajectories from the replay buffer, based on the disagreement between ensemble members.
    :param rb: The replay buffer
    :param reward_net: The reward network
    :param traj_len: The trajectory length
    :return:
    """
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
    """
    Sample two trajectories from the replay buffer, based on the entropy of the ensemble members.
    :param rb: The replay buffer
    :param reward_net: The reward network
    :param traj_len: The trajectory length
    :return:
    """
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


def slice_pairs(pairs, mini_batch_size):
    """
    Slices the pairs into mini-batches.
    :param pairs: The pairs to slice
    :param mini_batch_size: The size of the mini-batches
    :return:
    """
    return [
        pairs[i : i + mini_batch_size] for i in range(0, len(pairs), mini_batch_size)
    ]


def sample_pairs(
    size: int,
    rb: ReplayBuffer,
    sampling_strategy: str,
    reward_net: RewardNet,
    traj_len: int,
    batch_sampling: str,
    mini_batch_size: int,
):
    pairs = []
    for i in range(size):
        traj1, traj2 = sample_trajectories(rb, sampling_strategy, reward_net, traj_len)
        traj1_tuple = (traj1.replay_buffer_start_idx, traj1.replay_buffer_end_idx)
        traj2_tuple = (traj2.replay_buffer_start_idx, traj2.replay_buffer_end_idx)
        pairs.append((traj1_tuple, traj2_tuple))

    if batch_sampling == "minibatch":
        batch = slice_pairs(pairs, mini_batch_size)
    else:
        batch = [pairs]

    return batch


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
        raise ValueError(f"Invalid sampling strategy: {sampling_strategy}")

    return first_trajectory, second_trajectory

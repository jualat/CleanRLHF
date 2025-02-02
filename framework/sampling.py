import logging

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


def disagreement_sampling2(rb: ReplayBuffer, reward_net: RewardNet, traj_len: int, device, k=1):
    """
    Sample pairs of trajectories that maximize the ensemble disagreement.
    The disagreement is computed as the standard deviation (across ensemble members)
    of the Bradley-Terry preference probability for the first trajectory.

    This version vectorizes the computation over the sampled trajectories.
    """
    import time

    best_pairs = []

    for i in range(k):
        traj_mb1, traj_mb2 = rb.sample_trajectories(traj_len=traj_len)

        with (torch.no_grad()):
            obs1 = torch.stack([traj.samples.observations for traj in traj_mb1], dim=0).to(device) # (B, L, obs_dim)
            act1 = torch.stack([traj.samples.actions for traj in traj_mb1], dim=0).to(device) # (B, L, act_dim)
            obs2 = torch.stack([traj.samples.observations for traj in traj_mb2], dim=0).to(device) # (B, L, obs_dim)
            act2 = torch.stack([traj.samples.actions for traj in traj_mb2], dim=0).to(device) # (B, L, act_dim)

            B, L, obs_dim = obs1.shape
            _, _, act_dim = act1.shape
            ensemble_size = len(reward_net.ensemble)

            obs = torch.stack([obs1, obs2], dim=0).to(device)  # (2, B, L, obs_dim)
            acts = torch.stack([act1, act2], dim=0).to(device) # (2, B, L, act_dim)

            obs = obs.expand(ensemble_size, 2, B, L, obs_dim).reshape(ensemble_size, 2 * B * L, obs_dim)
            acts = acts.expand(ensemble_size, 2, B, L, act_dim).reshape(ensemble_size, 2 * B * L, act_dim)

            # start_time = time.time()
            r_ens = reward_net.forward_ensemble(obs, acts, device=device)  # (E, 2*B*L, 1)
            r_ens = r_ens.reshape(ensemble_size, 2, B, L).sum(dim=3)  # (E, 2, B)
            # logging.debug("Forward ensemble took %s seconds, obs shape: %s", time.time() - start_time, obs.shape)

            # Compute the preference probabilities
            probs = r_ens.softmax(dim=1) # (E, 2, B)
            p_first = probs[:, 0, :] # (E, B)

            # For each pair, compute the standard deviation (disagreement) over ensemble members.
            disagreement = p_first.std(dim=0)  # (B)
            max_idx = disagreement.argmax().item()

            best_pairs.append((traj_mb1[max_idx], traj_mb2[max_idx]))

    # Select the pairs with the highest disagreement
    return best_pairs


def disagreement_sampling(rb: ReplayBuffer, reward_net: RewardNet, traj_len: int, device, k=1):
    """
    Sample pairs of trajectories that maximize the ensemble disagreement.
    The disagreement is computed as the standard deviation (across ensemble members)
    of the Bradley-Terry preference probability for the first trajectory.

    This version vectorizes the computation over the sampled trajectories.
    """
    import time

    start_time = time.time()
    trajectory_batches = [rb.sample_trajectories(traj_len=traj_len) for _ in range(k)]
    logging.debug("Sampling trajectories took %s seconds", time.time() - start_time)

    with (torch.no_grad()):
        start_time = time.time()
        obs1 = torch.stack([torch.stack([traj.samples.observations for traj in traj_mb], dim=0) for traj_mb, _ in trajectory_batches], dim=0).to(device) # (k, B, L, obs_dim)
        act1 = torch.stack([torch.stack([traj.samples.actions for traj in traj_mb], dim=0) for traj_mb, _ in trajectory_batches], dim=0).to(device) # (k, B, L, act_dim)
        obs2 = torch.stack([torch.stack([traj.samples.observations for traj in traj_mb], dim=0) for _, traj_mb in trajectory_batches], dim=0).to(device) # (k, B, L, obs_dim)
        act2 = torch.stack([torch.stack([traj.samples.actions for traj in traj_mb], dim=0) for _, traj_mb in trajectory_batches], dim=0).to(device) # (k, B, L, act_dim)
        logging.debug("Stacking took %s seconds", time.time() - start_time)

        _, B, L, obs_dim = obs1.shape
        _, _, _, act_dim = act1.shape
        ensemble_size = len(reward_net.ensemble)

        obs = torch.stack([obs1, obs2], dim=0).to(device)  # (2, k, B, L, obs_dim)
        acts = torch.stack([act1, act2], dim=0).to(device) # (2, k, B, L, act_dim)

        start_time = time.time()
        obs = obs.expand(ensemble_size, 2, k, B, L, obs_dim).reshape(ensemble_size, 2 * k * B * L, obs_dim)
        acts = acts.expand(ensemble_size, 2, k, B, L, act_dim).reshape(ensemble_size, 2 * k * B * L, act_dim)
        logging.debug("Expanding took %s seconds", time.time() - start_time)

        start_time = time.time()
        r_ens = reward_net.forward_ensemble(obs, acts, device=device)  # (E, 2*k*B*L, 1)
        r_ens = r_ens.reshape(ensemble_size, 2, k*B, L).sum(dim=3)  # (E, 2, k*B)
        logging.debug("Forward ensemble took %s seconds, obs shape: %s", time.time() - start_time, obs.shape)

        # Compute the preference probabilities
        probs = r_ens.softmax(dim=1) # (E, 2, k*B)
        p_first = probs[:, 0, :] # (E, k*B)

        # For each pair, compute the standard deviation (disagreement) over ensemble members.
        disagreement = p_first.std(dim=0)  # (k*B)
        disagreement = disagreement.reshape(k, B)  # (k, B)
        max_idx = disagreement.argmax(dim=1) # (k,)

        logging.debug("disagreement.shape: %s", disagreement.shape)
        logging.debug("max_idx.shape: %s", max_idx.shape)

    # Select the pairs with the highest disagreement
    return [
        (trajectory_batches[i][0][max_idx[i].item()], trajectory_batches[i][1][max_idx[i].item()])
        for i in range(k)
    ]


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
    device
):
    pairs = []
    trajectory_batch = sample_trajectories(rb, sampling_strategy, reward_net, traj_len, device, k=size)
    for i in range(size):
        traj1, traj2 = trajectory_batch[i]
        traj1_tuple = (traj1.replay_buffer_start_idx, traj1.replay_buffer_end_idx)
        traj2_tuple = (traj2.replay_buffer_start_idx, traj2.replay_buffer_end_idx)
        pairs.append((traj1_tuple, traj2_tuple))

    if batch_sampling == "minibatch":
        batch = slice_pairs(pairs, mini_batch_size)
    else:
        batch = [pairs]

    return batch


def sample_trajectories(
    rb: ReplayBuffer, sampling_strategy: str, reward_net: RewardNet, traj_len: int, device, k=1
):
    import time

    if sampling_strategy == "disagree":
        start_time = time.time()
        samples = disagreement_sampling2(
            rb, reward_net, traj_len, device, k
        )

        logging.debug("Disagreement sampling took %s seconds", time.time() - start_time)

        return samples
    elif sampling_strategy == "entropy":
        first_trajectory, second_trajectory = entropy_sampling(rb, reward_net, traj_len)
    elif sampling_strategy == "uniform":
        first_trajectory, second_trajectory = uniform_sampling(rb, traj_len)
    else:
        raise ValueError(f"Invalid sampling strategy: {sampling_strategy}")

    return first_trajectory, second_trajectory

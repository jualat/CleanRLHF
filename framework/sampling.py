import torch

from replay_buffer import ReplayBuffer
from reward_net import RewardNet


def uniform_sampling(rb: ReplayBuffer):
    traj_mb1, traj_mb2 = rb.sample_trajectories()
    return traj_mb1[0], traj_mb2[0]


def disagreement_sampling(rb: ReplayBuffer, reward_net: RewardNet):
    traj_mb1, traj_mb2 = rb.sample_trajectories()

    disagrees = []
    with torch.no_grad():
        for traj_1, traj_2 in zip(traj_mb1, traj_mb2):
            probs = []
            for member in range(len(reward_net.ensemble)):
                probs.append(
                    reward_net.preference_prob_hat_member(traj_1, traj_2, member)
                )
            disagrees.append(np.std(probs, axis=0))

    disagrees_argmax = max(enumerate(disagrees), key=lambda x: x[1])[0]
    return traj_mb1[disagrees_argmax], traj_mb2[disagrees_argmax]


def entropy_sampling(rb: ReplayBuffer, reward_net: RewardNet):
    traj_mb1, traj_mb2 = rb.sample_trajectories()

    entropies = []
    with torch.no_grad():
        for traj_1, traj_2 in zip(traj_mb1, traj_mb2):
            probs = []
            for member in range(len(reward_net.ensemble)):
                probs.append(
                    reward_net.preference_hat_entropy_member(traj_1, traj_2, member)
                )
            entropies.append(np.mean(probs, axis=0))

    entropies_argmax = max(enumerate(entropies), key=lambda x: x[1])[0]
    return traj_mb1[entropies_argmax], traj_mb2[entropies_argmax]

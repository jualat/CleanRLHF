from dataclasses import replace

import numpy as np
from gymnasium import spaces
import torch
from typing import Union, Optional, NamedTuple

from numpy import dtype
from stable_baselines3.common.buffers import ReplayBuffer as SB3ReplayBuffer
from stable_baselines3.common.type_aliases import ReplayBufferSamples
from stable_baselines3.common.vec_env import VecNormalize
from sympy.codegen.ast import int32

from reward_net import RewardNet


class Trajectory(NamedTuple):
    replay_buffer_start_idx: int
    replay_buffer_end_idx: int
    samples: ReplayBufferSamples


class ReplayBuffer(SB3ReplayBuffer):
    """
    Replay buffer used in off-policy algorithms like SAC/TD3.
    It's based on the one from stable-baselines3, but with some modifications for preference based learning.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
    :param n_envs: Number of parallel environments
    :param optimize_memory_usage: Enable a memory efficient variant
        of the replay buffer which reduces by almost a factor two the memory used,
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
        and https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274
        Cannot be used in combination with handle_timeout_termination.
    :param handle_timeout_termination: Handle timeout termination (due to timelimit)
        separately and treat the task as infinite horizon task.
        https://github.com/DLR-RM/stable-baselines3/issues/284
    """

    ground_truth_rewards: np.ndarray

    def __init__(self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[torch.device, str] = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True
    ):
        super().__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs,
                         optimize_memory_usage=optimize_memory_usage,
                         handle_timeout_termination=handle_timeout_termination)
        self.ground_truth_rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

    def add(
            self,
            obs: np.ndarray,
            next_obs: np.ndarray,
            action: np.ndarray,
            reward: np.ndarray,
            done: np.ndarray,
            infos: list[dict[str, any]],
    ) -> None:
        super().add(obs, next_obs, action, reward, done, infos)
        self.ground_truth_rewards[self.pos] = np.array(reward)

    def sample_trajectories(self, env: Optional[VecNormalize] = None):
        """
        Sample trajectories from the replay buffer.
        :param env: associated gym VecEnv to normalize the observations/rewards when sampling
        :return: batch size many trajectories
        """
        done_indices = np.where(self.dones == 1)[0].astype(np.int32)

        if len(done_indices) < 2:
            raise ValueError("Replay buffer doesn't contain at least 2 trajectories.")

        # Compute start and end indices for each trajectory
        # First trajectory starts at index 0 and all done indices except the last one are transformed to start indices
        # by adding 1.
        starts = np.concatenate(([0], done_indices[:-1] + 1)).astype(np.int32)

        print("Done indices", done_indices)

        # The end indices are the done indices, including the last one.
        ends = done_indices + 1

        # Randomly select indices for the trajectories
        # Set replace=False to sample different trajectories
        indices = np.random.choice(len(done_indices), 2, replace=False)

        first_trajectory = self.get_trajectory(int(starts[indices[0]]), int(ends[indices[0]]), env)
        second_trajectory = self.get_trajectory(int(starts[indices[1]]), int(ends[indices[1]]), env)

        return first_trajectory, second_trajectory


    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None) -> ReplayBufferSamples:
        # Sample randomly the env idx
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))

        if self.optimize_memory_usage:
            next_obs = self._normalize_obs(self.observations[(batch_inds + 1) % self.buffer_size, env_indices, :], env)
        else:
            next_obs = self._normalize_obs(self.next_observations[batch_inds, env_indices, :], env)

        data = (
            self._normalize_obs(self.observations[batch_inds, env_indices, :], env),
            self.actions[batch_inds, env_indices, :],
            next_obs,
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            (self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])).reshape(-1, 1),
            self._normalize_reward(self.ground_truth_rewards[batch_inds, env_indices].reshape(-1, 1), env),
        )
        return ReplayBufferSamples(*tuple(map(self.to_torch, data)))

    def get_trajectory(self, start_idx: int, end_idx: int, env: Optional[VecNormalize] = None):
        trajectory_indices = np.arange(start_idx, end_idx)
        trajectory_samples = self._get_samples(trajectory_indices, env)

        return Trajectory(
            replay_buffer_start_idx=start_idx,
            replay_buffer_end_idx=end_idx,
            samples=trajectory_samples,
        )

    def relabel_rewards(self, reward_net: RewardNet):
        """
        Relabel rewards in the replay buffer to take into account the change in the reward function.
        :param reward_net: Reward network
        :return:
        """
        # Convert the entire observations and actions arrays to NumPy arrays
        observations = np.array(self.observations, dtype=np.float32).squeeze(axis=1)
        actions = np.array(self.actions, dtype=np.float32).squeeze(axis=1)

        # Calculate the new rewards using the reward_net
        new_rewards = reward_net.predict_reward(observations, actions)

        # Update the rewards in the replay buffer
        self.rewards = new_rewards
from dataclasses import replace

import numpy as np
from gymnasium import spaces
import torch
from typing import Union, Optional, NamedTuple, List
import logging

from stable_baselines3.common.buffers import ReplayBuffer as SB3ReplayBuffer
from stable_baselines3.common.vec_env import VecNormalize

from reward_net import RewardNet


class ReplayBufferSampleHF(NamedTuple):
    observations: torch.Tensor
    actions: torch.Tensor
    next_observations: torch.Tensor
    dones: torch.Tensor
    rewards: torch.Tensor
    ground_truth_rewards: torch.Tensor
    qpos: torch.Tensor
    qvel: torch.Tensor
    env_idx: int


class Trajectory(NamedTuple):
    replay_buffer_start_idx: int
    replay_buffer_end_idx: int
    samples: ReplayBufferSampleHF


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

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[torch.device, str] = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
        qpos_shape: int = 1,
        qvel_shape: int = 1,
        seed: int = None,
    ):
        super().__init__(
            buffer_size,
            observation_space,
            action_space,
            device,
            n_envs=n_envs,
            optimize_memory_usage=optimize_memory_usage,
            handle_timeout_termination=handle_timeout_termination,
        )
        self.ground_truth_rewards = np.zeros(
            (self.buffer_size, self.n_envs), dtype=np.float32
        )
        self.qpos = np.zeros(
            (self.buffer_size, self.n_envs, qpos_shape), dtype=np.float32
        )
        self.qvel = np.zeros(
            (self.buffer_size, self.n_envs, qvel_shape), dtype=np.float32
        )
        self.seed = seed

        if seed is not None:
            np.random.seed(seed)
            self.torch_generator = torch.Generator().manual_seed(seed)
        else:
            self.torch_generator = None

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        ground_truth_rewards: np.ndarray,
        done: np.ndarray,
        infos: list[dict[str, any]],
        qpos: np.ndarray,
        qvel: np.ndarray,
    ) -> None:
        """
        Extending the SB3 add method to include ground truth rewards, qpos and qvel.
        :param obs: observation
        :param next_obs: next observation
        :param action: action
        :param reward: reward
        :param ground_truth_rewards: ground truth rewards
        :param done: done
        :param infos: info
        :param qpos: qpos
        :param qvel: qvel
        :return:
        """
        super().add(obs, next_obs, action, reward, done, infos)
        self.ground_truth_rewards[self.pos] = ground_truth_rewards
        self.qpos[self.pos] = qpos
        self.qvel[self.pos] = qvel

    def sample_trajectories(
        self, env: Optional[VecNormalize] = None, mb_size: int = 20, traj_len: int = 32
    ) -> tuple[List[Trajectory], List[Trajectory]]:
        """
        Sample trajectories from the replay buffer.
        :param env: associated gym VecEnv to normalize the observations/rewards when sampling
        :param mb_size: amount of pairs of trajectories to be sampled
        :param traj_len: length of trajectories
        :return: two lists of mb_size many trajectories
        """
        max_valid_index = min(self.buffer_size, self.pos) - traj_len
        if max_valid_index <= 0:
            raise ValueError(
                f"Not enough valid data in buffer to sample trajectories. "
                f"self.pos={self.pos}, traj_len={traj_len}, capacity={self.buffer_size}"
            )

        if max_valid_index >= 2 * mb_size:
            indices = np.random.choice(max_valid_index, 2 * mb_size, replace=False)
        else:
            indices = np.random.choice(max_valid_index, 2 * mb_size, replace=True)

        trajectories = [
            self.get_trajectory(
                start,
                (start + traj_len),
                env,
            )
            for start in indices
        ]
        logging.debug(f"trajectory length: {traj_len}, ")
        return trajectories[:mb_size], trajectories[mb_size:]

    def _get_samples(
        self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None
    ) -> ReplayBufferSampleHF:
        """
        Helper function for get_trajectory() to get a single sample.
        :param batch_inds: Numpy array of indices of the trajectory
        :param env: Environment
        :return:
        """
        # Sample randomly the env idx
        env_indices = np.random.randint(0, high=self.n_envs)

        if self.optimize_memory_usage:
            next_obs = self._normalize_obs(
                self.observations[(batch_inds + 1) % self.buffer_size, env_indices, :],
                env,
            )
        else:
            next_obs = self._normalize_obs(
                self.next_observations[batch_inds, env_indices, :], env
            )

        return ReplayBufferSampleHF(
            observations=self.to_torch(
                self._normalize_obs(self.observations[batch_inds, env_indices, :], env)
            ),
            actions=self.to_torch(self.actions[batch_inds, env_indices, :]),
            next_observations=self.to_torch(next_obs),
            dones=self.to_torch(
                (
                    self.dones[batch_inds, env_indices]
                    * (1 - self.timeouts[batch_inds, env_indices])
                ).reshape(-1, 1)
            ),
            rewards=self.to_torch(
                self._normalize_reward(
                    self.rewards[batch_inds, env_indices].reshape(-1, 1), env
                )
            ),
            ground_truth_rewards=self.to_torch(
                self.ground_truth_rewards[batch_inds, env_indices].reshape(-1, 1)
            ),
            qpos=self.to_torch(self.qpos[batch_inds, env_indices, :]),
            qvel=self.to_torch(self.qvel[batch_inds, env_indices, :]),
            env_idx=env_indices,
        )

    def get_trajectory(
        self, start_idx: int, end_idx: int, env: Optional[VecNormalize] = None
    ):
        """
        Helper function of sample_trajectories() to get a single trajectory.
        :param start_idx: The start index of the trajectory
        :param end_idx: The end index of the trajectory
        :param env: The environment
        :return:
        """
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
        observations = self.observations.reshape(self.n_envs * self.buffer_size, -1)
        actions = self.actions.reshape(self.n_envs * self.buffer_size, -1)

        # Sample a set of indices before relabeling for comparison
        sample_indices = np.random.randint(0, self.pos, size=10)
        old_rewards = self.rewards[sample_indices].copy()

        # Calculate the new rewards using the reward_net
        with torch.no_grad():
            new_rewards = reward_net.predict_reward(observations, actions).reshape(
                self.buffer_size, self.n_envs
            )

        # Update the rewards in the replay buffer
        self.rewards = new_rewards

        # Check the new rewards at the same indices
        new_rewards = self.rewards[sample_indices]
        for i in range(len(sample_indices)):
            idx = sample_indices[i]
            logging.debug(
                f"Buffer Index {idx}: Old Reward = {old_rewards[i]}, New Reward = {new_rewards[i]}"
            )

        assert not np.allclose(
            old_rewards, new_rewards
        ), "No change in rewards after relabeling!"

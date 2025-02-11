import logging
from typing import List, NamedTuple, Optional, Union

import numpy as np
import torch
from gymnasium import spaces
from reward_training.reward_net import RewardNet
from stable_baselines3.common.buffers import ReplayBuffer as SB3ReplayBuffer
from stable_baselines3.common.vec_env import VecNormalize


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


class RolloutBufferSampleHF(NamedTuple):
    observations: torch.Tensor
    actions: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    values: torch.Tensor
    log_probs: torch.Tensor


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
        rune: bool = False,
        rune_beta: float = 0,
        rune_beta_decay: float = 0.999,
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
        self.rune = rune
        self.rune_beta_decay = rune_beta_decay

        if rune:
            self.rune_beta = rune_beta
        else:
            self.rune_beta = 0

        self.extrinsic_rewards = np.zeros(
            (self.buffer_size, self.n_envs), dtype=np.float32
        )
        self.intrinsic_rewards = np.zeros(
            (self.buffer_size, self.n_envs), dtype=np.float32
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
        self.qpos_shape = qpos_shape
        self.qvel_shape = qvel_shape
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
        extrinsic_reward: np.ndarray,
        intrinsic_reward: np.ndarray,
        ground_truth_reward: np.ndarray,
        done: np.ndarray,
        infos: list[dict[str, any]],
        global_step: int,
        qpos: np.ndarray,
        qvel: np.ndarray,
    ) -> None:
        """
        Extending the SB3 add method to include ground truth rewards, qpos and qvel.
        :param obs: observation
        :param next_obs: next observation
        :param action: action
        :param extrinsic_reward: extrinsic reward  (mean of the ensemble predictions)
        :param intrinsic_reward: intrinsic reward (standard deviation of the ensemble predictions
        :param ground_truth_reward: ground truth rewards
        :param done: done
        :param infos: info
        :param global_step:
        :param qpos: qpos
        :param qvel: qvel
        :return:
        """
        pos = self.pos
        super().add(obs, next_obs, action, extrinsic_reward, done, infos)
        self.intrinsic_rewards[pos] = intrinsic_reward
        self.ground_truth_rewards[pos] = ground_truth_reward
        self.qpos[pos] = qpos
        self.qvel[pos] = qvel
        self.extrinsic_rewards[pos] = extrinsic_reward
        self.rewards = (
            self.extrinsic_rewards
            + self.rune_beta
            * ((1 - self.rune_beta_decay) ** global_step)
            * self.intrinsic_rewards
        )
        if not self.rune:
            extrinsic_reward = extrinsic_reward.astype(np.float32)
            assert self.rewards[pos] == extrinsic_reward, (
                f"Reward must match the extrinsic reward. {self.rewards[pos]}, {extrinsic_reward}"
            )
        self.pos = pos + 1
        if self.pos == self.buffer_size:
            self.pos = 0

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

    def to_torch(self, array: np.ndarray, copy: bool = True) -> torch.Tensor:
        if copy:
            return torch.tensor(array, dtype=torch.float32, device=self.device)
        return torch.as_tensor(
            array, dtype=torch.float32, device=self.device
        )  # test if still needed

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
        assert start_idx < end_idx, "start_idx=%d, end_idx=%d" % (start_idx, end_idx)
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
        old_rewards = self.extrinsic_rewards[sample_indices].copy()

        # Calculate the new rewards using the reward_net
        with torch.no_grad():
            new_extrinsic_rewards, new_intrinsic_rewards = reward_net.predict_reward(
                observations, actions
            )
            new_extrinsic_rewards.reshape(self.buffer_size, self.n_envs)

        # Update the extrinsic rewards in the replay buffer
        self.extrinsic_rewards = new_extrinsic_rewards

        # Update the intrinsic rewards in the replay buffer
        self.intrinsic_rewards = new_intrinsic_rewards

        # Check the new rewards at the same indices
        new_rewards = self.extrinsic_rewards[sample_indices]

        for i in range(len(sample_indices)):
            idx = sample_indices[i]
            logging.debug(
                f"Buffer Index {idx}: Old Reward = {old_rewards[i]}, New Reward = {new_rewards[i]}"
            )

        assert not np.allclose(old_rewards, new_rewards), (
            "No change in rewards after relabeling!"
        )

    def temporal_data_augmentation(
        self, traj: Trajectory, H_max=55, H_min=45, env: Optional[VecNormalize] = None
    ):
        start_idx = traj.replay_buffer_start_idx
        end_idx = traj.replay_buffer_end_idx
        H = end_idx - start_idx

        H_prime = np.random.randint(low=H_min, high=min(H_max, H) + 1)
        offset = np.random.randint(low=0, high=H - H_prime + 1)

        slice_start_idx = start_idx + offset
        slice_end_idx = start_idx + offset + H_prime - 1
        assert slice_start_idx < slice_end_idx, "Invalid slice"
        assert slice_end_idx <= end_idx, "Index out of range"
        return self.get_trajectory(slice_start_idx, slice_end_idx, env=env)


class RolloutBuffer(ReplayBuffer):
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
        rune: bool = False,
        rune_beta: float = 0,
        rune_beta_decay: float = 0.999,
        seed: int = None,
        gae_lambda: float = 0.95,
        gamma: float = 0.99,
    ):
        super().__init__(
            buffer_size=buffer_size,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            n_envs=n_envs,
            optimize_memory_usage=optimize_memory_usage,
            handle_timeout_termination=handle_timeout_termination,
            qpos_shape=qpos_shape,
            qvel_shape=qvel_shape,
            rune=rune,
            rune_beta=rune_beta,
            rune_beta_decay=rune_beta_decay,
            seed=seed,
        )

        self.log_probs = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.values = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.advantages = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.returns = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.current_size = 0

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        extrinsic_reward: np.ndarray,
        intrinsic_reward: np.ndarray,
        ground_truth_reward: np.ndarray,
        done: np.ndarray,
        infos: list[dict[str, any]],
        global_step: int,
        qpos: np.ndarray,
        qvel: np.ndarray,
        logprobs: np.ndarray,
        values: np.ndarray,
    ):
        pos = self.pos
        super().add(
            obs=obs,
            next_obs=next_obs,
            action=action,
            extrinsic_reward=extrinsic_reward,
            intrinsic_reward=intrinsic_reward,
            ground_truth_reward=ground_truth_reward,
            done=done,
            infos=infos,
            global_step=global_step,
            qpos=qpos,
            qvel=qvel,
        )
        self.log_probs[pos] = logprobs
        self.values[pos] = values

        self.pos = pos + 1
        if self.pos == self.buffer_size:
            self.pos = 0
        if self.current_size < self.buffer_size:
            self.current_size += 1

    def compute_gae_and_returns(self, agent, next_obs, next_done):
        """
        Computes GAE advantages and returns using the rollout buffer data.

        Parameters:
            agent: The agent providing a value function (via agent.get_value)
            next_obs: The observation at the timestep after the rollout (as a torch tensor)
            next_done: The done flag for the next observation (scalar or tensor)
        """
        with torch.no_grad():
            next_obs = torch.as_tensor(
                next_obs, device=self.device, dtype=torch.float32
            )
            next_done = torch.as_tensor(
                next_done, device=self.device, dtype=torch.float32
            )
            next_value = agent.get_value(next_obs).reshape(1, -1)

            rewards = torch.tensor(
                self.rewards[: self.current_size],
                dtype=torch.float32,
                device=self.device,
            )
            dones = torch.tensor(
                self.dones[: self.current_size], dtype=torch.float32, device=self.device
            )
            values = torch.tensor(
                self.values[: self.current_size],
                dtype=torch.float32,
                device=self.device,
            )
            advantages = torch.zeros_like(rewards, device=self.device)
            lastgaelam = 0.0

            for t in reversed(range(self.current_size)):
                if t == self.current_size - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = (
                    rewards[t] + self.gamma * nextvalues * nextnonterminal - values[t]
                )

                lastgaelam = (
                    delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
                )
                advantages[t] = lastgaelam

            returns = advantages + values

        self.advantages[: self.current_size] = advantages.cpu().numpy()
        self.returns[: self.current_size] = returns.cpu().numpy()

    def get(
        self,
    ) -> RolloutBufferSampleHF:
        return RolloutBufferSampleHF(
            observations=torch.Tensor(self.observations)
            .to(self.device)
            .reshape((-1,) + self.obs_shape),
            log_probs=torch.Tensor(self.log_probs).reshape(-1).to(self.device),
            actions=torch.Tensor(self.actions)
            .to(self.device)
            .reshape((-1,) + self.action_space.shape),
            advantages=torch.Tensor(self.advantages)
            .to(self.device)
            .reshape(
                -1,
            ),
            returns=torch.Tensor(self.returns)
            .to(self.device)
            .reshape(
                -1,
            ),
            values=torch.Tensor(self.values)
            .to(self.device)
            .reshape(
                -1,
            ),
        )

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
        max_valid_index = max(self.current_size, self.pos) - traj_len
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

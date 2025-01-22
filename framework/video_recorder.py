import logging
import os

import gymnasium as gym
import numpy as np
import torch
from env import FlattenVectorObservationWrapper, is_mujoco_env
from gymnasium.utils.save_video import save_video
from replay_buffer import ReplayBuffer, Trajectory


class VideoRecorder:
    """
    Records videos from the replay buffer by replaying the actions.
    This allows you to visualize specific trajectories without having to generate videos
    for all trajectories during training.
    :param rb: The replay buffer storing the complete trajectories
    :param seed: Seed for the environment
    :param env_id: The gym environment ID to use for replaying the actions
    """

    def __init__(
        self,
        rb: ReplayBuffer,
        seed: int,
        env_id: str,
    ):
        self.rb = rb
        self.seed = seed
        self.env_id = env_id

    def record_trajectory(self, trajectory: Trajectory, run_name: str, fps=30):
        start_idx = trajectory.replay_buffer_start_idx
        end_idx = trajectory.replay_buffer_end_idx
        env_idx = trajectory.samples.env_idx
        length = trajectory.replay_buffer_end_idx - trajectory.replay_buffer_start_idx

        # Ensure the directory for videos exists
        video_folder = f"./videos/{run_name}/trajectories"
        os.makedirs(video_folder, exist_ok=True)
        out_path = f"{video_folder}/"

        env = gym.make(self.env_id, render_mode="rgb_array")
        env = FlattenVectorObservationWrapper(env)

        try:
            self._initialize_env_state(env, trajectory)
            frames = self._generate_frames(env, trajectory)
            save_video(
                frames=frames,
                video_length=length,
                video_folder=out_path,
                fps=fps,
                name_prefix=f"trajectory_{start_idx}_{end_idx}_{env_idx}",
            )
        except Exception as e:
            logging.error(
                f"Error recording trajectory (start_idx={start_idx}, end_idx={end_idx}, env_idx={env_idx}): {e}"
            )
        finally:
            env.close()

    def _initialize_env_state(self, env, trajectory):
        """Initialize the environment state based on Mujoco or non-Mujoco trajectories."""
        if is_mujoco_env(env):
            qpos_list = trajectory.samples.qpos
            qvel_list = trajectory.samples.qvel
            env.reset(seed=self.seed)
            qpos = (
                qpos_list[0].cpu().numpy()
                if isinstance(qpos_list[0], torch.Tensor)
                else qpos_list[0]
            )

            qvel = (
                qvel_list[0].cpu().numpy()
                if isinstance(qvel_list[0], torch.Tensor)
                else qvel_list[0]
            )
            # Append zeros to qpos for skipped qpos values
            try:
                skipped_qpos = env.unwrapped.observation_structure["skipped_qpos"]
            except (KeyError, AttributeError):
                skipped_qpos = 0

            qpos_extended = np.append(qpos, [0] * skipped_qpos)

            env.unwrapped.set_state(qpos_extended, qvel)
        else:
            env.reset(seed=self.seed)
            if hasattr(env, "state"):
                env.state = trajectory.samples.observations[0]

    def _generate_frames(self, env, trajectory):
        """Generate frames for the video from the trajectory."""
        frames = [env.render()]
        if is_mujoco_env(env):
            qpos_list = trajectory.samples.qpos
            qvel_list = trajectory.samples.qvel
            for i in range(1, len(qpos_list)):
                qpos = (
                    qpos_list[i].cpu().numpy()
                    if isinstance(qpos_list[i], torch.Tensor)
                    else qpos_list[i]
                )
                qvel = (
                    qvel_list[i].cpu().numpy()
                    if isinstance(qvel_list[i], torch.Tensor)
                    else qvel_list[i]
                )
                # Append zeros to qpos for skipped qpos values
                try:
                    skipped_qpos = env.unwrapped.observation_structure["skipped_qpos"]
                except (KeyError, AttributeError):
                    skipped_qpos = 0
                qpos_extended = np.append(qpos, [0] * skipped_qpos)

                env.unwrapped.set_state(qpos_extended, qvel)
                frames.append(env.render())
        else:
            actions = trajectory.samples.actions
            for action in actions:
                action = action.detach().cpu().numpy()
                env.step(action)
                frames.append(env.render())
        return frames

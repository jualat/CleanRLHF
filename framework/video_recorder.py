import os
import logging

import gymnasium as gym
import torch
from replay_buffer import ReplayBuffer, Trajectory
import cv2


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

        # Ensure the directory for videos exists
        video_folder = f"./videos/{run_name}/trajectories"
        os.makedirs(video_folder, exist_ok=True)
        out_path = f"{video_folder}/trajectory_{start_idx}_{end_idx}_{env_idx}.mp4"

        if os.path.exists(out_path):
            logging.info(f"Skipping {out_path}")
            return
        env = gym.make(self.env_id, render_mode="rgb_array")
        writer = None

        try:
            self._initialize_env_state(env, trajectory)
            writer = self._initialize_writer(env, out_path, fps)
            self._write_trajectory_to_video(env, trajectory, writer)
        except Exception as e:
            logging.error(
                f"Error recording trajectory (start_idx={start_idx}, end_idx={end_idx}, env_idx={env_idx}): {e}"
            )
        finally:
            if writer:
                writer.release()
            env.close()

    def _is_mujoco_env(self, env) -> bool:
        # Try to check the internal `mujoco` attribute
        return hasattr(env.unwrapped, "model") and hasattr(
            env.unwrapped, "do_simulation"
        )

    def _initialize_env_state(self, env, trajectory):
        """Initialize the environment state based on Mujoco or non-Mujoco trajectories."""
        if self._is_mujoco_env(env):
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
            env.unwrapped.set_state(qpos, qvel)
        else:
            env.reset(seed=self.seed)
            if hasattr(env, "state"):
                env.state = trajectory.samples.observations[0]

    def _initialize_writer(self, env, out_path, fps):
        """Initialize the video writer."""
        img = env.render()
        return cv2.VideoWriter(
            out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (img.shape[1], img.shape[0])
        )

    def _write_trajectory_to_video(self, env, trajectory, writer):
        """Write the trajectory to a video file."""
        if self._is_mujoco_env(env):
            qpos_list = trajectory.samples.qpos
            qvel_list = trajectory.samples.qvel
            for i in range(1, len(qpos_list)):
                print(type(qpos_list[i]))
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
                env.unwrapped.set_state(qpos, qvel)
                writer.write(env.render())
        else:
            actions = trajectory.samples.actions
            for action in actions:
                action = action.detach().cpu().numpy()
                env.step(action)
                writer.write(env.render())

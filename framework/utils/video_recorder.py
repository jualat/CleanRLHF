import logging
import os
import re
from typing import Optional

import torch
from env import is_mujoco_env, make_single_env
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
        dm_control: bool,
        teacher_feedback_mode: Optional[str] = None,
    ):
        self.rb = rb
        self.seed = seed
        self.env_id = env_id
        self.dm_control = dm_control
        self.teacher_feedback_mode = teacher_feedback_mode

    def record_trajectory(self, trajectory: Trajectory, run_name: str, fps=30):
        start_idx = trajectory.replay_buffer_start_idx
        end_idx = trajectory.replay_buffer_end_idx
        env_idx = trajectory.samples.env_idx
        length = trajectory.replay_buffer_end_idx - trajectory.replay_buffer_start_idx
        episode_index = 0
        # Ensure the directory for videos exists
        video_folder = f"./videos/{run_name}/trajectories/"
        os.makedirs(video_folder, exist_ok=True)
        name_prefix = f"trajectory_{start_idx}_{end_idx}_{env_idx}"
        env = make_single_env(
            env_id=self.env_id,
            render="rgb_array",
            video_recorder=True,
            teacher_feedback_mode=self.teacher_feedback_mode,
        )

        try:
            self._initialize_env_state(env, trajectory)
            frames = self._generate_frames(env, trajectory)
            save_video(
                frames=frames,
                video_length=length,
                video_folder=video_folder,
                fps=fps,
                name_prefix=name_prefix,
                episode_index=episode_index,
            )
            video_file_name = (
                f"{run_name}/trajectories/{name_prefix}-episode-{episode_index}.mp4"
            )
            logging.debug(f"Finished recording trajectory video: {video_file_name}")
            return video_file_name
        except Exception as e:
            logging.error(
                f"Error recording trajectory (start_idx={start_idx}, end_idx={end_idx}, env_idx={env_idx}): {e}"
            )
        finally:
            env.close()

    def _initialize_env_state(self, env, trajectory):
        """Initialize the environment state based on Mujoco or non-Mujoco trajectories."""
        if is_mujoco_env(env) or self.dm_control:
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
            if self.dm_control:
                env.unwrapped.physics.set_state(qpos)
            else:
                env.unwrapped.set_state(qpos, qvel)
        else:
            env.reset(seed=self.seed)
            if hasattr(env, "state"):
                env.state = trajectory.samples.observations[0]

    def _generate_frames(self, env, trajectory):
        """Generate frames for the video from the trajectory."""
        frames = [env.render()]
        if is_mujoco_env(env) or self.dm_control:
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
                if self.dm_control:
                    env.unwrapped.physics.set_state(qpos)
                    env.physics.forward()
                    frames.append(env.render())
                else:
                    env.unwrapped.set_state(qpos, qvel)
                    frames.append(env.render())
        else:
            actions = trajectory.samples.actions
            for action in actions:
                action = action.detach().cpu().numpy()
                env.step(action)
                frames.append(env.render())
        return frames


def parse_video_filename(video_name: str):
    """
    Parse the video file name and extract trajectory-related indices.

    Example filename: trajectory_100_200_1-episode-0.mp4
    """
    match = re.match(r"trajectory_(\d+)_(\d+)_(\d+)-episode-\d+\.mp4", video_name)
    if not match:
        raise ValueError(f"Invalid video file name format: {video_name}")

    start_idx = int(match.group(1))
    end_idx = int(match.group(2))
    env_idx = int(match.group(3))
    return start_idx, end_idx, env_idx


def retrieve_trajectory_by_video_name(
    replay_buffer: ReplayBuffer, video_name: str, env=None
) -> Trajectory:
    """
    Retrieve the trajectory object corresponding to a given video name.

    Args:
        replay_buffer (ReplayBuffer): The replay buffer.
        video_name (str): The video file name.
        env: Optional VecNormalize environment for normalizing samples.

    Returns:
        Trajectory: The corresponding trajectory object.
    """
    start_idx, end_idx, _ = parse_video_filename(video_name.split("/")[-1])
    return replay_buffer.get_trajectory(start_idx, end_idx, env)

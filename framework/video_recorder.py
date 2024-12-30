import os
import logging
import gymnasium as gym
import imageio
import torch
from replay_buffer import ReplayBuffer, Trajectory
from env import is_mujoco_env


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

        try:
            self._initialize_env_state(env, trajectory)
            self._write_trajectory_to_video(env, trajectory, out_path, fps)
        except Exception as e:
            logging.error(
                f"Error recording trajectory (start_idx={start_idx}, end_idx={end_idx}, env_idx={env_idx}): {e}",
                exc_info=True,
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
            env.unwrapped.set_state(qpos, qvel)
        else:
            env.reset(seed=self.seed)
            if hasattr(env, "state"):
                env.state = trajectory.samples.observations[0]

    def _write_trajectory_to_video(self, env, trajectory, out_path, fps=30):
        """Write the trajectory to a video file."""
        img = env.render()
        images = [img]
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
                env.unwrapped.set_state(qpos, qvel)
                images.append(env.render())
        else:
            actions = trajectory.samples.actions
            for action in actions:
                if isinstance(action, torch.Tensor):
                    action = action.detach().cpu().numpy()
                env.step(action)
                images.append(env.render())

        imageio.mimsave(uri=out_path, ims=images, fps=fps)

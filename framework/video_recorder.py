import os
import logging

import gymnasium as gym
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

        # Ensure the directory for videos exists
        video_folder = f"./videos/{run_name}/trajectories"
        os.makedirs(video_folder, exist_ok=True)
        out_path = f"{video_folder}/trajectory_{start_idx}_{end_idx}.mp4"
        if os.path.exists(out_path):
            logging.info(f"Skipping {out_path}")
            return
        env = gym.make(self.env_id, render_mode="rgb_array")
        # Start the video at the first observation of that trajectory
        qpos_list = trajectory.samples.qpos
        qvel_list = trajectory.samples.qvel

        # env.reset has to be called to be able to render
        env.reset()

        # Set state from mujoco positions
        env.unwrapped.set_state(qpos_list[0], qvel_list[0])

        img = env.render()
        writer = cv2.VideoWriter(
            out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (img.shape[1], img.shape[0])
        )
        writer.write(img)

        # Replay the actions
        for i in range(1, len(qpos_list)):
            env.unwrapped.set_state(qpos_list[i], qvel_list[i])
            img = env.render()
            writer.write(img)

        # Close the environment to ensure the video file is finalized
        writer.release()
        env.close()

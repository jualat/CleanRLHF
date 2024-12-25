import os
import random

import cv2
import torch
import numpy as np
import gymnasium as gym
import tyro
from scipy.stats import norm

from sac_rlhf import Actor, load_model_all
from dataclasses import dataclass
from tqdm import trange


@dataclass
class Args:
    path_to_model: str = ""
    """path to model"""
    env_id: str = "Hopper-v4"
    """the environment of the policy"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    run_name: str = ""
    """name of the run"""
    episodes: int = 30
    """number of episodes to run"""
    confidence: float = 0.95
    """confidence interval"""
    lowest_x_percent: float = 0.1
    """lowest x percent"""
    render: bool = True
    """render the videos"""


class Evaluation:
    def __init__(
        self,
        path_to_model=None,
        actor=None,
        env_id="Hopper-v4",
        render=False,
        seed=None,
        torch_deterministic=True,
        run_name=None,
    ):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.render = render
        self.seed = seed
        self.run_name = run_name
        if render:
            self.env = gym.make(env_id, render_mode="rgb_array")
        else:
            self.env = gym.make(env_id)
        self.env = gym.wrappers.RecordEpisodeStatistics(self.env)

        if path_to_model:
            self.actor = Actor(gym.vector.SyncVectorEnv([lambda: self.env]), 256, 4)
            state_dict = {"actor": self.actor}
            load_model_all(state_dict, path_to_model, self.device)
        else:
            self.actor = actor
        self.actor.eval()
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            self.env.action_space.seed(seed)

        torch.backends.cudnn.deterministic = torch_deterministic

    def evaluate_policy(self, episodes=30, fps=30, confidence=0.95, lowest_x_pct=0.1):
        actor = self.actor
        env = self.env
        run_name = self.run_name
        if self.render:
            video_folder = f"./videos/{run_name}/evaluation"
            os.makedirs(video_folder, exist_ok=True)

        episode_rewards = []
        for episode in trange(episodes, unit="episodes", desc="Evaluating"):
            obs, _ = env.reset(seed=self.seed)
            done = False
            total_episode_reward = 0
            if self.render:
                out_path = f"{video_folder}/{episode}.mp4"
                img = env.render()
                writer = cv2.VideoWriter(
                    out_path,
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps,
                    (img.shape[1], img.shape[0]),
                )
                writer.write(img)

            while not done:
                action, _, _ = actor.get_action(
                    torch.Tensor(obs).to(self.device).unsqueeze(0)
                )
                obs, reward, termination, truncation, _ = env.step(
                    action.detach().cpu().numpy().squeeze(0)
                )
                if self.render:
                    writer.write(env.render())
                total_episode_reward += reward
                done = termination or truncation

            if self.render:
                writer.release()
            episode_rewards.append(total_episode_reward)

        mean_episode_reward = np.mean(episode_rewards)
        std_episode_reward = np.std(episode_rewards)
        alpha = 1 - confidence
        z_value = norm.ppf(1 - alpha / 2)
        confidence_interval = z_value * std_episode_reward / np.sqrt(episodes)
        left_confidence_interval = mean_episode_reward - confidence_interval
        right_confidence_interval = mean_episode_reward + confidence_interval

        return {
            "mean_reward": mean_episode_reward,
            "std_reward": std_episode_reward,
            "ci_left": left_confidence_interval,
            "ci_right": right_confidence_interval,
            "max_reward": max(episode_rewards),
            "min_reward": min(episode_rewards),
            "median_reward": np.median(episode_rewards),
            f"lowest_{lowest_x_pct}_pct": np.quantile(episode_rewards, lowest_x_pct),
        }


if __name__ == "__main__":
    args = tyro.cli(Args)

    evaluation = Evaluation(
        path_to_model=args.path_to_model,
        env_id=args.env_id,
        render=args.render,
        seed=args.seed,
        torch_deterministic=args.torch_deterministic,
        run_name=args.run_name,
    )

    eval_dict = evaluation.evaluate_policy(
        episodes=args.episodes,
        confidence=args.confidence,
        lowest_x_pct=args.lowest_x_percent,
    )

    print(eval_dict)

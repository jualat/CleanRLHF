import os
import random
from dataclasses import dataclass

import gymnasium as gym
import imageio
import matplotlib

matplotlib.use("AGG")
import numpy as np
import pandas as pd
import torch
import tyro
import wandb
from actor import Actor
from env import is_pusher_v5, load_model_all, make_single_env
from plotnine import aes, geom_line, geom_point, ggplot, labs
from scipy.stats import norm
from tqdm import trange


@dataclass
class Args:
    path_to_model: str = ""
    """path to model"""
    env_id: str = "Hopper-v5"
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
    actor_net_hidden_dim: int = 256
    """the dimension of the hidden layers in the actor network"""
    actor_net_hidden_layers: int = 4
    """the number of hidden layers in the actor network"""


class Evaluation:
    def __init__(
        self,
        path_to_model=None,
        actor=None,
        env_id="Hopper-v4",
        seed=None,
        torch_deterministic=True,
        run_name=None,
        hidden_dim=256,
        hidden_layers=4,
    ):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.seed = seed
        self.run_name = run_name
        self.env_id = env_id
        self.DEFAULT_CAMERA_CONFIG_PUSHER = {
            "trackbodyid": -1,
            # "distance": 4.0,
            "distance": 3.0,
            "azimuth": 135.0,
            "elevation": -22.5,
        }
        env = make_single_env(env_id)
        env = gym.vector.SyncVectorEnv([lambda: env])
        if path_to_model:
            self.actor = Actor(env, hidden_dim, hidden_layers)
            state_dict = {"actor": self.actor}
            load_model_all(state_dict, path_to_model, self.device)
        else:
            self.actor = actor
        self.actor.eval()
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = torch_deterministic
        env.close()

    def evaluate_policy(
        self,
        episodes=30,
        fps=30,
        confidence=0.95,
        lowest_x_pct=0.1,
        step=None,
        render=False,
        actor=None,
        track=False,
    ):
        """
        Evaluate the policy
        :param episodes: The number of episodes to run
        :param fps: The frames per second of the video
        :param confidence: The confidence interval size
        :param lowest_x_pct: The lowest `lowest_x_pct` percentile of the episode rewards gets returned
        :param step: The global step
        :param render: Boolean if the videos should be rendered
        :param actor: The actor network
        :param track: Boolean if the best and worst video should be tracked
        :return:
        """
        actor = actor.eval() if actor is not None else self.actor
        env = self.make_env(render=render)
        run_name = self.run_name
        if render:
            video_folder = f"./videos/{run_name}/evaluation"
            os.makedirs(video_folder, exist_ok=True)

        episode_rewards = []
        video_paths = []
        for episode in trange(episodes, unit="episodes", desc="Evaluating"):
            obs, _ = env.reset(seed=self.seed)
            done = False
            total_episode_reward = 0
            if render:
                images = []
                img = env.render()[0]
                images.append(img)

            while not done:
                action, _, _ = actor.get_action(
                    torch.Tensor(obs).unsqueeze(0).to(self.device)
                )
                obs, reward, termination, truncation, _ = env.step(
                    action.detach().cpu().numpy().squeeze(0)
                )
                if render:
                    images.append(env.render()[0])
                total_episode_reward += reward.item()
                done = termination or truncation

            if render:
                if step is not None:
                    reward_path = f"{total_episode_reward:.0f}_{episode}_{step}.mp4"
                    out_path = os.path.join(video_folder, reward_path)
                else:
                    reward_path = f"{total_episode_reward:.0f}_{episode}.mp4"
                    out_path = os.path.join(video_folder, reward_path)
                imageio.mimsave(uri=out_path, ims=images, fps=fps)
                video_paths.append(reward_path)
            episode_rewards.append(total_episode_reward)
        if env is not None:
            env.close()
        mean_episode_reward = np.mean(episode_rewards)
        std_episode_reward = np.std(episode_rewards)
        alpha = 1 - confidence
        z_value = norm.ppf(1 - alpha / 2)
        confidence_interval = z_value * std_episode_reward / np.sqrt(episodes)
        left_confidence_interval = mean_episode_reward - confidence_interval
        right_confidence_interval = mean_episode_reward + confidence_interval

        if render and track:
            best_video = max(video_paths, key=lambda f: float(f.split("_")[0]))
            worst_video = min(video_paths, key=lambda f: float(f.split("_")[0]))
            best_video_path = os.path.join(video_folder, best_video)
            worst_video_path = os.path.join(video_folder, worst_video)
            wandb.log(
                {
                    "Best Video": wandb.Video(best_video_path, format="mp4"),
                    "Worst Video": wandb.Video(worst_video_path, format="mp4"),
                }
            )

        return {
            "mean_reward": mean_episode_reward,
            "std_reward": std_episode_reward,
            "ci_left": left_confidence_interval,
            "ci_right": right_confidence_interval,
            "max_reward": max(episode_rewards),
            "min_reward": min(episode_rewards),
            "median_reward": np.median(episode_rewards),
            f"lowest_{lowest_x_pct}_pct": np.quantile(episode_rewards, lowest_x_pct),
            "episode_rewards": episode_rewards,
            "episode": list(range(episodes)),
        }

    def plot(self, eval_dict, step=None):
        """
        Plot the evaluation
        :param eval_dict: Dictionary with the evaluation results
        :param step: The global step / total_timesteps
        :return:
        """
        model_folder = f"./models/{self.run_name}"
        os.makedirs(model_folder, exist_ok=True)
        if step is not None:
            out_path = f"{model_folder}/evaluation_{step}.png"
        else:
            out_path = f"{model_folder}/evaluation.png"
        df = pd.DataFrame(eval_dict)
        (
            ggplot(df, aes(x="episode", y="episode_rewards"))
            + geom_point(aes(color="'Episode Rewards'"))
            + geom_line(
                aes(x="episode", y="mean_reward", color="'Mean Reward'"), alpha=0.7
            )
            + labs(title="Evaluation", x="Episode", y="Episode Reward", color="Legend")
        ).save(out_path, width=10, height=6, dpi=300)

    def make_env(self, render):
        """
        Create the environment
        :param render: If videos should be rendered
        :return:
        """
        if render:
            if is_pusher_v5(self.env_id):
                env = gym.make(
                    self.env_id,
                    default_camera_config=self.DEFAULT_CAMERA_CONFIG_PUSHER,
                    render_mode="rgb_array",
                )
            else:
                env = make_single_env(self.env_id, render="rgb_array")
        else:
            env = make_single_env(self.env_id)
        env.action_space.seed(self.seed)
        env = gym.vector.SyncVectorEnv([lambda: env])
        return env


if __name__ == "__main__":
    args = tyro.cli(Args)

    evaluation = Evaluation(
        path_to_model=args.path_to_model,
        env_id=args.env_id,
        seed=args.seed,
        torch_deterministic=args.torch_deterministic,
        run_name=args.run_name,
        hidden_layers=args.actor_net_hidden_layers,
        hidden_dim=args.actor_net_hidden_dim,
    )

    eval_dict = evaluation.evaluate_policy(
        episodes=args.episodes,
        confidence=args.confidence,
        lowest_x_pct=args.lowest_x_percent,
        render=args.render,
    )

    print(eval_dict)
    evaluation.plot(eval_dict)

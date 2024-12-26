import gymnasium as gym
import os
import torch
import logging
from replay_buffer import ReplayBuffer
import gzip
import pickle


def save_model_all(run_name: str, step: int, state_dict: dict):
    model_folder = f"./models/{run_name}/{step}"
    os.makedirs(model_folder, exist_ok=True)
    out_path = f"{model_folder}/checkpoint.pth"

    total_state_dict = {name: obj.state_dict() for name, obj in state_dict.items()}
    torch.save(total_state_dict, out_path)
    logging.info(f"Saved all models and optimizers to {out_path}")


def load_model_all(state_dict: dict, path: str, device):
    assert os.path.exists(path), "Path to model does not exist"
    checkpoint = torch.load(path, device)
    for name, obj in state_dict.items():
        if name in checkpoint:
            obj.load_state_dict(checkpoint[name])
    logging.info(f"Models and optimizers loaded from {path}")


def save_replay_buffer(run_name: str, step: int, replay_buffer: ReplayBuffer):
    model_folder = f"./models/{run_name}/{step}"
    os.makedirs(model_folder, exist_ok=True)
    out_path = f"{model_folder}/replay_buffer.pth"

    buffer_data = {
        "observations": replay_buffer.observations,
        "next_observations": replay_buffer.next_observations,
        "actions": replay_buffer.actions,
        "rewards": replay_buffer.rewards,
        "ground_truth_rewards": replay_buffer.ground_truth_rewards,
        "dones": replay_buffer.dones,
    }
    with gzip.open(out_path, "wb") as f:
        pickle.dump(buffer_data, f)
    logging.info(f"Saved replay buffer to {out_path}")


def load_replay_buffer(replay_buffer: ReplayBuffer, path: str):
    assert os.path.exists(path), "Path to replay buffer does not exist"
    with gzip.open(path, "rb") as f:
        buffer_data = pickle.load(f)

    replay_buffer.observations = buffer_data["observations"]
    replay_buffer.next_observations = buffer_data["next_observations"]
    replay_buffer.actions = buffer_data["actions"]
    replay_buffer.rewards = buffer_data["rewards"]
    replay_buffer.ground_truth_rewards = buffer_data["ground_truth_rewards"]
    replay_buffer.dones = buffer_data["dones"]

    logging.info(f"Replay buffer loaded from {path}")


def make_env(env_id, seed):
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env

    return thunk

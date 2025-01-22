import gzip
import logging
import os
import pickle

import gymnasium as gym
import numpy as np
import shimmy
import torch
from dm_control import suite
from gym import Wrapper
from gymnasium.envs.registration import registry
from replay_buffer import ReplayBuffer


def save_model_all(run_name: str, step: int, state_dict: dict):
    """
    Save all models and optimizers to a checkpoint file
    :param run_name: The name of the run
    :param step: The current step
    :param state_dict: The current state dictionary
    :return:
    """
    model_folder = f"./models/{run_name}/{step}"
    os.makedirs(model_folder, exist_ok=True)
    out_path = f"{model_folder}/checkpoint.pth"

    total_state_dict = {name: obj.state_dict() for name, obj in state_dict.items()}
    torch.save(total_state_dict, out_path)
    logging.info(f"Saved all models and optimizers to {out_path}")


def load_model_all(state_dict: dict, path: str, device):
    """
    Load all models and optimizers from a checkpoint file
    :param state_dict: The state dictionary to load the models and optimizers into
    :param path: The path to the model
    :param device: The torch device
    :return:
    """
    assert os.path.exists(path), "Path to model does not exist"
    checkpoint = torch.load(path, device)
    for name, obj in state_dict.items():
        if name in checkpoint:
            obj.load_state_dict(checkpoint[name])
    logging.info(f"Models and optimizers loaded from {path}")


def save_replay_buffer(run_name: str, step: int, replay_buffer: ReplayBuffer):
    """
    Save the replay buffer to a file
    :param run_name: The name of the current run
    :param step: The current step
    :param replay_buffer: The current replay buffer
    :return:
    """
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
    """
    Load the replay buffer from a file
    :param replay_buffer: The replay buffer to load into
    :param path: The path to the replay buffer
    :return:
    """
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


def is_dm_control(env_id):
    DM_CONTROL_ENV_IDS = [
        env_id
        for env_id in registry
        if env_id.startswith("dm_control")
        and env_id != "dm_control/compatibility-env-v0"
    ]
    return isinstance(DM_CONTROL_ENV_IDS, list) and env_id in DM_CONTROL_ENV_IDS


def make_single_env(env_id, render=None, camera_settings=None):
    if is_dm_control(env_id):
        name = env_id.split("/")[1]
        domain = name.split("-")[0]
        task = name.split("-")[1]
        if render is not None and render != "human":
            render = "multi_camera"
        env = suite.load(domain_name=domain, task_name=task)
        env = shimmy.dm_control_compatibility.DmControlCompatibilityV0(
            env, render_mode=render
        )
    else:
        env = gym.make(
            env_id, render_mode=render, default_camera_config=camera_settings
        )
    return env


def make_env(env_id, seed, render=None):
    """
    Create an environment
    :param env_id: The run argument env_id
    :param seed: The seed for creating the environment
    :param render: Set rendering mode
    :return:
    """
    if render == "":
        render = None

    def thunk():
        env = make_single_env(env_id, render=render)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env

    return thunk


def is_mujoco_env(env) -> bool:
    """
    Check if the environment is a mujoco environment
    :param env: The environment to check
    :return:
    """
    # Try to check the internal `mujoco` attribute
    return hasattr(env.unwrapped, "model") and hasattr(env.unwrapped, "do_simulation")


class FlattenVectorObservationWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        original_obs_space = self.env.observation_space
        if isinstance(original_obs_space, gym.spaces.dict.Dict):
            low = np.concatenate(
                [
                    space.low.flatten()
                    for space in original_obs_space.spaces.values()
                    if space.shape != (0,)
                ]
            )
            high = np.concatenate(
                [
                    space.high.flatten()
                    for space in original_obs_space.spaces.values()
                    if space.shape != (0,)
                ]
            )
            self.single_observation_space = gym.spaces.Box(
                low=low, high=high, dtype=np.float64
            )
            self.shimmy = True
        else:
            self.shimmy = False
            self.single_observation_space = original_obs_space

    def reset(self, **kwargs):
        if self.shimmy:
            obs, x = self.env.reset(**kwargs)
            return self.flatten_observation(obs), x
        return self.env.reset(**kwargs)

    def step(self, action):
        if self.shimmy:
            obs, rewards, terminations, truncations, infos = self.env.step(action)
            return (
                self.flatten_observation(obs),
                rewards,
                terminations,
                truncations,
                infos,
            )
        return self.env.step(action)

    def flatten_observation(self, obs):
        # Apply flattening to all observations in the batch
        obs_dict = obs  # Obs is a dictionary with different keys and each obs is stored concatenated in a numpy array
        flat_obs = np.concatenate(
            [obs_dict[key].flatten() for key in obs.keys()]
        )  # This currently only works with single env
        flat_obs = flat_obs[np.newaxis, :]
        return flat_obs

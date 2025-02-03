import gzip
import logging
import os
import pickle
import warnings
from typing import Dict, Optional

import gymnasium as gym
import numpy as np
import shimmy
import torch
from dm_control import suite
from gymnasium import Env
from gymnasium.envs.registration import registry
from gymnasium.vector import SyncVectorEnv
from gymnasium.wrappers import FlattenObservation
from replay_buffer import ReplayBuffer


def save_model_all(run_name: str, step: int, state_dict: dict) -> None:
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


def load_model_all(state_dict: dict, path: str, device) -> None:
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


def save_replay_buffer(run_name: str, step: int, replay_buffer: ReplayBuffer) -> None:
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


def load_replay_buffer(replay_buffer: ReplayBuffer, path: str) -> None:
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


def make_single_env(
    env_id: str,
    render: Optional[str] = None,
    camera_settings: Optional[Dict] = None,
    video_recorder: Optional[bool] = False,
):
    if is_dm_control(env_id):
        name = env_id.split("/")[1]
        domain = name.split("-")[0]
        task = name.split("-")[1]
        if task != "v0":
            if render is not None and render != "human":
                render = "multi_camera"
            env = suite.load(domain_name=domain, task_name=task)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=DeprecationWarning)
                env = shimmy.dm_control_compatibility.DmControlCompatibilityV0(
                    env, render_mode=render
                )
        else:
            env = gym.make(env_id, render_mode=render)
        if not video_recorder:
            env = FlattenObservation(env)
    else:
        env = gym.make(
            env_id, render_mode=render, default_camera_config=camera_settings
        )
    return env


def make_env(env_id: str, seed: Optional[int], render: Optional[str] = None):
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


def make_env_ppo(
    env_id: str,
    gamma: float,
    seed: Optional[int],
    index: int,
    render_mode: Optional[str] = None,
):
    if render_mode == "":
        render_mode = None

    def thunk():
        if index == 0 and render_mode is not None:
            env = make_single_env(env_id, render=render_mode)
        else:
            env = make_single_env(env_id, render=render_mode)
        env = gym.wrappers.FlattenObservation(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(
            env, lambda obs: np.clip(obs, -10, 10), env.observation_space
        )
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env

    return thunk


def is_mujoco_env(env: Env) -> bool:
    """
    Check if the environment is a mujoco environment
    :param env: The environment to check
    :return:
    """
    # Try to check the internal `mujoco` attribute
    return hasattr(env.unwrapped, "model") and hasattr(env.unwrapped, "do_simulation")


def is_dm_control(env_id: str) -> bool:
    DM_CONTROL_ENV_IDS = [
        env_id
        for env_id in registry
        if env_id.startswith("dm_control")
        and env_id != "dm_control/compatibility-env-v0"
    ]
    return isinstance(DM_CONTROL_ENV_IDS, list) and env_id in DM_CONTROL_ENV_IDS


def initialize_qpos_qvel(
    envs: SyncVectorEnv, num_envs: int, dm_control_bool: bool
) -> (np.ndarray, np.ndarray):
    if is_mujoco_env(envs.envs[0]):
        try:
            qpos = np.zeros(
                (
                    num_envs,
                    envs.envs[0].unwrapped.observation_structure["qpos"]
                    + envs.envs[0].unwrapped.observation_structure["skipped_qpos"],
                ),
                dtype=np.float32,
            )
            qvel = np.zeros(
                (num_envs, envs.envs[0].unwrapped.observation_structure["qvel"]),
                dtype=np.float32,
            )
        except AttributeError:
            qpos = np.zeros(
                (num_envs, envs.envs[0].unwrapped.model.key_qpos.shape[1]),
                dtype=np.float32,
            )
            qvel = np.zeros(
                (num_envs, envs.envs[0].unwrapped.model.key_qvel.shape[1]),
                dtype=np.float32,
            )
    elif dm_control_bool:
        qpos = np.zeros(
            (num_envs, envs.envs[0].unwrapped.physics.get_state().shape[0]),
            dtype=np.float32,
        )
        qvel = np.zeros((num_envs, 1))
    else:
        qpos = np.zeros((num_envs, 1))
        qvel = np.zeros((num_envs, 1))

    return qpos, qvel


def get_qpos_qvel(
    envs: SyncVectorEnv, qpos: np.ndarray, qvel: np.ndarray, dm_control_bool: bool
) -> None:
    if is_mujoco_env(envs.envs[0]):
        for idx in range(qpos.shape[0]):
            single_env = envs.envs[idx]
            qpos[idx] = single_env.unwrapped.data.qpos.copy()
            qvel[idx] = single_env.unwrapped.data.qvel.copy()
    if dm_control_bool:
        for idx in range(qpos.shape[0]):
            single_env = envs.envs[idx]
            qpos[idx] = single_env.unwrapped.physics.get_state()


def is_pusher_v5(env_id: str) -> bool:
    """
    Check if the environment is the Pusher-v5 environment
    :param env_id: The environment id
    :return:
    """
    return env_id == "Pusher-v5"

# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
import os
import random
import time
from dataclasses import dataclass
import logging

import gymnasium as gym
from gymnasium.experimental.wrappers.rendering import RecordVideoV0
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
import pickle
import gzip

from video_recorder import VideoRecorder
from unsupervised_exploration import ExplorationRewardKNN
from preference_buffer import PreferenceBuffer
from replay_buffer import ReplayBuffer
from reward_net import RewardNet, train_reward
from torch.utils.tensorboard import SummaryWriter
from teacher import Teacher

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    num_envs: int = 12
    """the number of parallel environments to accelerate training. 
    Set this to the number of available CPU threads for best performance."""
    log_file: bool = True
    """if toggled, logger will write to a file"""
    log_level: str = "DEBUG"
    """the threshold level for the logger to print a message"""

    # Algorithm specific arguments
    env_id: str = "Hopper-v4"
    """the environment id of the task"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    learning_starts: int = 5e3
    """timestep to start learning"""
    policy_lr: float = 3e-4
    """the learning rate of the policy network optimizer"""
    q_lr: float = 1e-3
    """the learning rate of the Q network network optimizer"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    target_network_frequency: int = 1  # Denis Yarats' implementation delays this by 2.
    """the frequency of updates for the target nerworks"""
    alpha: float = 0.2
    """Entropy regularization coefficient."""
    autotune: bool = True
    """automatic tuning of the entropy coefficient"""

    # Human feedback arguments
    teacher_feedback_frequency: int = 5000
    """the frequency of teacher feedback (every K iterations)"""
    teacher_feedback_num_queries_per_session: int = 100
    """the number of queries per feedback session"""
    teacher_update_epochs: int = 20
    """the amount of gradient steps to take on the teacher feedback"""
    teacher_feedback_batch_size: int = 32
    """the batch size of the teacher feedback sampled from the feedback buffer"""
    teacher_learning_rate: float = 1e-3
    """the learning rate of the teacher"""

    # Simulated Teacher
    teacher_sim_beta: float = -1
    """this parameter controls how deterministic or random the teacher's preferences are"""
    teacher_sim_gamma: float = 1
    """the discount factor gamma, which models myopic behavior"""
    teacher_sim_epsilon: float = 0
    """with probability epsilon, the teacher's preference is flipped, introducing randomness"""
    teacher_sim_delta_skip: float = 0
    """skip two trajectories if neither segment demonstrates the desired behavior"""
    teacher_sim_delta_equal: float = 0
    """the range of two trajectories being equal"""

    # Unsupervised Exploration
    unsupervised_exploration: bool = True
    """toggle the unsupervised exploration"""
    total_explore_steps: int = 10000
    """total number of explore steps"""
    explore_batch_size: int = 256
    """the batch size of the explore sampled from the replay buffer"""
    explore_learning_starts: int = 512
    """timestep to start learning in the exploration"""

    # Load Model
    exploration_load: bool = False
    """skip exploration and load the pre-trained model and buffer from a file"""
    path_to_replay_buffer: str = ""
    """path to replay buffer"""
    path_to_model: str = ""
    """path to model"""


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            # You've to use experimental wrappers to record video to avoid black screen issue:
            # https://github.com/Farama-Foundation/Gymnasium/issues/455#issuecomment-1517900688
            env = RecordVideoV0(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env

    return thunk

def select_actions(obs, actor, device, step, learning_start, envs):
    if step < learning_start:
        return np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
    else:
        actions, _, _ = actor.get_action(torch.Tensor(obs).to(device))
        return actions.detach().cpu().numpy()

def train_q_network(data, qf1, qf2, qf1_target, qf2_target, alpha, gamma, q_optimizer):
    with torch.no_grad():
        real_rewards = data.rewards
        next_state_actions, next_state_log_pi, _ = actor.get_action(data.next_observations)
        qf1_next_target = qf1_target(data.next_observations, next_state_actions)
        qf2_next_target = qf2_target(data.next_observations, next_state_actions)
        min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
        next_q_value = real_rewards.flatten() + (1 - data.dones.flatten()) * gamma * (
            min_qf_next_target).view(-1)

    qf1_a_values = qf1(data.observations, data.actions).view(-1)
    qf2_a_values = qf2(data.observations, data.actions).view(-1)
    qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
    qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
    qf_loss = qf1_loss + qf2_loss

    # optimize the model
    q_optimizer.zero_grad()
    qf_loss.backward()
    q_optimizer.step()

    return qf_loss, qf1_a_values, qf2_a_values, qf1_loss, qf2_loss

def update_actor(data, actor, qf1, qf2, alpha, actor_optimizer):
    pi, log_pi, _ = actor.get_action(data.observations)
    qf1_pi = qf1(data.observations, pi)
    qf2_pi = qf2(data.observations, pi)
    min_qf_pi = torch.min(qf1_pi, qf2_pi)
    actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

    actor_optimizer.zero_grad()
    actor_loss.backward()
    actor_optimizer.step()
    if args.autotune:
        with torch.no_grad():
            _, log_pi, _ = actor.get_action(data.observations)
        alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()

        a_optimizer.zero_grad()
        alpha_loss.backward()
        a_optimizer.step()
        alpha = log_alpha.exp().item()
    return actor_loss, alpha, alpha_loss

def update_target_networks(source_net, target_net, tau):
    for param, target_param in zip(source_net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

def save_model_all(run_name: str, step: int,  state_dict: dict):
    model_folder = f"./models/{run_name}/{step}"
    os.makedirs(model_folder, exist_ok=True)
    out_path = f"{model_folder}/checkpoint.pth"

    total_state_dict = {name: obj.state_dict() for name, obj in state_dict.items()}
    torch.save(total_state_dict, out_path)
    logging.debug(f"Saved all models and optimizers to {out_path}")

def load_model_all(state_dict: dict, path: str, device):
    assert os.path.exists(path), "Path to model does not exist"
    checkpoint = torch.load(path, device)
    for name, obj in state_dict.items():
        if name in checkpoint:
            obj.load_state_dict(checkpoint[name])
    logging.debug(f"Models and optimizers loaded from {path}")

def save_replay_buffer(run_name: str, step:int, replay_buffer :ReplayBuffer):
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
    logging.debug(f"Saved replay buffer to {out_path}")

def load_replay_buffer(replay_buffer: ReplayBuffer, path:str):
    assert os.path.exists(path), "Path to replay buffer does not exist"
    with gzip.open(path, "rb") as f:
        buffer_data = pickle.load(f)

    replay_buffer.observations = buffer_data["observations"]
    replay_buffer.next_observations = buffer_data["next_observations"]
    replay_buffer.actions = buffer_data["actions"]
    replay_buffer.rewards = buffer_data["rewards"]
    replay_buffer.ground_truth_rewards = buffer_data["ground_truth_rewards"]
    replay_buffer.dones = buffer_data["dones"]

    logging.debug(f"Replay buffer loaded from {path}")


# ALGO LOGIC: initialize agent here:
class SoftQNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod() + np.prod(env.single_action_space.shape), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, np.prod(env.single_action_space.shape))
        self.fc_logstd = nn.Linear(256, np.prod(env.single_action_space.shape))
        # action rescaling
        self.register_buffer(
            "action_scale", torch.tensor((env.single_action_space.high - env.single_action_space.low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((env.single_action_space.high + env.single_action_space.low) / 2.0, dtype=torch.float32)
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean


if __name__ == "__main__":
    import stable_baselines3 as sb3

    if sb3.__version__ < "2.0":
        raise ValueError(
            """Ongoing migration: run the following command to install the new dependencies:
poetry run pip install "stable_baselines3==2.0.0a1"
"""
        )

    args = tyro.cli(Args)
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s", datefmt="%d/%m/%Y %H:%M:%S", level=args.log_level.upper())
    if args.log_file:
        os.makedirs(os.path.join("runs", run_name), exist_ok=True)
        logging.getLogger().addHandler(logging.FileHandler(filename=f"runs/{run_name}/logger.log"))
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed, i, args.capture_video, run_name) for i in range(args.num_envs)])
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    max_action = float(envs.single_action_space.high[0])

    actor = Actor(envs).to(device)
    qf1 = SoftQNetwork(envs).to(device)
    qf2 = SoftQNetwork(envs).to(device)
    qf1_target = SoftQNetwork(envs).to(device)
    qf2_target = SoftQNetwork(envs).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)

    # Automatic entropy tuning
    if args.autotune:
        target_entropy = -torch.prod(torch.Tensor(envs.single_action_space.shape).to(device)).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        alpha = args.alpha

    envs.single_observation_space.dtype = np.float32
    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        handle_timeout_termination=False,
        n_envs=args.num_envs,
    )
    start_time = time.time()

    pref_buffer = PreferenceBuffer((args.buffer_size // args.teacher_feedback_frequency) * args.teacher_feedback_num_queries_per_session)
    reward_net = RewardNet(hidden_dim=256, env=envs).to(device)
    reward_optimizer = optim.Adam(reward_net.parameters(), lr=args.teacher_learning_rate)
    video_recorder = VideoRecorder(rb, args.seed, args.env_id)

    # Init Teacher
    sim_teacher = Teacher(
        args.teacher_sim_beta,
        args.teacher_sim_gamma,
        args.teacher_sim_epsilon,
        args.teacher_sim_delta_skip,
        args.teacher_sim_delta_equal,
        args.seed
    )
    current_step = 0
    if args.unsupervised_exploration and not args.exploration_load:
        knn_estimator = ExplorationRewardKNN(k=3)
        current_step += 1
        obs, _ = envs.reset(seed=args.seed)
        for explore_step in range(args.total_explore_steps):

            actions = select_actions(obs, actor, device, explore_step, args.explore_learning_starts, envs)

            next_obs, ground_truth_reward, terminations, truncations, infos = envs.step(actions)

            real_next_obs = next_obs.copy()
            for idx, trunc in enumerate(truncations):
                if trunc:
                    real_next_obs[idx] = infos["final_observation"][idx]

            intrinsic_reward = knn_estimator.compute_intrinsic_rewards(next_obs)
            knn_estimator.update_states(next_obs)

            rb.add(obs, real_next_obs, actions, intrinsic_reward, ground_truth_reward, terminations, infos)

            obs = next_obs
            if explore_step > args.explore_learning_starts:
                data = rb.sample(args.explore_batch_size)
                qf_loss, qf1_a_values, qf2_a_values, qf1_loss, qf2_loss = train_q_network(data,
                                                                                          qf1,
                                                                                          qf2,
                                                                                          qf1_target,
                                                                                          qf2_target,
                                                                                          alpha,
                                                                                          args.gamma,
                                                                                          q_optimizer
                                                                                          )

                if explore_step % args.policy_frequency == 0:  # TD 3 Delayed update support
                    for _ in range(args.policy_frequency):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                        actor_loss = update_actor(data,
                                                  actor,
                                                  qf1,
                                                  qf2,
                                                  alpha,
                                                  actor_optimizer)

            if explore_step % args.target_network_frequency == 0:
                update_target_networks(qf1, qf1_target, args.tau)
                update_target_networks(qf2, qf2_target, args.tau)

            if explore_step % 100 == 0:
                writer.add_scalar("exploration/intrinsic_reward_mean", intrinsic_reward.mean(), explore_step)
                writer.add_scalar("exploration/terminations", terminations.sum(), explore_step)
                writer.add_scalar("exploration/state_coverage", len(knn_estimator.visited_states), explore_step)
                logging.debug(f"SPS: {int(explore_step / (time.time() - start_time))}")
                writer.add_scalar("exploration/SPS", int(explore_step / (time.time() - start_time)), explore_step)
                logging.debug(f"Exploration step: {explore_step}")

        state_dict = {
            "actor": actor,
            "qf1": qf1,
            "qf2": qf2,
            "qf1_target": qf1_target,
            "qf2_target": qf2_target,
            "reward_net": reward_net,
            "q_optimizer": q_optimizer,
            "actor_optimizer": actor_optimizer,
            "reward_optimizer": reward_optimizer
        }
        save_model_all(
            run_name,
            args.total_explore_steps,
            state_dict
        )
        save_replay_buffer(
            run_name,
            args.total_explore_steps,
            rb
        )

    if args.exploration_load:
        load_replay_buffer(
            rb,
            path = args.path_to_replay_buffer
        )
        state_dict = {
            "actor": actor,
            "qf1": qf1,
            "qf2": qf2,
            "qf1_target": qf1_target,
            "qf2_target": qf2_target,
            "reward_net": reward_net,
            "q_optimizer": q_optimizer,
            "actor_optimizer": actor_optimizer,
            "reward_optimizer": reward_optimizer
        }
        load_model_all(
            state_dict,
            path = args.path_to_model,
            device = device
        )

    try:
        obs, _ = envs.reset(seed=args.seed)
        for global_step in range(args.total_timesteps):
            ### REWARD LEARNING ###
            current_step += 1
            # If we pre-train we can start at step 0 with training our rewards
            if global_step % args.teacher_feedback_frequency == 0 and \
                    (global_step != 0 or args.exploration_load or args.unsupervised_exploration):
                for i in range(args.teacher_feedback_num_queries_per_session):
                    # Sample trajectories from replay buffer to query teacher
                    first_trajectory, second_trajectory = rb.sample_trajectories()

                    logging.debug(f"step {global_step}, {i}")

                    # Create video of the two trajectories. For now, we only render if capture_video is True.
                    # If we have a human teacher, we would render the video anyway and ask the teacher to compare the two trajectories.
                    if args.capture_video:
                        video_recorder.record_trajectory(first_trajectory, run_name)
                        video_recorder.record_trajectory(second_trajectory, run_name)

                    # Query instructor (normally a human who decides which trajectory is better, here we use ground truth)
                    preference = sim_teacher.give_preference(first_trajectory, second_trajectory)

                    # Trajectories are not added to the buffer if neither segment demonstrates the desired behavior
                    if preference is None:
                        continue

                    # Store preferences
                    pref_buffer.add(first_trajectory, second_trajectory, preference)

                train_reward(
                    reward_net,
                    reward_optimizer,
                    writer,
                    pref_buffer,
                    rb,
                    global_step,
                    args.teacher_update_epochs,
                    args.teacher_feedback_batch_size,
                )

                rb.relabel_rewards(reward_net)
                logging.debug("Rewards relabeled")

            ### AGENT LEARNING ###

            actions = select_actions(obs, actor, device, global_step, args.learning_starts, envs)

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, groundTruthRewards, terminations, truncations, infos = envs.step(actions)

            # TRY NOT TO MODIFY: record rewards for plotting purposes
            if infos and "final_info" in infos:
                for info in infos["final_info"]:
                    if info:
                        logging.debug(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                        break

            # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
            real_next_obs = next_obs.copy()
            for idx, trunc in enumerate(truncations):
                if trunc:
                    real_next_obs[idx] = infos["final_observation"][idx]

            rewards = reward_net.predict_reward(obs, actions)
            rb.add(obs, real_next_obs, actions, rewards.squeeze(), groundTruthRewards, terminations, infos)

            # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
            obs = next_obs

            # ALGO LOGIC: training.
            if global_step > args.learning_starts:
                data = rb.sample(args.batch_size)
                qf_loss, qf1_a_values, qf2_a_values, qf1_loss, qf2_loss = train_q_network(data,
                                                                                          qf1,
                                                                                          qf2,
                                                                                          qf1_target,
                                                                                          qf2_target,
                                                                                          alpha,
                                                                                          args.gamma,
                                                                                          q_optimizer
                                                                                          )

                if global_step % args.policy_frequency == 0:  # TD 3 Delayed update support
                    for _ in range(args.policy_frequency):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                        actor_loss, alpha, alpha_loss = update_actor(data,
                                                                     actor,
                                                                     qf1,
                                                                     qf2,
                                                                     alpha,
                                                                     actor_optimizer)

                # update the target networks
                if global_step % args.target_network_frequency == 0:
                    update_target_networks(qf1, qf1_target, args.tau)
                    update_target_networks(qf2, qf2_target, args.tau)

                if global_step % 100 == 0:
                    writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                    writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), global_step)
                    writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                    writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                    writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                    writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                    writer.add_scalar("losses/alpha", alpha, global_step)
                    logging.debug(f"SPS: {int(global_step / (time.time() - start_time))}")
                    writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
                    if args.autotune:
                        writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)

    except KeyboardInterrupt:
        logging.warning("KeyboardInterrupt caught! Saving progress...")
    finally:
        state_dict = {
            "actor": actor,
            "qf1": qf1,
            "qf2": qf2,
            "qf1_target": qf1_target,
            "qf2_target": qf2_target,
            "reward_net": reward_net,
            "q_optimizer": q_optimizer,
            "actor_optimizer": actor_optimizer,
            "reward_optimizer": reward_optimizer
        }
        save_model_all(
            run_name,
            current_step,
            state_dict
        )
        save_replay_buffer(
            run_name,
            current_step,
            rb
        )
        envs.close()
        writer.close()

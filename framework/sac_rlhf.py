import logging
import os
import random
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

import gymnasium as gym
import numpy as np
import pygame
import shimmy  # noqa
import torch
import torch.optim as optim
import tyro
from agent.actor import Actor, select_actions, update_actor, update_target_networks
from agent.critic import SoftQNetwork, train_q_network
from environment.env import (
    get_qpos_qvel,
    initialize_qpos_qvel,
    is_dm_control,
    load_model_all,
    load_replay_buffer,
    make_env,
    save_model_all,
    save_replay_buffer,
)
from feedback.feedback import collect_feedback
from feedback.feedback_util import start_feedback_server, stop_feedback_server
from feedback.teacher import Teacher, plot_feedback_schedule, teacher_feedback_schedule
from reward_training.preference_buffer import PreferenceBuffer
from reward_training.replay_buffer import ReplayBuffer
from reward_training.reward_net import RewardNet, train_reward
from reward_training.unsupervised_exploration import ExplorationRewardKNN
from tqdm import tqdm, trange
from utils.evaluation import Evaluation
from utils.performance_metrics import PerformanceMetrics
from utils.video_recorder import VideoRecorder


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
    wandb_project_name: str = ""
    """the wandb's project name"""
    wandb_entity: str = "cleanRLHF"
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    render_mode: str = ""
    """set render_mode to 'human' to watch training (it is not possible to render human and capture videos with the flag '--capture-video')"""
    log_file: bool = True
    """if toggled, logger will write to a file"""
    log_level: str = "INFO"
    """the threshold level for the logger to print a message"""
    play_sounds: bool = False
    """whether to play a alert when feedback is requested"""

    # Algorithm specific arguments
    env_id: str = "Hopper-v5"
    """the environment id of the task"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    buffer_size: int = int(total_timesteps)
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
    """the frequency of updates for the target networks"""
    alpha: float = 0.2
    """Entropy regularization coefficient."""
    autotune: bool = True
    """automatic tuning of the entropy coefficient"""

    ## Evaluation
    evaluation_frequency: int = 10000
    """the frequency of evaluation"""
    evaluation_episodes: int = 10
    """the number of evaluation episodes"""

    ## Early Stop
    early_stopping: bool = False
    """enable early stopping"""
    early_stopping_step: int = 500000
    """the number of steps before early stopping"""
    early_stop_patience: int = 5
    """the number of evaluation before early stopping"""
    early_stopping_mean: float = 900
    """the threshold of early stopping"""
    enable_greater_or_smaller_check: bool = False
    """stop if reward is greater/smaller then threshold (True: greater/ False: smaller)"""

    ## Arguments for the neural networks
    reward_net_hidden_dim: int = 128
    """the dimension of the hidden layers in the reward network"""
    reward_net_hidden_layers: int = 4
    """the number of hidden layers in the reward network"""
    reward_net_val_split: float = 0.2
    """the validation split for the reward network"""
    reward_net_dropout: float = 0.2
    """the dropout rate for the reward network"""
    actor_and_q_net_hidden_dim: int = 256
    """the dimension of the hidden layers in the actor network"""
    actor_and_q_net_hidden_layers: int = 4
    """the number of hidden layers in the actor network"""

    # Feedback server arguments
    feedback_server_url: str = "http://localhost:5001"
    """the url of the feedback server"""
    feedback_server_autostart: bool = False
    """toggle the autostart of a local feedback server"""

    # Teacher feedback mode
    teacher_feedback_mode: str = "simulated"
    """the mode of feedback, must be 'simulated' or 'human' """

    # Human feedback arguments
    teacher_feedback_schedule: str = "exponential"
    """the schedule of teacher feedback, must be 'exponential' or 'linear'"""
    teacher_feedback_total_queries: int = 1400
    """the total number of queries the teacher will provide"""
    teacher_feedback_num_queries_per_session: int = 20
    """the number of queries per feedback session"""
    teacher_feedback_exponential_lambda: float = 0.1
    """the lambda parameter for the exponential feedback schedule"""
    teacher_update_epochs: int = 16
    """the amount of gradient steps to take on the teacher feedback"""
    teacher_batch_strategy: str = "minibatch"
    """the sampling method for teacher training, must be 'minibatch' ,'batch' or 'full'"""
    teacher_minibatch_size: int = 10
    """the mini batch size of the teacher feedback sampled from the feedback buffer"""
    teacher_feedback_batch_size: int = 32
    """the batch size of the teacher feedback sampled from the feedback buffer"""
    teacher_learning_rate: float = 0.00082
    """the learning rate of the teacher"""
    pref_buffer_size_sessions: int = 7
    """the number of sessions to store in the preference buffer"""

    # Simulated Teacher
    trajectory_length: int = 64
    """the length of the trajectories that are sampled for human feedback"""
    preference_sampling: str = "disagree"
    """the sampling method for preferences, must be 'uniform', 'disagree' or 'entropy'"""
    teacher_sim_beta: float = -1
    """this parameter controls how deterministic or random the teacher's preferences are"""
    teacher_sim_gamma: float = 1
    """the discount factor gamma, which models myopic behavior"""
    teacher_sim_epsilon: float = 0
    """with probability epsilon, the teacher's preference is flipped, introducing randomness"""
    teacher_sim_delta_skip: float = -1e7
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

    # SURF
    surf: bool = True
    """Toggle SURF on/off"""
    unlabeled_batch_ratio: int = 1
    """Ratio of unlabeled to labeled batch size"""
    surf_sampling_strategy: str = "uniform"
    """the sampling method for SURF, must be 'uniform', 'disagree' or 'entropy'"""
    surf_tau: float = 0.999
    """Confidence threshold for pseudo-labeling"""
    lambda_ssl: float = 0.1
    """Weight for the unsupervised (pseudo-labeled) loss"""
    max_augmentation_offset: int = 10
    """Max offset for the data augmentation"""
    min_augmentation_offset: int = 5
    """Min offset for the data augmentation"""

    ### RUNE
    rune: bool = True
    """Toggle RUNE on/off"""
    rune_beta: float = 0.05
    """Beta parameter for RUNE"""
    rune_beta_decay: float = 0.0001
    """Beta decay parameter for RUNE"""

    # Load Model
    exploration_load: bool = False
    """skip exploration and load the pre-trained model and buffer from a file"""
    path_to_replay_buffer: str = ""
    """path to replay buffer"""
    path_to_model: str = ""
    """path to model"""


def run(args: Any):
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    if len(args.env_id.split("/")) > 1:
        env_id = args.env_id.split("/")[1]
    else:
        env_id = args.env_id
    run_name = f"{env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    logging.basicConfig(
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%d/%m/%Y %H:%M:%S",
        level=args.log_level.upper(),
    )

    if args.log_file:
        os.makedirs(os.path.join("runs", run_name), exist_ok=True)
        file_path = os.path.join("runs", run_name, "log.log")
        logging.getLogger().addHandler(
            logging.FileHandler(filename=file_path, encoding="utf-8", mode="a"),
        )
    if args.play_sounds:
        pygame.mixer.init()
    try:
        if args.feedback_server_autostart:
            if (
                "localhost" in args.feedback_server_url
                or "127.0.0" in args.feedback_server_url
            ):
                feedback_server_process = start_feedback_server(
                    args.feedback_server_url.split(":")[-1]
                )
            else:
                logging.error("feedback server autostart", args.feedback_server_url)
                raise ValueError(
                    "Feedback server autostart only works with localhost. Please start the feedback server manually."
                )
        else:
            logging.info("Feedback server autostart is disabled.")
        train(args, run_name)
    except Exception as e:
        logging.exception(e)
    finally:
        if args.feedback_server_autostart:
            stop_feedback_server(feedback_server_process)


def train(args: Any, run_name):
    """
    :param args: run arguments
    :param run_name: the name of the run
    :return:
    """
    import stable_baselines3 as sb3

    if sb3.__version__ < "2.0":
        raise ValueError(
            """Ongoing migration: run the following command to install the new dependencies:
poetry run pip install "stable_baselines3==2.0.0a1"
"""
        )

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

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    logging.info(f"Using device: {device}")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [
            make_env(
                args.env_id, args.seed, args.render_mode, args.teacher_feedback_mode
            )
            for i in range(
                1
            )  # number of environments is fixed to one in this implementation
        ]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), (
        "only continuous action space is supported"
    )
    dm_control_bool = is_dm_control(env_id=args.env_id)
    actor = Actor(
        env=envs,
        hidden_dim=args.actor_and_q_net_hidden_dim,
        hidden_layers=args.actor_and_q_net_hidden_layers,
    ).to(device)
    qf1 = SoftQNetwork(
        envs,
        hidden_dim=args.actor_and_q_net_hidden_dim,
        hidden_layers=args.actor_and_q_net_hidden_layers,
    ).to(device)
    qf2 = SoftQNetwork(
        envs,
        hidden_dim=args.actor_and_q_net_hidden_dim,
        hidden_layers=args.actor_and_q_net_hidden_layers,
    ).to(device)
    qf1_target = SoftQNetwork(
        envs,
        hidden_dim=args.actor_and_q_net_hidden_dim,
        hidden_layers=args.actor_and_q_net_hidden_layers,
    ).to(device)
    qf2_target = SoftQNetwork(
        envs,
        hidden_dim=args.actor_and_q_net_hidden_dim,
        hidden_layers=args.actor_and_q_net_hidden_layers,
    ).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(
        list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr
    )
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)

    # Automatic entropy tuning
    if args.autotune:
        target_entropy = -torch.prod(
            torch.Tensor(envs.single_action_space.shape).to(device)
        ).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        alpha = args.alpha

    envs.single_observation_space.dtype = np.float32

    obs, _ = envs.reset(seed=args.seed)

    qpos, qvel = initialize_qpos_qvel(
        envs=envs, num_envs=1, dm_control_bool=dm_control_bool
    )

    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        handle_timeout_termination=False,
        n_envs=1,
        qpos_shape=qpos.shape[1],
        qvel_shape=qvel.shape[1],
        rune=args.rune,
        rune_beta=args.rune_beta,
        rune_beta_decay=args.rune_beta_decay,
        seed=args.seed,
    )
    start_time = time.time()

    train_pref_buffer_size = (
        args.teacher_feedback_num_queries_per_session * args.pref_buffer_size_sessions
    )
    train_pref_buffer = PreferenceBuffer(
        buffer_size=train_pref_buffer_size, seed=args.seed
    )

    val_pref_buffer_size = int(train_pref_buffer_size * args.reward_net_val_split)
    val_pref_buffer = PreferenceBuffer(buffer_size=val_pref_buffer_size, seed=args.seed)

    reward_net = RewardNet(
        hidden_dim=args.reward_net_hidden_dim,
        hidden_layers=args.reward_net_hidden_layers,
        env=envs,
        dropout=args.reward_net_dropout,
    ).to(device)
    reward_optimizer = optim.Adam(
        reward_net.parameters(), lr=args.teacher_learning_rate, weight_decay=1e-4
    )
    video_recorder = VideoRecorder(
        rb,
        args.seed,
        args.env_id,
        dm_control_bool,
        teacher_feedback_mode=args.teacher_feedback_mode,
    )

    # Init Teacher
    sim_teacher = Teacher(
        args.teacher_sim_beta,
        args.teacher_sim_gamma,
        args.teacher_sim_epsilon,
        args.teacher_sim_delta_skip,
        args.teacher_sim_delta_equal,
        args.seed,
    )
    evaluate = Evaluation(
        actor=actor,
        env_id=args.env_id,
        seed=args.seed,
        torch_deterministic=args.torch_deterministic,
        run_name=run_name,
        teacher_feedback_mode=args.teacher_feedback_mode,
    )

    metrics = PerformanceMetrics(run_name, args, evaluate)
    surf_H_max = args.trajectory_length - args.min_augmentation_offset
    surf_H_min = args.trajectory_length - args.max_augmentation_offset

    current_step = 0
    if args.unsupervised_exploration and not args.exploration_load:
        knn_estimator = ExplorationRewardKNN(k=3)
        current_step += 1
        obs, _ = envs.reset(seed=args.seed)
        for explore_step in trange(
            args.total_explore_steps, desc="Exploration step", unit="steps", leave=False
        ):
            actions = select_actions(
                obs, actor, device, explore_step, args.explore_learning_starts, envs
            )

            next_obs, ground_truth_reward, terminations, truncations, infos = envs.step(
                actions
            )
            assert envs.observation_space.contains(next_obs), (
                "Observation is out of bounds!"
            )
            real_next_obs = next_obs.copy()

            get_qpos_qvel(envs, qpos, qvel, dm_control_bool)

            intrinsic_reward = knn_estimator.compute_intrinsic_rewards(next_obs)
            knn_estimator.update_states(next_obs)
            dones = terminations | truncations
            rb.add(
                obs,
                real_next_obs,
                actions,
                intrinsic_reward,
                np.zeros(
                    1
                ),  # number of environments is fixed to one in this implementation, there is no standard deviation during the exploration phase
                ground_truth_reward,
                dones,
                infos,
                current_step,
                qpos,
                qvel,
            )

            obs = next_obs
            if explore_step > args.explore_learning_starts:
                data = rb.sample(args.explore_batch_size)
                (
                    qf_loss,
                    qf1_a_values,
                    qf2_a_values,
                    qf1_loss,
                    qf2_loss,
                ) = train_q_network(
                    data,
                    qf1,
                    qf2,
                    qf1_target,
                    qf2_target,
                    alpha,
                    args.gamma,
                    q_optimizer,
                    actor,
                )

                if (
                    explore_step % args.policy_frequency == 0
                ):  # TD 3 Delayed update support
                    for _ in range(
                        args.policy_frequency
                    ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                        actor_loss, alpha, alpha_loss = update_actor(
                            data,
                            actor,
                            qf1,
                            qf2,
                            alpha,
                            actor_optimizer,
                            args.autotune,
                            log_alpha,
                            target_entropy,
                            a_optimizer,
                        )

            if explore_step % args.target_network_frequency == 0:
                update_target_networks(qf1, qf1_target, args.tau)
                update_target_networks(qf2, qf2_target, args.tau)

            metrics.log_exploration_metrics(
                explore_step,
                intrinsic_reward,
                knn_estimator,
                terminations,
                truncations,
                start_time,
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
            "reward_optimizer": reward_optimizer,
        }
        save_model_all(run_name, args.total_explore_steps, state_dict)
        save_replay_buffer(run_name, args.total_explore_steps, rb)

    if args.exploration_load:
        load_replay_buffer(rb, path=args.path_to_replay_buffer)
        state_dict = {
            "actor": actor,
            "qf1": qf1,
            "qf2": qf2,
            "qf1_target": qf1_target,
            "qf2_target": qf2_target,
            "reward_net": reward_net,
            "q_optimizer": q_optimizer,
            "actor_optimizer": actor_optimizer,
            "reward_optimizer": reward_optimizer,
        }
        load_model_all(state_dict, path=args.path_to_model, device=device)

    try:
        reward_means = deque(maxlen=args.early_stop_patience)
        obs, _ = envs.reset(seed=args.seed)
        total_steps = (
            args.total_timesteps - args.total_explore_steps
            if args.exploration_load or args.unsupervised_exploration
            else args.total_timesteps
        )

        teacher_total_queries = args.teacher_feedback_total_queries
        teacher_num_sessions = (
            teacher_total_queries // args.teacher_feedback_num_queries_per_session
        )
        teacher_exponential_lambda = args.teacher_feedback_exponential_lambda
        teacher_session_steps = teacher_feedback_schedule(
            num_sessions=teacher_num_sessions,
            total_steps=total_steps,
            schedule=args.teacher_feedback_schedule,
            lambda_=teacher_exponential_lambda,
        )

        model_folder = f"./models/{run_name}"
        os.makedirs(model_folder, exist_ok=True)

        sessions_steps_plt = plot_feedback_schedule(
            schedule=teacher_session_steps,
            num_queries=teacher_total_queries,
        )
        sessions_steps_plt.savefig(f"{model_folder}/feedback_schedule.png")

        next_session_idx = 0

        # Initialize the sub progress bar for the first teacher session, if any.
        if teacher_session_steps.any():
            sub_total = int(teacher_session_steps[next_session_idx])
            sub_bar = tqdm(
                total=sub_total,
                desc="Next Feedback Session",
                unit="steps",
                position=1,
                leave=False,
            )
        else:
            sub_bar = None

        for global_step in trange(
            total_steps, desc="Training steps", unit="steps", position=0
        ):
            ### REWARD LEARNING ###
            current_step += 1
            if sub_bar is not None:
                sub_bar.update(1)
            # If we pre-train we can start at step 0 with training our rewards
            if global_step >= teacher_session_steps[next_session_idx] and (
                global_step != 0
                or args.exploration_load
                or args.unsupervised_exploration
            ):
                if sub_bar is not None:
                    sub_bar.close()
                next_session_idx += 1

                if next_session_idx < len(teacher_session_steps):
                    new_total = int(
                        teacher_session_steps[next_session_idx]
                        - teacher_session_steps[next_session_idx - 1]
                    )
                    sub_bar = tqdm(
                        total=new_total,
                        desc="Next Feedback Session",
                        unit="steps",
                        position=1,
                        leave=False,
                    )
                else:
                    sub_bar = None  # No further teacher sessions.

                collect_feedback(
                    mode=args.teacher_feedback_mode,
                    feedback_server_url=args.feedback_server_url,
                    run_name=run_name,
                    preference_sampling=args.preference_sampling,
                    replay_buffer=rb,
                    trajectory_length=args.trajectory_length,
                    reward_net=reward_net,
                    train_pref_buffer=train_pref_buffer,
                    val_pref_buffer=val_pref_buffer,
                    reward_net_val_split=args.reward_net_val_split,
                    teacher_feedback_num_queries_per_session=args.teacher_feedback_num_queries_per_session,
                    capture_video=args.capture_video,
                    render_mode=args.render_mode,
                    video_recorder=video_recorder,
                    sim_teacher=sim_teacher,
                    play_sounds=args.play_sounds,
                )

                logging.debug(f"next_session_idx {next_session_idx}")
                train_reward(
                    model=reward_net,
                    optimizer=reward_optimizer,
                    metrics=metrics,
                    train_pref_buffer=train_pref_buffer,
                    val_pref_buffer=val_pref_buffer,
                    rb=rb,
                    global_step=global_step,
                    epochs=args.teacher_update_epochs,
                    batch_size=args.teacher_feedback_batch_size,
                    mini_batch_size=args.teacher_minibatch_size,
                    batch_sample_strategy=args.teacher_batch_strategy,
                    device=device,
                    surf=args.surf,
                    sampling_strategy=args.surf_sampling_strategy,
                    trajectory_length=args.trajectory_length,
                    unlabeled_batch_ratio=args.unlabeled_batch_ratio,
                    tau=args.surf_tau,
                    lambda_ssl=args.lambda_ssl,
                    H_max=surf_H_max,
                    H_min=surf_H_min,
                )
                rb.relabel_rewards(reward_net)
                logging.debug("Rewards relabeled")

            ### AGENT LEARNING ###

            actions = select_actions(
                obs, actor, device, global_step, args.learning_starts, envs
            )

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, groundTruthRewards, terminations, truncations, infos = envs.step(
                actions
            )

            get_qpos_qvel(envs, qpos, qvel, dm_control_bool)

            if infos and "episode" in infos:
                allocated, reserved = 0, 0
                cuda = False
                if args.cuda and torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated()
                    reserved = torch.cuda.memory_reserved()
                    cuda = True
                metrics.log_info_metrics(infos, global_step, allocated, reserved, cuda)

            # TRY NOT TO MODIFY: save data to reply buffer
            real_next_obs = next_obs.copy()

            dones = terminations | truncations
            with torch.no_grad():
                rewards, rewards_std = reward_net.predict_reward(obs, actions)

            rb.add(
                obs,
                real_next_obs,
                actions,
                rewards.squeeze(),
                rewards_std.squeeze(),
                groundTruthRewards,
                dones,
                infos,
                global_step,
                qpos,
                qvel,
            )

            # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
            obs = next_obs

            # ALGO LOGIC: training.
            if global_step > args.learning_starts:
                data = rb.sample(args.batch_size)
                (
                    qf_loss,
                    qf1_a_values,
                    qf2_a_values,
                    qf1_loss,
                    qf2_loss,
                ) = train_q_network(
                    data,
                    qf1,
                    qf2,
                    qf1_target,
                    qf2_target,
                    alpha,
                    args.gamma,
                    q_optimizer,
                    actor,
                )

                if (
                    global_step % args.policy_frequency == 0
                ):  # TD 3 Delayed update support
                    for _ in range(
                        args.policy_frequency
                    ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                        actor_loss, alpha, alpha_loss = update_actor(
                            data,
                            actor,
                            qf1,
                            qf2,
                            alpha,
                            actor_optimizer,
                            args.autotune,
                            log_alpha,
                            target_entropy,
                            a_optimizer,
                        )

                # update the target networks
                if global_step % args.target_network_frequency == 0:
                    update_target_networks(qf1, qf1_target, args.tau)
                    update_target_networks(qf2, qf2_target, args.tau)

                metrics.add_rewards(rewards.flatten(), groundTruthRewards)
                metrics.log_reward_metrics(rewards, groundTruthRewards, global_step)
                if global_step % 100 == 0:
                    metrics.log_training_metrics(
                        global_step,
                        args,
                        qf1_a_values,
                        qf2_a_values,
                        qf1_loss,
                        qf2_loss,
                        qf_loss,
                        actor_loss,
                        alpha,
                        alpha_loss,
                        start_time,
                        pearson=True,
                    )
            if global_step % args.evaluation_frequency == 0 and (
                global_step != 0
                or args.exploration_load
                or args.unsupervised_exploration
            ):
                render = (
                    global_step % 100000 == 0
                    and global_step != 0
                    and args.render_mode != "human"
                )
                track = (
                    global_step % 100000 == 0
                    and global_step != 0
                    and args.track
                    and args.render_mode != "human"
                )
                eval_dict = evaluate.evaluate_policy(
                    episodes=args.evaluation_episodes,
                    step=global_step,
                    actor=actor,
                    render=render,
                    track=track,
                )
                reward_means.append(eval_dict["mean_reward"])
                evaluate.plot(eval_dict, global_step)
                metrics.log_evaluate_metrics(global_step, eval_dict)
            if (
                global_step > args.early_stopping_step == 0
                and (
                    (
                        np.mean(reward_means) > args.early_stopping_mean
                        and args.enable_greater_or_smaller_check
                    )
                    or (
                        np.mean(reward_means) < args.early_stopping_mean
                        and not args.enable_greater_or_smaller_check
                    )
                )
                and args.early_stopping
            ):
                break
        eval_dict = evaluate.evaluate_policy(
            episodes=args.evaluation_episodes,
            step=args.total_timesteps,
            actor=actor,
            render=True,
            track=args.track,
        )
        evaluate.plot(eval_dict, args.total_timesteps)
        metrics.log_evaluate_metrics(args.total_timesteps, eval_dict)

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
            "reward_optimizer": reward_optimizer,
        }
        save_model_all(run_name, current_step, state_dict)
        save_replay_buffer(run_name, current_step, rb)
        envs.close()
        metrics.close()
        if args.play_sounds:
            pygame.mixer.quit()


if __name__ == "__main__":
    cli_args = tyro.cli(Args)
    run(cli_args)

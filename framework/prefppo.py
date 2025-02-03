import logging
import os
import random
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import torch.optim as optim
import tyro
from agent import Agent
from env import (
    get_qpos_qvel,
    initialize_qpos_qvel,
    is_dm_control,
    load_model_all,
    load_replay_buffer,
    make_env_ppo,
    save_model_all,
    save_replay_buffer,
)
from evaluation import Evaluation
from feedback import collect_feedback
from feedback_util import start_feedback_server, stop_feedback_server
from performance_metrics import PerformanceMetrics
from preference_buffer import PreferenceBuffer
from replay_buffer import RolloutBuffer
from reward_net import RewardNet, train_reward
from teacher import Teacher, plot_feedback_schedule, teacher_feedback_schedule
from tqdm import trange
from unsupervised_exploration import ExplorationRewardKNN
from video_recorder import VideoRecorder


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
    num_envs: int = 1
    """the number of parallel game environments"""
    log_file: bool = True
    """if toggled, logger will write to a file"""
    log_level: str = "INFO"
    """the threshold level for the logger to print a message"""

    # Algorithm specific arguments
    env_id: str = "Hopper-v5"
    """the id of the environment"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    num_steps: int = 2048
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 32
    """the number of mini-batches"""
    update_epochs: int = 10
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.2
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

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
    agent_net_hidden_dim: int = 128
    """the dimension of the hidden layers in the actor network"""
    agent_net_hidden_layers: int = 4
    """the number of hidden layers in the actor network"""

    # Feedback server arguments
    feedback_server_url: str = "http://localhost:5001"
    """the url of the feedback server"""
    feedback_server_autostart: bool = False
    """toggle the autostart of a local feedback server"""

    # Teacher feedback mode
    teacher_feedback_mode: str = "simulated"
    """the mode of feedback, must be 'simulated', 'human' or 'file'"""  # file is currently not supported

    # Human feedback arguments
    teacher_feedback_schedule: str = "linear"
    """the schedule of teacher feedback, must be 'exponential' or 'linear'"""
    teacher_feedback_total_queries: int = 5000
    """the total number of queries the teacher will provide"""
    teacher_feedback_num_queries_per_session: int = 40
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

    # SURF
    surf: bool = False
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
    rune: bool = False
    """Toggle RUNE on/off"""
    rune_beta: float = 100
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

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""
    num_iterations_exploration: int = 0
    """the number of exploration iterations (computed in runtime)"""
    buffer_size: int = 0
    """the replay memory buffer size (computed in runtime)"""


def run(args: Any):
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
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


def train(args: Any, run_name: str):
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = (
        args.total_timesteps - args.total_explore_steps
    ) // args.batch_size
    args.num_iterations_exploration = args.total_explore_steps // args.batch_size
    args.buffer_size = int(args.num_steps)

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
            make_env_ppo(
                args.env_id, args.gamma, args.seed, i, render_mode=args.render_mode
            )
            for i in range(args.num_envs)
        ]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), (
        "only continuous action space is supported"
    )
    envs.single_observation_space.dtype = np.float32
    dm_control_bool = is_dm_control(env_id=args.env_id)

    agent = Agent(
        envs, hidden_dim=args.agent_net_hidden_dim, layers=args.agent_net_hidden_layers
    ).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    global_step = 0
    start_time = time.time()
    obs, _ = envs.reset(seed=args.seed)

    qpos, qvel = initialize_qpos_qvel(
        envs=envs, num_envs=args.num_envs, dm_control_bool=dm_control_bool
    )

    rb = RolloutBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        handle_timeout_termination=False,
        n_envs=args.num_envs,
        qpos_shape=qpos.shape[1],
        qvel_shape=qvel.shape[1],
        rune=args.rune,
        rune_beta=args.rune_beta,
        rune_beta_decay=args.rune_beta_decay,
        seed=args.seed,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
    )

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
    video_recorder = VideoRecorder(rb, args.seed, args.env_id, dm_control_bool)

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
        actor=agent,
        env_id=args.env_id,
        seed=args.seed,
        torch_deterministic=args.torch_deterministic,
        run_name=run_name,
    )
    metrics = PerformanceMetrics(run_name, args, evaluate)
    surf_H_max = args.trajectory_length - args.min_augmentation_offset
    surf_H_min = args.trajectory_length - args.max_augmentation_offset

    current_step = 0
    if args.unsupervised_exploration and not args.exploration_load:
        knn_estimator = ExplorationRewardKNN(k=3)
        current_step += 1
        obs, _ = envs.reset(seed=args.seed)
        for iteration in trange(1, args.num_iterations_exploration + 1):
            if args.anneal_lr:
                frac = 1.0 - (iteration - 1.0) / args.num_iterations
                lrnow = frac * args.learning_rate
                optimizer.param_groups[0]["lr"] = lrnow

            for explore_step in range(0, args.num_steps):
                current_step += 1
                global_step += args.num_envs

                action, logprob, value = agent.select_action(obs, device)
                (
                    next_obs,
                    groundTruthRewards,
                    terminations,
                    truncations,
                    infos,
                ) = envs.step(action)

                get_qpos_qvel(envs, qpos, qvel, dm_control_bool)
                intrinsic_reward = knn_estimator.compute_intrinsic_rewards(next_obs)
                knn_estimator.update_states(next_obs)
                done = np.logical_or(terminations, truncations)
                rb.add(
                    obs=obs,
                    next_obs=next_obs,
                    action=action,
                    extrinsic_reward=intrinsic_reward,
                    intrinsic_reward=np.zeros(
                        args.num_envs
                    ),  # There is no standard deviation during the exploration phase
                    ground_truth_reward=groundTruthRewards,
                    done=done,
                    infos=infos,
                    global_step=global_step,
                    qpos=qpos,
                    qvel=qvel,
                    values=value,
                    logprobs=logprob,
                )
                obs = next_obs

                metrics.log_exploration_metrics(
                    explore_step,
                    intrinsic_reward,
                    knn_estimator,
                    terminations,
                    truncations,
                    start_time,
                )

            rb.compute_gae_and_returns(agent, obs, done)

            ppo_dict = agent.train_agent(rb, args, optimizer)

            metrics.log_training_metrics_ppo(
                global_step,
                optimizer,
                ppo_dict,
                start_time,
            )

    if args.exploration_load:
        load_replay_buffer(rb, path=args.path_to_replay_buffer)
        state_dict = {
            "agent": agent,
            "reward_net": reward_net,
            "optimizer": optimizer,
            "reward_optimizer": reward_optimizer,
        }
        load_model_all(state_dict, path=args.path_to_model, device=device)

    reward_means = deque(maxlen=3)
    try:
        total_steps = (
            args.total_timesteps - args.total_explore_steps
            if args.exploration_load
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

        for iteration in trange(1, args.num_iterations + 1):
            if args.anneal_lr:
                frac = 1.0 - (iteration - 1.0) / args.num_iterations
                lrnow = frac * args.learning_rate
                optimizer.param_groups[0]["lr"] = lrnow

            for step in range(0, args.num_steps):
                if global_step >= teacher_session_steps[next_session_idx] and (
                    global_step != 0 or args.exploration_load
                ):
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
                    )

                    next_session_idx += 1

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

                current_step += 1
                global_step += args.num_envs

                ## Agent steps
                action, logprob, value = agent.select_action(obs, device)

                with torch.no_grad():
                    rewards, rewards_std = reward_net.predict_reward(obs, action)

                (
                    next_obs,
                    groundTruthRewards,
                    terminations,
                    truncations,
                    infos,
                ) = envs.step(action)

                get_qpos_qvel(envs, qpos, qvel, dm_control_bool)
                done = np.logical_or(terminations, truncations)
                rb.add(
                    obs=obs,
                    next_obs=next_obs,
                    action=action,
                    extrinsic_reward=rewards.squeeze(),
                    intrinsic_reward=rewards_std.squeeze(),
                    ground_truth_reward=groundTruthRewards,
                    done=done,
                    infos=infos,
                    global_step=global_step,
                    qpos=qpos,
                    qvel=qvel,
                    values=value,
                    logprobs=logprob,
                )
                obs = next_obs
                if infos and "episode" in infos:
                    allocated, reserved = 0, 0
                    cuda = False
                    if args.cuda and torch.cuda.is_available():
                        allocated = torch.cuda.memory_allocated()
                        reserved = torch.cuda.memory_reserved()
                        cuda = True
                    metrics.log_info_metrics(
                        infos, global_step, allocated, reserved, cuda
                    )

                if global_step % args.evaluation_frequency == 0 and (
                    global_step != 0 or args.exploration_load
                ):
                    render = global_step % 100000 == 0 and global_step != 0
                    track = (
                        global_step % 100000 == 0 and global_step != 0 and args.track
                    )
                    eval_dict = evaluate.evaluate_policy(
                        episodes=args.evaluation_episodes,
                        step=global_step,
                        actor=agent,
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

            rb.compute_gae_and_returns(agent, obs, done)

            ppo_dict = agent.train_agent(rb, args, optimizer)

            metrics.log_training_metrics_ppo(
                global_step,
                optimizer,
                ppo_dict,
                start_time,
            )

        eval_dict = evaluate.evaluate_policy(
            episodes=args.evaluation_episodes,
            step=args.total_timesteps,
            actor=agent,
            render=True,
            track=args.track,
        )
        evaluate.plot(eval_dict, args.total_timesteps)
        metrics.log_evaluate_metrics(args.total_timesteps, eval_dict)
    except KeyboardInterrupt:
        logging.warning("KeyboardInterrupt caught! Saving progress...")
    finally:
        state_dict = {
            "agent": agent,
            "reward_net": reward_net,
            "optimizer": optimizer,
            "reward_optimizer": reward_optimizer,
        }
        save_model_all(run_name, current_step, state_dict)
        save_replay_buffer(run_name, current_step, rb)
        envs.close()
        metrics.close()


if __name__ == "__main__":
    cli_args = tyro.cli(Args)
    run(cli_args)

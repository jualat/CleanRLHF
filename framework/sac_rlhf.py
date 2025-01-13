import logging
import os
import random
import time
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import torch.optim as optim
import tyro
from actor import Actor, select_actions, update_actor, update_target_networks
from critic import SoftQNetwork, train_q_network
from env import (
    is_mujoco_env,
    load_model_all,
    load_replay_buffer,
    make_env,
    save_model_all,
    save_replay_buffer,
)
from evaluation import Evaluation
from performance_metrics import PerformanceMetrics
from preference_buffer import PreferenceBuffer
from replay_buffer import ReplayBuffer
from reward_net import RewardNet, train_reward
from sampling import disagreement_sampling, entropy_sampling, uniform_sampling
from teacher import Teacher
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
    num_envs: int = 1
    """the number of parallel environments to accelerate training.
    Set this to the number of available CPU threads for best performance."""
    log_file: bool = True
    """if toggled, logger will write to a file"""
    log_level: str = "INFO"
    """the threshold level for the logger to print a message"""

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
    """the frequency of updates for the target nerworks"""
    alpha: float = 0.2
    """Entropy regularization coefficient."""
    autotune: bool = True
    """automatic tuning of the entropy coefficient"""

    ## Evaluation
    evaluation_frequency: int = 10000
    """the frequency of evaluation"""
    evaluation_episodes: int = 30
    """the number of evaluation episodes"""

    ## Early Stop
    early_stopping: bool = True
    """enable early stopping"""
    early_stopping_step: int = 500000
    """the number of steps before early stopping"""
    early_stopping_mean: float = 900
    """the threshold of early stopping"""
    enable_greater_or_smaller_check: bool = False
    """stop if reward is greater/smaller then threshold (True: greater/ False: smaller)"""

    ## Arguments for the neural networks
    reward_net_hidden_dim: int = 256
    """the dimension of the hidden layers in the reward network"""
    reward_net_hidden_layers: int = 4
    """the number of hidden layers in the reward network"""
    actor_net_hidden_dim: int = 256
    """the dimension of the hidden layers in the actor network"""
    actor_net_hidden_layers: int = 4
    """the number of hidden layers in the actor network"""
    soft_q_net_hidden_dim: int = 256
    """the dimension of the hidden layers in the SoftQNetwork"""
    soft_q_net_hidden_layers: int = 4
    """the number of hidden layers in the SoftQNetwork"""

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
    trajectory_length: int = 32
    """the length of the trajectories that are sampled for human feedback"""
    preference_sampling: str = "disagree"
    """the sampling method for preferences, must be 'uniform', 'disagree' or 'entropy'"""
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


def train(args: Any):
    """
    The training function.
    :param args: run arguments
    :return:
    """
    import stable_baselines3 as sb3

    if sb3.__version__ < "2.0":
        raise ValueError(
            """Ongoing migration: run the following command to install the new dependencies:
poetry run pip install "stable_baselines3==2.0.0a1"
"""
        )

    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    logging.basicConfig(
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%d/%m/%Y %H:%M:%S",
        level=args.log_level.upper(),
    )
    if args.log_file:
        os.makedirs(os.path.join("runs", run_name), exist_ok=True)
        logging.getLogger().addHandler(
            logging.FileHandler(filename=f"runs/{run_name}/logger.log")
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

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    logging.info(f"Using device: {device}")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed) for i in range(args.num_envs)]
    )
    assert isinstance(
        envs.single_action_space, gym.spaces.Box
    ), "only continuous action space is supported"

    actor = Actor(
        env=envs,
        hidden_dim=args.actor_net_hidden_dim,
        hidden_layers=args.actor_net_hidden_layers,
    ).to(device)
    qf1 = SoftQNetwork(
        envs,
        hidden_dim=args.soft_q_net_hidden_dim,
        hidden_layers=args.soft_q_net_hidden_layers,
    ).to(device)
    qf2 = SoftQNetwork(
        envs,
        hidden_dim=args.soft_q_net_hidden_dim,
        hidden_layers=args.soft_q_net_hidden_layers,
    ).to(device)
    qf1_target = SoftQNetwork(
        envs,
        hidden_dim=args.soft_q_net_hidden_dim,
        hidden_layers=args.soft_q_net_hidden_layers,
    ).to(device)
    qf2_target = SoftQNetwork(
        envs,
        hidden_dim=args.soft_q_net_hidden_dim,
        hidden_layers=args.soft_q_net_hidden_layers,
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

    if is_mujoco_env(envs.envs[0]):
        try:
            qpos = np.zeros(
                (args.num_envs, envs.envs[0].unwrapped.observation_structure["qpos"]),
                dtype=np.float32,
            )
            qvel = np.zeros(
                (args.num_envs, envs.envs[0].unwrapped.observation_structure["qvel"]),
                dtype=np.float32,
            )
        except AttributeError:
            qpos = np.zeros(
                (args.num_envs, envs.envs[0].unwrapped.model.key_qpos.shape[1]),
                dtype=np.float32,
            )
            qvel = np.zeros(
                (args.num_envs, envs.envs[0].unwrapped.model.key_qvel.shape[1]),
                dtype=np.float32,
            )
    else:
        qpos = np.zeros((2, 2))
        qvel = np.zeros((2, 2))

    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        handle_timeout_termination=False,
        n_envs=1,
        qpos_shape=qpos.shape[1],
        qvel_shape=qvel.shape[1],
        seed=args.seed,
    )
    start_time = time.time()

    pref_buffer = PreferenceBuffer(
        (args.buffer_size // args.teacher_feedback_frequency)
        * args.teacher_feedback_num_queries_per_session
    )
    reward_net = RewardNet(
        hidden_dim=args.reward_net_hidden_dim,
        hidden_layers=args.reward_net_hidden_layers,
        env=envs,
    ).to(device)
    reward_optimizer = optim.Adam(
        reward_net.parameters(), lr=args.teacher_learning_rate, weight_decay=1e-4
    )
    video_recorder = VideoRecorder(rb, args.seed, args.env_id)

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
    )

    metrics = PerformanceMetrics(run_name, args, evaluate)

    current_step = 0
    if args.unsupervised_exploration and not args.exploration_load:
        knn_estimator = ExplorationRewardKNN(k=3)
        current_step += 1
        obs, _ = envs.reset(seed=args.seed)
        for explore_step in trange(
            args.total_explore_steps, desc="Exploration step", unit="steps"
        ):
            actions = select_actions(
                obs, actor, device, explore_step, args.explore_learning_starts, envs
            )

            next_obs, ground_truth_reward, terminations, truncations, infos = envs.step(
                actions
            )

            real_next_obs = next_obs.copy()

            if is_mujoco_env(envs.envs[0]):

                try:
                    skipped_qpos = envs.envs[0].unwrapped.observation_structure[
                        "skipped_qpos"
                    ]
                except (KeyError, AttributeError):
                    skipped_qpos = 0

                for idx in range(args.num_envs):
                    single_env = envs.envs[idx]
                    qpos[idx] = (
                        single_env.unwrapped.data.qpos[:-skipped_qpos].copy()
                        if skipped_qpos > 0
                        else single_env.unwrapped.data.qpos.copy()
                    )
                    qvel[idx] = single_env.unwrapped.data.qvel.copy()

            intrinsic_reward = knn_estimator.compute_intrinsic_rewards(next_obs)
            knn_estimator.update_states(next_obs)
            dones = terminations | truncations
            rb.add(
                obs,
                real_next_obs,
                actions,
                intrinsic_reward,
                ground_truth_reward,
                dones,
                infos,
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
        reward_means = np.zeros(3)
        obs, _ = envs.reset(seed=args.seed)
        total_steps = (
            args.total_timesteps - args.total_explore_steps
            if args.exploration_load or args.unsupervised_exploration
            else args.total_timesteps
        )
        for global_step in trange(total_steps, desc="Training steps", unit="steps"):
            ### REWARD LEARNING ###
            current_step += 1
            # If we pre-train we can start at step 0 with training our rewards
            if global_step % args.teacher_feedback_frequency == 0 and (
                global_step != 0
                or args.exploration_load
                or args.unsupervised_exploration
            ):
                for i in trange(
                    args.teacher_feedback_num_queries_per_session,
                    desc="Queries",
                    unit="queries",
                ):
                    # Sample trajectories from replay buffer to query teacher
                    if args.preference_sampling == "uniform":
                        first_trajectory, second_trajectory = uniform_sampling(
                            rb, args.trajectory_length
                        )
                    elif args.preference_sampling == "disagree":
                        first_trajectory, second_trajectory = disagreement_sampling(
                            rb, reward_net, args.trajectory_length
                        )
                    elif args.preference_sampling == "entropy":
                        first_trajectory, second_trajectory = entropy_sampling(
                            rb, reward_net, args.trajectory_length
                        )

                    logging.debug(f"step {global_step}, {i}")

                    # Create video of the two trajectories. For now, we only render if capture_video is True.
                    # If we have a human teacher, we would render the video anyway and ask the teacher to compare the two trajectories.
                    if args.capture_video:
                        video_recorder.record_trajectory(first_trajectory, run_name)
                        video_recorder.record_trajectory(second_trajectory, run_name)

                    # Query instructor (normally a human who decides which trajectory is better, here we use ground truth)
                    preference = sim_teacher.give_preference(
                        first_trajectory, second_trajectory
                    )

                    # Trajectories are not added to the buffer if neither segment demonstrates the desired behavior
                    if preference is None:
                        continue

                    # Store preferences
                    pref_buffer.add(first_trajectory, second_trajectory, preference)

                train_reward(
                    reward_net,
                    reward_optimizer,
                    metrics,
                    pref_buffer,
                    rb,
                    global_step,
                    args.teacher_update_epochs,
                    args.teacher_feedback_batch_size,
                    device,
                )

                rb.relabel_rewards(reward_net)
                logging.info("Rewards relabeled")

            ### AGENT LEARNING ###

            actions = select_actions(
                obs, actor, device, global_step, args.learning_starts, envs
            )

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, groundTruthRewards, terminations, truncations, infos = envs.step(
                actions
            )
            if is_mujoco_env(envs.envs[0]):

                try:
                    skipped_qpos = envs.envs[0].unwrapped.observation_structure[
                        "skipped_qpos"
                    ]
                except (KeyError, AttributeError):
                    skipped_qpos = 0

                for idx in range(args.num_envs):
                    single_env = envs.envs[idx]
                    qpos[idx] = (
                        single_env.unwrapped.data.qpos[:-skipped_qpos].copy()
                        if skipped_qpos > 0
                        else single_env.unwrapped.data.qpos.copy()
                    )
                    qvel[idx] = single_env.unwrapped.data.qvel.copy()

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
                rewards = reward_net.predict_reward(obs, actions)

            env_idx = 0
            single_env_info = {}
            for key, value in infos.items():
                if key != "episode":
                    single_env_info[key] = value[env_idx]

            rb.add(
                obs[env_idx : env_idx + 1],
                real_next_obs[env_idx : env_idx + 1],
                actions[env_idx : env_idx + 1],
                rewards[env_idx : env_idx + 1].squeeze(),
                groundTruthRewards[env_idx : env_idx + 1],
                dones[env_idx : env_idx + 1],
                single_env_info,
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
                    )
            if global_step % args.evaluation_frequency == 0 and (
                global_step != 0
                or args.exploration_load
                or args.unsupervised_exploration
            ):
                render = global_step % 100000 == 0 and global_step != 0
                track = global_step % 100000 == 0 and global_step != 0 and args.track
                eval_dict = evaluate.evaluate_policy(
                    episodes=args.evaluation_episodes,
                    step=global_step,
                    actor=actor,
                    render=render,
                    track=track,
                )
                reward_means[global_step % 3] = eval_dict["mean_reward"]
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


if __name__ == "__main__":
    cli_args = tyro.cli(Args)
    train(cli_args)

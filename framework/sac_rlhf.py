# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
import os
import random
import time
from dataclasses import dataclass
import logging
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import torch.optim as optim
import tyro

from tqdm import trange
from performance_metrics import PerformanceMetrics
from video_recorder import VideoRecorder
from unsupervised_exploration import ExplorationRewardKNN
from preference_buffer import PreferenceBuffer
from replay_buffer import ReplayBuffer
from reward_net import RewardNet, train_reward
from torch.utils.tensorboard import SummaryWriter
from teacher import Teacher
from sampling import uniform_sampling, disagreement_sampling, entropy_sampling
from actor import Actor, select_actions, update_actor, update_target_networks
from env import (
    make_env,
    save_replay_buffer,
    save_model_all,
    load_replay_buffer,
    load_model_all,
    is_mujoco_env,
)
from critic import SoftQNetwork, train_q_network
from evaluation import Evaluation


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
    env_id: str = "Hopper-v4"
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
    soft_q_net_hidden_layers: int = 2
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
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
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

    max_action = float(envs.single_action_space.high[0])

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
    metrics = PerformanceMetrics()

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
        qpos = np.zeros((args.num_envs, envs.envs[0].model.nq), dtype=np.float32)
        qvel = np.zeros((args.num_envs, envs.envs[0].model.nv), dtype=np.float32)
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
            for idx, trunc in enumerate(truncations):
                if trunc:
                    real_next_obs[idx] = infos["final_observation"][idx]
            if is_mujoco_env(envs.envs[0]):
                for idx in range(args.num_envs):
                    single_env = envs.envs[idx]
                    qpos[idx] = single_env.unwrapped.data.qpos.copy()
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

            if explore_step % 100 == 0:
                writer.add_scalar(
                    "exploration/intrinsic_reward_mean",
                    intrinsic_reward.mean(),
                    explore_step,
                )

                writer.add_scalar(
                    "exploration/state_coverage",
                    len(knn_estimator.visited_states),
                    explore_step,
                )
                logging.debug(f"SPS: {int(explore_step / (time.time() - start_time))}")
                writer.add_scalar(
                    "exploration/SPS",
                    int(explore_step / (time.time() - start_time)),
                    explore_step,
                )
                logging.debug(f"Exploration step: {explore_step}")

            writer.add_scalar(
                "exploration/dones",
                terminations.sum() + truncations.sum(),
                explore_step,
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
                    writer,
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
                for idx in range(args.num_envs):
                    single_env = envs.envs[idx]
                    qpos[idx] = single_env.unwrapped.data.qpos.copy()
                    qvel[idx] = single_env.unwrapped.data.qvel.copy()
            # TRY NOT TO MODIFY: record rewards for plotting purposes
            if infos and "final_info" in infos:
                for info in infos["final_info"]:
                    if info:
                        logging.info(
                            f"global_step={global_step}, episodic_return={info['episode']['r']}"
                        )
                        writer.add_scalar(
                            "charts/episodic_return", info["episode"]["r"], global_step
                        )
                        writer.add_scalar(
                            "charts/episodic_length", info["episode"]["l"], global_step
                        )

                        if args.cuda and torch.cuda.is_available():
                            allocated = torch.cuda.memory_allocated()
                            reserved = torch.cuda.memory_reserved()
                            logging.info(
                                f"Allocated cuda memory: {allocated / (1024 ** 2)} MB"
                            )
                            logging.info(
                                f"Reserved cuda memory: {reserved / (1024 ** 2)} MB"
                            )
                            writer.add_scalar(
                                "hardware/cuda_memory",
                                allocated / (1024**2),
                                global_step,
                            )
                        break

            # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
            real_next_obs = next_obs.copy()
            for idx, trunc in enumerate(truncations):
                if trunc:
                    real_next_obs[idx] = infos["final_observation"][idx]
            dones = terminations | truncations
            with torch.no_grad():
                rewards = reward_net.predict_reward(obs, actions)
            writer.add_scalar(
                "charts/predicted_rewards",
                rewards[0],
                global_step,
            )
            writer.add_scalar(
                "charts/groudTruthRewards",
                groundTruthRewards[0],
                global_step,
            )
            env_idx = 0
            single_env_info = {key: value[env_idx] for key, value in infos.items()}
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

                if global_step % 100 == 0:
                    writer.add_scalar(
                        "losses/qf1_values", qf1_a_values.mean().item(), global_step
                    )
                    writer.add_scalar(
                        "losses/qf2_values", qf2_a_values.mean().item(), global_step
                    )
                    writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                    writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                    writer.add_scalar(
                        "losses/qf_loss", qf_loss.item() / 2.0, global_step
                    )
                    writer.add_scalar(
                        "losses/actor_loss", actor_loss.item(), global_step
                    )
                    writer.add_scalar("losses/alpha", alpha, global_step)
                    logging.debug(
                        f"SPS: {int(global_step / (time.time() - start_time))}"
                    )
                    writer.add_scalar(
                        "charts/SPS",
                        int(global_step / (time.time() - start_time)),
                        global_step,
                    )
                    if args.autotune:
                        writer.add_scalar(
                            "losses/alpha_loss", alpha_loss.item(), global_step
                        )
                    writer.add_scalar(
                        "charts/pearson_correlation",
                        metrics.compute_pearson_correlation(),
                        global_step,
                    )
                    metrics.reset()
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
                evaluate.plot(eval_dict, global_step)
                writer.add_scalar(
                    "evaluate/mean", eval_dict["mean_reward"], global_step
                )
                writer.add_scalar("evaluate/std", eval_dict["std_reward"], global_step)
                writer.add_scalar("evaluate/max", eval_dict["max_reward"], global_step)
                writer.add_scalar("evaluate/min", eval_dict["min_reward"], global_step)
                writer.add_scalar(
                    "evaluate/median", eval_dict["median_reward"], global_step
                )
        eval_dict = evaluate.evaluate_policy(
            episodes=args.evaluation_episodes,
            step=args.total_timesteps,
            actor=actor,
            render=True,
            track=args.track,
        )
        evaluate.plot(eval_dict, args.total_timesteps)
        writer.add_scalar(
            "evaluate/mean", eval_dict["mean_reward"], args.total_timesteps
        )
        writer.add_scalar("evaluate/std", eval_dict["std_reward"], args.total_timesteps)
        writer.add_scalar("evaluate/max", eval_dict["max_reward"], args.total_timesteps)
        writer.add_scalar("evaluate/min", eval_dict["min_reward"], args.total_timesteps)
        writer.add_scalar(
            "evaluate/median", eval_dict["median_reward"], args.total_timesteps
        )

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
        writer.close()


if __name__ == "__main__":
    cli_args = tyro.cli(Args)
    train(cli_args)

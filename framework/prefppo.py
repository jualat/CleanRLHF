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
import torch.nn as nn
import torch.optim as optim
import tyro
from agent import Agent
from env import (
    is_mujoco_env,
    load_model_all,
    load_replay_buffer,
    make_env_ppo,
    save_model_all,
    save_replay_buffer,
)
from evaluation import Evaluation
from performance_metrics import PerformanceMetrics
from preference_buffer import PreferenceBuffer
from replay_buffer import ReplayBuffer
from reward_net import RewardNet, train_reward
from sampling import sample_trajectories
from teacher import Teacher
from tqdm import trange
from unsupervised_exploration import ExplorationRewardKNN
from video_recorder import VideoRecorder


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 2
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "HalfCheetah-PPO"
    """the wandb's project name"""
    wandb_entity: str = "cleanRLHF"
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    log_file: bool = True
    """if toggled, logger will write to a file"""
    log_level: str = "INFO"
    """the threshold level for the logger to print a message"""

    # Algorithm specific arguments
    env_id: str = "HalfCheetah-v5"
    """the id of the environment"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    buffer_size: int = int(total_timesteps)
    """the replay memory buffer size"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
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
    early_stopping_mean: float = 900
    """the threshold of early stopping"""
    enable_greater_or_smaller_check: bool = False
    """stop if reward is greater/smaller then threshold (True: greater/ False: smaller)"""

    ## Arguments for the neural networks
    reward_net_hidden_dim: int = 128
    """the dimension of the hidden layers in the reward network"""
    reward_net_hidden_layers: int = 4
    """the number of hidden layers in the reward network"""
    agent_net_hidden_dim: int = 128
    """the dimension of the hidden layers in the actor network"""
    agent_net_hidden_layers: int = 4
    """the number of hidden layers in the actor network"""

    # Human feedback arguments
    teacher_feedback_frequency: int = 5000
    """the frequency of teacher feedback (every K iterations)"""
    teacher_feedback_num_queries_per_session: int = 50
    """the number of queries per feedback session"""
    teacher_update_epochs: int = 16
    """the amount of gradient steps to take on the teacher feedback"""
    teacher_feedback_batch_size: int = 32
    """the batch size of the teacher feedback sampled from the feedback buffer"""
    teacher_learning_rate: float = 0.00082
    """the learning rate of the teacher"""

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


def train(args: Any):
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    args.num_iterations_exploration = args.total_explore_steps // args.batch_size
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

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    logging.info(f"Using device: {device}")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env_ppo(args.env_id, args.gamma) for _ in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), (
        "only continuous action space is supported"
    )
    # TODO ADAPT TO DM CONTROL ENVS
    agent = Agent(
        envs, hidden_dim=args.agent_net_hidden_dim, layers=args.agent_net_hidden_layers
    ).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # TODO: WRITE CUSTOM REPLAY BUFFER FOR THIS
    obs = torch.zeros(
        (args.num_steps, args.num_envs) + envs.single_observation_space.shape
    ).to(device)
    actions = torch.zeros(
        (args.num_steps, args.num_envs) + envs.single_action_space.shape
    ).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    # TODO ADAPT TO NEW VIDEO REC
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
    # TODO: ADAPT TO REPLAY BUFFER CHANGES
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
    # TODO: ADAPT TO NEW PREFERENCE BUFFER
    pref_buffer = PreferenceBuffer(
        (args.buffer_size // args.teacher_feedback_frequency)
        * args.teacher_feedback_num_queries_per_session,
        seed=args.seed,
    )
    # TODO: ADAPT TO NEW REWARD NET
    reward_net = RewardNet(
        hidden_dim=args.reward_net_hidden_dim,
        hidden_layers=args.reward_net_hidden_layers,
        env=envs,
        dropout=args.reward_net_dropout,
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
        actor=agent,
        env_id=args.env_id,
        seed=args.seed,
        torch_deterministic=args.torch_deterministic,
        run_name=run_name,
    )

    metrics = PerformanceMetrics(run_name, args, evaluate)
    current_step = 0
    if args.unsupervised_exploration and not args.exploration_load:
        knn_estimator = ExplorationRewardKNN(k=3)
        for iteration in trange(1, args.num_iterations_exploration + 1):
            if args.anneal_lr:
                frac = 1.0 - (iteration - 1.0) / args.num_iterations
                lrnow = frac * args.learning_rate
                optimizer.param_groups[0]["lr"] = lrnow
            # TODO: SPLIT UP THE METHODS
            for step in range(0, args.num_steps):
                current_step += 1
                global_step += args.num_envs
                with torch.no_grad():
                    action, logprob, _, value = agent.get_action_and_value(obs)
                    values = value.flatten()

                # TODO: ADAPT TO RUNE

                (
                    next_obs,
                    ground_truth_rewards,
                    terminations,
                    truncations,
                    infos,
                ) = envs.step(action.cpu().numpy())

                intrinsic_reward = knn_estimator.compute_intrinsic_rewards(next_obs)
                knn_estimator.update_states(next_obs)
                done = np.logical_or(terminations, truncations)

                if is_mujoco_env(envs.envs[0]):
                    for idx in range(args.num_envs):
                        single_env = envs.envs[idx]
                        qpos[idx] = single_env.unwrapped.data.qpos.copy()
                        qvel[idx] = single_env.unwrapped.data.qvel.copy()

                rb.add(
                    obs=obs,
                    next_obs=next_obs,
                    action=action,
                    intrinsic_reward=intrinsic_reward,
                    extrinsic_reward=np.zeros(
                        args.num_envs
                    ),  # There is no standard deviation during the exploration phase
                    ground_truth_reward=ground_truth_rewards,
                    done=done,
                    infos=infos,
                    qpos=qpos,
                    qvel=qvel,
                )
                obs = next_obs

                if infos and "final_info" in infos:
                    for info in infos["final_info"]:
                        if info:
                            break

            # bootstrap value if not done
            with torch.no_grad():
                next_value = agent.get_value(next_obs).reshape(1, -1)
                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = (
                        rewards[t]
                        + args.gamma * nextvalues * nextnonterminal
                        - values[t]
                    )
                    advantages[t] = lastgaelam = (
                        delta
                        + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                    )
                returns = advantages + values

            # flatten the batch

            b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)

            # Optimizing the policy and value network
            b_inds = np.arange(args.batch_size)
            clipfracs = []
            for epoch in range(args.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, args.batch_size, args.minibatch_size):
                    end = start + args.minibatch_size
                    mb_inds = b_inds[start:end]

                    _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                        b_obs[mb_inds], b_actions[mb_inds]
                    )
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [
                            ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()
                        ]

                    mb_advantages = b_advantages[mb_inds]
                    if args.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                            mb_advantages.std() + 1e-8
                        )

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(
                        ratio, 1 - args.clip_coef, 1 + args.clip_coef
                    )
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    newvalue = newvalue.view(-1)
                    if args.clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds],
                            -args.clip_coef,
                            args.clip_coef,
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    entropy_loss = entropy.mean()
                    loss = (
                        pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef
                    )

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    optimizer.step()

                if args.target_kl is not None and approx_kl > args.target_kl:
                    break

            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = (
                np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
            )
            metrics.log_training_metrics_ppo(
                global_step,
                optimizer,
                v_loss,
                pg_loss,
                entropy_loss,
                old_approx_kl,
                approx_kl,
                clipfracs,
                explained_var,
                start_time,
            )
        state_dict = {
            "agent": agent,
            "reward_net": reward_net,
            "optimizer": optimizer,
            "reward_optimizer": reward_optimizer,
        }
        save_model_all(run_name, args.total_explore_steps, state_dict)
        save_replay_buffer(run_name, args.total_explore_steps, rb)
        next_obs, _ = envs.reset(seed=args.seed)  #
        next_obs = torch.Tensor(next_obs).to(device)  #

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
        for iteration in trange(1, args.num_iterations + 1):
            if args.anneal_lr:
                frac = 1.0 - (iteration - 1.0) / args.num_iterations
                lrnow = frac * args.learning_rate
                optimizer.param_groups[0]["lr"] = lrnow
            # TODO: ADAPT TO NEW FEEDBACK SCHEDULING
            for step in range(0, args.num_steps):
                if global_step % args.teacher_feedback_frequency == 0 and (
                    global_step != 0
                ):
                    for i in trange(
                        args.teacher_feedback_num_queries_per_session,
                        desc="Queries",
                        unit="queries",
                    ):
                        # Sample trajectories from replay buffer to query teacher
                        first_trajectory, second_trajectory = sample_trajectories(
                            rb,
                            args.preference_sampling,
                            reward_net,
                            args.trajectory_length,
                        )

                        logging.debug(f"step {global_step}, {i}")

                        # Create video of the two trajectories. For now, we only render if capture_video is True.
                        # If we have a human teacher, we would render the video anyway and ask the teacher to compare the two trajectories.
                        if args.capture_video:
                            video_recorder.record_trajectory(first_trajectory, run_name)
                            video_recorder.record_trajectory(
                                second_trajectory, run_name
                            )

                        # Query instructor (normally a human who decides which trajectory is better, here we use ground truth)
                        preference = sim_teacher.give_preference(
                            first_trajectory, second_trajectory
                        )

                        # Trajectories are not added to the buffer if neither segment demonstrates the desired behavior
                        if preference is None:
                            continue

                        # Store preferences
                        pref_buffer.add(first_trajectory, second_trajectory, preference)
                    # TODO ADAPT TO NEW TRAINING
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

                current_step += 1
                global_step += args.num_envs
                obs[step] = next_obs
                dones[step] = next_done
                single_obs = next_obs
                with torch.no_grad():
                    action, logprob, _, value = agent.get_action_and_value(next_obs)
                    values[step] = value.flatten()
                actions[step] = action
                logprobs[step] = logprob
                with torch.no_grad():
                    reward = reward_net.predict_reward(
                        next_obs.cpu().numpy(), action.cpu().numpy()
                    )

                (
                    next_obs,
                    ground_truth_rewards,
                    terminations,
                    truncations,
                    infos,
                ) = envs.step(action.cpu().numpy())
                next_done = np.logical_or(terminations, truncations)

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

                next_obs = next_obs.astype(np.float32)
                rb.add(
                    obs=single_obs.cpu().numpy(),
                    next_obs=next_obs,
                    action=action.cpu().numpy(),
                    reward=reward,
                    ground_truth_rewards=ground_truth_rewards,
                    done=next_done,
                    infos=infos,
                    qpos=qpos,
                    qvel=qvel,
                )
                rewards[step] = torch.tensor(reward).to(device).view(-1)
                next_obs, next_done = (
                    torch.Tensor(next_obs).to(device),
                    torch.Tensor(next_done).to(device),
                )

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
                    global_step != 0
                    or args.exploration_load
                    or args.unsupervised_exploration
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

            # bootstrap value if not done
            with torch.no_grad():
                next_value = agent.get_value(next_obs).reshape(1, -1)
                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = (
                        rewards[t]
                        + args.gamma * nextvalues * nextnonterminal
                        - values[t]
                    )
                    advantages[t] = lastgaelam = (
                        delta
                        + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                    )
                returns = advantages + values

            # flatten the batch
            b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)

            # Optimizing the policy and value network
            b_inds = np.arange(args.batch_size)
            clipfracs = []
            for epoch in range(args.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, args.batch_size, args.minibatch_size):
                    end = start + args.minibatch_size
                    mb_inds = b_inds[start:end]

                    _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                        b_obs[mb_inds], b_actions[mb_inds]
                    )
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [
                            ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()
                        ]

                    mb_advantages = b_advantages[mb_inds]
                    if args.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                            mb_advantages.std() + 1e-8
                        )

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(
                        ratio, 1 - args.clip_coef, 1 + args.clip_coef
                    )
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    newvalue = newvalue.view(-1)
                    if args.clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds],
                            -args.clip_coef,
                            args.clip_coef,
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    entropy_loss = entropy.mean()
                    loss = (
                        pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef
                    )

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    optimizer.step()

                if args.target_kl is not None and approx_kl > args.target_kl:
                    break

            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = (
                np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
            )

            metrics.log_training_metrics_ppo(
                global_step,
                optimizer,
                v_loss,
                pg_loss,
                entropy_loss,
                old_approx_kl,
                approx_kl,
                clipfracs,
                explained_var,
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
    train(cli_args)

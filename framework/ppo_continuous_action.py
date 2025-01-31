import logging
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import shimmy  # noqa
import torch
import torch.optim as optim
import tyro
from agent import Agent
from env import (
    get_qpos_qvel,
    initialize_qpos_qvel,
    is_dm_control,
    make_env_ppo,
    save_model_all,
    save_replay_buffer,
)
from evaluation import Evaluation
from performance_metrics import PerformanceMetrics
from replay_buffer import RolloutBuffer
from tqdm import trange


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
    render_mode: str = "human"
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
    ent_coef: float = 0.0
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

    agent_net_hidden_dim: int = 128
    """the dimension of the hidden layers in the actor network"""
    agent_net_hidden_layers: int = 4
    """the number of hidden layers in the actor network"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


def train(args: Args):
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size

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
        [
            make_env_ppo(
                args.env_id,
                index=i,
                render_mode=args.render_mode,
                gamma=args.gamma,
                seed=args.seed,
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
        args.num_steps,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        handle_timeout_termination=False,
        n_envs=args.num_envs,
        qpos_shape=qpos.shape[1],
        qvel_shape=qvel.shape[1],
        rune=False,
        rune_beta=0,
        rune_beta_decay=0,
        seed=args.seed,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
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
    try:
        for iteration in trange(
            1, args.num_iterations + 1, desc="Training", unit="iteration"
        ):
            # Annealing the rate if instructed to do so.
            if args.anneal_lr:
                frac = 1.0 - (iteration - 1.0) / args.num_iterations
                lrnow = frac * args.learning_rate
                optimizer.param_groups[0]["lr"] = lrnow

            for step in range(0, args.num_steps):
                current_step += 1
                global_step += args.num_envs

                ## Agent steps
                action, logprob, value = agent.select_action(obs, device)

                (
                    next_obs,
                    rewards,
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
                    extrinsic_reward=rewards,
                    intrinsic_reward=np.zeros_like(rewards),
                    ground_truth_reward=np.zeros_like(rewards),
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

                if global_step % args.evaluation_frequency == 0 and (global_step != 0):
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
                    evaluate.plot(eval_dict, global_step)
                    metrics.log_evaluate_metrics(global_step, eval_dict)

            rb.compute_gae_and_returns(agent)

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
            "optimizer": optimizer,
        }
        save_model_all(run_name, current_step, state_dict)
        save_replay_buffer(run_name, current_step, rb)
        envs.close()
        metrics.close()


if __name__ == "__main__":
    cli_args = tyro.cli(Args)
    train(cli_args)

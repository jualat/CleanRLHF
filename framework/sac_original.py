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
from actor import Actor, select_actions, update_actor, update_target_networks
from critic import SoftQNetwork, train_q_network
from env import make_env, save_model_all
from evaluation import Evaluation
from performance_metrics import PerformanceMetrics
from stable_baselines3.common.buffers import ReplayBuffer
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
    render_mode: str = ""
    """set render_mode to 'human' to watch training (it is not possible to render human and capture videos with the flag '--capture-video')"""
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
    evaluation_episodes: int = 10
    """the number of evaluation episodes"""

    actor_and_q_net_hidden_dim: int = 256
    """the dimension of the hidden layers in the actor network"""
    actor_and_q_net_hidden_layers: int = 4
    """the number of hidden layers in the actor network"""


def train(args: Args):
    import stable_baselines3 as sb3

    if sb3.__version__ < "2.0":
        raise ValueError(
            """Ongoing migration: run the following command to install the new dependencies:
poetry run pip install "stable_baselines3==2.0.0a1"
"""
        )
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
    logging.info(f"device: {device}")

    # env setup
    # env setup
    envs = gym.vector.SyncVectorEnv(
        [
            make_env(args.env_id, args.seed, args.render_mode)
            for i in range(args.num_envs)
        ]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), (
        "only continuous action space is supported"
    )
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
    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        handle_timeout_termination=False,
    )
    start_time = time.time()

    evaluate = Evaluation(
        actor=actor,
        env_id=args.env_id,
        seed=args.seed,
        torch_deterministic=args.torch_deterministic,
        run_name=run_name,
    )

    metrics = PerformanceMetrics(run_name, args, evaluate)

    obs, _ = envs.reset(seed=args.seed)
    current_step = 0
    try:
        for global_step in trange(
            args.total_timesteps, desc="Training steps", unit="steps"
        ):
            # ALGO LOGIC: put action logic here
            actions = select_actions(
                obs, actor, device, global_step, args.learning_starts, envs
            )
            current_step += 1
            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, rewards, terminations, truncations, infos = envs.step(actions)

            if infos and "episode" in infos:
                allocated, reserved = 0, 0
                cuda = False
                if args.cuda and torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated()
                    reserved = torch.cuda.memory_reserved()
                    cuda = True
                metrics.log_info_metrics(infos, global_step, allocated, reserved, cuda)

            # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
            real_next_obs = next_obs.copy()
            for idx, trunc in enumerate(truncations):
                if trunc:
                    real_next_obs[idx] = infos["final_observation"][idx]
            rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

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
                        pearson=False,
                    )
            if global_step % args.evaluation_frequency == 0 and (global_step != 0):
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
                evaluate.plot(eval_dict, global_step)
                metrics.log_evaluate_metrics(global_step, eval_dict)

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
            "q_optimizer": q_optimizer,
            "actor_optimizer": actor_optimizer,
        }
        save_model_all(run_name, current_step, state_dict)
        envs.close()
        metrics.close()

    envs.close()
    metrics.close()


if __name__ == "__main__":
    cli_args = tyro.cli(Args)
    train(cli_args)

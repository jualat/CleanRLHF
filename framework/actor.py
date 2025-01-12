import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class Actor(nn.Module):
    LOG_STD_MAX = 2
    LOG_STD_MIN = -5

    def __init__(self, env, hidden_dim, hidden_layers):
        super().__init__()
        self.fc_first = nn.Linear(
            np.array(env.single_observation_space.shape).prod(), hidden_dim
        )
        self.hidden_layers = nn.ModuleList()
        for _ in range(hidden_layers):
            self.hidden_layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.hidden_layers.append(nn.ReLU())
        self.fc_mean = nn.Linear(hidden_dim, np.prod(env.single_action_space.shape))
        self.fc_logstd = nn.Linear(hidden_dim, np.prod(env.single_action_space.shape))
        # action rescaling
        self.register_buffer(
            "action_scale",
            torch.tensor(
                (env.single_action_space.high - env.single_action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor(
                (env.single_action_space.high + env.single_action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )

    def forward(self, x):
        """
        Forward pass of the actor network
        :param x: Parameter x
        :return:
        """
        x = F.relu(self.fc_first(x))
        for layer in self.hidden_layers:
            x = layer(x)
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = self.LOG_STD_MIN + 0.5 * (self.LOG_STD_MAX - self.LOG_STD_MIN) * (
            log_std + 1
        )  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_action(self, x):
        """
        Get action from the actor network
        :param x: Parameter x
        :return:
        """
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


def select_actions(obs, actor, device, step, learning_start, envs):
    """
    Select actions from the actor network
    :param obs: Obseration from the environment
    :param actor: The actor network
    :param device: The torch device
    :param step: The current step
    :param learning_start: Run argument - timestep to start learning
    :param envs: Vectorised environments
    :return:
    """
    if step < learning_start:
        return np.array(
            [envs.single_action_space.sample() for _ in range(envs.num_envs)]
        )
    else:
        actions, _, _ = actor.get_action(torch.Tensor(obs).to(device))
    return actions.detach().cpu().numpy()


def update_actor(
    data,
    actor,
    qf1,
    qf2,
    alpha,
    actor_optimizer,
    autotune,
    log_alpha,
    target_entropy,
    a_optimizer,
):
    """
    Update the actor network
    :param data: Data from the environment containg observations
    :param actor: The actor network
    :param qf1: the first Q network
    :param qf2: the second Q network
    :param alpha: The run argument alpha
    :param actor_optimizer: The actor optimizer created in train()
    :param autotune: Boolean if autotune of alpha is enabled
    :param log_alpha: The log alpha tensor
    :param target_entropy: The target entropy
    :param a_optimizer: The optimizer for alpha created in train()
    :return:
    """
    pi, log_pi, _ = actor.get_action(data.observations)
    qf1_pi = qf1(data.observations, pi)
    qf2_pi = qf2(data.observations, pi)
    min_qf_pi = torch.min(qf1_pi, qf2_pi)
    actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

    actor_optimizer.zero_grad()
    actor_loss.backward()
    actor_optimizer.step()
    if autotune:
        with torch.no_grad():
            _, log_pi, _ = actor.get_action(data.observations)
        alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()

        a_optimizer.zero_grad()
        alpha_loss.backward()
        a_optimizer.step()
        alpha = log_alpha.exp().item()
    return actor_loss, alpha, alpha_loss


def update_target_networks(source_net, target_net, tau):
    """
    Update the target networks
    :param source_net: The source q network
    :param target_net: The target q network
    :param tau: The run argument tau
    :return:
    """
    for param, target_param in zip(source_net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

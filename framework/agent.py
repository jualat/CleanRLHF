import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def gen_net(envs, hidden_dim, layers, output_dim, std):
    net = [
        layer_init(
            nn.Linear(np.array(envs.single_observation_space.shape).prod(), hidden_dim),
        ),
        nn.Tanh(),
    ]
    for _ in range(layers):
        net.append(layer_init(nn.Linear(hidden_dim, hidden_dim)))
        net.append(nn.Tanh())
    net.append(layer_init(nn.Linear(hidden_dim, output_dim), std))

    return net


class Agent(nn.Module):
    def __init__(self, envs, hidden_dim, layers):
        super().__init__()
        self.critic = nn.Sequential(*gen_net(envs, hidden_dim, layers, 1, std=0.0))
        self.actor_mean = nn.Sequential(
            *gen_net(
                envs,
                hidden_dim,
                layers,
                np.prod(envs.single_action_space.shape),
                std=0.01,
            )
        )
        self.actor_logstd = nn.Parameter(
            torch.zeros(1, np.prod(envs.single_action_space.shape))
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return (
            action,
            probs.log_prob(action).sum(1),
            probs.entropy().sum(1),
            self.critic(x),
        )

from torch import nn
import numpy as np
import torch
import torch.nn.functional as F


class SoftQNetwork(nn.Module):
    def __init__(self, env, hidden_dim, hidden_layers):
        super().__init__()
        self.fc_first = nn.Linear(
            np.array(env.single_observation_space.shape).prod()
            + np.prod(env.single_action_space.shape),
            hidden_dim,
        )

        self.hidden_layers = nn.ModuleList()
        for _ in range(hidden_layers):
            self.hidden_layers.append(nn.Linear(hidden_dim, hidden_dim))

        self.fc_last = nn.Linear(hidden_dim, 1)

    def forward(self, x, a):
        """
        The forward function of the critic network
        :param x: Parameter x to forward
        :param a: Parameter a to forward
        :return:
        """
        x = torch.cat([x, a], dim=1)
        x = F.relu(self.fc_first(x))
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
        x = self.fc_last(x)
        return x


def train_q_network(
    data, qf1, qf2, qf1_target, qf2_target, alpha, gamma, q_optimizer, actor
):
    """
    Train the Q network
    :param data: The data from the environment containing the observations, actions, rewards, next_observations, dones
    :param qf1: The first Q network
    :param qf2: The second Q network
    :param qf1_target: The target network for the first Q network
    :param qf2_target: The target network for the second Q network
    :param alpha: The alpha
    :param gamma: The run argument gamma
    :param q_optimizer: The optimizer for the Q network
    :param actor: The Actor network
    :return:
    """
    with torch.no_grad():
        real_rewards = data.rewards
        next_state_actions, next_state_log_pi, _ = actor.get_action(
            data.next_observations
        )
        qf1_next_target = qf1_target(data.next_observations, next_state_actions)
        qf2_next_target = qf2_target(data.next_observations, next_state_actions)
        min_qf_next_target = (
            torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
        )
        next_q_value = real_rewards.flatten() + (1 - data.dones.flatten()) * gamma * (
            min_qf_next_target
        ).view(-1)

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

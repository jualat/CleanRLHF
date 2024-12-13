from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from stable_baselines3.common.vec_env import VecNormalize


def gen_reward_net(hidden_dim, layers=4, env=None):
    reward_net = [
        nn.Linear(
            np.array(env.single_observation_space.shape).prod()
            + np.prod(env.single_action_space.shape),
            hidden_dim,
        )
    ]
    for _ in range(layers):
        reward_net.append(nn.Linear(hidden_dim, hidden_dim))
        reward_net.append(nn.LeakyReLU())
    reward_net.append(nn.Linear(hidden_dim, 1))
    reward_net.append(nn.Tanh())

    return reward_net


class RewardNet(nn.Module):
    def __init__(self, env, hidden_dim):
        super(RewardNet, self).__init__()
        self.ensemble = nn.ModuleList()

        for _ in range(3):
            model = nn.Sequential(*gen_reward_net(hidden_dim, env=env))
            self.ensemble.append(model)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        y = [model(x) for model in self.ensemble]
        y = torch.stack(y, dim=0)
        return torch.mean(y, dim=0)

    def preference_prob(self, r1, r2):
        # Probability based on Bradley-Terry model
        # r_{1,2} shape: (num_steps,)
        softmax = nn.Softmax(dim=0)
        exp1 = torch.sum(r1)
        exp2 = torch.sum(r2)
        prob = softmax(torch.tensor([exp1, exp2]))
        assert 0 <= prob[0] <= 1
        return prob[0]

    def preference_loss(self, predictions, preferences, epsilon=1e-7):
        # Compute binary cross entropy loss based on human feedback
        predictions = torch.clamp(predictions, epsilon, 1 - epsilon)
        return -torch.mean(
            preferences * torch.log(predictions)
            + (1 - preferences) * torch.log(1 - predictions)
        )

    def predict_reward(self, observations: np.ndarray, actions: np.ndarray):
        """
        Predict the reward for a given observation and action.
        :param observations: The observations as a numpy array
        :param actions: The action as a numpy array
        :return: The predicted as a numpy array
        """
        # Convert observations and actions to tensors
        device = next(self.parameters()).device
        observations = torch.tensor(observations, dtype=torch.float32).to(device)
        actions = torch.tensor(actions, dtype=torch.float32).to(device)

        # Forward pass through the network
        rewards = self.forward(observations, actions)
        return rewards.cpu().detach().numpy()


def train_reward(
    model,
    optimizer,
    writer,
    pref_buffer,
    rb,
    global_step,
    epochs,
    batch_size,
    env: Optional[VecNormalize] = None,
):
    for epoch in range(epochs):
        prefs = pref_buffer.sample(batch_size)
        total_loss = 0.0
        for pref_pair in prefs:
            t1_start_idx, t1_end_idx, t2_start_idx, t2_end_idx, pref = pref_pair
            pref = torch.tensor(pref, dtype=torch.float32)

            t1 = rb.get_trajectory(int(t1_start_idx), int(t1_end_idx), env=env)
            t2 = rb.get_trajectory(int(t2_start_idx), int(t2_end_idx), env=env)
            ensemble_loss = 0.0

            for single_model in model.ensemble:
                optimizer.zero_grad()

                r1 = single_model(
                    torch.cat([t1.samples.observations, t1.samples.actions], dim=1)
                )
                r2 = single_model(
                    torch.cat([t2.samples.observations, t2.samples.actions], dim=1)
                )

                prediction = model.preference_prob(r1, r2)
                prediction = prediction.clone().detach().requires_grad_(True)

                loss = model.preference_loss(prediction, pref)
                assert loss != float("inf")
                ensemble_loss += loss

            ensemble_loss /= len(model.ensemble)
            ensemble_loss.backward()
            optimizer.step()
            total_loss += ensemble_loss.item()

            writer.add_scalar("losses/reward_loss", ensemble_loss.item(), global_step)

        if epoch % 10 == 0:
            print(f"Reward epoch {epoch}, Loss {total_loss/(batch_size*0.5)}")
        if epoch % 100 == 0:
            print(f"Reward epoch {epoch}, Loss {loss.item()}")

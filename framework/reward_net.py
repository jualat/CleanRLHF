from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from stable_baselines3.common.vec_env import VecNormalize


class RewardNet(nn.Module):
    def __init__(self, env, hidden_dim):
        super(RewardNet, self).__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod() + np.prod(env.single_action_space.shape), hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, 1)

    def forward(self, x, a):
        # Concatenate observation and action
        x = torch.cat([x, a], 1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

    def preference_prob(self, r1, r2):
        # Probability based on Bradley-Terry model
        # r_{1,2} shape: (num_steps,)
        exp1 = torch.exp(torch.sum(r1))
        exp2 = torch.exp(torch.sum(r2))
        prop1 = exp1 / (exp1 + exp2)
        assert 0 <= prop1 <= 1
        return prop1

    def preference_loss(self, predictions, preferences, epsilon=1e-7):
        # Compute binary cross entropy loss based on human feedback
        predictions = torch.clamp(predictions, epsilon, 1 - epsilon)
        return -torch.mean(preferences * torch.log(predictions) + (1 - preferences) * torch.log(1 - predictions))

    def predict_reward(self, observations: np.ndarray, actions: np.ndarray):
        """
        Predict the reward for a given observation and action.
        :param observations: The observations as a numpy array
        :param action: The action as a numpy array
        :return: The predicted as a numpy array
        """

        # Convert observations and actions to tensors
        observations = torch.tensor(observations, dtype=torch.float32).to(self.fc1.weight.device)
        actions = torch.tensor(actions, dtype=torch.float32).to(self.fc1.weight.device)

        # Forward pass through the network
        rewards = self.forward(observations, actions)
        return rewards.cpu().detach().numpy()

def train_reward(model, optimizer, writer, pref_buffer, rb, global_step, epochs, batch_size, env: Optional[VecNormalize] = None):
    for epoch in range(epochs):
        prefs = pref_buffer.sample(batch_size)
        total_loss = 0.0
        for pref_pair in prefs:
            t1_start_idx, t1_end_idx, t2_start_idx, t2_end_idx, pref = pref_pair
            pref = torch.tensor(pref, dtype=torch.float32)

            t1 = rb.get_trajectory(int(t1_start_idx), int(t1_end_idx), env=env)
            t2 = rb.get_trajectory(int(t2_start_idx), int(t2_end_idx), env=env)

            optimizer.zero_grad()

            r1 = model(t1.samples.observations, t1.samples.actions)
            r2 = model(t2.samples.observations, t2.samples.actions)

            prediction = model.preference_prob(r1, r2)
            prediction = prediction.clone().detach().requires_grad_(True)

            loss = model.preference_loss(prediction, pref)
            assert loss != float('inf')
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            writer.add_scalar("losses/reward_loss", loss.item(), global_step)

        if epoch % 10 == 0:
            print(f"Reward epoch {epoch}, Loss {total_loss/(batch_size*0.5)}")
        if epoch % 100 == 0:
            print(f"Reward epoch {epoch}, Loss {loss.item()}")
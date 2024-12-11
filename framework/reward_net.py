from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from stable_baselines3.common.vec_env import VecNormalize

def gen_reward_net(hidden_dim = 256, layers = 2, env=None):
    reward_net = [nn.Linear(
        np.array(env.single_observation_space.shape).prod() + np.prod(env.single_action_space.shape),
                  hidden_dim)]
    for _ in range(layers):
        reward_net.append(nn.Linear(hidden_dim, hidden_dim))
        reward_net.append(nn.LeakyReLU())
    reward_net.append(nn.Linear(hidden_dim, 1))
    reward_net.append(nn.Tanh())

    return reward_net

class RewardNet(nn.Module):
    def __init__(self, env, hidden_dim):
        super(RewardNet, self).__init__()
        self.ensemble = []
        paramlst = []

        for _ in range(3):
            model = nn.Sequential(*gen_reward_net(hidden_dim, env=env))
            self.ensemble.append(model)
            paramlst.extend(model.parameters())

        self.parameters = nn.ParameterList(paramlst)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x0 = self.ensemble[0](x)
        x1 = self.ensemble[1](x)
        x2 = self.ensemble[2](x)
        #print(f"x0:{x0}, x1:{x1}, x2:{x2}")
        return x0

    def preference_prob(self, r1, r2):
        # Probability based on Bradley-Terry model
        # r_{1,2} shape: (num_steps,)

        if not r1.numel() == 0:
            exp1 = torch.exp(torch.clamp(torch.sum(r1)-torch.max(r1), max=85))
        else: exp1 = torch.tensor(0.0001)
        if  not r2.numel() == 0:
            exp2 = torch.exp(torch.clamp(torch.sum(r2)-torch.max(r2), max=85))
        else: exp2 = torch.tensor(0.0001)

        prob1 = exp1 / (exp1 + exp2)
        assert 0 <= prob1 <= 1
        return prob1

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
        device = next(self.parameters()).device
        observations = torch.tensor(observations, dtype=torch.float32).to(device)
        actions = torch.tensor(actions, dtype=torch.float32).to(device)

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
import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from stable_baselines3.common.vec_env import VecNormalize


def gen_reward_net(hidden_dim, layers, env=None, p=0.3):
    reward_net = [
        nn.Linear(
            np.array(env.single_observation_space.shape).prod()
            + np.prod(env.single_action_space.shape),
            hidden_dim,
        ),
        nn.LeakyReLU(),
    ]
    for _ in range(layers):
        reward_net.append(nn.Linear(hidden_dim, hidden_dim))
        reward_net.append(nn.LeakyReLU())
        reward_net.append(nn.Dropout(p))
    reward_net.append(nn.Linear(hidden_dim, 1))

    return reward_net


class RewardNet(nn.Module):
    def __init__(self, env, hidden_dim, hidden_layers):
        super(RewardNet, self).__init__()
        self.ensemble = nn.ModuleList()

        for _ in range(3):
            model = nn.Sequential(*gen_reward_net(hidden_dim, hidden_layers, env=env))
            self.ensemble.append(model)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        y = [model(x) for model in self.ensemble]
        y = torch.stack(y, dim=0)
        return torch.mean(y, dim=0)

    def preference_prob(self, r1, r2):
        # Probability based on Bradley-Terry model
        # r_{1,2} shape: (num_steps,)
        device = next(self.parameters()).device
        softmax = nn.Softmax(dim=0)
        exp1 = torch.sum(r1)
        exp2 = torch.sum(r2)
        prob = softmax(torch.stack([exp1, exp2]).to(device))
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

    def predict_reward_member(
        self, observations: np.ndarray, actions: np.ndarray, member: int = -1
    ):
        # Convert observations and actions to tensors
        device = next(self.parameters()).device
        observations = torch.as_tensor(observations, dtype=torch.float32, device=device)
        actions = torch.as_tensor(actions, dtype=torch.float32, device=device)

        x = torch.cat([observations, actions], 1)
        rewards = self.ensemble[member](x)
        return rewards

    def preference_prob_hat_member(self, traj1, traj2, member: int = -1):
        softmax = nn.Softmax(dim=0)
        r1 = self.predict_reward_member(
            traj1.samples.observations,
            traj1.samples.actions,
            member=member,
        )
        r2 = self.predict_reward_member(
            traj2.samples.observations,
            traj2.samples.actions,
            member=member,
        )
        exp1 = r1.sum()
        exp2 = r2.sum()
        prob = softmax(torch.stack([exp1, exp2]))
        assert 0 <= prob[0] <= 1
        return prob[0]

    def preference_hat_entropy_member(self, traj1, traj2, member: int = -1):
        r1 = self.predict_reward_member(
            traj1.samples.observations,
            traj1.samples.actions,
            member=member,
        )
        r2 = self.predict_reward_member(
            traj2.samples.observations,
            traj2.samples.actions,
            member=member,
        )
        r_hat1 = r1.sum(dim=1)
        r_hat2 = r2.sum(dim=1)
        r_hat = torch.cat([r_hat1, r_hat2], dim=-1)

        ent = nn.functional.softmax(r_hat, dim=-1) * nn.functional.log_softmax(
            r_hat, dim=-1
        )
        ent = -ent.sum(dim=-1)
        return ent


def train_or_val_pref_batch(
    model: RewardNet,
    optimizer: torch.optim.Optimizer,
    prefs: np.ndarray,  # shape (batch_size, 5),
    rb,
    device: torch.device,
    env: Optional[VecNormalize] = None,
    do_train: bool = True,
):
    """
    Runs over a set of preference samples, either in a training mode (with backward + optimizer step)
    or validation mode (no backward).

    :param model: The reward network model
    :param optimizer: The optimizer for the model
    :param prefs: The preferences as a numpy array of shape (batch_size, 5) => [t1_start_idx, t1_end_idx, t2_start_idx, t2_end_idx, pref]
    :param rb: The replay buffer, used to get the trajectories
    :param device: The device to run the model on (CPU or GPU)
    :param env: Optional VecNormalize to normalise observations
    :param do_train: If True, do a forward/backward pass. If False, only forward pass (e.g. for validation).
    :return: total_loss over this batch of preferences
    """

    # If we are in validation mode, we don't need to compute gradients
    if not do_train:
        torch.set_grad_enabled(False)

    ensemble_loss = 0.0
    total_loss = 0.0
    for pref_pair in prefs:
        t1_start_idx, t1_end_idx, t2_start_idx, t2_end_idx, pref = pref_pair
        pref = torch.tensor(pref, dtype=torch.float32).to(device)

        t1 = rb.get_trajectory(int(t1_start_idx), int(t1_end_idx), env=env)
        t2 = rb.get_trajectory(int(t2_start_idx), int(t2_end_idx), env=env)

        if do_train:
            optimizer.zero_grad()

        ensemble_loss = 0.0
        for single_model in model.ensemble:
            r1 = single_model(
                torch.cat([t1.samples.observations, t1.samples.actions], dim=1).to(
                    device
                )
            )
            r2 = single_model(
                torch.cat([t2.samples.observations, t2.samples.actions], dim=1).to(
                    device
                )
            )

            prediction = model.preference_prob(r1, r2)
            loss = model.preference_loss(prediction, pref)
            assert loss != float("inf")
            ensemble_loss += loss

        ensemble_loss /= len(model.ensemble)

        if do_train:
            # Backward pass in training mode
            ensemble_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += ensemble_loss.item()

    # Re-enable gradients if we are in training mode
    if not do_train:
        torch.set_grad_enabled(True)

    return {
        "ensemble_loss": ensemble_loss,
        "total_loss": total_loss,
    }


def train_reward(
    model: RewardNet,
    optimizer: torch.optim.Optimizer,
    metrics,
    train_pref_buffer,
    rb,
    global_step,
    epochs,
    batch_size,
    device,
    val_pref_buffer: Optional = None,
    env: Optional[VecNormalize] = None,
):
    for epoch in range(epochs):

        # ==== 1) TRAINING STEP ====
        train_prefs = train_pref_buffer.sample(batch_size)
        train_losses = train_or_val_pref_batch(
            model=model,
            optimizer=optimizer,
            prefs=train_prefs,
            rb=rb,
            device=device,
            env=env,
            do_train=True,
        )

        # ==== 2) VALIDATION STEP ====
        if val_pref_buffer is not None and val_pref_buffer.size > 0:
            # no need to compute gradients
            with torch.no_grad():
                val_prefs = val_pref_buffer.sample(batch_size)
                val_losses = train_or_val_pref_batch(
                    model=model,
                    optimizer=optimizer,
                    prefs=val_prefs,
                    rb=rb,
                    device=device,
                    env=env,
                    do_train=False,
                )

        train_total_loss = train_losses["total_loss"]
        total_val_loss = val_losses["total_loss"]
        val_avg_loss = total_val_loss / len(val_prefs)

        metrics.log_reward_net_losses(
            train_ensemble_loss=train_losses["ensemble_loss"],
            train_total_loss=train_losses["total_loss"],
            val_ensemble_loss=val_losses["ensemble_loss"],
            val_avg_loss=val_avg_loss,
            global_step=global_step,
            batch_size=batch_size,
        )

        if epoch % 10 == 0:
            if val_pref_buffer is not None and val_pref_buffer.size > 0:
                logging.info(
                    f"Reward epoch {epoch}, "
                    f"Train Loss {(train_total_loss / (batch_size * 0.5)):.4f}, "
                    f"Val Loss {val_avg_loss:.4f}"
                )
            else:
                logging.info(
                    f"Reward epoch {epoch}, "
                    f"Train Loss {(train_total_loss / (batch_size * 0.5)):.4f}"
                )

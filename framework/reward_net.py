from typing import Optional
import logging

import numpy as np
import torch
import torch.nn as nn
from stable_baselines3.common.vec_env import VecNormalize


def gen_reward_net(hidden_dim, layers, env=None):
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
    reward_net.append(nn.Dropout(p=0.1))
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
        observations = torch.tensor(observations, dtype=torch.float32).to(device)
        actions = torch.tensor(actions, dtype=torch.float32).to(device)

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


def train_reward(
    model,
    optimizer,
    writer,
    pref_buffer,
    rb,
    global_step,
    epochs,
    batch_size,
    device,
    env: Optional[VecNormalize] = None,
):
    for epoch in range(epochs):
        prefs = pref_buffer.sample(batch_size)
        total_loss = 0.0
        for pref_pair in prefs:
            t1_start_idx, t1_end_idx, t2_start_idx, t2_end_idx, pref = pref_pair
            pref = torch.tensor(pref, dtype=torch.float32).to(device)

            t1 = rb.get_trajectory(int(t1_start_idx), int(t1_end_idx), env=env)
            t2 = rb.get_trajectory(int(t2_start_idx), int(t2_end_idx), env=env)
            ensemble_loss = 0.0

            optimizer.zero_grad()

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
                assert (
                    prediction.requires_grad
                ), "prediction does not require gradients!"

                loss = model.preference_loss(prediction, pref)
                assert loss != float("inf")
                ensemble_loss += loss

            ensemble_loss /= len(model.ensemble)
            ensemble_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += ensemble_loss.item()

            writer.add_scalar("losses/reward_loss", ensemble_loss.item(), global_step)
        writer.add_scalar(
            "losses/total_loss", total_loss / (batch_size * 0.5), global_step
        )
        if epoch % 10 == 0:
            logging.info(f"Reward epoch {epoch}, Loss {total_loss/(batch_size*0.5)}")


def train_reward_surf(
    model,
    optimizer,
    writer,
    pref_buffer,
    rb,
    global_step,
    epochs,
    batch_size,
    device,
    env: Optional[VecNormalize] = None,
    sampling_strategy=None,
    trajectory_length=64,
    unlabeled_batch_ratio=1,
    tau=0.8,
    lambda_ssl=0.1,
    H_min=55,
    H_max=45,
):
    from sampling import sample_pairs
    from teacher import Preference

    for epoch in range(epochs):
        prefs = pref_buffer.sample(batch_size)
        sup_loss_accum = 0.0
        for pref_pair in prefs:
            t1_start_idx, t1_end_idx, t2_start_idx, t2_end_idx, pref = pref_pair
            pref = torch.tensor(pref, dtype=torch.float32).to(device)

            t1 = rb.get_trajectory(int(t1_start_idx), int(t1_end_idx), env=env)
            t2 = rb.get_trajectory(int(t2_start_idx), int(t2_end_idx), env=env)

            t1_aug = rb.temporal_data_augmentation(
                t1, H_max=H_max, H_min=H_min, env=env
            )
            t2_aug = rb.temporal_data_augmentation(
                t2, H_max=H_max, H_min=H_min, env=env
            )

            ensemble_loss = 0.0
            optimizer.zero_grad()

            for single_model in model.ensemble:
                r1 = single_model(
                    torch.cat(
                        [t1_aug.samples.observations, t1_aug.samples.actions], dim=1
                    ).to(device)
                )
                r2 = single_model(
                    torch.cat(
                        [t2_aug.samples.observations, t2_aug.samples.actions], dim=1
                    ).to(device)
                )

                prediction = model.preference_prob(r1, r2)
                assert (
                    prediction.requires_grad
                ), "prediction does not require gradients!"
                loss = model.preference_loss(prediction, pref)
                assert loss != float("inf")
                ensemble_loss += loss

            ensemble_loss /= len(model.ensemble)
            ensemble_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            sup_loss_accum += ensemble_loss.item()

        unsup_loss_accum = 0.0

        if unlabeled_batch_ratio > 0:
            unlabeled_pairs = sample_pairs(
                size=unlabeled_batch_ratio * batch_size,
                rb=rb,
                sampling_strategy="uniform",
                reward_net=model,
                traj_len=trajectory_length,
            )
            for (t1_start_idx, t1_end_idx), (
                t2_start_idx,
                t2_end_idx,
            ) in unlabeled_pairs:
                t1_u = rb.get_trajectory(int(t1_start_idx), int(t1_end_idx), env=env)
                t2_u = rb.get_trajectory(int(t2_start_idx), int(t2_end_idx), env=env)

                t1_u_aug = rb.temporal_data_augmentation(
                    t1_u, H_max=H_max, H_min=H_min, env=env
                )
                t2_u_aug = rb.temporal_data_augmentation(
                    t2_u, H_max=H_max, H_min=H_min, env=env
                )

                r1_u = t1_u_aug.samples.rewards
                r2_u = t2_u_aug.samples.rewards
                prob_u = model.preference_prob(r1_u, r2_u)

                if prob_u > tau:
                    pseudo_label = Preference.FIRST.value
                elif prob_u < (1.0 - tau):
                    pseudo_label = Preference.SECOND.value
                else:
                    continue  # skip uncertain pseudo-label

                ensemble_loss_u = 0.0
                optimizer.zero_grad()

                for single_model in model.ensemble:
                    r1_u = single_model(
                        torch.cat(
                            [t1_u_aug.samples.observations, t1_u_aug.samples.actions],
                            dim=1,
                        ).to(device)
                    )
                    r2_u = single_model(
                        torch.cat(
                            [t2_u_aug.samples.observations, t2_u_aug.samples.actions],
                            dim=1,
                        ).to(device)
                    )
                    pred_u = model.preference_prob(r1_u, r2_u)
                    loss_u = model.preference_loss(pred_u, pseudo_label)
                    ensemble_loss_u += loss_u

                ensemble_loss_u /= len(model.ensemble)
                unsup_loss_accum += ensemble_loss_u.item()

                (lambda_ssl * ensemble_loss_u).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
        total_loss = sup_loss_accum + lambda_ssl * unsup_loss_accum
        writer.add_scalar("losses/supervised_loss", sup_loss_accum, global_step)
        writer.add_scalar("losses/unsupervised_loss", unsup_loss_accum, global_step)
        writer.add_scalar(
            "losses/total_loss", total_loss / (batch_size * 0.5), global_step
        )

        if epoch % 10 == 0:
            logging.info(
                f"Semi-supervised epoch {epoch}, "
                f"Supervised Loss {sup_loss_accum:.3f}, Unsupervised Loss {unsup_loss_accum:.3f}"
            )

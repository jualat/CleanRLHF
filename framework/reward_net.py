import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from stable_baselines3.common.vec_env import VecNormalize


def gen_reward_net(hidden_dim, layers, env=None, p=0.3):
    """
    Generate a reward network with the given parameters.
    :param hidden_dim: The dimension of the hidden layers
    :param layers: The amount of hidden layers
    :param env: The environment
    :param p: Dropout value
    :return:
    """
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
    def __init__(self, env, hidden_dim, hidden_layers, dropout):
        super().__init__()
        self.ensemble = nn.ModuleList()

        for _ in range(3):
            model = nn.Sequential(
                *gen_reward_net(hidden_dim, hidden_layers, env=env, p=dropout)
            )
            self.ensemble.append(model)

    def forward(self, x, a):
        """
        Forward pass through the network
        :param x: Parameter x
        :param a: Parameter a
        :return:
        """
        x = torch.cat([x, a], 1)
        y = [model(x) for model in self.ensemble]
        y = torch.stack(y, dim=0)
        return torch.mean(y, dim=0)

    def preference_prob(self, r1, r2):
        """
        Compute the probability based on the Bradley-Terry model.
        :param r1: shape: (num_steps,)
        :param r2: shape: (num_steps,)
        :return:
        """
        device = next(self.parameters()).device
        softmax = nn.Softmax(dim=0)
        exp1 = torch.sum(r1)
        exp2 = torch.sum(r2)
        prob = softmax(torch.stack([exp1, exp2]).to(device))
        assert 0 <= prob[0] <= 1
        return prob[0]

    def preference_loss(self, predictions, preferences, epsilon=1e-7):
        """
        Compute the binary cross entropy loss based on human feedback.
        :param predictions: The predictions as a tensor
        :param preferences: The preferences as a tensor
        :param epsilon: The epsilon value
        :return:
        """
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
        r_hat_all_members = []

        for member in self.ensemble:
            x = torch.cat([observations, actions], 1)
            # Forward pass through the ensemble member network
            r_hat = member(x).cpu().detach().numpy()
            r_hat_all_members.append(r_hat)

        r_hat_all_members = np.array(r_hat_all_members)
        r_hat_std = np.std(r_hat_all_members, axis=0)
        rewards = np.mean(r_hat_all_members, axis=0)

        return rewards, r_hat_std

    def predict_reward_member(
        self, observations: np.ndarray, actions: np.ndarray, member: int = -1
    ):
        """
        Predict the reward for a given observation and action in a specific ensemble member.
        :param observations: The observations as a numpy array
        :param actions: The actions as a numpy array
        :param member: The member of the ensemble
        :return:
        """
        # Convert observations and actions to tensors
        device = next(self.parameters()).device
        observations = torch.as_tensor(observations, dtype=torch.float32, device=device)
        actions = torch.as_tensor(actions, dtype=torch.float32, device=device)

        x = torch.cat([observations, actions], 1)
        rewards = self.ensemble[member](x)
        return rewards

    def preference_prob_hat_member(self, traj1, traj2, member: int = -1):
        """
        Compute the probability based on the Bradley-Terry model for a specific ensemble member.
        :param traj1: Trajectory one
        :param traj2: Trajectory two
        :param member: The ensemble member
        :return:
        """
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
        """
        Compute the entropy for a specific ensemble member.
        :param traj1: Trajectory one
        :param traj2: Trajectory two
        :param member: The member of the ensemble
        :return:
        """
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
    prefs_batch: np.ndarray,  # shape (batch_size, 5),
    rb,
    device: torch.device,
    env: Optional[VecNormalize] = None,
    do_train: bool = True,
    surf: Optional[bool] = False,
    sampling_strategy: Optional[str] = None,
    trajectory_length: Optional[int] = 64,
    unlabeled_batch_ratio: Optional[int] = 1,
    tau: Optional[float] = 0.8,
    lambda_ssl: Optional[float] = 0.1,
    H_max: Optional[int] = 55,
    H_min: Optional[int] = 45,
    batch_sampling: Optional[str] = "full",
    mini_batch_size: Optional[int] = 10,
):
    """
    Runs over a set of preference samples, either in a training mode (with backward + optimizer step)
    or validation mode (no backward).

    :param model: The reward network model
    :param optimizer: The optimizer for the model
    :param prefs_batch: The preferences as a numpy array of shape (batch_size, 5) => [t1_start_idx, t1_end_idx, t2_start_idx, t2_end_idx, pref]
    :param rb: The replay buffer, used to get the trajectories
    :param device: The device to run the model on (CPU or GPU)
    :param env: Optional VecNormalize to normalise observations
    :param do_train: If True, do a forward/backward pass. If False, only forward pass (e.g. for validation).
    :param surf: Toggle surf
    :param sampling_strategy:
    :param trajectory_length: Trajectory length
    :param unlabeled_batch_ratio: Ratio of unlabeled to labeled batch size
    :param tau: Confidence threshold for pseudo-labeling
    :param lambda_ssl: Weight for the unsupervised (pseudo-labeled) loss
    :param H_max: Maximal length of the data augmented trajectory
    :param H_min: Minimal length of the data augmented trajectory
    :param batch_sampling: The batch sampling strategy
    :param mini_batch_size: The mini batch size
    :return: total_avg_loss over this batch of preferences
    """
    from sampling import sample_pairs
    from teacher import give_pseudo_label

    # If we are in validation mode, we don't need to compute gradients
    if not do_train:
        torch.set_grad_enabled(False)

    sup_loss_accum = 0.0
    sup_num_samples = 0
    for prefs in prefs_batch:
        if do_train:
            optimizer.zero_grad()
            batch_loss = torch.tensor(
                0.0, dtype=torch.float, device=device, requires_grad=True
            )
        else:
            batch_loss = torch.tensor(
                0.0, dtype=torch.float, device=device, requires_grad=False
            )

        for pref_pair in prefs:
            t1_start_idx, t1_end_idx, t2_start_idx, t2_end_idx, pref = pref_pair
            pref = torch.tensor(pref, dtype=torch.float32).to(device)

            t1 = rb.get_trajectory(int(t1_start_idx), int(t1_end_idx), env=env)
            t2 = rb.get_trajectory(int(t2_start_idx), int(t2_end_idx), env=env)
            if surf:
                t1 = rb.temporal_data_augmentation(
                    t1, H_max=H_max, H_min=H_min, env=env
                )
                t2 = rb.temporal_data_augmentation(
                    t2, H_max=H_max, H_min=H_min, env=env
                )

            if do_train:
                ensemble_loss = torch.tensor(
                    0.0, dtype=torch.float, device=device, requires_grad=True
                )
            else:
                ensemble_loss = torch.tensor(
                    0.0, dtype=torch.float, device=device, requires_grad=False
                )

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
                ensemble_loss = ensemble_loss + loss

            ensemble_loss /= len(model.ensemble)
            batch_loss = batch_loss + ensemble_loss
            sup_loss_accum += ensemble_loss.item()
            sup_num_samples += 1
        if do_train:
            # Backward pass in training mode
            batch_loss = batch_loss / len(prefs)
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
    sup_avg_loss = sup_loss_accum / sup_num_samples

    unsup_avg_loss = 0.0
    unsup_loss_accum = 0.0
    unsup_num_samples = 0
    if surf and unlabeled_batch_ratio > 0:
        unsup_batch_size = unlabeled_batch_ratio * len(prefs_batch)
        unlabeled_pairs_batch = sample_pairs(
            size=unsup_batch_size,
            rb=rb,
            sampling_strategy=sampling_strategy,
            reward_net=model,
            traj_len=trajectory_length,
            batch_sampling=batch_sampling,
            mini_batch_size=mini_batch_size,
        )
        for unlabeled_pairs in unlabeled_pairs_batch:
            if do_train:
                optimizer.zero_grad()
                batch_loss = torch.tensor(
                    0.0, dtype=torch.float, device=device, requires_grad=True
                )
            else:
                batch_loss = torch.tensor(
                    0.0, dtype=torch.float, device=device, requires_grad=False
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

                pseudo_label = give_pseudo_label(t1_u_aug, t2_u_aug, tau, model)
                if pseudo_label is None:
                    continue

                if do_train:
                    ensemble_loss_u = torch.tensor(
                        0.0, dtype=torch.float, device=device, requires_grad=True
                    )
                else:
                    ensemble_loss_u = torch.tensor(
                        0.0, dtype=torch.float, device=device, requires_grad=False
                    )

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
                    assert loss_u != float("inf")
                    ensemble_loss_u = ensemble_loss_u + loss_u

                ensemble_loss_u /= len(model.ensemble)
                batch_loss = batch_loss + ensemble_loss_u
                unsup_loss_accum += ensemble_loss_u.item()
                unsup_num_samples += 1

            if do_train:
                batch_loss = batch_loss / len(unlabeled_pairs)
                (lambda_ssl * batch_loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
        if unsup_num_samples > 0:
            unsup_avg_loss = unsup_loss_accum / unsup_num_samples

    # Re-enable gradients if we are in training mode
    if not do_train:
        torch.set_grad_enabled(True)

    total_avg_loss = sup_avg_loss + lambda_ssl * unsup_avg_loss
    if do_train:
        return {
            "train_total_loss": total_avg_loss,
            "train_supervised_loss": sup_avg_loss,
            "train_unsupervised_loss": unsup_avg_loss,
        }
    return {
        "val_total_loss": total_avg_loss,
        "val_supervised_loss": sup_avg_loss,
        "val_unsupervised_loss": unsup_avg_loss,
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
    mini_batch_size,
    batch_sample_strategy,
    device,
    val_pref_buffer: Optional = None,
    env: Optional[VecNormalize] = None,
    surf: bool = False,
    sampling_strategy=None,
    trajectory_length=64,
    unlabeled_batch_ratio=1,
    tau=0.8,
    lambda_ssl=0.1,
    H_max=55,
    H_min=45,
):
    """
    Train the reward network.
    :param model: The model to train
    :param optimizer: The optimizer for the training
    :param metrics: The metrics class
    :param train_pref_buffer: The preference buffer
    :param rb: The replay buffer
    :param global_step: The global step
    :param epochs: The amount of epochs
    :param batch_size: The batch size
    :param mini_batch_size: The mini batch size
    :param batch_sample_strategy: The batch sampling strategy
    :param device: The torch device
    :param val_pref_buffer: The validation preference buffer
    :param env: The environment
    :param surf: Toggle surf
    :param sampling_strategy: Sampling strategy
    :param trajectory_length: The trajectory length
    :param unlabeled_batch_ratio: Ratio of unlabeled to labeled batch size
    :param tau: Confidence threshold for pseudo-labeling
    :param lambda_ssl: Weight for the unsupervised (pseudo-labeled) loss
    :param H_max: Maximal length of the data augmented trajectory
    :param H_min: Minimal length of the data augmented trajectory
    :return:
    """
    for epoch in range(epochs):

        # ==== 1) TRAINING STEP ====
        train_prefs = train_pref_buffer.sample(
            sample_strategy=batch_sample_strategy,
            batch_size=batch_size,
            mini_batch_size=mini_batch_size,
        )

        train_loss_dict = train_or_val_pref_batch(
            model=model,
            optimizer=optimizer,
            prefs_batch=train_prefs,
            rb=rb,
            device=device,
            env=env,
            do_train=True,
            surf=surf,
            sampling_strategy=sampling_strategy,
            trajectory_length=trajectory_length,
            unlabeled_batch_ratio=unlabeled_batch_ratio,
            tau=tau,
            lambda_ssl=lambda_ssl,
            H_max=H_max,
            H_min=H_min,
            batch_sampling=batch_sample_strategy,
            mini_batch_size=mini_batch_size,
        )

        # ==== 2) VALIDATION STEP ====
        if val_pref_buffer is not None and val_pref_buffer.size > 0:
            # no need to compute gradients
            with torch.no_grad():
                val_prefs = val_pref_buffer.sample(
                    sample_strategy="full",
                    batch_size=batch_size,
                    mini_batch_size=mini_batch_size,
                )
                val_loss_dict = train_or_val_pref_batch(
                    model=model,
                    optimizer=optimizer,
                    prefs_batch=val_prefs,
                    rb=rb,
                    device=device,
                    env=env,
                    do_train=False,
                    surf=surf,
                    sampling_strategy=sampling_strategy,
                    trajectory_length=trajectory_length,
                    unlabeled_batch_ratio=unlabeled_batch_ratio,
                    tau=tau,
                    lambda_ssl=lambda_ssl,
                    H_max=H_max,
                    H_min=H_min,
                    batch_sampling=batch_sample_strategy,
                    mini_batch_size=mini_batch_size,
                )

        metrics.log_losses(
            loss_dict=train_loss_dict,
            global_step=global_step,
        )
        metrics.log_losses(
            loss_dict=val_loss_dict,
            global_step=global_step,
        )

        if epoch % 10 == 0:
            if (
                val_pref_buffer is not None
                and val_pref_buffer.size > 0
                and val_loss_dict is not None
            ):
                if surf:
                    logging.info(
                        f"Reward epoch {epoch}, "
                        f"Train Loss {train_loss_dict['train_total_loss'] :.4f}, "
                        f"Val Loss {val_loss_dict['val_total_loss'] :.4f}, "
                        f"Train Supervised Loss {train_loss_dict['train_supervised_loss']:.4f}, "
                        f"Val Supervised Loss {val_loss_dict['val_supervised_loss']:.4f}, "
                        f"Train Unsupervised Loss {train_loss_dict['train_unsupervised_loss']:.4f}, "
                        f"Val Unsupervised Loss {val_loss_dict['val_unsupervised_loss']:.4f}, "
                    )
                else:
                    logging.info(
                        f"Reward epoch {epoch}, "
                        f"Train Loss {train_loss_dict['train_total_loss'] :.4f}, "
                        f"Val Loss {val_loss_dict['val_total_loss'] :.4f}"
                    )
            else:
                if surf:
                    logging.info(
                        f"Reward epoch {epoch}, "
                        f"Train Loss {train_loss_dict['train_total_loss']:.4f}, "
                        f"Supervised Loss {train_loss_dict['train_supervised_loss']:.4f}, "
                        f"Unsupervised Loss {train_loss_dict['train_unsupervised_loss']:.4f}"
                    )
                else:
                    logging.info(
                        f"Reward epoch {epoch}, "
                        f"Train Loss {train_loss_dict['train_total_loss']:.4f}"
                    )

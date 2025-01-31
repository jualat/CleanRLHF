import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from stable_baselines3.common.vec_env import VecNormalize

from framework.sampling import sample_pairs
from framework.teacher import give_pseudo_label_ensemble


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

    def forward_ensemble(self, obs, acts):
        """
        Forward pass across *all* ensemble members simultaneously.
        :param obs: shape [ensemble_size, batch_size, traj_length, obs_dim]
        :param acts: shape [ensemble_size, batch_size, traj_length, act_dim]
        :return: shape [ensemble_size, batch_size, 1]
        """
        x = torch.cat([obs, acts], dim=2)  # shape: [ensemble_size, batch_size, obs_dim + act_dim]
        y = [model(x[i]) for i, model in enumerate(self.ensemble)]
        return torch.stack(y, dim=0)

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

    def preference_prob_ensemble(self, r_ens):
        """
        Compute the preference probabilities using the Bradley-Terry model.

        :param r_ens: The rewards as a tensor of shape (2, E, B, L)
        :return: The loss as a tensor of shape (E,B,)
        """
        r_ens_cum = r_ens.sum(dim=3)  # shape: (2, E, B, 1)
        r_ens_cum = r_ens_cum.squeeze(-1)  # shape: (2, E, B)

        r_ens_cum_t1 = r_ens_cum[0]  # shape: (E, B)
        r_ens_cum_t2 = r_ens_cum[1]  # shape: (E, B)

        # Compute the preference probabilities using the Bradley-Terry model
        exp1 = torch.exp(r_ens_cum_t1)
        exp2 = torch.exp(r_ens_cum_t2)
        return exp1 / (exp1 + exp2 + 1e-7)  # shape (E, B)

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

    def _compute_batch_pref_loss_ensemble(self,
        obs: torch.Tensor,
        acts: torch.Tensor,
        prefs: torch.Tensor,
        loss_method: str = "avg_prob",
    ):
        """
        Vectorized preference-loss computation across the entire ensemble in one shot.

        Shapes:
        E = ensemble size
        B = batch size
        L = trajectory length
        O = obs dim
        A = act dim

        :param obs: observations of shape (E, 2, B, L, O) of the first and second trajectory
        :param acts: actions of shape (E, 2, B, L, A) of the first and second trajectory
        :param prefs: preferences of shape (E, 2, B,) of the first and second trajectory
        :param loss_method: method to compute binary cross-entropy loss
                            - avg_prob: average the probabilities across ensemble
                            - avg_bce: compute BCE individually for each ensemble member, then average.
        """

        E = len(self.ensemble)
        B, L, O = obs.shape
        _, _, A = obs.shape

        # Flatten dimensions
        obs_flat = obs.reshape(E, 2 * B * L, O)
        acts_flat = acts.reshape(E, 2 * B * L, A)

        # Forward pass through the ensemble
        r_ens = self.forward_ensemble(obs_flat, acts_flat)  # shape: (E, 2 * B * L, 1)
        r_ens = r_ens.reshape(E, 2, B, L) # shape: (E, 2, B, L, 1)
        r_ens = r_ens.transpose(0, 1) # shape: (2, E, B, L, 1)

        epsilon = 1e-7
        prob_ens = self.preference_prob_ensemble(r_ens)  # shape: (E, B)

        # Compute the binary cross-entropy loss
        if loss_method == "avg_prob":
            # average the probabilities across ensemble
            prob = prob_ens.mean(dim=0)
            prob = torch.clamp(prob, epsilon, 1.0 - epsilon)
            bce = -(prefs * torch.log(prob) + (1.0 - prefs) * torch.log(1.0 - prob))
            return bce.mean()
        elif loss_method == "avg_bce":
            # compute BCE individually for each ensemble member, then average.
            prob_ens = torch.clamp(prob_ens, epsilon, 1.0 - epsilon)
            bce_ens = -(prefs * torch.log(prob_ens) + (1 - prefs) * torch.log(1 - prob_ens))
            return bce_ens.mean(dim=1).mean(dim=0)
        else:
            raise ValueError(f"Invalid method: {loss_method}")

    def _prepare_pref_pairs(self, ens_pref_batches, device, rb, env, surf, H_max, H_min):
        """
        Prepare the preference pairs for training and validation by transforming them into tensors and applying data augmentation.

        :param ens_pref_batches: A list of preference batches for each ensemble member. Each batch is a list of tuples
                                 of the form (t1_start_idx, t1_end_idx, t2_start_idx, t2_end_idx, [pref]) where the last
                                 element is optional and only present if the batch is labeled.
        :param device: The device to run the model on (CPU or GPU)
        :param rb: The replay buffer
        :param env: An optional VecNormalize environment to normalize observations
        :param surf: Whether to use SURF data augmentation
        :return: A tuple of tensors (ens_obs, ens_act, ens_prefs) of shape (E, 2, B, L, O), (E, 2, B, L, A), (E, B).
        The last tensor is meaningless if the pref_batch is not labeled.
        """
        ens_obs = []
        ens_act = []
        ens_prefs = []
        for pref_batch in ens_pref_batches:
            t1_obs_list, t1_act_list = [], []
            t2_obs_list, t2_act_list = [], []
            pref_list = []

            for pref_pair in pref_batch:
                is_labeled = len(pref_pair) == 5
                if is_labeled:
                    t1_start_idx, t1_end_idx, t2_start_idx, t2_end_idx, pref = pref_pair
                    pref = torch.tensor(pref, dtype=torch.float32).to(device)
                    pref_list.append(pref)
                else:
                    t1_start_idx, t1_end_idx, t2_start_idx, t2_end_idx = pref_pair

                t1 = rb.get_trajectory(int(t1_start_idx), int(t1_end_idx), env=env)
                t2 = rb.get_trajectory(int(t2_start_idx), int(t2_end_idx), env=env)

                if surf or not is_labeled:
                    t1 = rb.temporal_data_augmentation(
                        t1, H_max=H_max, H_min=H_min, env=env
                    )
                    t2 = rb.temporal_data_augmentation(
                        t2, H_max=H_max, H_min=H_min, env=env
                    )

                t1_obs_list.append(t1.samples.observations)
                t1_act_list.append(t1.samples.actions)
                t2_obs_list.append(t2.samples.observations)
                t2_act_list.append(t2.samples.actions)

            # shape: (B, L, 0)
            t1_obs = torch.stack(t1_obs_list, dim=0).to(device)
            t2_obs = torch.stack(t2_obs_list, dim=0).to(device)

            # shape: (B, L, A)
            t1_act = torch.stack(t1_act_list, dim=0).to(device)
            t2_act = torch.stack(t2_act_list, dim=0).to(device)


            prefs = torch.tensor(pref_list, dtype=torch.float32, device=device)  # shape: (B,)

            obs_cat = torch.cat([t1_obs, t2_obs], dim=1).to(device)  # shape: (2, B, L, O)
            act_cat = torch.cat([t1_act, t2_act], dim=1).to(device)  # shape: (2, B, L, A)

            ens_obs.append(obs_cat)
            ens_act.append(act_cat)
            ens_prefs.append(prefs)

        ens_obs = torch.stack(ens_obs, dim=0)  # shape: (E, 2, B, L, O)
        ens_act = torch.stack(ens_act, dim=0)  # shape: (E, 2, B, L, A)
        ens_prefs = torch.stack(ens_prefs, dim=0)  # shape: (E, B)

        return ens_obs, ens_act, ens_prefs


    def train_or_val_pref_batch_ensemble(
        self,
        optimizer: torch.optim.Optimizer,
        buffer,
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
        batch_size: Optional[int] = 32,
        loss_method: str = "avg_prob",  # or "avg_bce"
    ):
        """
        Train or validate the reward network with a batch of preferences.

        :param optimizer: The optimizer for the model
        :param buffer: The preference buffer
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
        :param batch_size: The batch size
        :param loss_method: method to compute binary cross-entropy loss
                            - avg_prob: average the probabilities across ensemble
                            - avg_bce: compute BCE individually for each ensemble member, then average.
        """

        # If we are not training, disable grad
        if not do_train:
            torch.set_grad_enabled(False)

        #############################
        # Labelled preference pairs #
        #############################

        labeled_batches = [buffer.sample(
            sample_strategy=batch_sampling,
            batch_size=batch_size,
            mini_batch_size=mini_batch_size,
        ) for _ in range(len(self.ensemble))]

        # Prepare the preference pairs for training and validation
        labeled_ens_obs, labeled_ens_act, labeled_ens_prefs = self._prepare_pref_pairs(
            labeled_batches, device, rb, env, surf, H_max=H_max, H_min=H_min
        )

        if do_train:
            optimizer.zero_grad()

        sup_loss = self._compute_batch_pref_loss_ensemble(labeled_ens_obs, labeled_ens_act, labeled_ens_prefs, loss_method=loss_method)

        if do_train:
            sup_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            optimizer.step()

        ##############################
        # Unlabeled preference pairs #
        ##############################

        unlabeled_batch_size = unlabeled_batch_ratio * batch_size
        unlabeled_pairs_batch = [sample_pairs(
            size=unlabeled_batch_size,
            rb=rb,
            sampling_strategy=sampling_strategy,
            reward_net=self,
            traj_len=trajectory_length,
            batch_sampling=batch_sampling,
            mini_batch_size=mini_batch_size,
        ) for _ in range(len(self.ensemble))]

        # Prepare the preference pairs for training and validation
        unlabeled_ens_obs, unlabeled_ens_act, _ = self._prepare_pref_pairs(
            unlabeled_pairs_batch, device, rb, env, surf, H_max=H_max, H_min=H_min
        )

        if do_train:
            optimizer.zero_grad()

        unlabeled_ens_prefs = give_pseudo_label_ensemble(unlabeled_ens_obs, unlabeled_ens_act, tau, self)
        unsup_loss = self._compute_batch_pref_loss_ensemble(unlabeled_ens_obs, unlabeled_ens_act, unlabeled_ens_prefs, loss_method=loss_method)

        if do_train:
            (lambda_ssl * unsup_loss).backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss = sup_loss + lambda_ssl * unsup_loss

        # Re-enable gradients if we are not in training mode
        if not do_train:
            torch.set_grad_enabled(True)

        prefix = "train" if do_train else "val"
        return {
            f"{prefix}_total_loss": total_loss,
            f"{prefix}_supervised_loss": sup_loss,
            f"{prefix}_unsupervised_loss": unsup_loss,
        }

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

        train_loss_dict = model.train_or_val_pref_batch_ensemble(
            optimizer=optimizer,
            buffer=train_pref_buffer,
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
            batch_size=batch_size,
            mini_batch_size=mini_batch_size,
            loss_method="avg_prob",
        )

        metrics.log_losses(
            loss_dict=train_loss_dict,
            global_step=global_step,
        )

        # ==== 2) VALIDATION STEP ====
        val_loss_dict = {
            "val_total_loss": 0,
            "val_supervised_loss": 0,
            "val_unsupervised_loss": 0,
        }
        if val_pref_buffer is not None and val_pref_buffer.size > 0:
            # no need to compute gradients
            with torch.no_grad():
                val_loss_dict_model = model.train_or_val_pref_batch_ensemble(
                    optimizer=optimizer,
                    rb=rb,
                    buffer=val_pref_buffer,
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
                    batch_sampling="full",
                    mini_batch_size=mini_batch_size,
                    batch_size=batch_size,
                )

                metrics.log_losses(
                    loss_dict=val_loss_dict,
                    global_step=global_step,
                )

        if epoch % 10 == 0:
            if val_pref_buffer is not None and val_pref_buffer.size > 0:
                if surf:
                    logging.info(
                        f"Reward epoch {epoch}, "
                        f"Train Loss {train_loss_dict['train_total_loss']:.4f}, "
                        f"Val Loss {val_loss_dict['val_total_loss']:.4f}, "
                        f"Train Supervised Loss {train_loss_dict['train_supervised_loss']:.4f}, "
                        f"Val Supervised Loss {val_loss_dict['val_supervised_loss']:.4f}, "
                        f"Train Unsupervised Loss {train_loss_dict['train_unsupervised_loss']:.4f}, "
                        f"Val Unsupervised Loss {val_loss_dict['val_unsupervised_loss']:.4f}, "
                    )
                else:
                    logging.info(
                        f"Reward epoch {epoch}, "
                        f"Train Loss {train_loss_dict['train_total_loss']:.4f}, "
                        f"Val Loss {val_loss_dict['val_total_loss']:.4f}"
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

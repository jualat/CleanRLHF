import logging
import random
from enum import Enum

import numpy as np
import torch
from matplotlib import pyplot as plt
from sympy.codegen.ast import float32

from replay_buffer import Trajectory
from reward_net import RewardNet


class Preference(Enum):
    SKIP = None  # Skip the query
    EQUAL = 0.5  # Segments are equally preferable
    FIRST = 1.0  # First segment is preferred
    SECOND = 0.0  # Second segment is preferred


class Teacher:
    """
    Models a human teacher's behavior with myopic (short-sighted) tendencies,
    the ability to skip queries, mark segments as equally preferable,
    and account for occasional errors in judgment.
    B-Pref paper: https://arxiv.org/pdf/2111.03026

    Parameters:
    - beta: A rationality constant that determines the teacher's consistency in preferring segments
      based on their underlying reward. A higher beta (β → ∞) makes the teacher perfectly rational
      and deterministic, always choosing the better segment. A beta of 0 leads to uniformly random
      choices, reflecting complete irrationality.

    - gamma: The discount factor used in the weighted sum of rewards.
      Determines how much past rewards influence the teacher’s decision relative to recent rewards.

    - epsilon: The error probability reflecting the likelihood of the teacher making a mistake.
      With probability epsilon, the teacher's preference is flipped, introducing randomness.

    - delta_skip: The skip threshold for queries. If the total reward for both segments
      is smaller than this threshold, the teacher marks the query as incomparable and skips it.
      This models scenarios where neither segment demonstrates the desired behavior.

    - delta_equal: The equality threshold for determining if two segments are equally preferable.
      If the absolute difference in total rewards between two segments is smaller than delta_equal,
      the teacher marks the segments as equally preferable and provides a uniform response.
    """

    def __init__(self, beta, gamma, epsilon, delta_skip, delta_equal, seed=None):
        self.beta = beta
        self.gamma = gamma
        self.epsilon = epsilon
        self.delta_skip = delta_skip
        self.delta_equal = delta_equal
        self.seed = seed

        if seed is not None:
            random.seed(seed)
            self.torch_generator = torch.Generator().manual_seed(seed)
        else:
            self.torch_generator = None

    def give_preference(
        self, first_trajectory: Trajectory, second_trajectory: Trajectory
    ) -> float:
        first_trajectory_reward = (
            first_trajectory.samples.ground_truth_rewards.sum().item()
        )
        second_trajectory_reward = (
            second_trajectory.samples.ground_truth_rewards.sum().item()
        )
        mistake = random.random()

        if max(first_trajectory_reward, second_trajectory_reward) < self.delta_skip:
            y = Preference.SKIP.value
        elif abs(first_trajectory_reward - second_trajectory_reward) < self.delta_equal:
            y = Preference.EQUAL.value
        elif self._stochastic_preference(
            first_trajectory,
            second_trajectory,
            first_trajectory_reward,
            second_trajectory_reward,
        ):
            y = (
                Preference.FIRST.value
                if (1 - self.epsilon) > mistake
                else Preference.SECOND.value
            )
        else:
            y = (
                Preference.SECOND.value
                if (1 - self.epsilon) > mistake
                else Preference.FIRST.value
            )

        return y

    def _stochastic_preference(
        self,
        first_trajectory: Trajectory,
        second_trajectory: Trajectory,
        first_trajectory_reward: float,
        second_trajectory_reward: float,
    ) -> bool:
        if self.beta > 0:
            sum1 = (
                self.beta
                * torch.sum(
                    self._trajectory_weights(first_trajectory)
                    * first_trajectory.samples.ground_truth_rewards
                ).item()
            )
            sum2 = (
                self.beta
                * torch.sum(
                    self._trajectory_weights(second_trajectory)
                    * second_trajectory.samples.ground_truth_rewards
                ).item()
            )

            combine_trajectories = torch.tensor([sum1, sum2])
            p1, p2 = torch.softmax(combine_trajectories, dim=0).tolist()

            assert 0 <= p1 <= 1, f"p1={p1} is out of bounds"
            assert 0 <= p2 <= 1, f"p2={p2} is out of bounds"

            return p1 > p2
        elif self.beta == 0:
            # Use a random float to choose uniformly between the two trajectories
            return random.random() < 0.5
        else:
            return first_trajectory_reward > second_trajectory_reward

    def _trajectory_weights(self, trajectory: Trajectory) -> torch.Tensor:
        h = trajectory.samples.ground_truth_rewards.size(dim=0)
        timestep = (
            torch.arange(1, h + 1)
            .to(trajectory.samples.ground_truth_rewards.device)
            .unsqueeze(-1)
        )
        weights_first_trajectory = self.gamma ** (h - timestep)

        assert (
            weights_first_trajectory.shape
            == trajectory.samples.ground_truth_rewards.shape
        ), (
            f"Shape mismatch: weights {weights_first_trajectory.shape}, rewards {trajectory.samples.ground_truth_rewards.shape}"
        )

        return weights_first_trajectory


def give_pseudo_label_ensemble(
    obs: torch.Tensor,
    acts: torch.Tensor,
    tau: float,
    model: RewardNet,
):
    """
    Given an ensemble of observations and actions, computes the pseudo-label for the ensemble.

    Shapes:
        E = ensemble size
        B = batch size
        L = trajectory length
        O = obs dim
        A = act dim

    :param obs: observations of shape (E, 2, B, L, O) of the first and second trajectory
    :param acts: actions of shape (E, 2, B, L, A) of the first and second trajectory
    :param tau:
    :param model:
    :return: pseudo-labels of shape (E, B)
    """

    E, _, B, L, O = obs.shape
    _, _, _, _, A = acts.shape

    # Pseudo-labeling
    with torch.no_grad():
        obs_flat = obs.reshape(E, 2 * B * L, O)
        acts_flat = acts.reshape(E, 2 * B * L, A)

        # Forward pass through the ensemble
        r_ens = model.forward_ensemble(obs_flat, acts_flat)  # shape: (E, 2 * B * L, 1)
        r_ens = r_ens.reshape(E, 2, B, L)  # shape: (E, 2, B, L, 1)

        # Compute the preference probabilities
        probs = model.preference_prob_ensemble(r_ens) # shape: (E, B)

        # logging.debug("probs.shape: %s", probs.shape)

        # Compute the pseudo-label
        # Create a tensor filled with the skip label (-1.0)
        pseudo_labels = -1.0 * torch.ones_like(probs)

        # Precomputed scalar values
        first_val = float(Preference.FIRST.value)
        second_val = float(Preference.SECOND.value)

        # Assign values based on conditions
        pseudo_labels[probs > tau] = first_val
        pseudo_labels[probs < (1.0 - tau)] = second_val

        return pseudo_labels

def give_pseudo_label(
    traj1: Trajectory, traj2: Trajectory, tau: float, model: RewardNet
):
    # Pseudo-labeling
    with torch.no_grad():
        r1_u = model.forward(traj1.samples.observations, traj1.samples.actions)
        r2_u = model.forward(traj2.samples.observations, traj2.samples.actions)

        prob_u = model.preference_prob(r1_u, r2_u)

    if prob_u > tau:
        pseudo_label = Preference.FIRST.value
    elif prob_u < (1.0 - tau):
        pseudo_label = Preference.SECOND.value
    else:
        pseudo_label = Preference.SKIP.value

    return pseudo_label


def teacher_feedback_schedule(
    total_steps: int,
    num_sessions: int,
    schedule: str = "exponential",
    lambda_: float = 0.1,
) -> np.ndarray:
    """
    Generates the steps at which a teacher provides feedback during a training run.
    The feedback schedule can be either exponential or linear.

    :param total_steps: Total number of training steps
    :param num_sessions: Number of feedback sessions
    :param schedule: Feedback schedule to use ("exponential" or "linear")
    :param lambda_: Exponential rate for the feedback schedule if using the exponential schedule, smaller values
                    result in more feedback early on
    :return: A numpy array containing the steps at which feedback is provided
    """
    # Last session at the end of training will never provide additional feedback,
    # so we increment the number of sessions by 1 to make sure we actually get num_sessions effective feedback sessions
    num_sessions += 1
    if schedule == "exponential":
        i = np.arange(1, num_sessions + 1)
        denom = np.exp(lambda_ * num_sessions) - 1.0
        fraction = (np.exp(lambda_ * i) - 1.0) / denom
        feedback_steps = total_steps * fraction

    elif schedule == "linear":
        feedback_steps = np.linspace(0, total_steps, num_sessions + 1)[1:]

    else:
        raise ValueError(f"Unknown schedule: {schedule}")

    return feedback_steps.astype(int)


def plot_feedback_schedule(
    schedule: np.array,
    num_queries: int,
):
    """
    Plots the feedback schedule generated by the teacher_feedback_schedule function.
    :param schedule: The feedback schedule array
    :param num_queries: The total number of queries
    :return:
    """
    # Remove the last element since it's the total number of steps (and this is not a real feedback session)
    schedule = schedule[:-1]
    cumulative_feedback = np.linspace(0, num_queries, len(schedule))

    plt.plot(schedule, cumulative_feedback, marker="o")
    plt.title("Schedule: Total Steps vs. Cumulative Feedback Queries")
    plt.xlabel("Total Steps")
    plt.ylabel("Cumulative Feedback")
    return plt

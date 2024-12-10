import random
import torch

from replay_buffer import Trajectory
from enum import Enum

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

    def give_preference(self, first_trajectory: Trajectory, second_trajectory: Trajectory) -> float:
        first_trajectory_reward = first_trajectory.samples.ground_truth_rewards.sum().item()
        second_trajectory_reward = second_trajectory.samples.ground_truth_rewards.sum().item()
        mistake = random.random()

        if max(first_trajectory_reward, second_trajectory_reward) < self.delta_skip:
            y = Preference.SKIP.value
        elif abs(first_trajectory_reward - second_trajectory_reward) < self.delta_equal:
            y = Preference.EQUAL.value
        elif self._stochastic_preference(first_trajectory, second_trajectory, first_trajectory_reward, second_trajectory_reward):
            y = Preference.FIRST.value if (1 - self.epsilon) > mistake else Preference.SECOND.value
        else:
            y = Preference.SECOND.value if (1 - self.epsilon) > mistake else Preference.FIRST.value

        return y

    def _stochastic_preference(self, first_trajectory: Trajectory, second_trajectory: Trajectory, first_trajectory_reward, second_trajectory_reward) -> bool:
        if self.beta > 0:
            sum1 = self.beta * torch.sum(self._trajectory_weights(first_trajectory) * first_trajectory.samples.ground_truth_rewards).item()
            sum2 = self.beta * torch.sum(self._trajectory_weights(second_trajectory) * second_trajectory.samples.ground_truth_rewards).item()

            combine_trajectories = torch.tensor([sum1, sum2])
            p1, p2 = torch.softmax(combine_trajectories, dim=0).tolist()

            assert 0 <= p1 <= 1, f"p1={p1} is out of bounds"
            assert 0 <= p2 <= 1, f"p2={p2} is out of bounds"

            return True if p1 > p2 else False
        elif self.beta == 0:
            # Use a random float to choose uniformly between the two trajectories
            return random.random() < 0.5
        else:
            return True if first_trajectory_reward > second_trajectory_reward else False

    def _trajectory_weights(self, trajectory: Trajectory) -> torch.Tensor:
        h = trajectory.samples.ground_truth_rewards.size(dim=0)
        timestep = torch.arange(1, h + 1).to(trajectory.samples.ground_truth_rewards.device).unsqueeze(-1)
        weights_first_trajectory = self.gamma ** (h - timestep)

        assert weights_first_trajectory.shape == trajectory.samples.rewards.shape, \
            f"Shape mismatch: weights {weights_first_trajectory.shape}, rewards {trajectory.samples.ground_truth_rewards.shape}"

        return weights_first_trajectory
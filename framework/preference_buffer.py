from replay_buffer import Trajectory
import numpy as np


class PreferenceBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        # Each row contains: first_start_idx, first_end_idx, second_start_idx, second_end_idx, preference
        self.buffer = np.zeros((self.buffer_size, 5), dtype=np.float32)
        self.size = 0
        self.next_idx = 0

    def add(
        self,
        first_trajectory: Trajectory,
        second_trajectory: Trajectory,
        preference: float,
    ):
        """
        Adds a new preference to the buffer. If the buffer is full, it overwrites the oldest entry.

        :param first_trajectory: First trajectory being compared
        :param second_trajectory: Second trajectory being compared
        :param preference: Preference score (e.g., 1 if first is preferred, -1 if second is preferred, 0 for equal preference)
        """
        # Store the preference data
        self.buffer[self.next_idx] = [
            first_trajectory.replay_buffer_start_idx,
            first_trajectory.replay_buffer_end_idx,
            second_trajectory.replay_buffer_start_idx,
            second_trajectory.replay_buffer_end_idx,
            preference,
        ]

        # Update the size and next index
        self.size = min(self.size + 1, self.buffer_size)
        self.next_idx = (self.next_idx + 1) % self.buffer_size

    def sample(self, batch_size: int):
        """
        Samples a batch of preferences from the buffer.

        :param batch_size: Number of preferences to sample
        :return: A numpy array of shape (batch_size, 5) containing the sampled preferences
        """
        if self.size == 0:
            raise ValueError("Preference buffer is empty, cannot sample.")

        # Randomly select indices to sample
        indices = np.random.choice(self.size, batch_size, replace=False)
        return self.buffer[indices]

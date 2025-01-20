import numpy as np
from replay_buffer import Trajectory


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
        indices = np.random.choice(self.size, min(self.size, batch_size), replace=False)
        return self.buffer[indices]

    def contains(self, trajectory_1: Trajectory, trajectory_2: Trajectory, preference: float):
        """
        Check if the preference buffer contains a given combination of trajectories and preference.

        :param trajectory_1: First trajectory being checked
        :param trajectory_2: Second trajectory being checked
        :param preference: Preference score to check for
        :return: True if the preference exists in the buffer, False otherwise
        """
        for i in range(self.size):
            buffer_entry = self.buffer[i]

            if (buffer_entry[0] == trajectory_1.replay_buffer_start_idx and
                    buffer_entry[1] == trajectory_1.replay_buffer_end_idx and
                    buffer_entry[2] == trajectory_2.replay_buffer_start_idx and
                    buffer_entry[3] == trajectory_2.replay_buffer_end_idx and
                    buffer_entry[4] == preference):
                return True

        return False

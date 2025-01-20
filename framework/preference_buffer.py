import numpy as np
from replay_buffer import Trajectory


class PreferenceBuffer:
    def __init__(self, buffer_size, seed):
        self.buffer_size = buffer_size
        # Each row contains: first_start_idx, first_end_idx, second_start_idx, second_end_idx, preference
        self.buffer = np.zeros((self.buffer_size, 5), dtype=np.float32)
        self.size = 0
        self.next_idx = 0
        self.seed = seed
        np.random.seed(seed)

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

    def sample_batch(self, batch_size: int):
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

    def sample_minibatch(self, mini_batch_size: int):
        """
        Samples mini-batches of preferences from the buffer.

        :param mini_batch_size:
        :return: A list of numpy arrays, each of shape (mini_batch_size, 5), containing the sampled preferences
        """
        # Shuffle the buffer
        permutation = np.arange(0, self.size)
        np.random.shuffle(permutation)
        mini_batches = []

        number_of_batches = self.size // mini_batch_size
        for i in range(0, number_of_batches):
            indices = permutation[i * mini_batch_size : (i + 1) * mini_batch_size]
            mini_batches.append(self.buffer[indices])

        # Handle last mini-batch, if necessary
        if number_of_batches * mini_batch_size < self.size:
            indices = permutation[number_of_batches * mini_batch_size :]
            mini_batches.append(self.buffer[indices])

        return mini_batches

    def sample_all(self):
        """
        Returns all preferences in the buffer.
        """
        return self.buffer[: self.size]

    def sample(self, sample_strategy: str, mini_batch_size: int, batch_size: int):
        """
        Samples preferences from the buffer based on the specified strategy.
        :param sample_strategy: the sampling strategy for reward training, must be 'minibatch' ,'batch' or 'full'
        :param mini_batch_size: size of the mini-batch
        :param batch_size: size of the sampled batch
        :return:
        """
        if sample_strategy == "full":
            return [self.sample_all()]
        elif sample_strategy == "minibatch":
            return self.sample_minibatch(mini_batch_size)
        elif sample_strategy == "batch":
            return [self.sample_batch(batch_size)]
        else:
            raise ValueError(
                "Invalid batch_mode. Choose 'full', 'minibatch', or 'sample'."
            )

    def reset(self):
        """
        Resets the buffer.
        """
        self.size = 0
        self.next_idx = 0

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

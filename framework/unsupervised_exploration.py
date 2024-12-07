import numpy as np
from scipy.spatial import KDTree

class UnsupervisedExploration:
    def __init__(self, k=3):
        self.k = k
        self.visited_states = []

    def update_states(self, states):
        """
        Add a batch of states to the visited states.
        :param states: Batch of states, shape (num_envs, obs_dim)
        """
        if len(states.shape) != 2:
            raise ValueError(f"States must have shape (num_envs, obs_dim), but got {states.shape}")
        self.visited_states.extend(states.tolist())  # Add batch of states

    def compute_intrinsic_rewards(self, states):
        """
        Compute intrinsic rewards for a batch of states.
        :param states: Batch of states, shape (num_envs, obs_dim)
        :return: Intrinsic rewards, shape (num_envs,)
        """
        if len(self.visited_states) < self.k + 1:
            # Not enough points to calculate k-NN distances
            return np.zeros(states.shape[0])

        # Build KDTree with normalized visited states
        visited_states = np.array(self.visited_states)
        visited_states = (visited_states - visited_states.mean(axis=0)) / (visited_states.std(axis=0) + 1e-8)
        tree = KDTree(visited_states)

        # Normalize the input states
        states = (states - visited_states.mean(axis=0)) / (visited_states.std(axis=0) + 1e-8)

        # Query the KDTree for distances
        distances, _ = tree.query(states, k=self.k + 1)
        distances_to_kth_neighbor = distances[:, -1]

        # Clip distances to avoid log(0)
        distances_to_kth_neighbor = np.clip(distances_to_kth_neighbor, 1e-8, None)

        # Compute intrinsic rewards
        intrinsic_rewards = np.log(distances_to_kth_neighbor)

        # Debugging: Check for inf or NaN values
        if not np.all(np.isfinite(intrinsic_rewards)):
            print(f"Non-finite intrinsic rewards detected: {intrinsic_rewards}")
            intrinsic_rewards = np.nan_to_num(intrinsic_rewards, nan=0.0, posinf=1e6, neginf=-1e6)

        return intrinsic_rewards




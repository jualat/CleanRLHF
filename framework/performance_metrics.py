import numpy as np


class PerformanceMetrics:
    """
    A class for calculating performance metrics for reinforcement learning.
    Includes:
    - Pearson correlation coefficient calculation
    """

    def __init__(self):
        self.predictions = []
        self.ground_truths = []

    def add_rewards(self, predictions, ground_truths):
        """
        Adds a batch of predictions and corresponding ground truth values.
        The size of the predictions must be the same as the size of the ground truths.

        Args:
            predictions (list or np.array): Predicted values.
            ground_truths (list or np.array): Corresponding ground truth values.
        """
        self.predictions.extend(predictions)
        self.ground_truths.extend(ground_truths)

    def reset(self):
        """
        Reset the stored data for predictions and ground truths.
        """
        self.predictions = []
        self.ground_truths = []

    def compute_pearson_correlation(self):
        """
        Compute the Pearson correlation coefficient between predictions and ground truths.

        Returns:
            float: Pearson correlation coefficient.
        """
        if len(self.predictions) == 0 or len(self.ground_truths) == 0:
            raise ValueError("No data available to compute Pearson correlation.")
        if len(self.predictions) != len(self.ground_truths):
            raise ValueError("Mismatch in the length of predictions and ground truths.")

        dev_predictions = self.predictions - np.mean(self.predictions)
        dev_ground_truths = self.ground_truths - np.mean(self.ground_truths)

        numerator = np.sum(dev_predictions * dev_ground_truths)
        denominator = np.sqrt(
            np.sum(dev_predictions**2) * np.sum(dev_ground_truths**2)
        )

        if denominator == 0:
            return 0

        return numerator / denominator

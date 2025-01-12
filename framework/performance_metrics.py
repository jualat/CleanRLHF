import logging
import time

import numpy as np
from torch.utils.tensorboard import SummaryWriter


class PerformanceMetrics:
    """
    A class for calculating performance metrics for reinforcement learning.
    Includes:
    - Pearson correlation coefficient calculation
    """

    def __init__(self, run_name, args, evaluate):
        self.predictions = []
        self.ground_truths = []

        self.writer = SummaryWriter(f"runs/{run_name}")
        self.writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s"
            % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )

        self.evaluate = evaluate

    def add_rewards(self, predictions, ground_truths):
        """
        Adds a batch of predictions and corresponding ground truth values.
        The size of the predictions must be the same as the size of the ground truths.

        Args:
            predictions (list or np.array): Predicted values.
            ground_truths (list or np.array): Corresponding ground truth values.
        """
        self.predictions.append(predictions[0])
        self.ground_truths.append(ground_truths[0])

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

    def log_exploration_metrics(
        self,
        explore_step,
        intrinsic_reward,
        knn_estimator,
        terminations,
        truncations,
        start_time,
    ):
        """
        Log exploration metrics to TensorBoard.
        :param explore_step: The current exploration step
        :param intrinsic_reward: The intrinsic reward
        :param knn_estimator: The KNN estimator
        :param terminations: The terminations
        :param truncations: The truncations
        :param start_time: The start time
        :return:
        """
        if explore_step % 100 == 0:
            self.writer.add_scalar(
                "exploration/intrinsic_reward_mean",
                intrinsic_reward.mean(),
                explore_step,
            )

            self.writer.add_scalar(
                "exploration/state_coverage",
                len(knn_estimator.visited_states),
                explore_step,
            )
            logging.debug(f"SPS: {int(explore_step / (time.time() - start_time))}")
            self.writer.add_scalar(
                "exploration/SPS",
                int(explore_step / (time.time() - start_time)),
                explore_step,
            )
            logging.debug(f"Exploration step: {explore_step}")

        self.writer.add_scalar(
            "exploration/dones",
            terminations.sum() + truncations.sum(),
            explore_step,
        )

    def log_reward_metrics(self, rewards, groundTruthRewards, global_step):
        """
        Log reward metrics to TensorBoard.
        :param rewards: The predicted rewards
        :param groundTruthRewards: The ground truth rewards
        :param global_step: The global step
        :return:
        """
        self.writer.add_scalar(
            "charts/predicted_rewards",
            rewards[0],
            global_step,
        )
        self.writer.add_scalar(
            "charts/groudTruthRewards",
            groundTruthRewards[0],
            global_step,
        )

    def log_training_metrics(
        self,
        global_step,
        args,
        qf1_a_values,
        qf2_a_values,
        qf1_loss,
        qf2_loss,
        qf_loss,
        actor_loss,
        alpha,
        alpha_loss,
        start_time,
    ):
        """
        Log training metrics to TensorBoard.
        :param global_step: The gloabl step
        :param args: The arguments for the run
        :param qf1_a_values: The q function 1 action values
        :param qf2_a_values: The q function action values
        :param qf1_loss: The q function 1 loss
        :param qf2_loss: The q function 2 loss
        :param qf_loss: The overall q function loss
        :param actor_loss: The actor loss
        :param alpha: The run parameter alpha
        :param alpha_loss: The alpha loss
        :param start_time: The start time
        :return:
        """
        self.writer.add_scalar(
            "losses/qf1_values", qf1_a_values.mean().item(), global_step
        )
        self.writer.add_scalar(
            "losses/qf2_values", qf2_a_values.mean().item(), global_step
        )
        self.writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
        self.writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
        self.writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
        self.writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
        self.writer.add_scalar("losses/alpha", alpha, global_step)
        logging.debug(f"SPS: {int(global_step / (time.time() - start_time))}")
        self.writer.add_scalar(
            "charts/SPS",
            int(global_step / (time.time() - start_time)),
            global_step,
        )
        if args.autotune:
            self.writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)
        self.writer.add_scalar(
            "charts/pearson_correlation",
            self.compute_pearson_correlation(),
            global_step,
        )
        self.reset()

    def log_info_metrics(self, info, global_step, allocated, reserved, cuda):
        """
        Log information metrics to TensorBoard.
        :param info: The info dictionary
        :param global_step: The global step
        :param allocated: The allocated memory for a cuda GPU
        :param reserved: The reserved memory for a cuda GPU
        :param cuda: If a cuda GPU is being used
        :return:
        """
        logging.info(
            f"global_step={global_step}, episodic_return={info['episode']['r']}"
        )
        self.writer.add_scalar(
            "charts/episodic_return", info["episode"]["r"], global_step
        )
        self.writer.add_scalar(
            "charts/episodic_length", info["episode"]["l"], global_step
        )

        if cuda:
            logging.debug(f"Allocated cuda memory: {allocated / (1024 ** 2)} MB")
            logging.debug(f"Reserved cuda memory: {reserved / (1024 ** 2)} MB")
            self.writer.add_scalar(
                "hardware/cuda_memory",
                allocated / (1024**2),
                global_step,
            )

    def log_evaluate_metrics(self, global_step, eval_dict):
        """
        Log evaluation metrics to TensorBoard.
        :param global_step: The global step
        :param eval_dict: The evaluation dictionary computed in train()
        :return:
        """
        self.writer.add_scalar("evaluate/mean", eval_dict["mean_reward"], global_step)
        self.writer.add_scalar("evaluate/std", eval_dict["std_reward"], global_step)
        self.writer.add_scalar("evaluate/max", eval_dict["max_reward"], global_step)
        self.writer.add_scalar("evaluate/min", eval_dict["min_reward"], global_step)
        self.writer.add_scalar(
            "evaluate/median", eval_dict["median_reward"], global_step
        )

    def log_reward_net_losses(self,
                              train_ensemble_loss,
                              train_total_loss,
                              val_ensemble_loss,
                              val_avg_loss,
                              global_step,
                              batch_size):
        """
        Log reward net losses to TensorBoard.
        :param train_ensemble_loss: The ensemble loss after training the reward_net
        :param train_total_loss: The total loss after training the reward_net
        :param val_ensemble_loss: The ensemble loss after validating the reward_net
        :param val_avg_loss: The average loss after validating the reward_net
        :param global_step: The global step
        :param batch_size: The batch size the reward net was trained on
        :return:
        """
        self.writer.add_scalar("losses/train_reward_loss", train_ensemble_loss.item(), global_step)
        self.writer.add_scalar(
            "losses/train_total_loss", train_total_loss / (batch_size * 0.5), global_step
        )

        self.writer.add_scalar("losses/val_reward_loss", val_ensemble_loss.item(), global_step)
        self.writer.add_scalar(
            "losses/val_avg_loss", val_avg_loss, global_step
        )

    def close(self):
        self.writer.close()

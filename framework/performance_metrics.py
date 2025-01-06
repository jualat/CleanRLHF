import numpy as np
from torch.utils.tensorboard import SummaryWriter
import logging
import time


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

    def log_training_metrics(
        self,
        global_step,
        args,
        rewards,
        groundTruthRewards,
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

        if global_step % 100 == 0:
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
                self.writer.add_scalar(
                    "losses/alpha_loss", alpha_loss.item(), global_step
                )
            self.writer.add_scalar(
                "charts/pearson_correlation",
                self.compute_pearson_correlation(),
                global_step,
            )
            self.reset()

    def log_info_metrics(self, info, global_step, allocated, reserved, cuda):
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
            logging.info(f"Allocated cuda memory: {allocated / (1024 ** 2)} MB")
            logging.info(f"Reserved cuda memory: {reserved / (1024 ** 2)} MB")
            self.writer.add_scalar(
                "hardware/cuda_memory",
                allocated / (1024**2),
                global_step,
            )

    def log_evaluate_metrics(self, global_step, args, eval_dict):
        if global_step % args.evaluation_frequency == 0 and (
            global_step != 0 or args.exploration_load or args.unsupervised_exploration
        ):
            self.evaluate.plot(eval_dict, global_step)
            self.writer.add_scalar(
                "evaluate/mean", eval_dict["mean_reward"], global_step
            )
            self.writer.add_scalar("evaluate/std", eval_dict["std_reward"], global_step)
            self.writer.add_scalar("evaluate/max", eval_dict["max_reward"], global_step)
            self.writer.add_scalar("evaluate/min", eval_dict["min_reward"], global_step)
            self.writer.add_scalar(
                "evaluate/median", eval_dict["median_reward"], global_step
            )
        eval_dict = self.evaluate.evaluate_policy(
            episodes=args.evaluation_episodes,
            step=args.total_timesteps,
            actor=self.evaluate.actor,
            render=True,
            track=args.track,
        )
        self.evaluate.plot(eval_dict, args.total_timesteps)
        self.writer.add_scalar(
            "evaluate/mean", eval_dict["mean_reward"], args.total_timesteps
        )
        self.writer.add_scalar(
            "evaluate/std", eval_dict["std_reward"], args.total_timesteps
        )
        self.writer.add_scalar(
            "evaluate/max", eval_dict["max_reward"], args.total_timesteps
        )
        self.writer.add_scalar(
            "evaluate/min", eval_dict["min_reward"], args.total_timesteps
        )
        self.writer.add_scalar(
            "evaluate/median", eval_dict["median_reward"], args.total_timesteps
        )

    def close(self):
        self.writer.close()

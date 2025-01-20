import json
import logging
import time

import numpy as np
import requests
from sampling import disagreement_sampling, entropy_sampling, uniform_sampling, sample_trajectories
from tqdm import tqdm, trange
from video_recorder import retrieve_trajectory_by_video_name


def collect_feedback(
        mode,
        feedback_server_url,
        replay_buffer,
        pref_buffer,
        run_name,
        preference_sampling,
        teacher_feedback_num_queries_per_session,
        trajectory_length,
        train_pref_buffer,
        val_pref_buffer,
        reward_net_val_split,
        sim_teacher=None,
        reward_net=None,
        feedback_file=None,
        capture_video=True,
        video_recorder=None,
):
    """
    Collect feedback based on the given mode.

    Parameters:
        mode (str):
            Specifies the feedback mode. Possible values:
            - "simulated_teacher": Uses a simulated teacher to generate trajectory preferences.
            - "human_feedback": Requests feedback from a human evaluator via a feedback server.

        feedback_server_url (str):
            Url including port of the feedback server to use

        replay_buffer (ReplayBuffer):
            Replay buffer containing the trajectories available for sampling.

        pref_buffer (PreferenceBuffer):
            Preference buffer where the collected feedback will be recorded.

        run_name (str):


        preference_sampling (str):
            The trajectory-pair sampling strategy. Options include:
            - "disagree": Pairs where the reward network shows high disagreement.
            - "entropy": Pairs with high uncertainty or entropy in predicted rewards.
            - Any other value defaults to uniform random sampling.

        teacher_feedback_num_queries_per_session (int):
            The number of feedback queries to collect per session.

        sim_teacher (SimulatedTeacher, optional):
            A simulated teacher object used to generate preferences automatically.
            Required for the "simulated_teacher" mode.

        reward_net (RewardNet, optional):
            Reward network used for advanced sampling strategies like "disagree" or "entropy"
            (only used if applicable).

        trajectory_length (int):
            The number of steps in each sampled trajectory.

        capture_video (bool, optional):
            Whether to record videos of sampled trajectory pairs. Defaults to True.

        video_recorder (VideoRecorder, optional):
            An object responsible for recording trajectory videos.
            Used only when `capture_video` is True or "human_feedback" mode is selected.
    Returns:
        PreferenceBuffer: Updated preference buffer with collected feedback.
        :param mode:
        :param video_recorder:
        :param capture_video:
        :param feedback_file:
        :param reward_net:
        :param sim_teacher:
        :param val_pref_buffer:
        :param trajectory_length:
        :param teacher_feedback_num_queries_per_session:
        :param preference_sampling:
        :param run_name:
        :param pref_buffer:
        :param replay_buffer:
        :param feedback_server_url:
            Url including port of the feedback server to use
        :param next_session_idx:
        :param reward_net_val_split:
        :param train_pref_buffer:

    """

    if mode == "simulated":
        if not sim_teacher:
            raise ValueError(
                "sim_teacher must be provided for 'simulated_teacher' mode."
            )
        logging.info("Collecting feedback from simulated teacher")

        for i in trange(
                teacher_feedback_num_queries_per_session,
                desc="Queries",
                unit="queries",
        ):
            # Sample trajectories from replay buffer to query teacher
            first_trajectory, second_trajectory = sample_trajectories(
                replay_buffer, preference_sampling, reward_net, trajectory_length
            )

            # Create video of the two trajectories. For now, we only render if capture_video is True.
            # If we have a human teacher, we would render the video anyway and ask the teacher to compare the two trajectories.
            if capture_video:
                video_recorder.record_trajectory(first_trajectory, run_name)
                video_recorder.record_trajectory(second_trajectory, run_name)

            # Query instructor (normally a human who decides which trajectory is better, here we use ground truth)
            preference = sim_teacher.give_preference(
                first_trajectory, second_trajectory
            )

            # Trajectories are not added to the buffer if neither segment demonstrates the desired behavior
            if preference is None:
                continue

            # Store preferences
            if np.random.rand() < (
                    1 - reward_net_val_split
            ):  # 1 - (Val Split)% for training
                train_pref_buffer.add(
                    first_trajectory, second_trajectory, preference
                )
            else:  # (Val Split)% for validation
                val_pref_buffer.add(
                    first_trajectory, second_trajectory, preference
                )

        # todo rm
        # for i in trange(
        #         teacher_feedback_num_queries_per_session,
        #         desc="Queries",
        #         unit="queries",
        # ):
        #     # Sample trajectories from replay buffer to query teacher
        #     if preference_sampling == "disagree":
        #         first_trajectory, second_trajectory = disagreement_sampling(
        #             replay_buffer, reward_net, trajectory_length
        #         )
        #     elif preference_sampling == "entropy":
        #         first_trajectory, second_trajectory = entropy_sampling(
        #             replay_buffer, reward_net, trajectory_length
        #         )
        #     else:
        #         first_trajectory, second_trajectory = uniform_sampling(
        #             replay_buffer, trajectory_length
        #         )
        #     if capture_video:
        #         video_recorder.record_trajectory(first_trajectory, run_name)
        #         video_recorder.record_trajectory(second_trajectory, run_name)
        #
        # for _ in range(teacher_feedback_num_queries_per_session):
        #
        #     # Sample trajectories (modify to use your preferred sampling strategy)
        #     first_trajectory, second_trajectory = uniform_sampling(
        #         replay_buffer, trajectory_length
        #     )
        #     # Query simulated teacher
        #     preference = sim_teacher.give_preference(
        #         first_trajectory, second_trajectory
        #     )
        #     if preference is not None:
        #         pref_buffer.add(first_trajectory, second_trajectory, preference)

    elif mode == "human":
        logging.info("Collecting human feedback")
        human_feedback_preparation(
            feedback_server_url,
            teacher_feedback_num_queries_per_session,
            preference_sampling,
            replay_buffer,
            reward_net,
            run_name,
            trajectory_length,
            video_recorder,
        )

        try:
            collected_feedback = 0
            while collected_feedback < teacher_feedback_num_queries_per_session:
                with tqdm(
                        total=teacher_feedback_num_queries_per_session,
                        initial=collected_feedback,
                        desc="Collecting Feedback",
                        unit="feedback",
                ) as pbar:
                    response = requests.get(feedback_server_url + "/api/get_feedback/" + run_name)
                    if response.status_code == 200:
                        feedback_items = response.json()
                        if feedback_items:
                            for feedback in feedback_items:
                                first_trajectory = retrieve_trajectory_by_video_name(replay_buffer,
                                                                                     feedback["trajectory_1"])
                                second_trajectory = retrieve_trajectory_by_video_name(replay_buffer,
                                                                                      feedback["trajectory_2"])
                                preference = feedback["preference"]
                                if preference == "-1": preference = None

                                if not pref_buffer.contains(first_trajectory, second_trajectory, preference):
                                    pref_buffer.add(first_trajectory, second_trajectory, preference)
                                    logging.info(
                                        f"Adding to pref_buffer: {first_trajectory, second_trajectory, preference}")
                                    collected_feedback += 1
                                    pbar.update(1)
                    else:
                        logging.debug("Could not fetch feedback; retrying...")
                    time.sleep(5)

        except KeyboardInterrupt:
            logging.error("Human feedback process interrupted.")

    elif mode == "file":
        if not feedback_file:
            raise ValueError("feedback_file must be provided for 'file' mode.")
        logging.info(f"Loading feedback from file: {feedback_file}...")
        with open(feedback_file, "r") as f:
            feedback_data = json.load(f)
        for feedback in feedback_data:
            first_trajectory = feedback["trajectory_1"]
            second_trajectory = feedback["trajectory_2"]
            preference = feedback["preference"]
            pref_buffer.add(first_trajectory, second_trajectory, preference)

    else:
        raise ValueError(f"Unknown feedback mode: {mode}")

    logging.info(f"Collected feedback with mode: {mode}")
    return pref_buffer


def human_feedback_preparation(
        feedback_server_url,
        teacher_feedback_num_queries_per_session,
        preference_sampling,
        replay_buffer,
        reward_net,
        run_name,
        trajectory_length,
        video_recorder,
):
    sampled_trajectory_videos = []
    for i in trange(
            teacher_feedback_num_queries_per_session,
            desc="Feedback pairs prepared:",
            # unit="",
    ):
        """
        Prepares human feedback data by sampling trajectory pairs, recording their videos,
        and sending them to the server for human evaluation.

        Parameters:
        ----------
        num_queries : int
            The number of trajectory queries to prepare for human feedback.

        preference_sampling : str
            The sampling strategy to use for selecting trajectory pairs. Options:
            - "disagree": Use trajectories where the reward network has high disagreement.
            - "entropy": Use trajectories with high entropy in the reward distribution.
            - Any other value: Defaults to uniform sampling from the replay buffer.

        replay_buffer : ReplayBuffer
            The replay buffer containing experience data to sample trajectories from.

        reward_net : RewardNetwork
            The reward network used for disagreement or entropy-based sampling
            (only used when preference_sampling is "disagree" or "entropy").

        run_name : str
            A name or identifier for the current run. Used for recording trajectory videos.

        trajectory_length : int
            The length of the trajectories to sample.

        video_recorder : VideoRecorder
            An object responsible for recording trajectory videos to the file system.
            It should expose a `record_trajectory` method that returns the recorded
            video file path and corresponding `env_id`.

        Returns:
        --------
        None
            The method does not return anything. It prepares and sends trajectory videos as a
            POST request to a server.

        Workflow:
        ---------
        1. **Sample Trajectories**:
           - Based on the `preference_sampling` strategy, it samples trajectory pairs
             using disagreement sampling, entropy sampling, or uniform sampling.

        2. **Record Videos**:
           - Records videos of each sampled trajectory using the `video_recorder` object.
           - The videos are stored locally, and the paths are retrieved.

        3. **Send Data to Server**:
           - Sends the generated video filenames as a `POST` request to the server
             under the corresponding `env_id` for human evaluation.

        Example:
        --------
        prepare_human_feedback_data(
            num_queries=50,
            preference_sampling="disagree",
            replay_buffer=replay_buffer,
            reward_net=reward_net,
            run_name="experiment_01",
            sampled_trajectories=[],
            trajectory_length=100,
            video_recorder=video_recorder
        )
        """
        if preference_sampling == "disagree":
            first_trajectory, second_trajectory = disagreement_sampling(
                replay_buffer, reward_net, trajectory_length
            )
        elif preference_sampling == "entropy":
            first_trajectory, second_trajectory = entropy_sampling(
                replay_buffer, reward_net, trajectory_length
            )
        else:
            first_trajectory, second_trajectory = uniform_sampling(
                replay_buffer, trajectory_length
            )
        first_trajectory_video = video_recorder.record_trajectory(
            first_trajectory, run_name
        )
        if not first_trajectory_video:
            logging.error(
                "Failed to record video for the first trajectory. Skipping this pair..."
            )
            continue  # Skip this trajectory pair if there is no video

        second_trajectory_video = video_recorder.record_trajectory(
            second_trajectory, run_name
        )
        if not second_trajectory_video:
            logging.error(
                "Failed to record video for the second trajectory. Skipping this pair..."
            )
            continue  # Skip this trajectory pair if there is no video

        sampled_trajectory_videos.append(
            [first_trajectory_video, second_trajectory_video]
        )

    try:
        payload = {"video_pairs": sampled_trajectory_videos}
        response = requests.post(
            f"{feedback_server_url}/api/add_videos/{run_name}", json=payload
        )

        logging.debug(f"Response Status Code: {response.status_code}")
        logging.debug(f"Response Content: {response.text}")

        if response.status_code == 200:
            response_data = response.json()
            logging.debug(f"Server Response: {response_data}")
        else:
            logging.error(f"Failed to add videos: {response.text}")

    except json.JSONDecodeError:
        logging.error("Invalid JSON response received from server.")
    except requests.RequestException as e:
        logging.error(f"An error occurred while sending the videos: {str(e)}")

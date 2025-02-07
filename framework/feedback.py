import json
import logging
import time

import numpy as np
import pygame
import requests
from preference_buffer import PreferenceBuffer
from replay_buffer import ReplayBuffer
from sampling import sample_trajectories
from tqdm import tqdm, trange
from video_recorder import retrieve_trajectory_by_video_name


def collect_feedback(
    mode,
    feedback_server_url,
    replay_buffer: ReplayBuffer,
    run_name,
    teacher_feedback_num_queries_per_session,
    trajectory_length,
    train_pref_buffer: PreferenceBuffer,
    val_pref_buffer: PreferenceBuffer,
    reward_net_val_split,
    preference_sampling="disagree",
    sim_teacher=None,
    reward_net=None,
    feedback_file=None,
    capture_video=True,
    render_mode="simulated",
    video_recorder=None,
    play_sounds=False,
):
    """
    Collects feedback for trajectories from either a simulated teacher, human interaction or a file.
    The function handles querying various sources, manages preferences,
    and updates training and validation buffers as appropriate.

    :param mode: Specifies the feedback collection mode. It can be one of the following:
        - "simulated": Collect feedback using a simulated teacher.
        - "human": Collect feedback from a human teacher via a feedback server.
        - "file": Load feedback from a pre-recorded file.
    :type mode: str

    :param feedback_server_url: URL of the feedback server for hosting human feedback collection. Required only
        if mode is set to "human".
    :type feedback_server_url: str, optional

    :param replay_buffer: A buffer containing trajectories from which samples are drawn for feedback.
    :type replay_buffer: object

    :param run_name: Unique identifier for the current feedback collection session.
    :type run_name: str

    :param preference_sampling: A strategy to select trajectory pairs for sampling and feedback.
    :type preference_sampling: str

    :param teacher_feedback_num_queries_per_session: Total number of feedback queries to perform
        during one collection session.
    :type teacher_feedback_num_queries_per_session: int

    :param trajectory_length: The length or number of steps in one trajectory.
    :type trajectory_length: int

    :param train_pref_buffer: Buffer to store preferences for training purposes.
    :type train_pref_buffer: object

    :param val_pref_buffer: Buffer to store preferences for validation purposes.
    :type val_pref_buffer: object

    :param reward_net_val_split: Fraction of collected preferences to dedicate for validation.
    :type reward_net_val_split: float

    :param sim_teacher: A simulated teacher providing preferences based on trajectory pairs.
        Required if mode is set to "simulated".
    :type sim_teacher: object, optional

    :param reward_net: Reward network used for evaluating trajectories during preference query.
    :type reward_net: object, optional

    :param feedback_file: Path to a file containing pre-recorded feedback to load. Required only if
        mode is set to "file".
    :type feedback_file: str, optional

    :param capture_video: If True, videos of sampled trajectories will be captured for feedback.
    :type capture_video: bool

    :param render_mode: Specifies the mode for rendering sampled trajectories. By default, this is
        set to "simulated". In "human" mode, trajectories are rendered for humans to annotate.
    :type render_mode: str

    :param video_recorder: Object responsible for recording videos of sampled trajectories.
        Used only if `capture_video` or mode `human` is enabled.
    :type video_recorder: object, optional

    :param play_sounds: If True, plays sound when feedback is required.
    :type video_recorder: object, optional

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
            position=2,
            leave=False,
        ):
            # Sample trajectories from replay buffer to query teacher
            first_trajectory, second_trajectory = sample_trajectories(
                replay_buffer, preference_sampling, reward_net, trajectory_length
            )

            # Create video of the two trajectories. For now, we only render if capture_video is True.
            # If we have a human teacher, we would render the video anyway and ask the teacher to compare the two trajectories.
            if capture_video and render_mode != "human":
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
                train_pref_buffer.add(first_trajectory, second_trajectory, preference)
            else:  # (Val Split)% for validation
                val_pref_buffer.add(first_trajectory, second_trajectory, preference)

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
        if play_sounds:
            pygame.mixer.music.load("sound.wav")
            pygame.mixer.music.play()
        try:
            collected_feedback = 0
            with tqdm(
                total=teacher_feedback_num_queries_per_session,
                initial=collected_feedback,
                desc="Collecting Feedback",
                unit="feedback",
                position=2,
                leave=False,
            ) as pbar:
                while collected_feedback < teacher_feedback_num_queries_per_session:
                    response = requests.get(
                        feedback_server_url + "/api/get_feedback/" + run_name
                    )
                    if response.status_code == 200:
                        feedback_items = response.json()
                        if feedback_items:
                            for feedback in feedback_items:
                                first_trajectory = retrieve_trajectory_by_video_name(
                                    replay_buffer, feedback["trajectory_1"]
                                )
                                second_trajectory = retrieve_trajectory_by_video_name(
                                    replay_buffer, feedback["trajectory_2"]
                                )
                                preference = feedback["preference"]
                                if preference == -1:
                                    preference = None
                                    collected_feedback += 1
                                    pbar.update(1)

                                if (
                                    not (
                                        val_pref_buffer.contains(
                                            first_trajectory,
                                            second_trajectory,
                                            preference,
                                        )
                                        or train_pref_buffer.contains(
                                            first_trajectory,
                                            second_trajectory,
                                            preference,
                                        )
                                    )
                                ) and preference is not None:
                                    # Store preferences
                                    if np.random.rand() < (
                                        1 - reward_net_val_split
                                    ):  # 1 - (Val Split)% for training
                                        train_pref_buffer.add(
                                            first_trajectory,
                                            second_trajectory,
                                            preference,
                                        )
                                    else:  # (Val Split)% for validation
                                        val_pref_buffer.add(
                                            first_trajectory,
                                            second_trajectory,
                                            preference,
                                        )
                                    collected_feedback += 1
                                    pbar.update(1)

                    elif response.status_code == 422:
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
            if np.random.rand() < (
                1 - reward_net_val_split
            ):  # 1 - (Val Split)% for training
                train_pref_buffer.add(first_trajectory, second_trajectory, preference)
            else:  # (Val Split)% for validation
                val_pref_buffer.add(first_trajectory, second_trajectory, preference)

    else:
        raise ValueError(f"Unknown feedback mode: {mode}")
    logging.debug(f"Collected feedback with mode: {mode}")


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
    """
    Prepares human feedback data by sampling trajectory pairs, recording their videos,
    and sending them to the server for human evaluation.

    Parameters:
    ----------
    feedback_server_url: str
        The url of the feedback server.

    teacher_feedback_num_queries_per_session : int
        The number of trajectory queries to prepare for human feedback.

    preference_sampling : str
        The sampling strategy to use for selecting trajectory pairs. Options:
        - "disagree": Use trajectories where the reward network has high disagreement.
        - "entropy": Use trajectories with high entropy in the reward distribution.
        - "uniform: Use random trajectories. This is the default strategy.

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
    """
    sampled_trajectory_videos = []
    for i in trange(
        teacher_feedback_num_queries_per_session,
        desc="Feedback pairs prepared",
        position=2,
        leave=False,
    ):
        first_trajectory, second_trajectory = sample_trajectories(
            replay_buffer, preference_sampling, reward_net, trajectory_length
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

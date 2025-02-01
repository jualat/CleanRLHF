import argparse
import logging
import os
import threading

from flask import Flask, jsonify, render_template, request, send_from_directory
from flask_cors import CORS

# Flask app setup
app = Flask(__name__)
CORS(app)

VIDEO_FOLDER = "../framework/videos"
sampled_videos = {}
feedback_buffers = {}
lock = threading.Lock()


def safe_path_join(base, *paths):
    """
    Safely join one or more path components to the base directory.
    Prevents directory traversal attacks by ensuring the final path is within the base directory.
    """
    final_path = os.path.join(base, *paths)
    if not os.path.commonprefix(
        [os.path.abspath(final_path), os.path.abspath(base)]
    ) == os.path.abspath(base):
        raise ValueError("Attempted directory traversal attack detected.")
    return final_path


@app.route("/")
def index():
    """Serve the web interface to view and compare videos."""
    return render_template("index.html")


@app.route("/video/<path:filename>")
def fetch_video(filename):
    """Endpoint to serve the video files."""
    try:
        full_path = safe_path_join(VIDEO_FOLDER, filename)
        directory, file_name = os.path.split(full_path)
        return send_from_directory(directory, file_name)
    except ValueError as ve:
        logging.error(f"Security error: {ve}")
        return jsonify({"status": "error", "message": "Invalid file path."}), 400
    except Exception as e:
        logging.error(f"Failed to fetch video: {e}")
        return (
            jsonify({"status": "error", "message": f"File not found: {filename} "}),
            404,
        )


@app.route("/api/add_videos/<env_id>", methods=["POST"])
def add_videos(env_id):
    """
    Add new sampled video pairs to the stack for the given env_id.
    Expects a JSON input with a list of video pairs (filenames of the videos).
    Example Input: { "video_pairs": [["video1.mp4", "video2.mp4"], ["video3.mp4", "video4.mp4"]] }
    """
    data = request.json

    if not data or "video_pairs" not in data:
        return (
            jsonify(
                {
                    "status": "error",
                    "message": "Missing or invalid input. Please provide video_pairs as a list of pairs.",
                }
            ),
            400,
        )

    if not isinstance(data["video_pairs"], list) or not all(
        isinstance(pair, list) and len(pair) == 2 for pair in data["video_pairs"]
    ):
        return (
            jsonify(
                {
                    "status": "error",
                    "message": "Invalid video_pairs format. Must be a list of pairs (each pair has exactly 2 elements).",
                }
            ),
            400,
        )

    with lock:
        if env_id not in sampled_videos:
            sampled_videos[env_id] = []
        sampled_videos[env_id].extend(data["video_pairs"])

    return (
        jsonify(
            {
                "status": "success",
                "message": f"Added {len(data['video_pairs'])} video pairs to env_id {env_id}.",
                "total_pairs": len(sampled_videos[env_id]),
            }
        ),
        200,
    )


@app.route("/api/get_videos", methods=["GET"])
def get_videos():
    """
    Endpoint to get the next video filenames for comparison.
    Searches through all env_ids in sampled_videos, skipping pairs that already have feedback.
    """
    with lock:
        for run_name, video_stack in sampled_videos.items():
            while video_stack:
                video_pair = video_stack.pop(0)

                # Check if this pair already has feedback
                feedback_exists = any(
                    feedback["trajectory_1"] == video_pair[0]
                    and feedback["trajectory_2"] == video_pair[1]
                    for feedback in feedback_buffers.get(run_name, [])
                )

                if not feedback_exists:
                    # If the pair has no feedback, return this pair and put it to the end of the list (this is only done
                    # to get feedback for this pair later if no feedback is given by the person who requested this pair)
                    video_stack.append(video_pair)
                    return (
                        jsonify(
                            {
                                "video1": f"/video/{video_pair[0]}",
                                "video2": f"/video/{video_pair[1]}",
                                "run_name": run_name,
                            }
                        ),
                        200,
                    )
                else:
                    logging.info(
                        f"Skipped pair {video_pair} for run: {run_name} as feedback already exists."
                    )
        return "", 204


@app.route("/api/submit_feedback/<run_name>", methods=["POST"])
def submit_feedback(run_name):
    """
    Endpoint to collect user feedback for a specific environment ID (env_id).
    Expects JSON: { "trajectory_1": ..., "trajectory_2": ..., "preference": ... }
    Ensures atomic checks and updates to avoid race conditions.
    """
    data = request.json
    if not data:
        return jsonify({"error": "No feedback data provided"}), 400

    # Validate feedback format
    if "trajectory_1" in data and "trajectory_2" in data and "preference" in data:
        trajectory_1 = data["trajectory_1"].removeprefix("/video/")
        trajectory_2 = data["trajectory_2"].removeprefix("/video/")
        pair_to_remove = [trajectory_1, trajectory_2]

        with lock:
            if run_name not in feedback_buffers:
                feedback_buffers[run_name] = []

            for feedback in feedback_buffers[run_name]:
                if (
                    feedback["trajectory_1"] == trajectory_1
                    and feedback["trajectory_2"] == trajectory_2
                ):
                    return jsonify(
                        {
                            "status": "error",
                            "message": f"Feedback already exists for pair: ({trajectory_1}, {trajectory_2}) in run: {run_name}.",
                        }
                    ), 400

            feedback_buffers[run_name].append(data)

            # Remove the pair from video stack if it exists
            if run_name in sampled_videos:
                video_stack = sampled_videos[run_name]
                try:
                    video_stack.remove(pair_to_remove)
                except ValueError:
                    logging.warning(
                        f"Pair {pair_to_remove} not found in queue for run: {run_name}"
                    )

        return (
            jsonify(
                {
                    "status": "success",
                    "message": f"Feedback received and pair removed for run: {run_name}.",
                }
            ),
            200,
        )
    else:
        return jsonify({"status": "error", "message": "Invalid feedback format"}), 400


@app.route("/api/get_feedback/<run_name>", methods=["GET"])
def get_feedback(run_name):
    """
    Endpoint to fetch accumulated feedback data for a given environment ID (env_id).
    """
    with lock:
        if run_name in feedback_buffers:
            try:
                buffer_copy = feedback_buffers[run_name][:]
                return jsonify(buffer_copy), 200
            finally:
                feedback_buffers[run_name].clear()
        else:
            return (
                jsonify(
                    {
                        "status": "error",
                        "message": f"No feedback buffer found for env_id {run_name}.",
                    }
                ),
                404,
            )


@app.route("/favicon.ico")
def favicon():
    return send_from_directory(
        os.path.join(app.root_path, "static"),
        "favicon.ico",
        mimetype="image/vnd.microsoft.icon",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Feedback Server")
    parser.add_argument(
        "--port",
        type=int,
        default=5001,
        help="Port to run the feedback server on (default: 5001)",
    )
    args = parser.parse_args()
    logging.getLogger("werkzeug").setLevel(logging.ERROR)
    app.run(debug=True, port=args.port)

import threading

from flask import Flask, jsonify, render_template, request, send_from_directory
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

VIDEO_STORAGE = "../videos"
sampled_videos = {}
feedback_buffers = {}
lock = threading.Lock()


@app.route("/")
def index():
    """Serve the web interface to view and compare videos."""
    return render_template("index.html")


@app.route("/videos/<filename>")
def fetch_video(filename):
    """Endpoint to serve the video files."""
    return send_from_directory(VIDEO_STORAGE, filename)


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
                "total_pairs": len(
                    sampled_videos[env_id]
                ),  # Optional: Return total pairs in stack
            }
        ),
        200,
    )


@app.route("/api/get_videos", methods=["GET"])
def get_videos():
    """
    Endpoint to get the next video filenames for comparison.
    Searches through all env_ids in sampled_videos and serves the next available pair.
    """
    with lock:
        for env_id, video_stack in sampled_videos.items():
            if video_stack:
                video_pair = video_stack.pop(0)
                if len(video_pair) == 2:
                    return (
                        jsonify(
                            {
                                "video1": f"/videos/{video_pair[0]}",
                                "video2": f"/videos/{video_pair[1]}",
                                "env_id": env_id,  # currently not used but could be displayed in the ui for clarification
                            }
                        ),
                        200,
                    )
        return (
            jsonify(
                {"status": "error", "message": "No videos available in any environment"}
            ),
            404,
        )


@app.route("/api/submit_feedback/<env_id>", methods=["POST"])
def submit_feedback(env_id):
    """
    Endpoint to collect user feedback for a specific environment ID (env_id).
    Expects: { "trajectory_1": ..., "trajectory_2": ..., "preference": ... }
    """
    data = request.json
    if "trajectory_1" in data and "trajectory_2" in data and "preference" in data:
        with lock:
            if env_id not in feedback_buffers:
                feedback_buffers[env_id] = []

            feedback_buffers[env_id].append(data)

        return (
            jsonify(
                {
                    "status": "success",
                    "message": f"Feedback received for env_id {env_id}!",
                }
            ),
            200,
        )
    else:
        return jsonify({"status": "error", "message": "Invalid feedback format"}), 400


@app.route("/api/get_feedback/<env_id>", methods=["GET"])
def get_feedback(env_id):
    """
    Endpoint to fetch accumulated feedback data for a given environment ID (env_id).
    """
    with lock:
        if env_id in feedback_buffers:
            buffer_copy = feedback_buffers[env_id][:]
            feedback_buffers[env_id].clear()
            return jsonify(buffer_copy), 200
        else:
            return (
                jsonify(
                    {
                        "status": "error",
                        "message": f"No feedback buffer found for env_id {env_id}.",
                    }
                ),
                404,
            )


if __name__ == "__main__":
    app.run(debug=True, port=5000)

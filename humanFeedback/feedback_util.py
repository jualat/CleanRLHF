import json
import logging
from subprocess import Popen


# TODO start feedback server at the start of training or manual?
def start_feedback_server():
    """
    Start the feedback server as a separate process/thread.
    Example:
    --------
    server_process = start_feedback_server()
    """
    logging.info("Starting feedback server...")
    server_process = Popen(["python3", "feedback_server.py"])

    return server_process


def stop_feedback_server(server_process):
    """
    Stop the feedback server gracefully.
    Example:
    --------
    stop_feedback_server(server_process)
    """
    logging.info("Stopping feedback server...")
    server_process.terminate()
    server_process.wait()
    logging.info("Feedback server stopped.")


def save_feedback_to_file(feedback_data, filename="feedback.json"):
    with open(filename, "w") as f:
        json.dump(feedback_data, f)
    print(f"Saved {len(feedback_data)} feedback items to {filename}")
